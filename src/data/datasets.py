from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import tonic
import tonic.transforms as transforms
from torch.utils.data import DataLoader, Dataset, Subset
import lightning.pytorch as pl

from esn.augmentations import AudioPad, AudioTransform, FrameTransform  # pyright: ignore[reportMissingImports]
from braille_dataset import Braille, RandomOffsetPad # pyright: ignore[reportMissingImports]

from .task_config import TaskConfig


class _DataModule(pl.LightningDataModule):
    task_name: str = "task"

    def __init__(
        self,
        cfg: TaskConfig,
        num_workers: int = 4,
        cv_folds: int = 0,
        cv_fold_idx: int = 0,
    ):
        super().__init__()
        self.cfg = cfg
        self.num_workers = num_workers
        self.cv_folds = cv_folds
        self.cv_fold_idx = cv_fold_idx
        # populated in setup()
        self.train_set: Optional[Dataset] = None
        self.val_set: Optional[Dataset] = None
        self.test_set: Optional[Dataset] = None

    # Subclasses override these ----------------------------------------------

    def setup(self, stage: Optional[str] = None) -> None:
        raise NotImplementedError

    def _train_collate(self) -> Callable:
        raise NotImplementedError

    def _val_collate(self) -> Callable:
        raise NotImplementedError

    # DataLoader factories ---------------------------------------------------

    def train_dataloader(self) -> DataLoader:
        assert self.train_set is not None
        return DataLoader(
            self.train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            collate_fn=self._train_collate(),
            num_workers=self.num_workers,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        assert self.val_set is not None
        return DataLoader(
            self.val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=self._val_collate(),
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
        )

    def test_dataloader(self) -> DataLoader:
        assert self.test_set is not None
        return DataLoader(
            self.test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            collate_fn=self._val_collate(),
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,  # final eval -> don't throw away the tail batch
        )


class _SubjectSplitDataModule(_DataModule):
    """Subject-stratified train/val carved from the official train pool, plus official test.

    Subclasses must provide:
        * ``_build_dataset(train: bool, train_augment: bool)`` -> tonic dataset
            instance with the appropriate transform applied. ``train`` selects
            the official split; ``train_augment`` toggles augmentations.
        * ``_subject_ids(dataset)`` -> per-sample subject/speaker ids of the
            given dataset.
        * ``_train_collate()`` / ``_val_collate()`` -> collate fns.

    ``split_seed`` is intentionally separate from the user's global ``--seed``:
    the subject split should stay fixed across model-init seeds so different
    training runs are comparable on the same validation subjects.
    """

    def __init__(
        self,
        cfg: TaskConfig,
        num_workers: int = 4,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        cv_folds: int = 0,
        cv_fold_idx: int = 0,
    ):
        super().__init__(cfg, num_workers, cv_folds, cv_fold_idx)
        self.val_fraction = val_fraction
        self.split_seed = split_seed

    # Subclasses override these ----------------------------------------------

    def _build_dataset(self, train: bool, train_augment: bool) -> Dataset:
        raise NotImplementedError

    def _subject_ids(self, dataset) -> Sequence[int]:
        raise NotImplementedError

    # ------------------------------------------------------------------------

    @staticmethod
    def split_subjects_by_ratio(
        subjects: Sequence[int],
        val_fraction: float,
        seed: int,
    ) -> Tuple[List[int], List[int], float]:
        """Hold out ``floor(n * val_fraction)`` whole subjects for val."""
        unique = sorted({int(s) for s in subjects})
        n = len(unique)
        n_val = int(n * val_fraction)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n).tolist()
        val_subjects = sorted(unique[i] for i in perm[:n_val])
        train_subjects = sorted(unique[i] for i in perm[n_val:])
        return train_subjects, val_subjects, n_val / n

    @staticmethod
    def split_subjects_kfold(
        subjects: Sequence[int],
        k: int,
        fold_idx: int,
        seed: int,
    ) -> Tuple[List[int], List[int], float]:
        """k-fold subject split; fold ``fold_idx`` is val, the rest train."""
        unique = sorted({int(s) for s in subjects})
        n = len(unique)
        rng = np.random.default_rng(seed)
        perm = rng.permutation(n).tolist()
        base, rem = divmod(n, k)
        sizes = [base + (1 if i < rem else 0) for i in range(k)]
        starts = np.cumsum([0] + sizes).tolist()
        val_positions = perm[starts[fold_idx] : starts[fold_idx + 1]]
        val_set = {unique[i] for i in val_positions}
        val_subjects = sorted(val_set)
        train_subjects = sorted(s for s in unique if s not in val_set)
        return train_subjects, val_subjects, len(val_subjects) / n

    @staticmethod
    def _indices_for_subjects(subjects_per_sample: Sequence[int], keep: Sequence[int]) -> List[int]:
        keep_set = {int(s) for s in keep}
        return [i for i, s in enumerate(subjects_per_sample) if int(s) in keep_set]

    @staticmethod
    def _speaker_disjoint_indices(
        subjects_per_sample: Sequence[int],
        train_subjects: Sequence[int],
        val_subjects: Sequence[int],
    ) -> Tuple[List[int], List[int]]:
        """Speaker-disjoint val: whole held-out subjects only."""
        train_idx = _SubjectSplitDataModule._indices_for_subjects(subjects_per_sample, train_subjects)
        val_idx = _SubjectSplitDataModule._indices_for_subjects(subjects_per_sample, val_subjects)
        return train_idx, val_idx

    @staticmethod
    def _log_val_split(
        task: str,
        subjects: Sequence[int],
        train_subjects: Sequence[int],
        val_subjects: Sequence[int],
        actual_ratio: float,
        mode: str,
        n_val: int,
        n_speaker_val: int,
        n_tail: int,
    ) -> None:
        n = len({int(s) for s in subjects})
        print(
            f"[data:{task}] {mode} split: "
            f"{len(train_subjects)} train / {len(val_subjects)} held-out subjects out of {n} "
            f"(subject val ratio = {actual_ratio:.3f})."
        )
        if n_tail > 0:
            print(
                f"[data:{task}]   val samples: {n_val} ({n_speaker_val} held-out-subject + {n_tail} train-subject tails)"
            )
        else:
            print(f"[data:{task}]   val samples: {n_val}")


    def _compute_split(self, subjects: Sequence[int]) -> Tuple[List[int], List[int], float, str]:
        if self.cv_folds and self.cv_folds > 1:
            mode = f"k-fold(k={self.cv_folds}, fold={self.cv_fold_idx})"
            train_s, val_s, ratio = self.split_subjects_kfold(
                subjects, self.cv_folds, self.cv_fold_idx, self.split_seed,
            )
        else:
            mode = f"ratio(target={self.val_fraction:.3f})"
            train_s, val_s, ratio = self.split_subjects_by_ratio(
                subjects, self.val_fraction, self.split_seed,
            )
        return train_s, val_s, ratio, mode

    def _build_sample_indices(
        self,
        subjects: Sequence[int],
        train_subjects: Sequence[int],
        val_subjects: Sequence[int],
    ) -> Tuple[List[int], List[int], int, int, float]:
        """Map subject split to per-sample train/val indices.

        Returns ``(train_idx, val_idx, n_speaker_val, n_tail)``.
        """
        train_idx, val_idx = self._speaker_disjoint_indices(
            subjects, train_subjects, val_subjects,
        )
        return train_idx, val_idx, len(val_idx), 0

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            if self.train_set is None or self.val_set is None:
                train_augmented = self._build_dataset(train=True, train_augment=True)
                train_deterministic = self._build_dataset(train=True, train_augment=False)

                subjects = self._subject_ids(train_augmented)
                train_subjects, val_subjects, ratio, mode = self._compute_split(subjects)

                train_idx, val_idx, n_speaker_val, n_tail = (
                    self._build_sample_indices(subjects, train_subjects, val_subjects)
                )
                self._log_val_split(
                    self.task_name,
                    subjects,
                    train_subjects,
                    val_subjects,
                    ratio,
                    mode,
                    len(val_idx),
                    n_speaker_val,
                    n_tail,
                )

                self.train_set = Subset(train_augmented, train_idx)
                self.val_set = Subset(train_deterministic, val_idx)

        if stage in (None, "test"):
            if self.test_set is None:
                self.test_set = self._build_dataset(train=False, train_augment=False)


# Notes on the SHD split
# - tonic.datasets.SHD(train=False) contains all 12 speakers — speakers 4 and 5 make up ~81% of test (they're held out from training entirely), and the other 10 have a small tail of test samples each. That's the dataset's own convention, we use it as-is.
# - SHD overrides ``_build_sample_indices`` to mix held-out subjects with per-subject tails (~5.5% each) so val tracks official test (~81% unseen / ~19% same-subject tail).
class SHDDataModule(_SubjectSplitDataModule):
    task_name = "shd"

    def __init__(
        self,
        cfg: TaskConfig,
        num_workers: int = 4,
        val_fraction: float = 0.2,
        split_seed: int = 42,
        cv_folds: int = 0,
        cv_fold_idx: int = 0,
        val_tail_fraction: float = 0.055, # ~18.7% tail share in val (matches official SHD test)
    ):
        super().__init__(cfg, num_workers, val_fraction, split_seed, cv_folds, cv_fold_idx)
        self.val_tail_fraction = val_tail_fraction


    def _val_tail_indices(self,
        subjects_per_sample: Sequence[int],
        train_subjects: Sequence[int],
    ) -> List[int]:
        """Hold out a per-subject sample fraction from train subjects (val tail)."""
        rng = np.random.default_rng(self.split_seed + 1)
        tail_idx: List[int] = []
        for subject in sorted({int(s) for s in train_subjects}):
            idx = [i for i, s in enumerate(subjects_per_sample) if int(s) == subject]
            if not idx:
                continue
            n_tail = max(1, int(len(idx) * self.val_tail_fraction))
            chosen = rng.choice(idx, size=n_tail, replace=False).tolist()
            tail_idx.extend(chosen)
        return sorted(tail_idx)


    def _build_train_val_indices(self,
        subjects_per_sample: Sequence[int],
        train_subjects: Sequence[int],
        val_subjects: Sequence[int],
    ) -> Tuple[List[int], List[int]]:
        """Held-out subjects + per-subject sample tails. Mirrors the official SHD test mix."""
        speaker_val_idx = self._indices_for_subjects(subjects_per_sample, val_subjects)
        train_idx = self._indices_for_subjects(subjects_per_sample, train_subjects)
        tail_idx = self._val_tail_indices(subjects_per_sample, train_subjects)
        tail_set = set(tail_idx)
        train_idx = [i for i in train_idx if i not in tail_set]
        val_idx = sorted(set(speaker_val_idx) | tail_set)
        return train_idx, val_idx

    def _build_sample_indices(self,
        subjects: Sequence[int],
        train_subjects: Sequence[int],
        val_subjects: Sequence[int],
    ) -> Tuple[List[int], List[int], int, int, float]:
        train_idx, val_idx = self._build_train_val_indices(
            subjects, train_subjects, val_subjects,
        )
        n_speaker_val = len(self._indices_for_subjects(subjects, val_subjects))
        n_tail = len(val_idx) - n_speaker_val
        return train_idx, val_idx, n_speaker_val, n_tail

    def _transform(self, train_augment: bool):
        extra = self.cfg.extra
        all_transforms = []
        if train_augment and extra.get("time_jitter", 0) > 0:
            all_transforms.append(
                transforms.TimeJitter(std=extra["time_jitter"], clip_negative=True)
            )
        if train_augment and extra.get("event_dropping", 0) > 0:
            all_transforms.append(transforms.DropEvent(p=extra["event_dropping"]))
        all_transforms.append(AudioTransform(
            original_sensor_size=[700, 1, 1],
            desired_sensor_size=extra["desired_sensor_size"],
            dt=extra["dt"],
            random_time_scale=extra.get("random_time_scale", [1.0, 1.0]) if train_augment else [1.0, 1.0],
            squeeze_thresh=extra.get("squeeze_thresh", 1),
        ))
        return transforms.Compose(all_transforms)

    def _build_dataset(self, train: bool, train_augment: bool) -> Dataset:
        return tonic.datasets.SHD(
            save_to=self.cfg.dataset_path,
            train=train,
            transform=self._transform(train_augment=train_augment),
        )

    def _subject_ids(self, dataset) -> Sequence[int]:
        return np.asarray(dataset.speaker).tolist()

    def _train_collate(self):
        return AudioPad(batch_first=False, noise=self.cfg.extra.get("noise", 0.0))

    def _val_collate(self):
        return AudioPad(batch_first=False, noise=0.0)


class DVSGestureDataModule(_SubjectSplitDataModule):
    task_name = "dvsgesture"

    def _transform(self, train_augment: bool):
        extra = self.cfg.extra
        all_transforms = []
        if train_augment and extra.get("time_jitter", 0) > 0:
            all_transforms.append(
                transforms.TimeJitter(std=extra["time_jitter"], clip_negative=True)
            )
        if train_augment and extra.get("event_dropping", 0) > 0:
            all_transforms.append(transforms.DropEvent(p=extra["event_dropping"]))
        all_transforms.append(FrameTransform(
            original_sensor_size=tonic.datasets.DVSGesture.sensor_size,
            desired_sensor_size=extra["desired_sensor_size"],
            dt=extra["dt"],
            n_steps=extra["n_steps"],
            random_start_offset=extra.get("random_start_offset", False) if train_augment else False,
            noise=extra.get("noise", 0.0) if train_augment else 0.0,
            random_time_scale=extra.get("random_time_scale", [1.0, 1.0]) if train_augment else [1.0, 1.0],
            random_image_scale=extra.get("random_image_scale", [1.0, 1.0]) if train_augment else [1.0, 1.0],
            random_image_offset=extra.get("random_image_offset", [0.0, 0.0]) if train_augment else [0.0, 0.0],
        ))
        return transforms.Compose(all_transforms)

    def _build_dataset(self, train: bool, train_augment: bool) -> Dataset:
        return tonic.datasets.DVSGesture(
            save_to=self.cfg.dataset_path,
            train=train,
            transform=self._transform(train_augment=train_augment),
        )

    def _subject_ids(self, dataset) -> Sequence[int]:
        return np.asarray(dataset.users).tolist()

    def _train_collate(self):
        return tonic.collation.PadTensors(batch_first=False)

    def _val_collate(self):
        return tonic.collation.PadTensors(batch_first=False)


class BrailleDataModule(_DataModule):
    task_name = "braille"

    def __init__(self, 
        cfg: TaskConfig, 
        num_workers: int = 4, 
        cv_folds: int = 0, 
        cv_fold_idx: int = 0
    ):
        super().__init__(cfg, num_workers, cv_folds, cv_fold_idx)
        assert cv_folds in (0, 1, 5), "Braille dataset currently only supports 5-fold cross validation"
        assert cv_fold_idx == 0 or (0 <= cv_fold_idx < cv_folds)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit", "validate"):
            if self.train_set is None or self.val_set is None:
                self.train_set = self._build_dataset(split=f"train{self.cv_fold_idx + 1}", train_augment=True)
                self.val_set = self._build_dataset(split=f"val{self.cv_fold_idx + 1}", train_augment=False)
        if stage in (None, "test"):
            if self.test_set is None:
                self.test_set = self._build_dataset(split="test", train_augment=False)

    def _transform(self, train_augment: bool):
        extra = self.cfg.extra
        all_transforms = []
        if train_augment and extra.get("random_time_scale", (1.0, 1.0)) != (1.0, 1.0):
            all_transforms.append(transforms.TimeSkew(coefficient=extra["random_time_scale"]))
        if train_augment and extra.get("time_jitter", 0) != 0:
            all_transforms.append(transforms.TimeJitter(std=extra["time_jitter"], clip_negative=True))
        if train_augment and extra.get("event_dropping", 0) != 0:
            all_transforms.append(transforms.DropEvent(p=extra["event_dropping"]))
        if train_augment and extra.get("noise", 0) > 0:
            noise_absolute = self._sample_uniform_noise()
            all_transforms.append(transforms.UniformNoise(Braille.sensor_size, noise_absolute))
        all_transforms.append(
            transforms.ToFrame(sensor_size=Braille.sensor_size, n_time_bins=extra["n_steps"])
        )
        return transforms.Compose(all_transforms)

    def _sample_uniform_noise(self) -> int:
        extra = self.cfg.extra
        p = float(extra.get("noise", 0))
        means = extra.get("mean_events_per_sample", 0)
        threshold = int(extra["threshold"])
        mean_ev = float(means[threshold])
        return int(round(p * mean_ev))

    def _build_dataset(self, split: str, train_augment: bool) -> Dataset:
        return Braille(
            save_to=self.cfg.dataset_path,
            threshold=self.cfg.extra["threshold"],
            split=split,
            transform=self._transform(train_augment),
        )

    def _train_collate(self):
        if self.cfg.extra.get("random_start_offset", False):
            return RandomOffsetPad(n_time_bins=self.cfg.extra.get("n_steps", 128), batch_first=False)
        else:
            return tonic.collation.PadTensors(batch_first=False)

    def _val_collate(self):
        return tonic.collation.PadTensors(batch_first=False)


def build_datamodule(
    cfg: TaskConfig,
    num_workers: int,
    val_fraction: float = 0.2,
    split_seed: int = 42,
    cv_folds: int = 0,
    cv_fold_idx: int = 0,
) -> pl.LightningDataModule:
    """Factory for the per-task LightningDataModule."""
    if cfg.name == "shd":
        return SHDDataModule(cfg, num_workers, val_fraction, split_seed, cv_folds, cv_fold_idx)
    if cfg.name == "dvsgesture":
        return DVSGestureDataModule(cfg, num_workers, val_fraction, split_seed, cv_folds, cv_fold_idx)
    if cfg.name == "braille":
        return BrailleDataModule(cfg, num_workers, cv_folds=cv_folds, cv_fold_idx=cv_fold_idx)
    raise ValueError(f"Unsupported task: {cfg.name}")
