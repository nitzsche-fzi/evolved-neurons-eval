from dataclasses import dataclass, field
from typing import Callable, Optional


@dataclass
class TaskConfig:
    name: str
    input_size: int
    n_classes: int
    inp_mean: float
    inp_var: float
    # `esn_surrogate_*` values mirror the GA task configs and are applied only
    # to esn neurons (`atan_derivative` isn't a Norse surrogate name).
    esn_surrogate_method: str = "atan_derivative"
    esn_surrogate_alpha: float = 1.0
    # User-explicit overrides from the CLI, applied to every neuron type.
    surrogate_method: Optional[str] = None
    surrogate_alpha: Optional[float] = None
    lr: float = 1e-3
    weight_decay: float = 1e-5
    grad_clip_norm: float = 10.0
    spike_rate_reg: float = 1e-4
    # Optional LR scheduling.
    lr_scheduler: str = "cosine"  # "none" | "cosine"
    lr_min_factor: float = 0.1  # cosine eta_min = lr * factor
    lr_warmup_epochs: int = 5
    lr_warmup_start_factor: float = 0.1
    batch_size: int = 32
    n_epochs: int = 10
    default_hidden_size: int = 256
    # populated by subclasses
    dataset_path: str = "/path/to/datasets"
    extra: dict = field(default_factory=dict)


def _shd_config() -> TaskConfig:
    return TaskConfig(
        name="shd",
        input_size=350,
        n_classes=20,
        inp_mean=0.876514,
        inp_var=1.60399,
        spike_rate_reg=5.0,
        batch_size=128,
        n_epochs=30,
        default_hidden_size=256,
        extra={
            "desired_sensor_size": [350, 1, 1],
            "dt": 10,
            "test_set_size": 2264,
            "time_jitter": 4128.170910580887,
            "noise": 0.0003954653996250096,
            # Augmentation magnitudes the per-neuron HPO co-tunes.
            "use_augmentations": ("time_jitter", "noise"),
        },
    )


def _dvs_config() -> TaskConfig:
    return TaskConfig(
        name="dvsgesture",
        input_size=32 * 32 * 2,
        n_classes=11,
        inp_mean=1.02272,
        inp_var=2.1217,
        lr=1e-3,
        spike_rate_reg=5.0,
        batch_size=128,
        n_epochs=150,
        default_hidden_size=256,
        extra={
            "desired_sensor_size": [32, 32, 2],
            "dt": 8000,
            "n_steps": 200,
            "test_set_size": 264,
            "random_time_scale": (0.97, 1.03),
            "random_image_scale": [0.9978, 1.0023],
            "random_image_offset": [-0.0654, 0.0654],
            # Augmentation magnitudes the per-neuron HPO co-tunes.
            "use_augmentations": ("random_time_scale", "random_image_scale", "random_image_offset",),
        },
    )


def _braille_config() -> TaskConfig:
    return TaskConfig(
        name="braille",
        input_size=12 * 1 * 2,
        n_classes=27,
        inp_mean=3.94081, # Recalculate for changes in n_steps, threshold
        inp_var=34.8037,
        lr=1e-3,
        spike_rate_reg=5.0,
        batch_size=128,
        n_epochs=150,
        default_hidden_size=256,
        extra={
            "n_steps": 64, # effective dt is 1,275s / n_steps
            "test_set_size": 810,
            "threshold": 1, # forwarded to dataset for event sampling, (1,2,5,10)
            "mean_events_per_sample": {1: 1028.64, 2: 446.35, 5: 124.28, 10: 40.37},
            "time_jitter": 8409.605884014742, # STD
            # Augmentation magnitudes the per-neuron HPO co-tunes.
            "use_augmentations": ("time_jitter",),
        },
    )


TASK_CONFIGS: dict[str, Callable[[], TaskConfig]] = {
    "shd": _shd_config,
    "dvsgesture": _dvs_config,
    "braille": _braille_config,
}

