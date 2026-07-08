from typing import Callable, List, Optional
from dataclasses import dataclass

import torch
import pytorch_lightning as pl
from torch.optim.lr_scheduler import CosineAnnealingLR
from torchmetrics.classification import MulticlassF1Score

import norse.torch as snn

from esn.initializers.linear import neuromorphic_dense_layer  # pyright: ignore[reportMissingImports]
from esn.neurons import Integrator  # pyright: ignore[reportMissingImports]
from esn.neuron_classes import PMSN_CLR  # pyright: ignore[reportMissingImports]
from esn.surrogate_gradient.Heaviside import get_heaviside  # pyright: ignore[reportMissingImports]
from esn.neurons.clr import LIFBox, N1D1, N1D2, N1D3, N2D1, N2D2, N2D3, N3D1, N3D2  # pyright: ignore[reportMissingImports]

from src.data.datasets import TaskConfig
from src.utils.lr_scheduler import WarmupScheduler

# Readout modules (non-spiking). They all follow the same interface expected by
# `SequentialState`: `out, new_state = module(input, state)`. The ESN Integrator
# needs an explicit zero state, Norse `LICell`/`LIBoxCell` accept `None` state.
READOUT_FACTORIES: dict[str, Callable[[], torch.nn.Module]] = {
    "esn_integrator": Integrator,
    "norse_li": snn.LICell,
    "norse_libox": snn.LIBoxCell,
}

@dataclass
class NeuronSpec:
    """Metadata for a spiking-neuron factory.

    Attributes:
        factory: Callable that produces the `nn.Module` used as the spiking
            non-linearity. Receives a `NeuronBuildCtx` with the CLI / task
            surrogate overrides.
        default_spike_rate: Fallback spike rate used by the neuromorphic
            weight initialiser when the neuron itself does not expose one.
    """
    factory: Callable[["NeuronBuildCtx"], torch.nn.Module]
    default_spike_rate: float

@dataclass
class NeuronBuildCtx:
    # User-explicit CLI overrides; applied to every neuron type when set.
    surrogate_method: Optional[str] = None
    surrogate_alpha: Optional[float] = None
    # Task-level defaults (from the GA pipeline) that only make sense for esn.
    esn_surrogate_method: Optional[str] = None
    esn_surrogate_alpha: Optional[float] = None

    def effective_for_esn(self) -> tuple[Optional[str], Optional[float]]:
        method = self.surrogate_method or self.esn_surrogate_method
        alpha = self.surrogate_alpha if self.surrogate_alpha is not None else self.esn_surrogate_alpha
        return method, alpha


# Neurons we recognise as "spiking hidden" layers (as opposed to the readout).
# Norse cell classes handle a `None` initial state themselves, so we don't need
# to pre-initialise them the way we have to for PMSN_CLR / Integrator.
_SPIKING_TYPES: tuple[type, ...] = (
    PMSN_CLR,
    snn.LIFBoxCell,
    snn.LIFCell,
)

def _build_esn(factory: Callable[[], torch.nn.Module]) -> Callable[[NeuronBuildCtx], torch.nn.Module]:
    """Wrap an esn factory so surrogate overrides are applied uniformly."""
    def build(ctx: NeuronBuildCtx) -> torch.nn.Module:
        neuron = factory()
        method, alpha = ctx.effective_for_esn()
        if isinstance(neuron, PMSN_CLR) and (method is not None or alpha is not None):
            params = neuron.params
            if method is not None:
                params.surrogate_method = method
            if alpha is not None:
                params.surrogate_alpha = float(alpha)
            neuron.threshold_fn = get_heaviside(params.surrogate_method, params.surrogate_alpha)
            neuron.surrogate_alpha = params.surrogate_alpha
        return neuron
    return build

def _build_norse_lifbox(ctx: NeuronBuildCtx) -> torch.nn.Module:
    kwargs: dict = {}
    if ctx.surrogate_method is not None:
        kwargs["p"] = snn.LIFBoxParameters(method=ctx.surrogate_method,
                                           alpha=torch.tensor(ctx.surrogate_alpha or 100.0))
    elif ctx.surrogate_alpha is not None:
        kwargs["p"] = snn.LIFBoxParameters(alpha=torch.tensor(ctx.surrogate_alpha))
    return snn.LIFBoxCell(**kwargs)


def _build_norse_lif(ctx: NeuronBuildCtx) -> torch.nn.Module:
    kwargs: dict = {}
    if ctx.surrogate_method is not None or ctx.surrogate_alpha is not None:
        p_kwargs = {}
        if ctx.surrogate_method is not None:
            p_kwargs["method"] = ctx.surrogate_method
        if ctx.surrogate_alpha is not None:
            p_kwargs["alpha"] = torch.tensor(ctx.surrogate_alpha)
        kwargs["p"] = snn.LIFParameters(**p_kwargs)
    return snn.LIFCell(**kwargs)



_ESN_CLR_SPECS: tuple[tuple[str, type[PMSN_CLR], float], ...] = (
    # fallback default_spike_rate: 0 → neuromorphic init uses neuron.params.spike_rate when > 0
    ("n1d1", N1D1, 0.0),
    ("n1d2", N1D2, 0.0),
    ("n1d3", N1D3, 0.0),
    ("n2d1", N2D1, 0.0),
    ("n2d2", N2D2, 0.0),
    ("n2d3", N2D3, 0.0),
    ("n3d1", N3D1, 0.0),
    ("n3d2", N3D2, 0.0),
    ("esn_lifbox", LIFBox, 0.1),
)

NEURON_SPECS: dict[str, NeuronSpec] = {
    name: NeuronSpec(_build_esn(neuron_cls), default_spike_rate=fallback_sr)
    for name, neuron_cls, fallback_sr in _ESN_CLR_SPECS
} | {
    # Norse neurons
    "norse_lifbox": NeuronSpec(_build_norse_lifbox, default_spike_rate=0.1),
    "norse_lif": NeuronSpec(_build_norse_lif, default_spike_rate=0.1),
}


def _infer_neuron_spike_rate(neuron: torch.nn.Module, fallback: float) -> float:
    rate = getattr(getattr(neuron, "params", None), "spike_rate", None)
    if rate is None or rate <= 0.0:
        return fallback if fallback > 0.0 else 0.1
    return float(rate)


def build_sequential_net(
    neuron: torch.nn.Module,
    input_size: int,
    hidden_size: int,
    n_hidden_layers: int,
    output_size: int,
    inp_mean: float,
    inp_var: float,
    fallback_spike_rate: float,
    readout: str = "esn_integrator",
) -> snn.SequentialState:
    """Input -> [Linear, Spiking] * n_hidden -> Linear -> <readout>.

    The readout is a non-spiking module selected by name from `READOUT_FACTORIES`.
    """
    assert n_hidden_layers >= 1, "need at least one hidden spiking layer"
    if readout not in READOUT_FACTORIES:
        raise ValueError(f"Unknown readout '{readout}'. Expected one of {sorted(READOUT_FACTORIES)}.")
    spike_rate = _infer_neuron_spike_rate(neuron, fallback=fallback_spike_rate)

    modules: List[torch.nn.Module] = []
    prev = input_size
    for i in range(n_hidden_layers):
        if i == 0:
            dense = neuromorphic_dense_layer(prev, hidden_size, inp_mean=inp_mean, inp_var=inp_var)
        else:
            dense = neuromorphic_dense_layer(prev, hidden_size, spike_rate=spike_rate)
        modules.append(dense)
        modules.append(neuron)  # weights of the spiking neuron are shared across layers (same as the GA)
        prev = hidden_size

    modules.append(neuromorphic_dense_layer(prev, output_size, spike_rate=spike_rate))
    modules.append(READOUT_FACTORIES[readout]())
    return snn.SequentialState(*modules)


def initial_state_for(net: snn.SequentialState, batch_size: int, device: torch.device) -> list:
    """Build an explicit initial state list for modules that can't handle `None`.

    `PMSN_CLR.forward` and `Integrator.forward` both dereference `state` on the
    first call, so we must pre-fill those. Norse cells accept `None` and will
    lazily initialise, so we leave those slots as `None`.
    """
    modules = list(net)
    state: list = [None] * len(modules)
    for i, m in enumerate(modules):
        prev = modules[i - 1] if i > 0 else None
        n_features = prev.out_features if isinstance(prev, torch.nn.Linear) else None
        if isinstance(m, PMSN_CLR):
            assert n_features is not None, "PMSN_CLR must follow a Linear layer"
            state[i] = m.init_state(batch_size, n_features).to(device)
        elif isinstance(m, Integrator):
            assert n_features is not None, "Integrator must follow a Linear layer"
            state[i] = torch.zeros(batch_size, n_features, device=device)
    return state



class DenseSNN(pl.LightningModule):
    def __init__(
        self,
        cfg: TaskConfig,
        neuron_name: str,
        hidden_size: int,
        n_hidden_layers: int,
        readout: str = "esn_integrator",
    ):
        super().__init__()
        self.save_hyperparameters({
            "task": cfg.name,
            "neuron": neuron_name,
            "hidden_size": hidden_size,
            "n_hidden_layers": n_hidden_layers,
            "lr": cfg.lr,
            "weight_decay": cfg.weight_decay,
            "grad_clip_norm": cfg.grad_clip_norm,
            "spike_rate_reg": cfg.spike_rate_reg,
            "lr_scheduler": cfg.lr_scheduler,
            "lr_warmup_epochs": cfg.lr_warmup_epochs,
            "lr_warmup_start_factor": cfg.lr_warmup_start_factor,
            "lr_min_factor": cfg.lr_min_factor,
            "batch_size": cfg.batch_size,
            "n_epochs": cfg.n_epochs,
            "surrogate_method": cfg.surrogate_method,
            "surrogate_alpha": cfg.surrogate_alpha,
            "esn_surrogate_method": cfg.esn_surrogate_method,
            "esn_surrogate_alpha": cfg.esn_surrogate_alpha,
            "readout": readout,
        })
        self.cfg = cfg
        self.neuron_name = neuron_name

        spec = NEURON_SPECS[neuron_name]
        self.neuron = spec.factory(NeuronBuildCtx(
            surrogate_method=cfg.surrogate_method,
            surrogate_alpha=cfg.surrogate_alpha,
            esn_surrogate_method=cfg.esn_surrogate_method,
            esn_surrogate_alpha=cfg.esn_surrogate_alpha,
        ))
        self.net = build_sequential_net(
            self.neuron,
            input_size=cfg.input_size,
            hidden_size=hidden_size,
            n_hidden_layers=n_hidden_layers,
            output_size=cfg.n_classes,
            inp_mean=cfg.inp_mean,
            inp_var=cfg.inp_var,
            fallback_spike_rate=spec.default_spike_rate,
            readout=readout,
        )
        self.n_hidden_neurons = hidden_size * n_hidden_layers
        self.f1_metrics = torch.nn.ModuleDict({
            "train_f1": MulticlassF1Score(num_classes=cfg.n_classes, average="macro"),
            "val_f1": MulticlassF1Score(num_classes=cfg.n_classes, average="macro"),
            "test_f1": MulticlassF1Score(num_classes=cfg.n_classes, average="macro"),
        })

        
    # ----- forward over time -----

    def forward_sequence(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Unroll the network over the time dimension.

        Args:
            x: shape [T, B, input_size].
        Returns:
            logits (integrator readout at the last timestep) and total spike count
            across hidden layers (for spike-rate regularisation).
        """
        T, B = x.shape[0], x.shape[1]
        state = initial_state_for(self.net, B, x.device)
        spike_count = torch.zeros((), device=x.device)
        out = None
        spiking_indices = {i for i, m in enumerate(self.net) if isinstance(m, _SPIKING_TYPES)}
        for t in range(T):
            input = x[t]
            for i, module in enumerate(self.net):
                if self.net.stateful_layers[i]:
                    input, s = module(input, state[i])
                    state[i] = s
                else:
                    input = module(input)
                if i in spiking_indices:
                    # inp at this point is the spikes from that hidden layer
                    spike_count = spike_count + input.sum()
            out = input
        assert out is not None
        return out, spike_count

    # ----- step helpers -----

    @staticmethod
    def _prepare_batch(batch) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        x = x.to(dtype=torch.float32)
        x = x.reshape(x.shape[0], x.shape[1], -1)
        return x, y.long()

    def _step(self, batch, stage: str):
        x, y = self._prepare_batch(batch)
        logits, spike_count = self.forward_sequence(x)
        ce = torch.nn.functional.cross_entropy(logits, y)
        n_timesteps = x.shape[0]
        norm_count = spike_count / (self.n_hidden_neurons * n_timesteps * x.shape[1])
        loss = ce + self.cfg.spike_rate_reg * norm_count
        preds = logits.argmax(dim=-1)
        acc = (preds == y).float().mean()
        f1 = self.f1_metrics[f"{stage}_f1"](preds, y)
        batch_size = x.shape[1]
        self.log(f"{stage}_loss", loss, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_ce", ce, prog_bar=False, batch_size=batch_size)
        self.log(f"{stage}_acc", acc, prog_bar=True, batch_size=batch_size)
        self.log(f"{stage}_f1", f1, prog_bar=False, batch_size=batch_size)
        self.log(f"{stage}_spike_rate", norm_count, prog_bar=False, batch_size=batch_size)
        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.cfg.lr, weight_decay=self.cfg.weight_decay)

        scheduler_name = getattr(self.cfg, "lr_scheduler", "none")
        if scheduler_name in (None, "none"):
            return optimizer
            
        min_factor = float(getattr(self.cfg, "lr_min_factor", 0.1))
        if scheduler_name == "cosine":
            cosine_epochs = int(self.cfg.n_epochs)
            scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=self.cfg.lr * min_factor)
        elif scheduler_name == "warmup_cosine":
            warmup_epochs = int(getattr(self.cfg, "lr_warmup_epochs", 0))
            start_factor = float(getattr(self.cfg, "lr_warmup_start_factor", 0.0))
            # Main phase uses cosine decay over the non-warmup tail
            cosine_epochs = max(1, int(self.cfg.n_epochs) - warmup_epochs)
            main_scheduler = CosineAnnealingLR(optimizer, T_max=cosine_epochs, eta_min=self.cfg.lr * min_factor)
            scheduler = WarmupScheduler(optimizer, main_scheduler=main_scheduler,
                warmup_epochs=max(1, warmup_epochs),
                warmup_start_lr=self.cfg.lr * start_factor,
            )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            },
        }

    def configure_gradient_clipping(
        self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None,
    ) -> None:
        # Mirror the GA training loop which uses clip_grad_value_.
        torch.nn.utils.clip_grad_value_(self.parameters(), self.cfg.grad_clip_norm)
