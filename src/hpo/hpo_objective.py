"""HPO trial selection objective (``val_comp`` and related helpers)."""
from __future__ import annotations

from functools import lru_cache

from src.utils.energy import get_neuron_energy, get_supported_neurons

DEFAULT_SELECTION_LAMBDA: float = 0.5


def selection_lambda_for(override: float | None) -> float:
    if override is not None:
        return float(override)
    return DEFAULT_SELECTION_LAMBDA


@lru_cache(maxsize=1)
def _max_esn_spike_energy() -> float:
    """Max per-timestep spike energy (pJ) across registered ESN CLR neurons."""
    return max(
        get_neuron_energy(name).energy_spike
        for name in get_supported_neurons()
    )


def spike_energy_weight(neuron: str) -> float:
    """Normalize ``energy_spike`` to [0, 1] over all ESN neurons."""
    spike = get_neuron_energy(neuron).energy_spike
    return float(spike) / _max_esn_spike_energy()


def val_comp_score(
    acc: float,
    spike_rate: float,
    lam: float,
    spike_energy_weight: float = 1.0,
) -> float:
    return float(acc) - float(lam) * float(spike_energy_weight) * float(spike_rate)
