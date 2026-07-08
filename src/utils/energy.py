"""Hardware energy estimates for ESN spiking neurons in SNN evaluation."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import yaml

from src.data.task_config import TASK_CONFIGS
from src.hpo.hpo_params import load_hpo_yaml
from src.utils.test_framing import test_neuron_timesteps_from_run_info

from esn.neuron_classes import PMSN_CLR  # pyright: ignore[reportMissingImports]
from esn.neurons.clr import (  # pyright: ignore[reportMissingImports]
    LIFBox,
    N1D1,
    N1D2,
    N1D3,
    N2D1,
    N2D2,
    N2D3,
    N3D1,
    N3D2,
)


@dataclass(frozen=True)
class NeuronEnergyConstants:
    energy_idle: float
    energy_spike: float


_NEURON_CLASSES: dict[str, Callable[[], PMSN_CLR]] = {
    "n1d1": N1D1,
    "n1d2": N1D2,
    "n1d3": N1D3,
    "n2d1": N2D1,
    "n2d2": N2D2,
    "n2d3": N2D3,
    "n3d1": N3D1,
    "n3d2": N3D2,
    "esn_lifbox": LIFBox,
}


def get_supported_neurons() -> list[str]:
    return sorted(_NEURON_CLASSES)


def get_neuron_energy(neuron_name: str) -> NeuronEnergyConstants:
    """Return per-timestep idle/spike energy (pJ) for a registered ESN CLR neuron."""
    factory = _NEURON_CLASSES.get(neuron_name)
    if factory is None:
        raise ValueError(
            f"No hardware energy map for neuron {neuron_name!r}. "
            f"Supported: {sorted(_NEURON_CLASSES)}"
        )
    cell = factory()
    idle = float(cell.energy_idle)
    spike = float(cell.energy_spike)
    return NeuronEnergyConstants(energy_idle=idle, energy_spike=spike)


PJ_PER_UJ = 1_000_000.0


def pj_to_uj(energy_pj: float) -> float:
    return float(energy_pj) / PJ_PER_UJ


def per_neuron_energy_per_step(
    spike_rate: float,
    energy_idle: float,
    energy_spike: float,
    *,
    ignore_idle_energy: bool = True,
) -> float:
    """Energy in pJ for one hidden neuron update at mean spike rate in [0, 1]."""
    r = float(spike_rate)
    if ignore_idle_energy:
        return r * float(energy_spike)
    return (1.0 - r) * float(energy_idle) + r * float(energy_spike)


def network_energy_per_sample(
    *,
    n_hidden_neurons: int,
    n_timesteps: int,
    spike_rate: float,
    energy_idle: float,
    energy_spike: float,
    ignore_idle_energy: bool = True,
) -> float:
    """Total hidden-neuron energy in pJ for one inference (one test sample)."""
    e_step = per_neuron_energy_per_step(
        spike_rate,
        energy_idle,
        energy_spike,
        ignore_idle_energy=ignore_idle_energy,
    )
    return int(n_hidden_neurons) * int(n_timesteps) * e_step


def _load_run_info(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError:
        return {}
    return data if isinstance(data, dict) else {}


def resolve_run_energy_config(run_info_path: Path) -> dict[str, Any]:
    """Infer network topology and timestep count from a ``run_info.yaml`` file."""
    info = _load_run_info(run_info_path)
    args = info.get("args") if isinstance(info.get("args"), dict) else {}

    task = args.get("task")
    neuron = args.get("neuron")
    n_hidden_layers = int(args.get("n_hidden_layers") or 2)

    hidden_size = args.get("hidden_size")
    n_timesteps: Optional[int] = None
    if task and task in TASK_CONFIGS:
        cfg = TASK_CONFIGS[task]()
        if hidden_size is None:
            hidden_size = cfg.default_hidden_size
        raw_steps = cfg.extra.get("n_steps")
        if raw_steps is not None:
            n_timesteps = int(raw_steps)

    hpo_path = args.get("hpo_params")
    if hpo_path:
        params = load_hpo_yaml(hpo_path).get("params") or {}
        if isinstance(params, dict):
            if params.get("n_steps") is not None:
                n_timesteps = int(params["n_steps"])

    n_hidden_neurons = None
    if hidden_size is not None:
        n_hidden_neurons = int(hidden_size) * n_hidden_layers

    test_set_size: Optional[int] = None
    test_neuron_timesteps: Optional[int] = None
    if task and task in TASK_CONFIGS:
        raw_size = TASK_CONFIGS[task]().extra.get("test_set_size")
        if raw_size is not None:
            test_set_size = int(raw_size)

    if n_hidden_neurons is not None:
        test_neuron_timesteps = test_neuron_timesteps_from_run_info(
            run_info_path,
            n_hidden_neurons=int(n_hidden_neurons),
        )
        if test_neuron_timesteps is not None and test_set_size is None:
            n_steps = n_timesteps
            if n_steps is not None and n_steps > 0:
                test_set_size = test_neuron_timesteps // (int(n_hidden_neurons) * int(n_steps))

    return {
        "task": task,
        "neuron": neuron,
        "hidden_size": int(hidden_size) if hidden_size is not None else None,
        "n_hidden_layers": n_hidden_layers,
        "n_hidden_neurons": n_hidden_neurons,
        "n_timesteps": n_timesteps,
        "test_set_size": test_set_size,
        "test_neuron_timesteps": test_neuron_timesteps,
    }


def energy_test_set_pj(
    *,
    total_spikes: float,
    neuron: str,
    ignore_idle_energy: bool = True,
) -> Optional[float]:
    """Total hidden-layer spike energy in pJ for a full test-set forward pass."""
    try:
        constants = get_neuron_energy(neuron)
    except ValueError:
        return None
    if not ignore_idle_energy:
        return None
    return float(total_spikes) * constants.energy_spike


def energy_test_set_for_run(
    *,
    config: dict[str, Any],
    metrics: dict[str, float],
    ignore_idle_energy: bool = True,
) -> Optional[float]:
    """Return hidden-network test-set energy in pJ from ``test_spike_rate`` and offline framing."""
    neuron = config.get("neuron")
    spike_rate = metrics.get("test_spike_rate")
    neuron_timesteps = config.get("test_neuron_timesteps")
    if neuron is None or spike_rate is None or neuron_timesteps is None:
        return None
    total_spikes = float(spike_rate) * float(neuron_timesteps)
    return energy_test_set_pj(
        total_spikes=total_spikes,
        neuron=str(neuron),
        ignore_idle_energy=ignore_idle_energy,
    )


def energy_per_sample_for_run(
    *,
    config: dict[str, Any],
    spike_rate: float,
    ignore_idle_energy: bool = True,
) -> Optional[float]:
    """Return hidden-network energy (pJ) for one test sample, or None if unknown."""
    neuron = config.get("neuron")
    n_hidden = config.get("n_hidden_neurons")
    n_steps = config.get("n_timesteps")
    if neuron is None or n_hidden is None or n_steps is None or n_steps <= 0:
        return None
    try:
        energy = get_neuron_energy(str(neuron))
    except ValueError:
        return None
    return network_energy_per_sample(
        n_hidden_neurons=int(n_hidden),
        n_timesteps=int(n_steps),
        spike_rate=float(spike_rate),
        energy_idle=energy.energy_idle,
        energy_spike=energy.energy_spike,
        ignore_idle_energy=ignore_idle_energy,
    )
