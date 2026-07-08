from __future__ import annotations

import ast


def ensure_single_device(devices: object, *, raw: str | None = None) -> object:
    """Reject multi-GPU / DDP device specs (e.g. from --select-gpus)."""
    if devices == 1 or (isinstance(devices, list) and len(devices) == 1):
        return devices
    label = f"'{raw}'" if raw is not None else repr(devices)
    raise ValueError(f"{label}: exactly one GPU required (--devices 1 or --devices [N]).")


def parse_devices_arg(raw: str) -> object:
    """Parse --devices as ``1`` or ``[N]`` (single GPU)."""
    value = str(raw).strip()
    if value == "1":
        return 1
    if value.startswith("[") and value.endswith("]"):
        parsed = ast.literal_eval(value)
        if not isinstance(parsed, (list, tuple)) or len(parsed) != 1:
            raise ValueError(f"'{raw}': exactly one GPU required (--devices 1 or --devices [N]).")
        return [int(parsed[0])]
    raise ValueError(f"'{raw}': invalid --devices; use 1 or [N].")
