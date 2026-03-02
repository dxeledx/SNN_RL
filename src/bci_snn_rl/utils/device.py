from __future__ import annotations

from typing import Any


def _normalize_device_str(device: str) -> str:
    device = str(device).strip().lower()
    if device in {"auto", "cuda", "cpu"}:
        return device
    if device.startswith("cuda:"):
        return device
    return device


def normalize_devices_in_cfg(cfg: dict[str, Any]) -> dict[str, Any]:
    """
    Normalize `project.device`:
      - auto -> cuda if available else cpu
      - cuda -> cpu if cuda unavailable
    """
    project = cfg.setdefault("project", {})
    requested = _normalize_device_str(project.get("device", "auto"))

    try:
        import torch

        cuda_ok = bool(torch.cuda.is_available())
    except Exception:
        cuda_ok = False

    if requested == "auto":
        project["device"] = "cuda" if cuda_ok else "cpu"
    elif requested.startswith("cuda") and (not cuda_ok):
        project["device"] = "cpu"
    else:
        project["device"] = requested

    return cfg

