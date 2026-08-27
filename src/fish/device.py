"""Compute-device resolution. Resolve once, here, and log the result.

Never write `torch.device("cuda")` anywhere else: it fails on a CPU-only box
and on Apple silicon, and teammates run on all three.
"""

from __future__ import annotations

import logging
import platform
from dataclasses import dataclass

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    kind: str          # "cuda" | "mps" | "cpu"
    name: str
    total_memory_gb: float | None
    ultralytics: str   # what to pass as Ultralytics' `device=`

    def __str__(self) -> str:
        mem = f", {self.total_memory_gb:.1f} GB" if self.total_memory_gb else ""
        return f"{self.kind} ({self.name}{mem})"


def resolve_device(prefer: str = "auto") -> DeviceInfo:
    """Pick a device: cuda -> mps -> cpu.

    `prefer` may be "auto", "cuda", "mps" or "cpu". An explicit choice that is
    unavailable raises rather than silently falling back to CPU -- a training
    run that quietly drops to CPU wastes days before anyone notices.
    """
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - environment problem
        raise RuntimeError(
            "torch is not installed. Run: python scripts/bootstrap.py"
        ) from exc

    has_cuda = torch.cuda.is_available()
    has_mps = getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available()

    if prefer not in ("auto", "cuda", "mps", "cpu"):
        raise ValueError(f"Unknown device preference: {prefer!r}")

    if prefer == "cuda" and not has_cuda:
        raise RuntimeError(
            "device=cuda was requested but torch.cuda.is_available() is False. "
            "The installed torch is probably a CPU-only build -- check "
            "`python scripts/check_env.py`."
        )
    if prefer == "mps" and not has_mps:
        raise RuntimeError("device=mps was requested but MPS is not available.")

    if prefer == "cpu" or (prefer == "auto" and not has_cuda and not has_mps):
        return DeviceInfo("cpu", platform.processor() or platform.machine(), None, "cpu")

    if (prefer in ("auto", "cuda")) and has_cuda:
        props = torch.cuda.get_device_properties(0)
        return DeviceInfo("cuda", props.name, props.total_memory / (1024**3), "0")

    return DeviceInfo("mps", "Apple Silicon GPU", None, "mps")


def suggested_batch(device: DeviceInfo, imgsz: int = 640, model: str = "yolov8n") -> int:
    """A batch size that should fit, so a teammate's first run does not OOM.

    Deliberately conservative. Override in the config; this is only the default
    when none is given.
    """
    if device.kind != "cuda" or not device.total_memory_gb:
        return 4 if device.kind == "mps" else 2

    scale = {"yolov8n": 1.0, "yolov8s": 0.55, "yolov8m": 0.30}.get(model, 0.5)
    # ~4 images per usable GB for yolov8n at 640, reserving 1.5 GB of headroom.
    usable = max(device.total_memory_gb - 1.5, 1.0)
    batch = int(usable * 4 * scale * (640 / imgsz) ** 2)
    return max(2, min(64, batch))
