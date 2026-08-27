"""Shared, platform-independent code for the fish detection project."""

from .device import DeviceInfo, resolve_device, suggested_batch
from .paths import DATA_ROOT, REPO_ROOT, RUNS_ROOT, ensure_dirs, free_gb, rel, require_free_gb

__all__ = [
    "DATA_ROOT", "REPO_ROOT", "RUNS_ROOT",
    "DeviceInfo", "resolve_device", "suggested_batch",
    "ensure_dirs", "free_gb", "rel", "require_free_gb",
]
