"""Machine profiles and run provenance.

A profile is a YAML file under configs/env/ selected by $FISH_ENV. Adding a new
machine means adding a YAML file -- never editing code.
"""

from __future__ import annotations

import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .paths import CONFIG_DIR, DATA_ROOT, REPO_ROOT, free_gb

DEFAULT_PROFILE = "workstation"


def profile_name() -> str:
    return os.environ.get("FISH_ENV") or DEFAULT_PROFILE


def load_profile(name: str | None = None) -> dict[str, Any]:
    """Load configs/env/<name>.yaml. Missing profile is an error, not a default.

    Silently falling back would hide a typo in $FISH_ENV and produce a run
    labelled with the wrong machine.
    """
    name = name or profile_name()
    path = CONFIG_DIR / "env" / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (CONFIG_DIR / "env").glob("*.yaml"))
        raise FileNotFoundError(
            f"No machine profile {name!r} at {path}. "
            f"Available: {', '.join(available) or '(none)'}. "
            f"Set $FISH_ENV to one of these, or add a new YAML file."
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def apply_profile(profile: dict[str, Any]) -> None:
    """Export the profile's cache redirections into the process environment.

    Caches must not land on a full system drive. Doing this here means no
    script hardcodes a cache location and no teammate needs a shell snippet.
    """
    for key, value in (profile.get("env") or {}).items():
        resolved = str(value).replace("${REPO_ROOT}", str(REPO_ROOT)).replace(
            "${DATA_ROOT}", str(DATA_ROOT)
        )
        os.environ.setdefault(key, resolved)
        if key.endswith(("_HOME", "_DIR", "_CACHE")):
            Path(resolved).mkdir(parents=True, exist_ok=True)


def git_sha() -> str:
    """Short git SHA, or a marker. Never raises -- provenance must not break a run."""
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=10, check=False,
        )
        sha = out.stdout.strip()
        if not sha:
            return "not-a-git-repo"
        dirty = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "status", "--porcelain"],
            capture_output=True, text=True, timeout=10, check=False,
        ).stdout.strip()
        return f"{sha}-dirty" if dirty else sha
    except Exception:
        return "unknown"


def describe_environment() -> dict[str, Any]:
    """Everything needed to trace a number back to the machine that produced it."""
    info: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "fish_env": profile_name(),
        "git_sha": git_sha(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": sys.version.split()[0],
        "data_root_free_gb": round(free_gb(), 1),
    }
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_build"] = torch.version.cuda
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        info["torch"] = None
    try:
        import ultralytics
        info["ultralytics"] = ultralytics.__version__
    except ImportError:
        info["ultralytics"] = None

    # Jetson boards identify themselves here; absent on desktops.
    model = Path("/proc/device-tree/model")
    try:
        if model.exists():
            info["board"] = model.read_text(errors="ignore").strip("\x00").strip()
    except OSError:
        pass
    return info


def write_env_file(run_dir: Path) -> dict[str, Any]:
    """Write env.txt (human) + env.json (machine) into a run directory."""
    info = describe_environment()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "env.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    (run_dir / "env.txt").write_text(
        "\n".join(f"{k}: {v}" for k, v in info.items()) + "\n", encoding="utf-8"
    )
    return info
