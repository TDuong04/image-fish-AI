"""Single source of truth for every path in this project.

Nothing else may compute a path from scratch. See CLAUDE.md, "Portability".

The repo root is derived from this file's own location, so every script works
regardless of the current working directory or which machine it runs on.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path

# <repo>/src/fish/paths.py -> parents[2] == <repo>
REPO_ROOT: Path = Path(__file__).resolve().parents[2]

#: Datasets. Override with $FISH_DATA_ROOT to put data on another volume.
DATA_ROOT: Path = Path(os.environ.get("FISH_DATA_ROOT") or REPO_ROOT / "data").resolve()

RAW_DIR = DATA_ROOT / "raw"
INTERIM_DIR = DATA_ROOT / "interim"
PROCESSED_DIR = DATA_ROOT / "processed"
EVAL_DIR = DATA_ROOT / "eval"

RUNS_ROOT: Path = Path(os.environ.get("FISH_RUNS_ROOT") or REPO_ROOT / "runs").resolve()
MODELS_ROOT: Path = Path(os.environ.get("FISH_MODELS_ROOT") or REPO_ROOT / "models").resolve()

CONFIG_DIR = REPO_ROOT / "configs"
SPLITS_DIR = REPO_ROOT / "splits"
DOCS_DIR = REPO_ROOT / "docs"

DATA_DIRS = (RAW_DIR, INTERIM_DIR, PROCESSED_DIR, EVAL_DIR)


def ensure_dirs() -> None:
    """Create the standard directory layout. Idempotent."""
    for d in (*DATA_DIRS, RUNS_ROOT, MODELS_ROOT, SPLITS_DIR):
        d.mkdir(parents=True, exist_ok=True)


def free_gb(path: Path | None = None) -> float:
    """Free space in GB on the volume holding `path` (default: DATA_ROOT).

    Walks up to the nearest existing parent, so it works before the directory
    has been created.
    """
    target = Path(path) if path is not None else DATA_ROOT
    while not target.exists() and target != target.parent:
        target = target.parent
    return shutil.disk_usage(target).free / (1024**3)


def require_free_gb(needed_gb: float, path: Path | None = None) -> None:
    """Refuse up front rather than filling the volume mid-download.

    Raises RuntimeError naming the shortfall. Every fetch script calls this.
    """
    available = free_gb(path)
    if available < needed_gb:
        target = path if path is not None else DATA_ROOT
        raise RuntimeError(
            f"Not enough free space at {target}: need {needed_gb:.1f} GB, "
            f"have {available:.1f} GB (short by {needed_gb - available:.1f} GB)."
        )


def rel(path: Path) -> str:
    """Path relative to the repo root, forward slashes, for logs and manifests.

    Never write an absolute path into a committed file.
    """
    p = Path(path).resolve()
    try:
        return p.relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return p.as_posix()
