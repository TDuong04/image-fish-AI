"""FD-006: keep the code runnable on machines that are not this one.

These are cheap greps, but they catch the regression that actually happens:
someone debugging on their own box hardcodes a path or a device, it works for
them, and it breaks for everyone else.
"""

from __future__ import annotations

import ast
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

CODE_DIRS = ("src", "scripts", "edge", "configs")

# A drive letter only counts when it starts a token. Without the leading
# boundary, ordinary prose like "Reinstall with:\n" matches on "h:" + backslash.
ABSOLUTE_PATH = re.compile(
    r"""(?:^|[\s'"(\[=])(?:[A-Za-z]:[\\/]|/home/|/Users/|/kaggle/|/content/)""")

SOURCE_FILES = [
    p for d in CODE_DIRS for p in (REPO / d).rglob("*")
    if p.suffix in {".py", ".yaml", ".yml"} and "__pycache__" not in p.parts
]


def _lines(p: Path):
    return p.read_text(encoding="utf-8", errors="replace").splitlines()


@pytest.mark.parametrize("path", SOURCE_FILES, ids=lambda p: str(p.relative_to(REPO)))
def test_no_absolute_paths(path):
    """Every path derives from the repo root or an env var, never a drive letter."""
    offenders = []
    for i, line in enumerate(_lines(path), 1):
        stripped = line.strip()
        if stripped.startswith("#"):
            continue
        # URLs legitimately contain '/', and pytorch wheel indexes contain paths.
        if "http://" in line or "https://" in line:
            continue
        if ABSOLUTE_PATH.search(line):
            offenders.append(f"{i}: {stripped[:90]}")
    assert not offenders, (
        f"{path.relative_to(REPO)} contains absolute path(s):\n  "
        + "\n  ".join(offenders)
        + "\nUse fish.paths (DATA_ROOT / REPO_ROOT) instead."
    )


@pytest.mark.parametrize(
    "path", [p for p in SOURCE_FILES if p.suffix == ".py"],
    ids=lambda p: str(p.relative_to(REPO)))
def test_no_hardcoded_cuda_device(path):
    """`torch.device("cuda")` fails on CPU-only boxes and on Apple silicon.

    fish/device.py is the one place allowed to name a backend.
    """
    if path.name == "device.py":
        return
    for i, line in enumerate(_lines(path), 1):
        if line.strip().startswith("#"):
            continue
        assert 'torch.device("cuda")' not in line and "torch.device('cuda')" not in line, (
            f"{path.relative_to(REPO)}:{i} hardcodes cuda. "
            f"Use fish.device.resolve_device()."
        )


@pytest.mark.parametrize(
    "path", [p for p in SOURCE_FILES if p.suffix == ".py" and p.parent.name == "scripts"],
    ids=lambda p: p.name)
def test_scripts_guard_main(path):
    """Windows and macOS spawn DataLoader workers; an unguarded script re-executes."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    has_guard = any(
        isinstance(n, ast.If) and ast.unparse(n.test).replace("'", '"') == '__name__ == "__main__"'
        for n in tree.body
    )
    assert has_guard, (
        f"{path.name} has no `if __name__ == '__main__':` guard. "
        f"Without it, DataLoader workers re-import and re-run it on Windows/macOS."
    )


def test_paths_resolve_without_cwd_dependency():
    """Importing from any working directory must yield the same roots."""
    import os

    from fish import paths

    before = paths.REPO_ROOT
    cwd = Path.cwd()
    try:
        os.chdir(REPO.parent)
        import importlib
        importlib.reload(paths)
        assert paths.REPO_ROOT == before
    finally:
        os.chdir(cwd)


def test_data_root_honours_env(monkeypatch, tmp_path):
    monkeypatch.setenv("FISH_DATA_ROOT", str(tmp_path))
    import importlib

    from fish import paths
    importlib.reload(paths)
    assert paths.DATA_ROOT == tmp_path.resolve()
    monkeypatch.delenv("FISH_DATA_ROOT")
    importlib.reload(paths)


def test_every_env_profile_parses():
    """A broken profile must fail at load, not halfway through a training run."""
    import yaml

    from fish import env as fenv
    profiles = sorted((REPO / "configs" / "env").glob("*.yaml"))
    assert profiles, "no machine profiles found"
    for p in profiles:
        data = yaml.safe_load(p.read_text(encoding="utf-8"))
        assert "name" in data and "device" in data, f"{p.name} missing name/device"
        fenv.load_profile(p.stem)
