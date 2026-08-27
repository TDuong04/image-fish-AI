#!/usr/bin/env python
"""Report whether this machine can train, and exit non-zero if it cannot.

Run this first, on every machine, before anything else. It is deliberately
readable rather than clever: the failure it exists to catch is a CPU-only torch
build silently training at 1/100th speed for two days.

    python scripts/check_env.py
"""

from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish import env as fenv  # noqa: E402
from fish import paths  # noqa: E402

MIN_PY = (3, 9)
MAX_PY = (3, 13)  # exclusive upper bound is checked as > MAX_PY


def ok(msg: str) -> None:
    print(f"  [ ok ] {msg}")


def warn(msg: str) -> None:
    print(f"  [warn] {msg}")


def bad(msg: str) -> None:
    print(f"  [FAIL] {msg}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-gpu", action="store_true",
                    help="Exit non-zero if no GPU is available (use in CI / before a real run).")
    ap.add_argument("--min-free-gb", type=float, default=10.0,
                    help="Minimum free space required at DATA_ROOT.")
    args = ap.parse_args()

    errors: list[str] = []
    warnings: list[str] = []

    print("\n=== Machine ===")
    print(f"  platform : {platform.platform()}")
    print(f"  machine  : {platform.machine()}")
    print(f"  python   : {sys.version.split()[0]}  ({sys.executable})")

    pyver = sys.version_info[:2]
    if pyver < MIN_PY:
        bad(f"Python {pyver[0]}.{pyver[1]} is too old; need >= {MIN_PY[0]}.{MIN_PY[1]}")
        errors.append("python-too-old")
    elif pyver > MAX_PY:
        bad(f"Python {pyver[0]}.{pyver[1]} is too new. Ultralytics and CUDA torch "
            f"wheels do not support it. Use Python 3.10 or 3.11.")
        errors.append("python-too-new")
    else:
        ok(f"Python {pyver[0]}.{pyver[1]} is supported")

    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    if in_venv:
        ok("running inside a virtual environment")
    else:
        warn("not running inside a virtual environment (expected .venv)")
        warnings.append("no-venv")

    print("\n=== Profile ===")
    try:
        profile = fenv.load_profile()
        fenv.apply_profile(profile)
        ok(f"FISH_ENV={fenv.profile_name()}  ({profile.get('description', '')})")
        if profile.get("training_allowed") is False:
            warn("this profile is inference-only; do not train here")
            warnings.append("inference-only-profile")
    except FileNotFoundError as exc:
        bad(str(exc))
        errors.append("bad-profile")

    print("\n=== Paths ===")
    print(f"  repo root : {paths.REPO_ROOT}")
    print(f"  data root : {paths.DATA_ROOT}")
    print(f"  runs root : {paths.RUNS_ROOT}")
    free = paths.free_gb()
    if free < args.min_free_gb:
        bad(f"only {free:.1f} GB free at DATA_ROOT (want >= {args.min_free_gb:.0f} GB). "
            f"Set $FISH_DATA_ROOT to a larger volume.")
        errors.append("low-disk")
    else:
        ok(f"{free:.1f} GB free at DATA_ROOT")

    print("\n=== PyTorch ===")
    try:
        import torch
        print(f"  torch    : {torch.__version__}")
        print(f"  cuda build: {torch.version.cuda}")
        if torch.cuda.is_available():
            ok(f"GPU available: {torch.cuda.get_device_name(0)}")
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            ok("Apple MPS backend available")
        else:
            msg = "no GPU available -- torch reports CPU only."
            if "+cpu" in torch.__version__:
                msg += (" The installed wheel is a CPU-only build. Reinstall with:\n"
                        "         pip install torch torchvision "
                        "--index-url https://download.pytorch.org/whl/cu124")
            if args.require_gpu:
                bad(msg)
                errors.append("no-gpu")
            else:
                warn(msg)
                warnings.append("no-gpu")
    except ImportError:
        bad("torch is not installed. Run: python scripts/bootstrap.py")
        errors.append("no-torch")

    print("\n=== Training deps ===")
    for mod in ("ultralytics", "cv2", "yaml", "numpy"):
        try:
            __import__(mod)
            ok(f"{mod} importable")
        except ImportError:
            if mod == "ultralytics":
                bad(f"{mod} not installed. Run: python scripts/bootstrap.py")
                errors.append(f"no-{mod}")
            else:
                warn(f"{mod} not installed")
                warnings.append(f"no-{mod}")

    print("\n=== Device resolution ===")
    try:
        from fish.device import resolve_device, suggested_batch
        dev = resolve_device("auto")
        ok(f"resolved device: {dev}")
        print(f"  suggested batch (yolov8n @640): {suggested_batch(dev)}")
    except Exception as exc:
        warn(f"could not resolve a device: {exc}")

    print("\n=== GPU smoke test ===")
    try:
        import torch
        if torch.cuda.is_available():
            a = torch.randn(512, 512, device="cuda")
            (a @ a).sum().item()
            torch.cuda.synchronize()
            ok("512x512 matmul on GPU succeeded")
        else:
            print("  skipped (no GPU)")
    except Exception as exc:
        bad(f"GPU matmul failed: {exc}")
        errors.append("gpu-smoke-failed")

    print("\n" + "=" * 52)
    if errors:
        print(f"NOT READY -- {len(errors)} blocking problem(s): {', '.join(errors)}")
        if warnings:
            print(f"plus {len(warnings)} warning(s): {', '.join(warnings)}")
        return 1
    if warnings:
        print(f"USABLE, with {len(warnings)} warning(s): {', '.join(warnings)}")
        return 0
    print("READY -- this machine can train.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
