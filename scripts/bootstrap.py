#!/usr/bin/env python
"""One cross-platform installer. Picks the right torch wheel for this machine.

    python scripts/bootstrap.py             # install training deps
    python scripts/bootstrap.py --dry-run   # show what it would do
    python scripts/bootstrap.py --edge      # Jetson / inference-only deps

Why this exists: `pip install torch` silently gives a CPU-only wheel on Windows,
and gives a broken wheel on ARM64 Jetson. Neither failure is announced. This
script branches on platform so a teammate does not have to know that.
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

MIN_PY, MAX_PY = (3, 9), (3, 13)

CUDA_INDEX = {
    "cu121": "https://download.pytorch.org/whl/cu121",
    "cu124": "https://download.pytorch.org/whl/cu124",
    "cu126": "https://download.pytorch.org/whl/cu126",
    "cpu": "https://download.pytorch.org/whl/cpu",
}


def is_jetson() -> bool:
    if platform.machine() not in ("aarch64", "arm64") or platform.system() != "Linux":
        return False
    try:
        model = Path("/proc/device-tree/model")
        return model.exists() and "jetson" in model.read_text(errors="ignore").lower()
    except OSError:
        return False


def detect_cuda_tag() -> str | None:
    """Read the driver's CUDA version from nvidia-smi and map to a wheel tag.

    Returns None when no NVIDIA driver is present.
    """
    try:
        out = subprocess.run(["nvidia-smi"], capture_output=True, text=True, timeout=20)
    except (FileNotFoundError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    # "CUDA Version: 12.4" appears in the nvidia-smi header
    text = out.stdout
    if "CUDA Version:" not in text:
        return "cu124"
    try:
        ver = text.split("CUDA Version:")[1].split()[0]
        major, minor = (int(x) for x in ver.split(".")[:2])
    except (ValueError, IndexError):
        return "cu124"
    # Driver CUDA is backward compatible: pick the newest wheel it supports.
    if (major, minor) >= (12, 6):
        return "cu126"
    if (major, minor) >= (12, 4):
        return "cu124"
    if major >= 12:
        return "cu121"
    return None  # CUDA 11 and older: no supported wheel here, tell the user


def run(cmd: list[str], dry: bool) -> None:
    print("  $ " + " ".join(cmd))
    if dry:
        return
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(f"\nCommand failed with exit code {result.returncode}.")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true", help="print commands without running them")
    ap.add_argument("--edge", action="store_true", help="install inference-only deps (Jetson)")
    ap.add_argument("--dev", action="store_true", help="also install dev/test deps")
    ap.add_argument("--cpu", action="store_true", help="force the CPU-only torch wheel")
    args = ap.parse_args()

    pyver = sys.version_info[:2]
    print(f"Python  : {pyver[0]}.{pyver[1]}  ({sys.executable})")
    print(f"Platform: {platform.system()} {platform.machine()}")

    if pyver < MIN_PY or pyver > MAX_PY:
        print(f"\nERROR: Python {pyver[0]}.{pyver[1]} is not supported.")
        print(f"Use Python 3.10 or 3.11. Ultralytics and the CUDA torch wheels do")
        print(f"not publish builds for Python {pyver[0]}.{pyver[1]}.")
        print("\n  py -3.10 -m venv .venv                  # Windows")
        print("  python3.10 -m venv .venv                 # Linux / macOS")
        return 1

    if sys.prefix == getattr(sys, "base_prefix", sys.prefix):
        print("\nWARNING: not inside a virtual environment. Create one first:")
        print("  python -m venv .venv")
        print("  .venv\\Scripts\\activate     (Windows)")
        print("  source .venv/bin/activate   (Linux / macOS)")
        if not args.dry_run:
            print("\nRefusing to install into the system Python. Re-run inside a venv,")
            print("or pass --dry-run to see the commands.")
            return 1

    pip = [sys.executable, "-m", "pip"]
    run([*pip, "install", "--upgrade", "pip"], args.dry_run)

    if args.edge or is_jetson():
        print("\n--- Jetson / edge install ---")
        print("NOTE: torch is NOT installed from PyPI on ARM64+JetPack -- the PyPI")
        print("wheel has no CUDA support for Jetson. Use NVIDIA's prebuilt wheels")
        print("matched to your JetPack version, or the dusty-nv containers:")
        print("  https://developer.download.nvidia.com/compute/redist/jp/")
        print("Edge inference needs TensorRT + numpy + opencv only -- no torch.")
        run([*pip, "install", "-r", str(REPO_ROOT / "requirements-edge.txt")], args.dry_run)
        print("\nDone. Verify with: python scripts/check_env.py")
        return 0

    print("\n--- Training install ---")
    if args.cpu:
        tag = "cpu"
        print("CPU wheel forced by --cpu. Training will be ~100x slower; smoke tests only.")
    else:
        tag = detect_cuda_tag()
        if tag is None:
            print("No usable NVIDIA driver detected (or CUDA < 12).")
            if platform.system() == "Darwin":
                print("On macOS, torch uses the MPS backend -- the default wheel is correct.")
                tag = "cpu"
            else:
                print("Installing the CPU wheel. If this machine HAS an NVIDIA GPU,")
                print("update the driver and re-run -- do not train on the CPU wheel.")
                tag = "cpu"
        else:
            print(f"Detected CUDA driver -> torch wheel index: {tag}")

    torch_cmd = [*pip, "install", "torch", "torchvision"]
    if not (tag == "cpu" and platform.system() == "Darwin"):
        torch_cmd += ["--index-url", CUDA_INDEX[tag]]
    run(torch_cmd, args.dry_run)
    run([*pip, "install", "-r", str(REPO_ROOT / "requirements-train.txt")], args.dry_run)
    if args.dev:
        run([*pip, "install", "-r", str(REPO_ROOT / "requirements-dev.txt")], args.dry_run)

    print("\nDone. Verify with:")
    print("  python scripts/check_env.py --require-gpu")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
