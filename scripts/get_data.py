#!/usr/bin/env python
"""Download, extract and prepare every dataset. One command, safe to re-run.

    python scripts/get_data.py

Downloads ~1.15 GB from the YOLO-Fish authors' Google Drive, extracts it, and
builds three ready-to-train datasets. Takes 5-15 minutes on a normal connection.

Every step is skipped if its output already exists, so re-running after an
interrupted download costs nothing and resumes where it stopped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish import paths  # noqa: E402

# Google Drive file ids, taken from fish-research/README.md (the YOLO-Fish repo).
DOWNLOADS = [
    {
        "id": "10Pr4lLeSGTfkjA40ReGSC8H3a9onfMZ0",
        "zip": "deepfish_darknet.zip",
        "extract_to": "interim/deepfish_darknet",
        "size_gb": 1.09,
        "what": "DeepFish, 6517 images with boxes, 20 sites, incl. 2012 no-fish frames",
    },
    {
        "id": "1C_7l2YFc5fXt1DMsuVZPDFKTJLm0syX3",
        "zip": "ozfish_test.zip",
        "extract_to": "interim/ozfish_test",
        "size_gb": 0.06,
        "what": "OzFish BRUVS, 352 images with boxes -- the sealed test set",
    },
]

# (args to prepare_dataset.py, output name, description)
PREPARATIONS = [
    (["deepfish", "--group", "site"], "deepfish", "training + validation"),
    (["ozfish_test", "--all-val", "--name", "ozfish_eval"], "ozfish_eval",
     "sealed cross-dataset test set"),
    (["deepfish", "--group", "site", "--max-per-group", "12", "--name", "smoke"], "smoke",
     "tiny subset for pipeline checks"),
]

TOTAL_GB = sum(d["size_gb"] for d in DOWNLOADS)
# Downloads + extracted copies. Prepared datasets are hard-linked, so ~free.
NEEDED_GB = TOTAL_GB * 2.2


def run(cmd: list[str], what: str) -> None:
    print(f"    $ {' '.join(str(c) for c in cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        raise SystemExit(
            f"\n{what} failed (exit {result.returncode}).\n"
            f"If a download failed, just run this script again -- it resumes."
        )


def download(entry: dict, raw_dir: Path) -> Path:
    dest = raw_dir / entry["zip"]
    if dest.exists() and dest.stat().st_size > 1024:
        print(f"  [skip] {entry['zip']} already downloaded "
              f"({dest.stat().st_size / 1e9:.2f} GB)")
        return dest
    print(f"  [get ] {entry['zip']}  (~{entry['size_gb']} GB)  {entry['what']}")
    run([sys.executable, "-m", "gdown", entry["id"], "-O", str(dest)], "Download")
    return dest


def extract(entry: dict, zip_path: Path) -> None:
    out = paths.DATA_ROOT / entry["extract_to"]
    if out.exists() and any(out.rglob("*.jpg")):
        print(f"  [skip] {entry['extract_to']} already extracted")
        return
    print(f"  [open] {entry['zip']} -> {entry['extract_to']}")
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as z:
        z.extractall(out)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--force", action="store_true",
                    help="rebuild the prepared datasets even if they exist")
    args = ap.parse_args()

    print("=" * 62)
    print("  Fish detection -- dataset setup")
    print("=" * 62)

    paths.ensure_dirs()
    print(f"\nData goes to: {paths.DATA_ROOT}")
    print("(set FISH_DATA_ROOT to put it elsewhere)")

    free = paths.free_gb()
    print(f"Free space  : {free:.1f} GB   (need about {NEEDED_GB:.1f} GB)")
    if free < NEEDED_GB:
        raise SystemExit(
            f"\nNot enough free space: need ~{NEEDED_GB:.1f} GB, have {free:.1f} GB.\n"
            f"Free some space, or set FISH_DATA_ROOT to a bigger drive and re-run."
        )

    try:
        import gdown  # noqa: F401
    except ImportError:
        raise SystemExit(
            "\n`gdown` is not installed. Run this first:\n"
            "  python scripts/bootstrap.py"
        ) from None

    print("\n--- 1/3  Downloading ---")
    zips = [(e, download(e, paths.RAW_DIR)) for e in DOWNLOADS]

    print("\n--- 2/3  Extracting ---")
    for entry, zip_path in zips:
        extract(entry, zip_path)

    print("\n--- 3/3  Building datasets ---")
    prep = Path(__file__).parent / "prepare_dataset.py"
    for cmd_args, name, desc in PREPARATIONS:
        out = paths.PROCESSED_DIR / name
        if out.exists() and not args.force:
            print(f"  [skip] {name} already built ({desc}) -- use --force to rebuild")
            continue
        print(f"  [make] {name}  ({desc})")
        run([sys.executable, str(prep), *cmd_args], f"Preparing {name}")

    print("\n" + "=" * 62)
    print("  Ready. Next:")
    print("    python scripts/train.py --config smoke        # ~2 min, checks it works")
    print("    python scripts/train.py --config yolov8n_640  # the real run, hours")
    print("=" * 62)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
