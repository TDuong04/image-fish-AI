#!/usr/bin/env python
"""Build a DELIBERATELY LEAKY DeepFish split, only to place our numbers next to published ones.

    python scripts/prepare_random_split.py

Published DeepFish detection results use about 4,505 images (our export minus its 2,012 empty frames) and, as far
as the papers we could read describe, split frames at random. Frames of one clip are near-identical, so a random
per-image split puts the same scene on both sides and inflates mAP by an unknown margin. scripts/prepare_dataset.py
refuses to do that on purpose; this script does it on purpose, into a dataset named `deepfish_randomsplit`, so we
can measure how much of the gap between our site-split numbers and published ones is the split.

   NOTHING TRAINED ON THIS DATASET MAY BE REPORTED AS A RESULT. It is a diagnostic. Its numbers are inflated by
   construction and must be labelled as such wherever they appear.

Only images with at least one fish are kept (the published convention). The split is a seeded random 80/20 per image.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from fish import paths  # noqa: E402
from fish.datasets import iter_pairs, read_label  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_dataset import link_or_copy  # noqa: E402

NAME = "deepfish_randomsplit"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src_root = paths.DATA_ROOT / "interim" / "deepfish_darknet"
    if not src_root.exists():
        sys.exit(f"Source not found: {src_root}\nRun: python scripts/get_data.py")
    pairs = [(i, l) for i, l in iter_pairs(src_root) if read_label(l)[0]]
    print(f"fish-containing images: {len(pairs)}")

    names = sorted(p[0].name for p in pairs)
    random.Random(args.seed).shuffle(names)
    val_names = set(names[: int(len(names) * args.val_frac)])

    out_root = paths.PROCESSED_DIR / NAME
    if out_root.exists():
        shutil.rmtree(out_root)
    stats = {"train": {"images": 0, "boxes": 0}, "val": {"images": 0, "boxes": 0}}
    for img, lbl in pairs:
        split = "val" if img.name in val_names else "train"
        link_or_copy(img, out_root / "images" / split / img.name)
        link_or_copy(lbl, out_root / "labels" / split / f"{img.stem}.txt")
        stats[split]["images"] += 1
        stats[split]["boxes"] += len(read_label(lbl)[0])

    data_yaml = {"path": out_root.relative_to(paths.DATA_ROOT).as_posix(), "train": "images/train",
                 "val": "images/val", "names": {0: "fish"}}
    (paths.CONFIG_DIR / "data" / f"{NAME}.yaml").write_text(
        "# DIAGNOSTIC ONLY: random per-image split, leaks near-identical frames by design.\n"
        "# Never report a number trained or validated on this as a result.\n"
        + yaml.safe_dump(data_yaml, sort_keys=False), encoding="utf-8")
    (out_root / "prepare.json").write_text(json.dumps({
        "name": NAME, "inflated": True, "seed": args.seed, "val_frac": args.val_frac, "stats": stats,
        "warning": "random per-image split of near-duplicate video frames; scores are inflated by construction"}, indent=2))
    for split, s in stats.items():
        print(f"  {split:5s} {s['images']:5d} images  {s['boxes']:6d} boxes")
    print("\nWARNING: this split leaks by design. Use it only as a diagnostic.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
