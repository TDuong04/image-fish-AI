#!/usr/bin/env python
"""FD-011 + FD-013: build an Ultralytics dataset with a leakage-free split.

    python scripts/prepare_dataset.py deepfish
    python scripts/prepare_dataset.py ozfish_test --all-val
    python scripts/prepare_dataset.py deepfish --group clip --val-frac 0.2

Source frames come from video. Splitting per-image puts adjacent, near-identical
frames on both sides of the boundary and inflates mAP by an unknown margin, so
the split is made over whole groups -- by default the capture *site*, which is
stricter than by clip because two clips from one site share water, lighting and
substrate.

Images and labels are hard-linked (not copied) into the Ultralytics layout, so
this costs no meaningful disk and re-running is cheap. Splits are written to
splits/ and committed, so a run is reproducible rather than re-randomised.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from fish import paths  # noqa: E402
from fish.datasets import group_key, iter_pairs, read_label, site_key  # noqa: E402

SOURCES = {
    "deepfish": {
        "root": "interim/deepfish_darknet",
        "description": "DeepFish, darknet/YOLO boxes, 20 sites incl. explicit negatives",
    },
    "ozfish_test": {
        "root": "interim/ozfish_test",
        "description": "OzFish BRUVS test frames, darknet/YOLO boxes",
    },
}


def link_or_copy(src: Path, dst: Path) -> None:
    """Hard-link when possible; fall back to copying across volumes."""
    if dst.exists():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)


def stable_shuffle(items: list[str], seed: int) -> list[str]:
    """Deterministic order that does not depend on filesystem enumeration order."""
    rng = random.Random(seed)
    ordered = sorted(items, key=lambda s: hashlib.sha1(s.encode()).hexdigest())
    rng.shuffle(ordered)
    return ordered


def assign_groups(counts: dict[str, int], val_frac: float, seed: int,
                  min_val_groups: int = 4) -> dict[str, str]:
    """Fill the validation side to ~val_frac of images, whole groups only.

    Deliberately *not* largest-first. Taking the biggest groups hits the image
    target with the fewest groups, which is the worst outcome for validation:
    a val set drawn from two capture sites measures those two sites, not the
    dataset. Random order tends to admit more, smaller groups, and
    `min_val_groups` enforces a floor on diversity even when that overshoots
    the image fraction slightly.
    """
    total = sum(counts.values())
    target = total * val_frac
    order = stable_shuffle(list(counts), seed)

    assignment: dict[str, str] = {g: "train" for g in order}
    val_n = 0
    val_groups = 0
    for g in order:
        need_diversity = val_groups < min_val_groups
        room = val_n + counts[g] <= target * 1.35
        if (val_n < target and room) or need_diversity:
            assignment[g] = "val"
            val_n += counts[g]
            val_groups += 1
        if val_n >= target and val_groups >= min_val_groups:
            break
    return assignment


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", choices=sorted(SOURCES), help="which interim dataset to prepare")
    ap.add_argument("--group", choices=("site", "clip"), default="site",
                    help="grouping level the split may not straddle (default: site)")
    ap.add_argument("--val-frac", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--all-val", action="store_true",
                    help="put everything in val (for a held-out cross-dataset test set)")
    ap.add_argument("--name", default=None, help="output dataset name (default: source)")
    ap.add_argument("--max-per-group", type=int, default=None,
                    help="cap frames taken from each group -- builds a small, "
                         "still-diverse subset for smoke testing (FD-015)")
    args = ap.parse_args()

    name = args.name or args.source
    src_root = paths.DATA_ROOT / SOURCES[args.source]["root"]
    if not src_root.exists():
        raise SystemExit(f"Source not found: {src_root}\nExtract the archive there first.")

    pairs = iter_pairs(src_root)
    if not pairs:
        raise SystemExit(f"No image/label pairs under {src_root}")

    keyfn = site_key if args.group == "site" else group_key
    by_group: dict[str, list[tuple[Path, Path]]] = defaultdict(list)
    for img, lbl in pairs:
        by_group[keyfn(img)].append((img, lbl))
    counts = {g: len(v) for g, v in by_group.items()}

    print(f"source     : {paths.rel(src_root)}")
    print(f"images     : {len(pairs)}")
    print(f"groups     : {len(counts)}  (by {args.group})")

    if args.all_val:
        assignment = {g: "val" for g in counts}
    else:
        if len(counts) < 3:
            raise SystemExit(
                f"Only {len(counts)} group(s) by {args.group} -- too few for a "
                f"leakage-free split. Use --group clip, or --all-val if this is a "
                f"held-out test set."
            )
        assignment = assign_groups(counts, args.val_frac, args.seed)

    out_root = paths.PROCESSED_DIR / name
    if out_root.exists():
        shutil.rmtree(out_root)

    split_files: dict[str, list[str]] = {"train": [], "val": []}
    stats = {"train": {"images": 0, "boxes": 0, "empty": 0},
             "val": {"images": 0, "boxes": 0, "empty": 0}}

    for g, items in by_group.items():
        split = assignment[g]
        if args.max_per_group:
            # Evenly spaced rather than the first N: consecutive frames of one
            # clip are near-identical, so the first N is barely more varied
            # than a single frame.
            step = max(1, len(items) // args.max_per_group)
            items = items[::step][: args.max_per_group]
        for img, lbl in items:
            dst_img = out_root / "images" / split / img.name
            dst_lbl = out_root / "labels" / split / f"{img.stem}.txt"
            if dst_img.exists():
                raise SystemExit(
                    f"Filename collision: {img.name} appears twice across groups. "
                    f"Names must be unique -- prefix them with their group first."
                )
            link_or_copy(img, dst_img)
            link_or_copy(lbl, dst_lbl)
            boxes, bad = read_label(lbl)
            if bad:
                print(f"  !! {len(bad)} malformed line(s) in {lbl.name}")
            stats[split]["images"] += 1
            stats[split]["boxes"] += len(boxes)
            stats[split]["empty"] += (len(boxes) == 0)
            split_files[split].append(f"{g}\t{img.name}")

    # --- leakage assertion: no group may appear on both sides -------------
    groups_in = {s: {a.split("\t")[0] for a in v} for s, v in split_files.items()}
    overlap = groups_in["train"] & groups_in["val"]
    if overlap:
        raise SystemExit(f"LEAKAGE: {len(overlap)} group(s) in both splits: {sorted(overlap)[:5]}")

    paths.SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in split_files.items():
        (paths.SPLITS_DIR / f"{name}_{split}.txt").write_text(
            "\n".join(sorted(rows)) + "\n", encoding="utf-8"
        )

    # The committed config stores a path RELATIVE to DATA_ROOT. An absolute path
    # here would point at whichever machine happened to build the dataset, which
    # is the portability bug this project is trying not to have. train.py
    # resolves it against the local DATA_ROOT at run time.
    rel_path = out_root.relative_to(paths.DATA_ROOT).as_posix()
    data_yaml = {
        "path": rel_path,
        "train": "images/train" if split_files["train"] else "images/val",
        "val": "images/val",
        "names": {0: "fish"},
    }
    cfg_dir = paths.CONFIG_DIR / "data"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    (cfg_dir / f"{name}.yaml").write_text(yaml.safe_dump(data_yaml, sort_keys=False),
                                          encoding="utf-8")
    # The copy inside the dataset dir is machine-local (gitignored) and absolute,
    # so `yolo` CLI invocations against it work without going through train.py.
    (out_root / "data.yaml").write_text(
        yaml.safe_dump({**data_yaml, "path": str(out_root.resolve())}, sort_keys=False),
        encoding="utf-8")

    meta = {
        "name": name, "source": SOURCES[args.source]["description"],
        "grouping": args.group, "seed": args.seed, "val_frac": args.val_frac,
        "groups_train": sorted(groups_in["train"]), "groups_val": sorted(groups_in["val"]),
        "stats": stats,
    }
    (out_root / "prepare.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print()
    for split in ("train", "val"):
        s = stats[split]
        if not s["images"]:
            continue
        pct = 100 * s["empty"] / s["images"]
        print(f"  {split:5s} {s['images']:5d} images  {s['boxes']:6d} boxes  "
              f"{s['empty']:4d} empty ({pct:.1f}%)  {len(groups_in[split])} groups")
    print(f"\n  leakage check: PASS (no group in both splits)")
    print(f"  dataset  : {paths.rel(out_root)}")
    print(f"  config   : {paths.rel(cfg_dir / f'{name}.yaml')}  -> use as `data: {name}`")
    print(f"  splits   : {paths.rel(paths.SPLITS_DIR)}/{name}_[train|val].txt")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
