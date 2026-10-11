#!/usr/bin/env python
"""Time-ordered splits of the converted DEEPECOMAR set for the adaptation experiment (docs/eval_protocol.md item 12).

    python scripts/prepare_adaptation.py          # needs data/processed/deepecomar from scripts/prepare_deepecomar.py

The `vivo` frames (886 of the 1,516) are single snapshots from one fixed camera taken on different dates between August 2018 and August 2019, which is what a deployed camera
would give: the camera and scene are fixed and the water, light and fish change over time. They are sorted by their time stamp and split chronologically:

  train pool   the earliest 60%  (the data a deployment could label first)
  val          the next 10%      (epoch selection only)
  test_vivo    the latest 30%    (same camera, later months)
  test_other   every frame of the other 17 groups (other cameras and dates), never used for training

Training subsets of N frames are drawn uniformly from the train pool with a fixed seed (nested: each larger subset contains the smaller). Lists are written as Ultralytics text files
(`./../images/val/<stem>.jpg` relative to the list, so they are portable) under data/processed/deepecomar/splits_adapt/, with a meta.json recording the counts and date ranges.
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish import paths  # noqa: E402

SIZES = (25, 100)      # plus the whole pool


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = paths.DATA_ROOT / "processed" / "deepecomar"
    src_file, grp_file = root / "sources.json", root / "groups.json"
    if not src_file.exists():
        sys.exit(f"{src_file} missing: run scripts/prepare_deepecomar.py (the current version writes sources.json)")
    sources = json.loads(src_file.read_text())
    groups = json.loads(grp_file.read_text())

    def stamp(base: str):
        m = re.match(r"vivo_(\d{8})_(\d{6})M", base)
        return (m.group(1), m.group(2)) if m else None

    vivo = sorted((st for st, base in sources.items() if group_is_vivo(groups[st])), key=lambda st: stamp(sources[st]))
    other = sorted(st for st in sources if not group_is_vivo(groups[st]))
    n = len(vivo)
    n_train, n_val = int(n * 0.6), int(n * 0.1)
    pool, val, test_vivo = vivo[:n_train], vivo[n_train:n_train + n_val], vivo[n_train + n_val:]
    order = pool[:]
    random.Random(args.seed).shuffle(order)
    subsets = {f"train_{k}": sorted(order[:k]) for k in SIZES}
    subsets["train_all"] = sorted(pool)

    out = root / "splits_adapt"
    out.mkdir(parents=True, exist_ok=True)

    def write(name, stems):
        (out / f"{name}.txt").write_text("".join(f"./../images/val/{s}.jpg\n" for s in stems))

    for name, stems in {**subsets, "val": val, "test_vivo": test_vivo, "test_other": other}.items():
        write(name, stems)

    def span(stems):
        d = [stamp(sources[s])[0] for s in stems]
        return [min(d), max(d)]

    meta = {"seed": args.seed, "vivo_frames": n, "pool": len(pool), "pool_dates": span(pool), "val": len(val), "val_dates": span(val),
            "test_vivo": len(test_vivo), "test_vivo_dates": span(test_vivo), "test_other": len(other),
            "train_sizes": {k: len(v) for k, v in subsets.items()}, "nested": True}
    (out / "meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps(meta, indent=2))
    return 0


def group_is_vivo(g: str) -> bool:
    return g == "vivo"


if __name__ == "__main__":
    sys.exit(main())
