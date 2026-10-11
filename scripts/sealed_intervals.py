#!/usr/bin/env python
"""Interval-only second pass on the sealed OzFish test set (docs/eval_protocol.md item 11, declared before it was run).

    python scripts/sealed_intervals.py --confirm-protocol-item-11

The sealed set was scored once (F33, Ultralytics' built-in validation, which keeps no boxes). This pass exists only to put uncertainty on those numbers: it runs
the same three fixed YOLOv8s checkpoints, matches their predictions to the labels, and resamples whole deployments (the id before the first underscore of the
file name, so stereo _L/_R partners are one cluster) with a cluster bootstrap. It chooses nothing, tunes nothing, and the F33 numbers stay the headline. The AP routine
is the one of scripts/bootstrap_ci.py, so its point values are printed next to the F33 values and any difference is reported, not corrected.
It refuses to run without the confirmation flag and refuses any model that is not in the fixed list.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402
import bootstrap_ci as bc  # noqa: E402
import error_analysis as ea  # noqa: E402

FIXED_MODELS = {   # the three YOLOv8s seeds of F33 and their first-pass Ultralytics values (mAP50, mAP50-95)
    "20261005-yolov8s-960-kakadu-ozfish-s0.pt": (0.6953, 0.3759),
    "20261007-yolov8s-960-kakadu-ozfish-s1.pt": (0.7005, 0.3805),
    "20261007-yolov8s-960-kakadu-ozfish-s2.pt": (0.7049, 0.3846),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--confirm-protocol-item-11", action="store_true", help="required: you have read docs/eval_protocol.md item 11")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if not args.confirm_protocol_item_11:
        sys.exit("refusing: this reads the sealed OzFish test set. Read docs/eval_protocol.md item 11 and pass --confirm-protocol-item-11.")

    root = paths.DATA_ROOT / "processed" / "ozfish_eval"
    img_dir, lab_dir = root / "images" / "val", root / "labels" / "val"
    images = sorted(img_dir.glob("*.jpg"))
    if len(images) != 352:
        sys.exit(f"expected the 352 sealed frames, found {len(images)} in {img_dir}")
    import cv2
    gts = []
    for p in images:
        h, w = cv2.imread(str(p)).shape[:2]
        gts.append(ea.read_labels(lab_dir / (p.stem + ".txt"), w, h))
    n_gt = np.array([len(g) for g in gts], dtype=float)
    keys = [p.stem.split("_")[0] for p in images]
    uniq = sorted(set(keys))
    cid = np.array([uniq.index(k) for k in keys])
    print(f"{len(images)} frames, {int(n_gt.sum())} boxes, {len(uniq)} deployments")

    device = resolve_device("auto")
    cache_dir = paths.DATA_ROOT / ".cache" / "sealed_intervals"
    scorers = {}
    for m in FIXED_MODELS:
        print(f"  scoring {m} ...", flush=True)
        boxes, confs = bc.predict_all(paths.MODELS_ROOT / m, images, device, 960, 0.005, cache_dir / f"{m}.npz")
        scorers[m] = bc.Scorer(gts, boxes, confs)

    rng = np.random.default_rng(args.seed)
    full = np.ones(len(images))
    i50, iall = [0], list(range(len(bc.IOUS)))
    point = {m: (scorers[m].ap(full, n_gt, i50), scorers[m].ap(full, n_gt, iall)) for m in FIXED_MODELS}
    boots = {m: ([], []) for m in FIXED_MODELS}
    for b in range(args.boot):
        draw = rng.integers(0, len(uniq), len(uniq))
        w = np.bincount(draw, minlength=len(uniq)).astype(float)[cid]
        for m in FIXED_MODELS:
            boots[m][0].append(scorers[m].ap(w, n_gt, i50))
            boots[m][1].append(scorers[m].ap(w, n_gt, iall))

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    report = {"protocol": "docs/eval_protocol.md item 11", "frames": len(images), "boxes": int(n_gt.sum()), "clusters": len(uniq), "cluster_by": "deployment (id before the first underscore)",
              "boot": args.boot, "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "note": "second pass, intervals only; the F33 point estimates are the first pass's (Ultralytics validation); AP here is the 101-point routine of bootstrap_ci.py",
              "per_seed": {}, "mean_of_seeds": {}}
    for m, (f50, f5095) in FIXED_MODELS.items():
        report["per_seed"][m] = {"AP50": round(point[m][0], 4), "AP50_ci95": ci(boots[m][0]), "AP50_95": round(point[m][1], 4), "AP50_95_ci95": ci(boots[m][1]),
                                 "first_pass_ultralytics_mAP50": f50, "first_pass_ultralytics_mAP50_95": f5095}
    mean50 = [np.mean([boots[m][0][b] for m in FIXED_MODELS]) for b in range(args.boot)]
    mean5095 = [np.mean([boots[m][1][b] for m in FIXED_MODELS]) for b in range(args.boot)]
    report["mean_of_seeds"] = {"AP50": round(float(np.mean([point[m][0] for m in FIXED_MODELS])), 4), "AP50_ci95": ci(mean50),
                               "AP50_95": round(float(np.mean([point[m][1] for m in FIXED_MODELS])), 4), "AP50_95_ci95": ci(mean5095)}
    out = args.out or (paths.RUNS_ROOT / f"sealed-intervals-{datetime.now().strftime('%Y%m%d')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "intervals.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for m, d in report["per_seed"].items():
        print(f"{m:44s} AP50 {d['AP50']:.4f} {d['AP50_ci95']} (first pass {d['first_pass_ultralytics_mAP50']})  AP50-95 {d['AP50_95']:.4f} {d['AP50_95_ci95']} (first pass {d['first_pass_ultralytics_mAP50_95']})")
    d = report["mean_of_seeds"]
    print(f"{'mean of the three seeds':44s} AP50 {d['AP50']:.4f} {d['AP50_ci95']}  AP50-95 {d['AP50_95']:.4f} {d['AP50_95_ci95']}")
    print(f"wrote {out / 'intervals.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
