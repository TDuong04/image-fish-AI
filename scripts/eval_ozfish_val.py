#!/usr/bin/env python
"""Score checkpoints on the held-out OzFish deployments (decision-only validation, NOT the sealed test).

    python scripts/eval_ozfish_val.py models/A.pt models/B.pt --imgsz 960

`scripts/prepare_ozfish_train.py --val-frac 0.2` holds out whole OzFish deployments in
`data/processed/ozfish_train/images/val`. Models trained with and without OzFish can be compared on them to see what
OzFish adds, without touching the sealed OzFish test set (data/processed/ozfish_eval, which this script refuses).
The held-out deployments are disjoint from the training deployments but not guaranteed to be different reefs, so a gain here
is an optimistic estimate. DataLoader workers are off, so it is safe on Windows.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("models", nargs="+", type=Path)
    ap.add_argument("--data", default="ozfish_train", help="processed dataset with images/val; ozfish_eval is refused")
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()
    if "ozfish_eval" in args.data:
        sys.exit("refusing: ozfish_eval is the sealed test set.")

    root = (paths.DATA_ROOT / "processed" / args.data).resolve()
    val = root / "images" / "val"
    if not val.exists() or not any(val.iterdir()):
        sys.exit(f"{val} is missing or empty. Build it with: python scripts/prepare_ozfish_train.py --val-frac 0.2 --drop-below 4.0")
    n_val = len(list(val.iterdir()))
    n_box = sum(len(p.read_text().split("\n")) - 1 for p in (root / "labels" / "val").glob("*.txt"))
    print(f"held-out images: {n_val} | boxes: {n_box}", flush=True)

    out = paths.DATA_ROOT / ".cache" / "ozfish_val_eval"
    out.mkdir(parents=True, exist_ok=True)
    (out / "data.yaml").write_text(yaml.safe_dump({"path": str(root), "train": "images/train", "val": "images/val",
                                                   "names": {0: "fish"}}))
    from ultralytics import YOLO
    device = resolve_device("auto")
    for model in args.models:
        if not model.exists():
            sys.exit(f"model not found: {model}")
        r = YOLO(str(model)).val(data=str(out / "data.yaml"), imgsz=args.imgsz, conf=0.001, iou=0.7,
                                 device=device.ultralytics, workers=0, plots=False, verbose=False)
        print(f"{model.name:46s} held-out OzFish: mAP@50 {r.box.map50:.4f}  mAP@50-95 {r.box.map:.4f}  "
              f"P {r.box.mp:.3f}  R {r.box.mr:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
