#!/usr/bin/env python
"""Re-score checkpoints on the validation images that contain at least one fish.

    python scripts/eval_positive_only.py models/A.pt models/B.pt --imgsz 960

Published DeepFish detection results use about 4,505 images, which is the 6,517-image export minus its
2,012 empty frames. Our validation split keeps empty frames, so a false alarm on one costs precision. This
re-scores the same checkpoints on the fish-only subset of the validation split to show how much of any gap to
published numbers that convention explains. It reads the existing split and writes only a list of file names to
`<DATA_ROOT>/.cache/val_positive/`. DataLoader workers are off, so it is safe on Windows.
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
    ap.add_argument("--data", default="deepfish", help="processed dataset; ozfish_eval is refused")
    ap.add_argument("--imgsz", type=int, default=960)
    args = ap.parse_args()
    if "ozfish_eval" in args.data:
        sys.exit("refusing: ozfish_eval is the sealed test set.")

    root = (paths.DATA_ROOT / "processed" / args.data).resolve()
    img_dir, lab_dir = root / "images" / "val", root / "labels" / "val"
    if not img_dir.exists():
        sys.exit(f"{img_dir} does not exist. Run: python scripts/get_data.py")
    imgs = sorted(img_dir.glob("*.jpg"))
    pos = [p for p in imgs if (lab_dir / (p.stem + ".txt")).exists() and (lab_dir / (p.stem + ".txt")).read_text().strip()]
    print(f"val images: {len(imgs)} | with fish: {len(pos)} | empty: {len(imgs) - len(pos)}", flush=True)

    out = paths.DATA_ROOT / ".cache" / "val_positive"
    out.mkdir(parents=True, exist_ok=True)
    (out / "val_pos.txt").write_text("\n".join(str(p) for p in pos))
    (out / "data.yaml").write_text(yaml.safe_dump({"path": str(root), "train": "images/train",
                                                   "val": str((out / "val_pos.txt").resolve()), "names": {0: "fish"}}))
    from ultralytics import YOLO
    device = resolve_device("auto")
    for model in args.models:
        if not model.exists():
            sys.exit(f"model not found: {model}")
        r = YOLO(str(model)).val(data=str(out / "data.yaml"), imgsz=args.imgsz, conf=0.001, iou=0.7,
                                 device=device.ultralytics, workers=0, plots=False, verbose=False)
        print(f"{model.name:44s} fish-only val: mAP@50 {r.box.map50:.4f}  mAP@50-95 {r.box.map:.4f}  "
              f"P {r.box.mp:.3f}  R {r.box.mr:.3f}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
