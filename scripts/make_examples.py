#!/usr/bin/env python
"""Draw example detections (a figure for the paper): two frames from each of three sources, picked by a fixed random rule, not by looks.

    python scripts/make_examples.py --model models/20261007-yolov8s-960-kakadu-ozfish-s2.pt

Sources (never the sealed OzFish test set, protocol item 1):
  DeepFish validation          data/processed/deepfish/images/val                (sites held out of training)
  OzFish held-out deployments  data/processed/ozfish_train/images/val            (the decision-only validation set, not the sealed test)
  Mediterranean (unseen)       data/processed/deepecomar/images/val              (the F39 set, no training data from it)
Rule: frames with at least one labelled fish, sorted by file name, drawn with random.Random(--seed); the same draw is reported in the figure caption.
Green boxes are labels, red boxes are detections at the operating confidence (default 0.25). The panel title gives found / labelled fish and false alarms at that threshold.
Writes docs/report/figures/fig5_examples.png and .pdf, and examples.json (the chosen files and counts) beside them.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patches as patches  # noqa: E402
import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402
import error_analysis as ea  # noqa: E402

SOURCES = [("DeepFish validation (sites held out)", "deepfish"), ("OzFish held-out deployments", "ozfish_train"), ("Mediterranean set (unseen source)", "deepecomar")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--per-source", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=paths.REPO_ROOT / "docs" / "report" / "figures")
    args = ap.parse_args()
    if not args.model.exists():
        sys.exit(f"model not found: {args.model}")
    from ultralytics import YOLO

    model = YOLO(str(args.model))
    device = resolve_device("auto")
    rng = random.Random(args.seed)
    picks = []
    for title, name in SOURCES:
        root = paths.DATA_ROOT / "processed" / name
        imgs = sorted((root / "images" / "val").glob("*.jpg"))
        withfish = [p for p in imgs if (root / "labels" / "val" / (p.stem + ".txt")).exists() and (root / "labels" / "val" / (p.stem + ".txt")).read_text().strip()]
        if not withfish:
            sys.exit(f"no labelled frames under {root}")
        for p in rng.sample(withfish, args.per_source):
            picks.append((title, root, p))

    fig, axes = plt.subplots(len(SOURCES), args.per_source, figsize=(4.6 * args.per_source, 2.9 * len(SOURCES)))
    record = []
    for ax, (title, root, p) in zip(np.array(axes).reshape(-1), picks):
        im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = im.shape[:2]
        gt = ea.read_labels(root / "labels" / "val" / (p.stem + ".txt"), w, h)
        r = model.predict(str(p), imgsz=args.imgsz, conf=0.005, device=device.ultralytics, verbose=False)[0]
        boxes = r.boxes.xyxy.cpu().numpy() if len(r.boxes) else np.zeros((0, 4))
        conf = r.boxes.conf.cpu().numpy() if len(r.boxes) else np.zeros(0)
        tp, fp, _ = ea.match_at(gt, boxes, conf, args.conf)
        ax.imshow(im)
        for b in gt:
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#00e676", lw=1.1))
        for k in np.where(conf >= args.conf)[0]:
            b = boxes[k]
            ax.add_patch(patches.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec="#ff1744", lw=1.1, ls="--"))
        ax.set_title(f"{title}\nfound {tp} of {len(gt)} labelled, {len(fp)} false alarms", fontsize=8)
        ax.axis("off")
        record.append({"source": title, "file": p.name, "labelled": int(len(gt)), "found": int(tp), "false_alarms": int(len(fp))})
    fig.tight_layout()
    args.out.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(args.out / f"fig5_examples.{ext}", dpi=140, bbox_inches="tight")
    (args.out / "examples.json").write_text(json.dumps({"model": args.model.name, "conf": args.conf, "imgsz": args.imgsz, "seed": args.seed, "panels": record}, indent=2), encoding="utf-8")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
