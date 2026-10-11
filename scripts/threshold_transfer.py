#!/usr/bin/env python
"""Does a threshold chosen on validation transfer to an unseen source? (declared before the unseen run was read, docs/eval_protocol.md item 10, second addendum)

    python scripts/threshold_transfer.py          # uses the predictions cached by scripts/eval_unseen.py

For each YOLOv8s seed, found fish and false alarms on the DEEPECOMAR set at the fixed confidence 0.25 and at the seed's own best-F1 confidence on the DeepFish validation split
(seed 0: 0.25, seed 1: 0.5, seed 2: 0.2, from runs/analysis-*). A reading, not a selection: nothing is chosen from it.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fish import paths  # noqa: E402
import error_analysis as ea  # noqa: E402

SEEDS = (("20261005-yolov8s-960-kakadu-ozfish-s0", 0.25), ("20261007-yolov8s-960-kakadu-ozfish-s1", 0.5), ("20261007-yolov8s-960-kakadu-ozfish-s2", 0.2))


def main() -> int:
    root = paths.DATA_ROOT / "processed" / "deepecomar"
    images = sorted((root / "images" / "val").glob("*.jpg"))
    gts = []
    for p in images:
        h, w = cv2.imread(str(p)).shape[:2]
        gts.append(ea.read_labels(root / "labels" / "val" / (p.stem + ".txt"), w, h))
    n = sum(len(g) for g in gts)
    out = {"data": "deepecomar", "boxes": n, "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "models": {}}
    for m, thr in SEEDS:
        z = np.load(paths.DATA_ROOT / ".cache" / "unseen" / f"deepecomar-{m}.pt-{len(images)}.npz", allow_pickle=True)
        row = {"val_best_f1_threshold": thr}
        for t in sorted({0.25, thr}):
            tp = fp = 0
            for gt, b, c in zip(gts, z["boxes"], z["conf"]):
                a, f, _ = ea.match_at(gt, b, c, t)
                tp, fp = tp + a, fp + len(f)
            row[str(t)] = {"found": tp, "false_alarms": fp, "precision": round(tp / max(tp + fp, 1), 3), "recall": round(tp / n, 3)}
        out["models"][m] = row
    d = paths.RUNS_ROOT / "unseen-20261011-deepecomar-labels"
    d.mkdir(parents=True, exist_ok=True)
    (d / "threshold_transfer.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
