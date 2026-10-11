#!/usr/bin/env python
"""How well do the detections count fish? Per-frame counts and the clip-level maximum (MaxN) against the labels.

    python scripts/count_accuracy.py                  # YOLOv8s seeds and the YOLOv8n arm on the DeepFish validation split

Ecologists use the count, not the boxes: abundance per frame and MaxN (the largest number of fish seen in one frame of a clip). This turns the cached predictions of
scripts/bootstrap_ci.py (run that first) into those quantities at a confidence threshold chosen on the validation split itself (the best-F1 confidence of each model from
its error analysis, so the numbers are optimistic for validation and are meant to show the bias, not to rank models) and reports the mean error, the mean absolute error,
the share of fish counted (sum of predicted over sum of true) and the correlation per frame, and the MaxN error per clip. Frames are not independent; the clip rows are the
more honest unit. The same numbers feed any downstream consumer of the count (for example a controller that uses the fish count as an observation).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from fish import paths  # noqa: E402
import bootstrap_ci as bc  # noqa: E402
import error_analysis as ea  # noqa: E402


def best_f1_threshold(analysis_json: Path) -> float:
    return float(json.loads(analysis_json.read_text())["best_f1_threshold"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=paths.CONFIG_DIR / "stats_arms.yaml")
    ap.add_argument("--data", default="deepfish")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--min-conf", type=float, default=0.005)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if "ozfish_eval" in args.data:
        sys.exit("refusing: ozfish_eval is the sealed test set.")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = paths.DATA_ROOT / "processed" / args.data
    img_dir, lab_dir = root / "images" / args.split, root / "labels" / args.split
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    import cv2
    gt_count = []
    for p in images:
        h, w = cv2.imread(str(p)).shape[:2]
        gt_count.append(len(ea.read_labels(lab_dir / (p.stem + ".txt"), w, h)))
    gt_count = np.array(gt_count)
    clips = np.array([bc.clip_of(p.stem) for p in images])
    uniq = sorted(set(clips))

    rows = {}
    for arm, models in cfg["arms"].items():
        for m in models:
            tag = f"{m}-{args.data}-{args.split}-{args.imgsz}-{args.min_conf}-{len(images)}"
            cache = paths.DATA_ROOT / ".cache" / "bootstrap" / f"{tag}.npz"
            if not cache.exists():
                sys.exit(f"missing cache {cache.name}: run scripts/bootstrap_ci.py first")
            z = np.load(cache, allow_pickle=True)
            confs = list(z["conf"])
            ana = sorted(paths.RUNS_ROOT.glob(f"analysis-*-{Path(m).stem}"))
            thr = best_f1_threshold(ana[-1] / "analysis.json") if ana else 0.25
            pred = np.array([int((c >= thr).sum()) for c in confs])
            err = pred - gt_count
            maxn_true = np.array([gt_count[clips == c].max() for c in uniq])
            maxn_pred = np.array([pred[clips == c].max() for c in uniq])
            rows[Path(m).stem] = {
                "arm": arm, "threshold": thr, "frames": len(images), "true_fish": int(gt_count.sum()), "counted": int(pred.sum()),
                "share_counted": round(float(pred.sum() / max(gt_count.sum(), 1)), 3), "mean_error_per_frame": round(float(err.mean()), 3),
                "mae_per_frame": round(float(np.abs(err).mean()), 3),
                "frame_corr": round(float(np.corrcoef(pred, gt_count)[0, 1]), 3),
                "clips": len(uniq), "maxn_mean_error": round(float((maxn_pred - maxn_true).mean()), 2),
                "maxn_mean_abs_error": round(float(np.abs(maxn_pred - maxn_true).mean()), 2),
                "maxn_share": round(float(maxn_pred.sum() / max(maxn_true.sum(), 1)), 3),
            }
    out = args.out or (paths.RUNS_ROOT / "stats-counting")
    out.mkdir(parents=True, exist_ok=True)
    (out / "counts.json").write_text(json.dumps({"data": args.data, "split": args.split, "rows": rows,
                                                 "note": "threshold = each model's best-F1 confidence on this same split (optimistic); frames are not independent"}, indent=2))
    for n, r in rows.items():
        print(f"{r['arm']:22s} {n[:42]:42s} thr {r['threshold']:.2f} counted {r['share_counted']:.0%}  MAE/frame {r['mae_per_frame']:.2f}  r {r['frame_corr']:.2f}  MaxN share {r['maxn_share']:.0%}")
    print(f"wrote {out / 'counts.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
