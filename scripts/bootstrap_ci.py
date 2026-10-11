#!/usr/bin/env python
"""Confidence intervals that respect the structure of the data: resample whole clips (or sites), not frames.

    python scripts/bootstrap_ci.py                         # arms in configs/stats_arms.yaml, DeepFish validation
    python scripts/bootstrap_ci.py --limit 60 --boot 50    # smoke test

Frames from one video are near-duplicates, so a frame-level interval is far too narrow. This runs every checkpoint once, matches its
predictions to the labels at IoU 0.50 to 0.95, and then repeats the scoring on resamples of whole clips (cluster bootstrap, the same resample
for every model, so differences between arms are paired). An arm score is the mean over its seeds; the interval therefore reflects the sampling of
clips and does not include seed variance (that is reported separately, see PROGRESS F18, F24, F32). Clips are the video ids in the file names
(`<site>_<clip>_f<frame>`); `--by site` resamples the six sites instead (very few clusters: the intervals are wide and honest).

AP here is a 101-point AP on predictions above `--min-conf`, close to but not identical to the Ultralytics mAP. Refuses the sealed OzFish test set.
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
import yaml  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402
import error_analysis as ea  # noqa: E402

IOUS = np.round(np.arange(0.5, 0.96, 0.05), 2)


def clip_of(stem: str) -> str:
    return "_".join(stem.split("_")[:-1])


def match_image(gt: np.ndarray, pred: np.ndarray, conf: np.ndarray):
    """Greedy match by confidence at every IoU threshold. Returns (order, tp[n_pred, n_iou]) for predictions sorted by confidence."""
    order = np.argsort(-conf)
    tp = np.zeros((len(order), len(IOUS)), dtype=bool)
    if len(gt) and len(order):
        ious = ea.iou_matrix(pred[order], gt)
        for t, thr in enumerate(IOUS):
            used = np.zeros(len(gt), dtype=bool)
            for i in range(len(order)):
                row = np.where(used, -1.0, ious[i])
                j = int(row.argmax())
                if row[j] >= thr:
                    used[j] = True
                    tp[i, t] = True
    return order, tp


def predict_all(model_path: Path, images, device, imgsz: int, min_conf: float, cache: Path):
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["boxes"]), list(z["conf"])
    from ultralytics import YOLO
    model = YOLO(str(model_path))
    boxes, confs = [], []
    for i in range(0, len(images), 16):         # a list source is one batch, so chunk it explicitly
        for r in model.predict([str(p) for p in images[i:i + 16]], imgsz=imgsz, conf=min_conf, iou=0.7,
                               device=device.ultralytics, verbose=False):
            boxes.append(r.boxes.xyxy.cpu().numpy().astype(float).reshape(-1, 4))
            confs.append(r.boxes.conf.cpu().numpy().astype(float))
    cache.parent.mkdir(parents=True, exist_ok=True)
    obj_b = np.empty(len(boxes), dtype=object)
    obj_c = np.empty(len(confs), dtype=object)
    for i, (b, c) in enumerate(zip(boxes, confs)):
        obj_b[i], obj_c[i] = b, c
    np.savez(cache, boxes=obj_b, conf=obj_c)
    return boxes, confs


class Scorer:
    """Pre-sorted predictions of one model, so a resample is a weighted cumulative sum."""

    def __init__(self, gts, boxes, confs):
        conf_all, tp_all, img_all = [], [], []
        for k, (gt, b, c) in enumerate(zip(gts, boxes, confs)):
            order, tp = match_image(gt, b, c)
            conf_all.append(c[order])
            tp_all.append(tp)
            img_all.append(np.full(len(order), k))
        conf = np.concatenate(conf_all)
        self.tp = np.concatenate(tp_all)
        self.img = np.concatenate(img_all)
        o = np.argsort(-conf, kind="stable")
        self.tp, self.img = self.tp[o], self.img[o]
        self.grid = np.linspace(0, 1, 101)

    def ap(self, w_img: np.ndarray, n_gt_img: np.ndarray, iou_idx):
        w = w_img[self.img]
        n_gt = float((w_img * n_gt_img).sum())
        if n_gt == 0:
            return 0.0
        out = []
        for t in iou_idx:
            tpw = np.cumsum(w * self.tp[:, t])
            fpw = np.cumsum(w * ~self.tp[:, t])
            rec = tpw / n_gt
            prec = np.maximum.accumulate((tpw / np.maximum(tpw + fpw, 1e-12))[::-1])[::-1]
            idx = np.searchsorted(rec, self.grid, side="left")
            out.append(np.where(idx < len(prec), prec[np.minimum(idx, len(prec) - 1)], 0.0).mean())
        return float(np.mean(out))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=paths.CONFIG_DIR / "stats_arms.yaml")
    ap.add_argument("--data", default="deepfish")
    ap.add_argument("--split", default="val")
    ap.add_argument("--by", choices=("clip", "site"), default="clip")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--min-conf", type=float, default=0.005)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    if "ozfish_eval" in args.data:
        sys.exit("refusing: ozfish_eval is the sealed test set and must not be used for anything but its single planned run.")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    root = paths.DATA_ROOT / "processed" / args.data
    img_dir, lab_dir = root / "images" / args.split, root / "labels" / args.split
    if not img_dir.exists():
        sys.exit(f"{img_dir} does not exist. Run: python scripts/get_data.py")
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        images = images[: args.limit]
    import cv2  # noqa: E402
    gts = []
    for p in images:
        h, w = cv2.imread(str(p)).shape[:2]
        gts.append(ea.read_labels(lab_dir / (p.stem + ".txt"), w, h))
    n_gt = np.array([len(g) for g in gts], dtype=float)
    keys = [clip_of(p.stem) if args.by == "clip" else p.stem.split("_")[0] for p in images]
    uniq = sorted(set(keys))
    cid = np.array([uniq.index(k) for k in keys])
    print(f"{len(images)} images, {len(uniq)} {args.by}s, {int(n_gt.sum())} boxes")

    device = resolve_device("auto")
    cache_dir = paths.DATA_ROOT / ".cache" / "bootstrap"
    scorers = {}
    for arm, models in cfg["arms"].items():
        for m in models:
            mp = paths.MODELS_ROOT / m
            if not mp.exists():
                sys.exit(f"missing checkpoint: {mp}")
            tag = f"{m}-{args.data}-{args.split}-{args.imgsz}-{args.min_conf}-{len(images)}"
            print(f"  scoring {m} ...", flush=True)
            boxes, confs = predict_all(mp, images, device, args.imgsz, args.min_conf, cache_dir / f"{tag}.npz")
            scorers[m] = Scorer(gts, boxes, confs)

    rng = np.random.default_rng(args.seed)
    full = np.ones(len(images))
    idx50, idx5095 = [0], list(range(len(IOUS)))

    def arm_scores(w, iou_idx):
        return {arm: float(np.mean([scorers[m].ap(w, n_gt, iou_idx) for m in models])) for arm, models in cfg["arms"].items()}

    point = {"AP50": arm_scores(full, idx50), "AP50_95": arm_scores(full, idx5095)}
    boots = {"AP50": [], "AP50_95": []}
    for b in range(args.boot):
        draw = rng.integers(0, len(uniq), len(uniq))
        w = np.bincount(draw, minlength=len(uniq)).astype(float)[cid]
        boots["AP50"].append(arm_scores(w, idx50))
        boots["AP50_95"].append(arm_scores(w, idx5095))
        if (b + 1) % 100 == 0:
            print(f"  bootstrap {b + 1}/{args.boot}", flush=True)

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    report = {"model_set": "configs/stats_arms.yaml", "data": args.data, "split": args.split, "cluster": args.by, "clusters": len(uniq),
              "images": len(images), "boot": args.boot, "min_conf": args.min_conf, "imgsz": args.imgsz,
              "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "note": "arm score = mean over the arm's seeds; interval = clip/site sampling only, not seed variance; 101-point AP on predictions above min_conf",
              "arms": {}, "contrasts": []}
    for metric in ("AP50", "AP50_95"):
        for arm in cfg["arms"]:
            v = [x[arm] for x in boots[metric]]
            report["arms"].setdefault(arm, {})[metric] = {"point": round(point[metric][arm], 4), "ci95": ci(v)}
        for a, b_ in cfg["contrasts"]:
            d = [x[b_] - x[a] for x in boots[metric]]
            report["contrasts"].append({"metric": metric, "from": a, "to": b_, "diff": round(point[metric][b_] - point[metric][a], 4),
                                        "ci95": ci(d), "share_of_resamples_positive": round(float(np.mean(np.array(d) > 0)), 3)})
    out = args.out or (paths.RUNS_ROOT / f"stats-{datetime.now().strftime('%Y%m%d')}-bootstrap-{args.by}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "ci.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for arm, d in report["arms"].items():
        print(f"{arm:24s} AP50 {d['AP50']['point']:.4f} {d['AP50']['ci95']}   AP50-95 {d['AP50_95']['point']:.4f} {d['AP50_95']['ci95']}")
    for c in report["contrasts"]:
        print(f"{c['metric']:8s} {c['from']} -> {c['to']}: {c['diff']:+.4f} {c['ci95']}  positive in {c['share_of_resamples_positive']:.1%}")
    print(f"wrote {out / 'ci.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
