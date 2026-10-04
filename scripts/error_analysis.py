#!/usr/bin/env python
"""Step 0 of the improvement plan: find out HOW a checkpoint fails before changing anything.

    python scripts/error_analysis.py --model models/20260923-yolov8n-960-s2.pt
    python scripts/error_analysis.py --model models/X.pt --conf 0.25 --worst 12

mAP says a model is weak; it does not say why. This splits every ground-truth box and every
prediction on the DeepFish *validation* split into outcomes, at IoU 0.5:

  missed fish    no prediction at all / found only under the threshold / found but badly placed
  false alarms   duplicate of a matched fish / badly placed / on background
  by box size    recall per size bucket, sizes measured as seen at 640 input (as in PROGRESS F3)
  confidence     precision, recall and F1 across thresholds, to check 0.25 is not hiding recall

The remedies differ by cause: low-confidence hits are a threshold problem, bad placement points at
labels or resolution, background false alarms point at data. Refuses the sealed OzFish set: nothing
here may be tuned against it (docs/eval_protocol.md). Writes JSON plus the worst frames drawn.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402

NAME = re.compile(r"yolov8n-(?P<imgsz>\d+)-s(?P<seed>\d+)")
IOU_MATCH = 0.5
BUCKETS = [("<16", 0, 16), ("16-32", 16, 32), ("32-96", 32, 96), (">96", 96, 1e9)]
THRESHOLDS = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6]


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(n,4) x (m,4) xyxy -> (n,m) IoU."""
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)))
    lt = np.maximum(a[:, None, :2], b[None, :, :2])
    rb = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.clip(rb - lt, 0, None).prod(axis=2)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (area_a[:, None] + area_b[None, :] - inter + 1e-9)


def read_labels(path: Path, w: int, h: int) -> np.ndarray:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) == 5:
                cx, cy, bw, bh = (float(x) for x in p[1:])
                rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=float).reshape(-1, 4)


def match_at(gt: np.ndarray, pred: np.ndarray, conf: np.ndarray, thr: float):
    """Greedy match by confidence. Returns (tp, fp_idx, fn_idx, gt_to_pred) for predictions >= thr."""
    keep = np.where(conf >= thr)[0]
    keep = keep[np.argsort(-conf[keep])]
    ious = iou_matrix(pred[keep], gt)
    used, tp, fp = set(), 0, []
    for row, k in enumerate(keep):
        if ious.shape[1] == 0:
            fp.append(k)
            continue
        cand = [(ious[row, j], j) for j in range(ious.shape[1]) if j not in used and ious[row, j] >= IOU_MATCH]
        if cand:
            used.add(max(cand)[1])
            tp += 1
        else:
            fp.append(k)
    fn = [j for j in range(len(gt)) if j not in used]
    return tp, fp, fn


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--data", default="deepfish", help="processed dataset name; ozfish_eval is refused")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.25, help="operating threshold used for the breakdown")
    ap.add_argument("--imgsz", type=int, default=None, help="default: read from the checkpoint name")
    ap.add_argument("--worst", type=int, default=12, help="how many worst frames to draw")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if "ozfish_eval" in args.data:
        sys.exit("refusing: ozfish_eval is the sealed test set and must not be analysed or tuned against.")
    if not args.model.exists():
        sys.exit(f"model not found: {args.model}")
    m = NAME.search(args.model.name)
    imgsz = args.imgsz or (int(m["imgsz"]) if m else None)
    if imgsz is None:
        sys.exit("cannot tell the training resolution from the file name; pass --imgsz.")

    root = paths.DATA_ROOT / "processed" / args.data
    img_dir, lab_dir = root / "images" / args.split, root / "labels" / args.split
    if not img_dir.exists():
        sys.exit(f"{img_dir} does not exist. Run: python scripts/get_data.py")
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    if args.limit:
        images = images[: args.limit]

    import cv2  # noqa: E402
    from ultralytics import YOLO  # noqa: E402

    device = resolve_device("auto")
    print(f"model {args.model.name}  imgsz {imgsz}  device {device}  images {len(images)}")
    model = YOLO(str(args.model))

    per_image = []
    for i in range(0, len(images), 32):
        batch = images[i:i + 32]
        results = model.predict([str(p) for p in batch], imgsz=imgsz, conf=0.001, iou=0.7,
                                device=device.ultralytics, verbose=False)
        for p, r in zip(batch, results):
            h, w = r.orig_shape
            per_image.append({
                "path": p, "w": w, "h": h,
                "gt": read_labels(lab_dir / (p.stem + ".txt"), w, h),
                "pred": r.boxes.xyxy.cpu().numpy().astype(float).reshape(-1, 4),
                "conf": r.boxes.conf.cpu().numpy().astype(float),
            })

    # confidence sweep
    sweep = []
    n_gt = sum(len(d["gt"]) for d in per_image)
    for t in THRESHOLDS:
        tp = fp = 0
        for d in per_image:
            a, b, _ = match_at(d["gt"], d["pred"], d["conf"], t)
            tp, fp = tp + a, fp + len(b)
        prec, rec = tp / max(tp + fp, 1), tp / max(n_gt, 1)
        sweep.append({"conf": t, "precision": round(prec, 4), "recall": round(rec, 4),
                      "f1": round(2 * prec * rec / max(prec + rec, 1e-9), 4), "tp": tp, "fp": fp,
                      "fn": n_gt - tp})

    # outcome breakdown at the operating threshold
    t = args.conf
    fn_kind = {"no prediction": 0, "found but under threshold": 0, "badly placed": 0}
    fp_kind = {"duplicate": 0, "badly placed": 0, "background": 0}
    size = {name: [0, 0] for name, _, _ in BUCKETS}          # [found, total]
    empty_frames = empty_fp = 0
    scored = []
    for d in per_image:
        tp, fp_idx, fn_idx = match_at(d["gt"], d["pred"], d["conf"], t)
        scale = 640.0 / max(d["w"], d["h"])
        gt_iou_all = iou_matrix(d["gt"], d["pred"])
        for j in range(len(d["gt"])):
            wj = (d["gt"][j, 2] - d["gt"][j, 0]) * scale
            hj = (d["gt"][j, 3] - d["gt"][j, 1]) * scale
            s = float(np.sqrt(max(wj * hj, 0)))
            for name, lo, hi in BUCKETS:
                if lo <= s < hi:
                    size[name][1] += 1
                    size[name][0] += 0 if j in fn_idx else 1
        for j in fn_idx:
            above = d["conf"] >= t
            if gt_iou_all.shape[1] and (gt_iou_all[j][~above] >= IOU_MATCH).any():
                fn_kind["found but under threshold"] += 1
            elif gt_iou_all.shape[1] and (gt_iou_all[j][above] >= 0.1).any():
                fn_kind["badly placed"] += 1
            else:
                fn_kind["no prediction"] += 1
        for k in fp_idx:
            row = iou_matrix(d["pred"][k:k + 1], d["gt"])[0] if len(d["gt"]) else np.zeros(0)
            if len(row) and row.max() >= IOU_MATCH:
                fp_kind["duplicate"] += 1
            elif len(row) and row.max() >= 0.1:
                fp_kind["badly placed"] += 1
            else:
                fp_kind["background"] += 1
        if len(d["gt"]) == 0:
            empty_frames += 1
            empty_fp += len(fp_idx)
        scored.append((len(fn_idx) + len(fp_idx), d, fn_idx, fp_idx))

    total_fn, total_fp = sum(fn_kind.values()), sum(fp_kind.values())
    best = max(sweep, key=lambda r: r["f1"])
    report = {
        "model": args.model.name, "imgsz": imgsz, "data": args.data, "split": args.split,
        "images": len(per_image), "gt_boxes": n_gt, "iou_match": IOU_MATCH, "operating_conf": t,
        "device": str(device), "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "sweep": sweep, "best_f1_threshold": best["conf"],
        "missed_fish": {"total": total_fn, **fn_kind},
        "false_alarms": {"total": total_fp, **fp_kind},
        "recall_by_size_at_640_input": {k: {"found": v[0], "total": v[1],
                                            "recall": round(v[0] / max(v[1], 1), 4)} for k, v in size.items()},
        "empty_frames": {"frames": empty_frames, "false_alarms": empty_fp},
        "note": "single split of near-duplicate video frames: counts are descriptive, not independent samples",
    }

    out = args.out or (paths.RUNS_ROOT / f"analysis-{datetime.now().strftime('%Y%m%d')}-{args.model.stem}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "analysis.json").write_text(json.dumps(report, indent=2))

    scored.sort(key=lambda x: -x[0])
    for rank, (_, d, fn_idx, fp_idx) in enumerate(scored[: args.worst]):
        img = cv2.imread(str(d["path"]))
        for g in d["gt"]:
            cv2.rectangle(img, (int(g[0]), int(g[1])), (int(g[2]), int(g[3])), (0, 200, 0), 2)
        for k in fp_idx:
            b = d["pred"][k]
            cv2.rectangle(img, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 0, 255), 2)
        cv2.imwrite(str(out / f"worst_{rank:02d}_{d['path'].stem}.jpg"), img)

    print(json.dumps({k: report[k] for k in ("missed_fish", "false_alarms", "recall_by_size_at_640_input",
                                             "empty_frames", "best_f1_threshold")}, indent=2))
    for r in sweep:
        print(f"  conf {r['conf']:.2f}  precision {r['precision']:.3f}  recall {r['recall']:.3f}  f1 {r['f1']:.3f}")
    print(f"\nwrote {out}  (green = ground truth, red = false alarms)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
