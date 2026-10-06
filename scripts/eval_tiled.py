#!/usr/bin/env python
"""Does tiled inference find more small fish than one pass over the whole frame?

    python scripts/eval_tiled.py --model models/20261005-yolov8s-960-kakadu-ozfish-s0.pt
    python scripts/eval_tiled.py --model models/X.pt --limit 40          # smoke test

At 960 px a 1920x1080 frame is shrunk 2x, so a 20 px fish is seen as 10 px. Tiling runs the same
checkpoint on overlapping crops (each resized to `--imgsz`), so small fish are seen at a larger
scale, then merges the boxes with NMS. No training, no new weights.

Scored on the DeepFish *validation* split only (the sealed OzFish set is refused), with one AP routine
for every variant so the untiled row is a like-for-like control: AP at IoU 0.5 and mean over
0.50:0.95 (101-point interpolation, so close to but not identical to Ultralytics' own mAP), fish found
and false alarms at the operating confidence, recall by box size, and mean seconds per frame.
Seconds per frame are from whatever machine this runs on and say nothing about the Orin.

Tile boxes that touch an inner tile edge are dropped by default (`--edge-drop`): a fish cut by the tile
border yields a partial box that duplicates the whole fish found by a neighbour or the full frame.
A variant without the drop is scored as well, to show what the rule changes.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402
import error_analysis as ea  # noqa: E402

IOUS = np.arange(0.5, 0.96, 0.05)


def grid_tiles(w: int, h: int, cols: int, rows: int, overlap: float):
    """Tile boxes (x0, y0, x1, y1) covering the frame; neighbours overlap by `overlap` of a tile."""
    tw = w / (cols - (cols - 1) * overlap)
    th = h / (rows - (rows - 1) * overlap)
    out = []
    for r in range(rows):
        for c in range(cols):
            x0 = 0 if cols == 1 else c * (w - tw) / (cols - 1)
            y0 = 0 if rows == 1 else r * (h - th) / (rows - 1)
            out.append((int(round(x0)), int(round(y0)), int(round(x0 + tw)), int(round(y0 + th))))
    return out


def nms(boxes: np.ndarray, conf: np.ndarray, thr: float):
    if len(boxes) == 0:
        return boxes, conf
    order = np.argsort(-conf)
    keep = []
    while len(order):
        i = order[0]
        keep.append(i)
        if len(order) == 1:
            break
        ious = ea.iou_matrix(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < thr]
    keep = np.array(keep, dtype=int)
    return boxes[keep], conf[keep]


def average_precision(per_image, iou_thr: float) -> float:
    """101-point AP at one IoU, pooled over images, greedy matching by confidence."""
    records, n_gt = [], 0
    for gt, pred, conf in per_image:
        n_gt += len(gt)
        order = np.argsort(-conf)
        ious = ea.iou_matrix(pred[order], gt) if len(gt) and len(pred) else np.zeros((len(order), len(gt)))
        used = set()
        for row, k in enumerate(order):
            best, bj = iou_thr, -1
            for j in range(ious.shape[1]):
                if j not in used and ious[row, j] >= best:
                    best, bj = ious[row, j], j
            if bj >= 0:
                used.add(bj)
            records.append((conf[k], bj >= 0))
    if not records or n_gt == 0:
        return 0.0
    records.sort(key=lambda r: -r[0])
    tp = np.cumsum([r[1] for r in records])
    fp = np.cumsum([not r[1] for r in records])
    rec, prec = tp / n_gt, tp / np.maximum(tp + fp, 1)
    prec = np.maximum.accumulate(prec[::-1])[::-1]
    grid = np.linspace(0, 1, 101)
    idx = np.searchsorted(rec, grid, side="left")
    return float(np.mean([prec[i] if i < len(prec) else 0.0 for i in idx]))


def summarise(per_image, sizes_wh, conf_op: float) -> dict:
    ap = [average_precision(per_image, t) for t in IOUS]
    n_gt = sum(len(g) for g, _, _ in per_image)
    tp = fp = empty_fp = 0
    size = {name: [0, 0] for name, _, _ in ea.BUCKETS}
    for (gt, pred, conf), (w, h) in zip(per_image, sizes_wh):
        a, fp_idx, fn_idx = ea.match_at(gt, pred, conf, conf_op)
        tp, fp = tp + a, fp + len(fp_idx)
        if len(gt) == 0:
            empty_fp += len(fp_idx)
        scale = 640.0 / max(w, h)
        for j in range(len(gt)):
            s = float(np.sqrt(max((gt[j, 2] - gt[j, 0]) * (gt[j, 3] - gt[j, 1]), 0))) * scale
            for name, lo, hi in ea.BUCKETS:
                if lo <= s < hi:
                    size[name][1] += 1
                    size[name][0] += 0 if j in fn_idx else 1
    return {
        "map50": round(ap[0], 4), "map50_95": round(float(np.mean(ap)), 4),
        "found": tp, "gt_boxes": n_gt, "false_alarms": fp, "false_alarms_on_empty_frames": empty_fp,
        "recall_by_size_at_640_input": {k: round(v[0] / max(v[1], 1), 4) for k, v in size.items()},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, type=Path)
    ap.add_argument("--data", default="deepfish")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--conf", type=float, default=0.25, help="operating threshold for found / false alarms")
    ap.add_argument("--nms-iou", type=float, default=0.7)
    ap.add_argument("--overlap", type=float, default=0.2, help="fraction of a tile shared with its neighbour")
    ap.add_argument("--edge-drop", type=int, default=4, help="drop tile boxes within N px of an inner tile edge")
    ap.add_argument("--grids", nargs="+", default=["2x2", "3x3"], help="COLSxROWS tilings to try")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if "ozfish_eval" in args.data:
        sys.exit("refusing: ozfish_eval is the sealed test set and must not be analysed or tuned against.")
    if not args.model.exists():
        sys.exit(f"model not found: {args.model}")
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
    print(f"model {args.model.name}  imgsz {args.imgsz}  device {device}  images {len(images)}  grids {args.grids}")
    model = YOLO(str(args.model))

    def predict(arrs):
        res = model.predict(arrs, imgsz=args.imgsz, conf=0.001, iou=args.nms_iou, device=device.ultralytics,
                            verbose=False)
        return [(r.boxes.xyxy.cpu().numpy().astype(float).reshape(-1, 4), r.boxes.conf.cpu().numpy().astype(float))
                for r in res]

    parsed = [tuple(int(x) for x in g.lower().split("x")) for g in args.grids]
    sets = ["full"] + [f"{c}x{r}" for c, r in parsed] + [f"{c}x{r}_keepedges" for c, r in parsed]
    pred = {s: [] for s in sets}
    gts, sizes_wh, secs = [], [], {s: 0.0 for s in sets}
    for n, p in enumerate(images):
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        gts.append(ea.read_labels(lab_dir / (p.stem + ".txt"), w, h))
        sizes_wh.append((w, h))
        t0 = time.perf_counter()
        pred["full"].append(predict([img])[0])
        secs["full"] += time.perf_counter() - t0
        for (c, r) in parsed:
            t0 = time.perf_counter()
            tiles = grid_tiles(w, h, c, r, args.overlap)
            outs = predict([img[y0:y1, x0:x1] for x0, y0, x1, y1 in tiles])
            elapsed = time.perf_counter() - t0
            secs[f"{c}x{r}"] += elapsed
            secs[f"{c}x{r}_keepedges"] += elapsed
            drop, keep = [], []
            for (x0, y0, x1, y1), (b, s) in zip(tiles, outs):
                b = b + np.array([x0, y0, x0, y0])
                keep.append((b, s))
                inner = np.zeros(len(b), dtype=bool)
                d = args.edge_drop
                inner |= (b[:, 0] <= x0 + d) & (x0 > 0)
                inner |= (b[:, 1] <= y0 + d) & (y0 > 0)
                inner |= (b[:, 2] >= x1 - d) & (x1 < w)
                inner |= (b[:, 3] >= y1 - d) & (y1 < h)
                drop.append((b[~inner], s[~inner]))
            pred[f"{c}x{r}"].append(tuple(map(np.concatenate, zip(*drop))))
            pred[f"{c}x{r}_keepedges"].append(tuple(map(np.concatenate, zip(*keep))))
        if (n + 1) % 100 == 0:
            print(f"  {n + 1}/{len(images)}", flush=True)

    variants = {}
    for s in sets:
        variants[s] = [(*nms(b, c, args.nms_iou),) for b, c in pred[s]]
        if s != "full":
            variants[s + "+full"] = [nms(np.concatenate([b, fb]), np.concatenate([c, fc]), args.nms_iou)
                                     for (b, c), (fb, fc) in zip(pred[s], pred["full"])]
            secs[s + "+full"] = secs[s] + secs["full"]

    report = {
        "model": args.model.name, "data": args.data, "split": args.split, "images": len(images),
        "imgsz": args.imgsz, "operating_conf": args.conf, "nms_iou": args.nms_iou, "overlap": args.overlap,
        "edge_drop_px": args.edge_drop, "device": str(device),
        "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "variants": {}, "note": "AP here is a 101-point pooled AP shared by all rows, close to but not identical to "
                                "Ultralytics' mAP. Seconds per frame are from this machine, not the Orin. Frames from one "
                                "video are near-duplicates: descriptive, not independent samples.",
    }
    for s, per in variants.items():
        rec = summarise([(g, b, c) for g, (b, c) in zip(gts, per)], sizes_wh, args.conf)
        rec["seconds_per_frame_this_machine"] = round(secs[s] / max(len(images), 1), 3)
        report["variants"][s] = rec
        print(f"{s:22s} AP50 {rec['map50']:.4f}  AP50-95 {rec['map50_95']:.4f}  found {rec['found']}/{rec['gt_boxes']}  "
              f"FA {rec['false_alarms']}  16-32px {rec['recall_by_size_at_640_input']['16-32']:.3f}  "
              f"{rec['seconds_per_frame_this_machine']:.2f}s/frame")

    out = args.out or (paths.RUNS_ROOT / f"tiled-{datetime.now().strftime('%Y%m%d')}-{args.model.stem}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "tiled.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out / 'tiled.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
