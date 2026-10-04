#!/usr/bin/env python
"""FD-105: look at the labels behind the errors, instead of assuming they are right.

    python scripts/label_audit.py --model models/20260923-yolov8n-960-s2.pt
    python scripts/label_audit.py --model models/X.pt --missed 48 --found 24 --false-alarms 36

error_analysis.py says most misses are small fish with no prediction at all. That is only
a model problem if the labelled box really contains a findable fish. This draws a seeded
random sample into contact sheets for a person to review:

  M## a labelled fish the model MISSED      F## a labelled fish it FOUND (the control)
  B## a confident box on background (a false alarm, possibly an unlabelled fish)

Each tile is a zoomed crop around the box with the box outlined. The control group matters:
it shows what "clearly a fish" looks like in this footage, so a missed box can be judged
against it. Verdicts go in verdicts.json (id -> clear | ambiguous | nothing-visible, and for
B## also fish | not-fish) and `--summarise` turns them into counts. A verdict is one
reviewer's judgement of one still frame; it is evidence about label quality, not a measurement.

Validation split only; the sealed OzFish set is refused.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from error_analysis import NAME, iou_matrix, match_at, read_labels  # noqa: E402
from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402

TILE, COLS, ROWS = 256, 4, 3


def tile_for(img, box, tag):
    import cv2
    h, w = img.shape[:2]
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    side = int(max(160, 3 * max(box[2] - box[0], box[3] - box[1])))
    x0 = int(min(max(cx - side / 2, 0), max(w - side, 0)))
    y0 = int(min(max(cy - side / 2, 0), max(h - side, 0)))
    crop = img[y0:y0 + side, x0:x0 + side]
    k = TILE / max(crop.shape[:2])
    crop = cv2.resize(crop, (int(crop.shape[1] * k), int(crop.shape[0] * k)), interpolation=cv2.INTER_CUBIC)
    canvas = np.full((TILE, TILE, 3), 30, dtype=np.uint8)
    canvas[:crop.shape[0], :crop.shape[1]] = crop
    p1 = (int((box[0] - x0) * k), int((box[1] - y0) * k))
    p2 = (int((box[2] - x0) * k), int((box[3] - y0) * k))
    cv2.rectangle(canvas, p1, p2, (0, 255, 255), 1)
    cv2.putText(canvas, tag, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)
    cv2.putText(canvas, tag, (4, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 1)
    return canvas


def summarise(out: Path) -> int:
    manifest = json.loads((out / "manifest.json").read_text())
    vfile = out / "verdicts.json"
    if not vfile.exists():
        sys.exit(f"no {vfile}: write verdicts first (id -> clear | ambiguous | nothing-visible).")
    verdicts = json.loads(vfile.read_text())
    groups = {"missed": "M", "found": "F", "false_alarm": "B"}
    res = {}
    for name, prefix in groups.items():
        ids = [m["id"] for m in manifest if m["id"].startswith(prefix)]
        counts: dict = {}
        for i in ids:
            v = verdicts.get(i, "unreviewed")
            counts[v] = counts.get(v, 0) + 1
        res[name] = {"n": len(ids), **counts}
    (out / "summary.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", type=Path)
    ap.add_argument("--data", default="deepfish")
    ap.add_argument("--split", default="val")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--missed", type=int, default=48)
    ap.add_argument("--found", type=int, default=24)
    ap.add_argument("--false-alarms", type=int, default=36)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--summarise", type=Path, default=None, help="summarise verdicts.json in this audit folder")
    args = ap.parse_args()

    if args.summarise:
        return summarise(args.summarise)
    if "ozfish_eval" in args.data:
        sys.exit("refusing: ozfish_eval is the sealed test set.")
    if not args.model or not args.model.exists():
        sys.exit("--model is required and must exist.")
    m = NAME.search(args.model.name)
    if not m:
        sys.exit("cannot read the training resolution from the file name.")
    imgsz = int(m["imgsz"])

    root = paths.DATA_ROOT / "processed" / args.data
    img_dir, lab_dir = root / "images" / args.split, root / "labels" / args.split
    if not img_dir.exists():
        sys.exit(f"{img_dir} does not exist. Run: python scripts/get_data.py")
    images = sorted(p for p in img_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})

    import cv2
    from ultralytics import YOLO

    device = resolve_device("auto")
    model = YOLO(str(args.model))
    pool = {"M": [], "F": [], "B": []}
    for i in range(0, len(images), 32):
        batch = images[i:i + 32]
        results = model.predict([str(p) for p in batch], imgsz=imgsz, conf=0.001, iou=0.7,
                                device=device.ultralytics, verbose=False)
        for p, r in zip(batch, results):
            h, w = r.orig_shape
            gt = read_labels(lab_dir / (p.stem + ".txt"), w, h)
            pred = r.boxes.xyxy.cpu().numpy().astype(float).reshape(-1, 4)
            conf = r.boxes.conf.cpu().numpy().astype(float)
            _, fp_idx, fn_idx = match_at(gt, pred, conf, args.conf)
            for j in range(len(gt)):
                pool["M" if j in fn_idx else "F"].append((p, gt[j].tolist(), None))
            for k in fp_idx:
                row = iou_matrix(pred[k:k + 1], gt)[0] if len(gt) else np.zeros(0)
                if not len(row) or row.max() < 0.1:
                    pool["B"].append((p, pred[k].tolist(), float(conf[k])))

    rng = random.Random(args.seed)
    want = {"M": args.missed, "F": args.found, "B": args.false_alarms}
    picked = []
    for key in ("M", "F", "B"):
        items = pool[key]
        picked += [(key, it) for it in rng.sample(items, min(want[key], len(items)))]
        print(f"{key}: {len(items)} available, {min(want[key], len(items))} sampled")

    out = args.out or (paths.RUNS_ROOT / f"audit-labels-{args.model.stem}")
    out.mkdir(parents=True, exist_ok=True)
    manifest, tiles, counter = [], [], {"M": 0, "F": 0, "B": 0}
    for key, (p, box, conf) in picked:
        counter[key] += 1
        tag = f"{key}{counter[key]:02d}"
        img = cv2.imread(str(p))
        tiles.append(tile_for(img, box, tag))
        side = max(box[2] - box[0], box[3] - box[1]) * 640.0 / max(img.shape[:2])
        manifest.append({"id": tag, "frame": p.name, "clip": p.stem.rsplit("_f", 1)[0],
                         "box_xyxy": [round(v, 1) for v in box],
                         "size_at_640": round(float(side), 1), "conf": conf})
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    per = COLS * ROWS
    for s in range(0, len(tiles), per):
        chunk = tiles[s:s + per]
        chunk += [np.zeros((TILE, TILE, 3), dtype=np.uint8)] * (per - len(chunk))
        rows = [np.hstack(chunk[r * COLS:(r + 1) * COLS]) for r in range(ROWS)]
        cv2.imwrite(str(out / f"sheet_{s // per:02d}.jpg"), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"wrote {len(tiles)} tiles in {-(-len(tiles) // per)} sheets to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
