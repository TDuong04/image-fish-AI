#!/usr/bin/env python
"""Score detectors of several kinds on a source none of them saw (the unseen-source test), with one AP routine.

    python scripts/eval_unseen.py --data deepecomar                       # arms in configs/unseen_arms.yaml
    python scripts/eval_unseen.py --data deepecomar --labels labels_nosmall
    python scripts/eval_unseen.py --limit 40 --boot 20                    # smoke test

Handles Ultralytics checkpoints, the published YOLO-Fish darknet detectors (cfg + weights, run with OpenCV's DNN module on the CPU) and torchvision
state dicts. Every model is run once on every image; AP at IoU 0.50 and 0.50 to 0.95 uses the same matching and 101-point interpolation as
scripts/bootstrap_ci.py, found fish and false alarms are counted at a fixed confidence of 0.25 (nothing is tuned on this set), and intervals resample
groups of images (`groups.json` from scripts/prepare_deepecomar.py) with the same resample for every model. An arm score is the mean over its models.
Refuses the sealed OzFish test. See docs/eval_protocol.md item 10 for the rules fixed before this was run.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402
import bootstrap_ci as bc  # noqa: E402
import error_analysis as ea  # noqa: E402

MIN_CONF = 0.005
OPERATING_CONF = 0.25


def darknet_input_size(cfg: Path) -> tuple[int, int]:
    w = h = None
    for line in cfg.read_text().splitlines():
        m = re.match(r"\s*(width|height)\s*=\s*(\d+)", line)
        if m and m.group(1) == "width" and w is None:
            w = int(m.group(2))
        if m and m.group(1) == "height" and h is None:
            h = int(m.group(2))
        if w and h:
            break
    return w or 416, h or 416


def predict_darknet(cfg: Path, weights: Path, images, threads: int = 2):
    import cv2
    cv2.setNumThreads(threads)       # OpenCV's DNN otherwise takes most cores of a laptop for an hour per model
    net = cv2.dnn.readNetFromDarknet(str(cfg), str(weights))
    out_names = net.getUnconnectedOutLayersNames()
    iw, ih = darknet_input_size(cfg)
    boxes, confs = [], []
    for p in images:
        img = cv2.imread(str(p))
        H, W = img.shape[:2]
        net.setInput(cv2.dnn.blobFromImage(img, 1 / 255.0, (iw, ih), swapRB=True, crop=False))
        cand, score = [], []
        for out in net.forward(out_names):
            for row in out:
                s = float(row[4] * row[5:].max()) if row.shape[0] > 5 else float(row[4])
                if s >= MIN_CONF:
                    cx, cy, bw, bh = row[0] * W, row[1] * H, row[2] * W, row[3] * H
                    cand.append([cx - bw / 2, cy - bh / 2, bw, bh])
                    score.append(s)
        if cand:
            keep = np.array(cv2.dnn.NMSBoxes(cand, score, MIN_CONF, 0.45)).reshape(-1)
            b = np.array(cand, dtype=float)[keep]
            b[:, 2:] += b[:, :2]
            boxes.append(b)
            confs.append(np.array(score, dtype=float)[keep])
        else:
            boxes.append(np.zeros((0, 4)))
            confs.append(np.zeros(0))
    return boxes, confs


def predict_torchvision(weights: Path, images, device):
    import torch
    import train_tv as tv
    model = tv.build_model(960, pretrained=False)
    model.load_state_dict(torch.load(weights, map_location="cpu"))
    return tv.predict_images(model.to(device.torch), images, device)


def get_predictions(kind, spec, images, device, imgsz, cache_dir: Path, tag: str):
    cache = cache_dir / f"{tag}.npz"
    if cache.exists():
        z = np.load(cache, allow_pickle=True)
        return list(z["boxes"]), list(z["conf"])
    if kind == "ultralytics":
        boxes, confs = bc.predict_all(paths.MODELS_ROOT / spec, images, device, imgsz, MIN_CONF, cache)
        return boxes, confs
    if kind == "darknet":
        boxes, confs = predict_darknet(paths.REPO_ROOT / spec["cfg"], paths.REPO_ROOT / spec["weights"], images)
    elif kind == "torchvision":
        boxes, confs = predict_torchvision(paths.MODELS_ROOT / spec, images, device)
    else:
        raise SystemExit(f"unknown kind {kind!r}")
    cache.parent.mkdir(parents=True, exist_ok=True)
    ob, oc = np.empty(len(boxes), dtype=object), np.empty(len(confs), dtype=object)
    for i, (b, c) in enumerate(zip(boxes, confs)):
        ob[i], oc[i] = b, c
    np.savez(cache, boxes=ob, conf=oc)
    return boxes, confs


def found_and_false(gts, boxes, confs, thr):
    tp = fp = 0
    for gt, b, c in zip(gts, boxes, confs):
        a, fpi, _ = ea.match_at(gt, b, c, thr)
        tp, fp = tp + a, fp + len(fpi)
    return tp, fp


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=paths.CONFIG_DIR / "unseen_arms.yaml")
    ap.add_argument("--data", default="deepecomar")
    ap.add_argument("--labels", default="labels", help="label folder under the dataset (labels or labels_nosmall)")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--arms", nargs="*", default=None, help="only these arms of the config (used to cache slow CPU models in parallel processes)")
    ap.add_argument("--predict-only", action="store_true", help="stop after the predictions are cached (no scoring)")
    args = ap.parse_args()
    if "ozfish_eval" in args.data:
        sys.exit("refusing: ozfish_eval is the sealed test set.")

    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    if args.arms:
        cfg["arms"] = {k: v for k, v in cfg["arms"].items() if k in args.arms}
    root = paths.DATA_ROOT / "processed" / args.data
    img_dir, lab_dir = root / "images" / "val", root / args.labels / "val"
    if not img_dir.exists():
        sys.exit(f"{img_dir} does not exist. Run scripts/prepare_{args.data}.py")
    images = sorted(img_dir.glob("*.jpg"))
    if args.limit:
        images = images[: args.limit]
    import cv2
    gts = []
    for p in images:
        h, w = cv2.imread(str(p)).shape[:2]
        gts.append(ea.read_labels(lab_dir / (p.stem + ".txt"), w, h))
    n_gt = np.array([len(g) for g in gts], dtype=float)
    groups_file = root / "groups.json"
    gmap = json.loads(groups_file.read_text()) if groups_file.exists() else {}
    keys = [gmap.get(p.stem, p.stem) for p in images]
    uniq = sorted(set(keys))
    cid = np.array([uniq.index(k) for k in keys])
    print(f"{len(images)} images, {int(n_gt.sum())} boxes, {len(uniq)} groups ({'file-name groups' if gmap else 'frames'})")

    device = resolve_device("auto")
    cache_dir = paths.DATA_ROOT / ".cache" / "unseen"
    scorers, models_info = {}, {}
    for arm, a in cfg["arms"].items():
        for m in a["models"]:
            name = m if isinstance(m, str) else Path(m["weights"]).stem
            tag = f"{args.data}-{name}-{len(images)}"
            print(f"  {arm}: {name} ...", flush=True)
            boxes, confs = get_predictions(a["kind"], m, images, device, a.get("imgsz", 960), cache_dir, tag)
            scorers[name] = bc.Scorer(gts, boxes, confs)
            tp, fp = found_and_false(gts, boxes, confs, OPERATING_CONF)
            models_info[name] = {"arm": arm, "found_at_0.25": tp, "false_alarms_at_0.25": fp}

    if args.predict_only:
        print("predictions cached; no scoring (--predict-only)")
        return 0

    rng = np.random.default_rng(args.seed)
    full = np.ones(len(images))
    idx50, idx5095 = [0], list(range(len(bc.IOUS)))
    arms = {arm: [m if isinstance(m, str) else Path(m["weights"]).stem for m in a["models"]] for arm, a in cfg["arms"].items()}

    def arm_scores(w, ious):
        return {arm: float(np.mean([scorers[n].ap(w, n_gt, ious) for n in names])) for arm, names in arms.items()}

    point = {"AP50": arm_scores(full, idx50), "AP50_95": arm_scores(full, idx5095)}
    boots = {"AP50": [], "AP50_95": []}
    for b in range(args.boot):
        draw = rng.integers(0, len(uniq), len(uniq))
        w = np.bincount(draw, minlength=len(uniq)).astype(float)[cid]
        boots["AP50"].append(arm_scores(w, idx50))
        boots["AP50_95"].append(arm_scores(w, idx5095))

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    per_model = {}
    for n, sc in scorers.items():
        per_model[n] = {**models_info[n], "AP50": round(sc.ap(full, n_gt, idx50), 4), "AP50_95": round(sc.ap(full, n_gt, idx5095), 4)}
    report = {"data": args.data, "labels": args.labels, "images": len(images), "boxes": int(n_gt.sum()), "groups": len(uniq), "boot": args.boot,
              "min_conf": MIN_CONF, "operating_conf": OPERATING_CONF, "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
              "note": "arm score = mean over the arm's models; interval = group sampling only; 101-point AP on predictions above min_conf",
              "arms": {}, "models": per_model}
    for arm in arms:
        report["arms"][arm] = {m: {"point": round(point[m][arm], 4), "ci95": ci([x[arm] for x in boots[m]])} for m in ("AP50", "AP50_95")}
    out = args.out or (paths.RUNS_ROOT / f"unseen-{datetime.now().strftime('%Y%m%d')}-{args.data}-{args.labels}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "unseen.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for arm, d in report["arms"].items():
        print(f"{arm:22s} AP50 {d['AP50']['point']:.4f} {d['AP50']['ci95']}  AP50-95 {d['AP50_95']['point']:.4f} {d['AP50_95']['ci95']}")
    print(f"wrote {out / 'unseen.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
