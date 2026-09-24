#!/usr/bin/env python
"""Build the data behind the comparison demo: detections, matches, timings.

    python scripts/build_demo.py
    python scripts/build_demo.py --per-case 3 --out demo/demo_data.json

Runs one 640 checkpoint and one 960 checkpoint over a small set of real frames,
matches every prediction against ground truth, and records what each model found
and missed. The web page draws boxes from this JSON, so it ships one image per
frame rather than one per model.

**Frames are selected on input properties only** -- box size, object count,
emptiness -- never on which model happens to do better. Selecting frames where
960 wins would manufacture the result the demo claims to show. The selection
rule for each case is recorded in the output and printed on the page.

Latency is measured end-to-end (preprocess + inference + NMS) on whatever GPU is
present, with warm-up discarded. That is a desktop measurement and is labelled as
one; it is not a Jetson number.
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import statistics as st
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish import paths  # noqa: E402
from fish.datasets import group_key, read_label  # noqa: E402
from fish.device import resolve_device  # noqa: E402

IOU_MATCH = 0.5          # a prediction counts as finding a fish at this overlap
CONF_SHOW = 0.25         # operating threshold for the visual comparison
WEB_WIDTH = 1024         # frames are 1920 wide; this keeps the page under budget

# Buckets are measured on the NATIVE 1920-wide frame, but named for what the box
# becomes at 640 input -- which is the scale that decides whether a stride-8 head
# can represent it at all. Frames are 1920 wide, so 640 input scales by 1/3:
# a 48 px native box is 16 px to a 640 model, and 24 px to a 960 model.
NATIVE_W = 1920
S640 = NATIVE_W / 640          # 3.0
BUCKETS = [
    (0, 16 * S640, "<16px @640"),          # native <48   -- below the stride-8 floor
    (16 * S640, 32 * S640, "16-32px @640"),  # native 48-96  -- the marginal band
    (32 * S640, 96 * S640, "32-96px @640"),  # native 96-288
    (96 * S640, 10**9, ">96px @640"),
]


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    ua = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / ua if ua > 0 else 0.0


def gt_boxes(label_path: Path, w: int, h: int) -> list[list[float]]:
    boxes, _ = read_label(label_path)
    out = []
    for _c, cx, cy, bw, bh in boxes:
        out.append([(cx - bw / 2) * w, (cy - bh / 2) * h,
                    (cx + bw / 2) * w, (cy + bh / 2) * h])
    return out


def bucket_of(box) -> str:
    side = ((box[2] - box[0]) * (box[3] - box[1])) ** 0.5
    for lo, hi, name in BUCKETS:
        if lo <= side < hi:
            return name
    return BUCKETS[-1][2]


def pick_frames(dataset: str, per_case: int) -> list[dict]:
    """Choose frames by INPUT properties. Never by model performance."""
    from PIL import Image
    root = paths.PROCESSED_DIR / dataset / "images" / "val"
    lroot = paths.PROCESSED_DIR / dataset / "labels" / "val"
    if not root.exists():
        return []

    stats = []
    for img in sorted(root.glob("*.jpg")):
        lab = lroot / f"{img.stem}.txt"
        if not lab.exists():
            continue
        with Image.open(img) as im:
            w, h = im.size
        gts = gt_boxes(lab, w, h)
        sides = [((b[2] - b[0]) * (b[3] - b[1])) ** 0.5 for b in gts]
        stats.append({"image": img, "label": lab, "w": w, "h": h,
                      "n": len(gts), "median_side": st.median(sides) if sides else 0.0})
    return stats


def _spread(items: list[dict], n: int) -> list[dict]:
    """At most one frame per source clip. Consecutive frames of one clip are
    near-identical, so taking the top N gives N copies of the same picture."""
    seen, out = set(), []
    for s in items:
        clip = group_key(s["image"])
        if clip in seen:
            continue
        seen.add(clip)
        out.append(s)
        if len(out) >= n:
            break
    return out


def select_cases(deep: list[dict], oz: list[dict], per_case: int) -> list[dict]:
    chosen = []

    # 1. The band where the resolution effect is predicted to live: boxes around
    #    16-40 px, small enough to be hard at 640 but not so small that both
    #    models fail. Sorting by the extreme tail picked frames where BOTH find
    #    nothing, which is honest but shows nothing.
    band = [s for s in deep if s["n"] >= 2 and 16 * S640 <= s["median_side"] <= 40 * S640]
    band.sort(key=lambda s: -s["n"])           # prefer more fish per frame
    for s in _spread(band, per_case):
        chosen.append({**s, "case": "small-fish",
                       "rule": "DeepFish frames whose median box is 48-120px native, i.e. "
                               "16-40px once resized to a 640 input -- the band a stride-8 "
                               "head handles only marginally"})

    # 2. Typical sparse DeepFish (median object count)
    withfish = sorted([s for s in deep if s["n"] >= 1], key=lambda s: s["n"])
    mid = withfish[len(withfish) // 2:]
    for s in _spread(mid, per_case):
        chosen.append({**s, "case": "typical",
                       "rule": "DeepFish frames at the median object count"})

    # 3. Empty water -- the false-alarm case OzFish cannot measure
    empties = [s for s in deep if s["n"] == 0]
    for s in _spread(empties, per_case):
        chosen.append({**s, "case": "empty",
                       "rule": "DeepFish frames with zero annotated fish (first N by filename)"})

    # 4. Crowded cross-dataset (OzFish, sorted by most objects)
    crowd = sorted(oz, key=lambda s: -s["n"])
    for s in _spread(crowd, per_case):
        chosen.append({**s, "case": "crowded-cross-dataset",
                       "rule": "OzFish frames sorted by most annotated fish"})

    return chosen


def encode_image(path: Path, width: int) -> tuple[str, int, int]:
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        scale = width / w
        im = im.convert("RGB").resize((width, int(h * scale)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=80, optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), w, h


def measure_latency(model, imgsz: int, sample: Path, device, n: int = 60) -> dict:
    """End-to-end: preprocess + inference + NMS. Warm-up discarded."""
    for _ in range(10):
        model.predict(str(sample), imgsz=imgsz, device=device.ultralytics,
                      verbose=False, conf=CONF_SHOW)
    times = []
    for _ in range(n):
        t0 = time.perf_counter()
        model.predict(str(sample), imgsz=imgsz, device=device.ultralytics,
                      verbose=False, conf=CONF_SHOW)
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"median_ms": round(st.median(times), 2),
            "p95_ms": round(times[int(len(times) * 0.95)], 2),
            "fps_median": round(1000 / st.median(times), 1), "n": n}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-case", type=int, default=3)
    ap.add_argument("--out", type=Path, default=Path("demo/demo_data.json"))
    ap.add_argument("--seed-tag", default="s0",
                    help="which seed's checkpoints to visualise (same for both sides)")
    args = ap.parse_args()

    ck = {}
    for mp in sorted(paths.MODELS_ROOT.glob(f"*-{args.seed_tag}.pt")):
        if "-640-" in mp.name:
            ck[640] = mp
        elif "-960-" in mp.name:
            ck[960] = mp
    if len(ck) != 2:
        raise SystemExit(f"Need a 640 and a 960 checkpoint tagged {args.seed_tag} in "
                         f"{paths.rel(paths.MODELS_ROOT)}/; found {list(ck)}")

    device = resolve_device("auto")
    print(f"device     : {device}")
    print(f"checkpoints: {ck[640].name} | {ck[960].name}")

    deep = pick_frames("deepfish", args.per_case)
    oz = pick_frames("ozfish_eval", args.per_case)
    cases = select_cases(deep, oz, args.per_case)
    print(f"frames     : {len(cases)}")

    from ultralytics import YOLO
    models = {sz: YOLO(str(p)) for sz, p in ck.items()}

    frames = []
    for c in cases:
        img, lab, w, h = c["image"], c["label"], c["w"], c["h"]
        gts = gt_boxes(lab, w, h)
        entry = {"name": img.stem, "case": c["case"], "rule": c["rule"],
                 "width": w, "height": h, "gt": [[round(v, 1) for v in b] for b in gts],
                 "gt_buckets": [bucket_of(b) for b in gts], "models": {}}

        for sz, model in models.items():
            r = model.predict(str(img), imgsz=sz, device=device.ultralytics,
                              conf=CONF_SHOW, verbose=False)[0]
            preds = []
            for b, cf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
                preds.append({"box": [round(v, 1) for v in b], "conf": round(float(cf), 3)})
            # Greedy match: each GT claimed by at most one prediction.
            matched_gt, used = [], set()
            for gi, g in enumerate(gts):
                best, bi = 0.0, None
                for pi, p in enumerate(preds):
                    if pi in used:
                        continue
                    v = iou(g, p["box"])
                    if v > best:
                        best, bi = v, pi
                if bi is not None and best >= IOU_MATCH:
                    used.add(bi)
                    matched_gt.append(gi)
            entry["models"][str(sz)] = {
                "pred": preds,
                "found_gt": matched_gt,
                "false_positives": [i for i in range(len(preds)) if i not in used],
            }
        frames.append(entry)
        print(f"  {c['case']:22s} {img.stem[:38]:38s} gt={len(gts):3d} "
              f"640={len(entry['models']['640']['found_gt']):3d} "
              f"960={len(entry['models']['960']['found_gt']):3d}")

    # Recall by size bucket over the FULL validation set. Twelve illustration
    # frames are far too few to support a claim; the pictures illustrate, this
    # measures.
    print("\nscoring the full DeepFish val set for size-stratified recall ...")
    recall = {str(sz): {b[2]: {"found": 0, "total": 0} for b in BUCKETS} for sz in models}
    all_val = [s for s in deep if s["n"] > 0]
    for i, s_ in enumerate(all_val):
        if i % 100 == 0:
            print(f"    {i}/{len(all_val)}", flush=True)
        gts = gt_boxes(s_["label"], s_["w"], s_["h"])
        for sz, model in models.items():
            r = model.predict(str(s_["image"]), imgsz=sz, device=device.ultralytics,
                              conf=CONF_SHOW, verbose=False)[0]
            preds = r.boxes.xyxy.tolist()
            used = set()
            for g in gts:
                best, bi = 0.0, None
                for pi, pb in enumerate(preds):
                    if pi in used:
                        continue
                    v = iou(g, pb)
                    if v > best:
                        best, bi = v, pi
                bk = bucket_of(g)
                recall[str(sz)][bk]["total"] += 1
                if bi is not None and best >= IOU_MATCH:
                    used.add(bi)
                    recall[str(sz)][bk]["found"] += 1
    print("  full-set recall by bucket:")
    for sz in models:
        parts = [f"{b}: {recall[str(sz)][b]['found']}/{recall[str(sz)][b]['total']}"
                 for _l, _h, b in BUCKETS]
        print(f"    {sz}px  " + "  ".join(parts))

    sample = cases[0]["image"]
    timing = {str(sz): measure_latency(models[sz], sz, sample, device) for sz in models}
    print(f"\nlatency (desktop, end-to-end, warm-up discarded):")
    for sz, t in timing.items():
        print(f"  {sz}px: median {t['median_ms']} ms  p95 {t['p95_ms']} ms  "
              f"({t['fps_median']} fps)")

    print("\nencoding frames ...")
    for f, c in zip(frames, cases):
        b64, _, _ = encode_image(c["image"], WEB_WIDTH)
        f["jpeg_b64"] = b64
        f["web_width"] = WEB_WIDTH

    payload = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "checkpoints": {str(k): v.name for k, v in ck.items()},
        "seed_tag": args.seed_tag,
        "conf": CONF_SHOW, "iou_match": IOU_MATCH,
        "device": str(device),
        "timing_desktop": timing,
        "recall_by_bucket": recall,
        "recall_source": "full DeepFish val split (1339 frames), not the illustration frames",
        "frames": frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")
    mb = args.out.stat().st_size / 1e6
    print(f"\nwritten: {args.out}  ({mb:.1f} MB)")
    if mb > 14:
        print("  WARNING: approaching the 16 MB artifact limit -- lower --per-case")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
