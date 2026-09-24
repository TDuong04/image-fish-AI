#!/usr/bin/env python
"""Build the frame-by-frame data behind the device demo.

    python scripts/build_device_demo.py
    python scripts/build_device_demo.py --clip 7623_F2 --frames 40 --stride 3

Runs the detector over consecutive frames of one real clip and records, per
frame, the boxes it found and the running count. The device page replays that
sequence, so what you watch is a real detection run rather than an animation.

Also records true GPU-only inference time separately from end-to-end time. The
difference matters: on a desktop GPU the fixed overhead (decode, letterbox, NMS,
host transfers) dominates and hides the cost of the extra pixels, which is why
desktop end-to-end timing tells you almost nothing about a constrained board.
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
from fish.datasets import read_label  # noqa: E402
from fish.device import resolve_device  # noqa: E402

CONF = 0.25
WEB_WIDTH = 720          # smaller than the comparison page: many frames, one page


def encode(path: Path, width: int) -> str:
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        im = im.convert("RGB").resize((width, int(h * width / w)), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=70, optimize=True)
    return base64.b64encode(buf.getvalue()).decode()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", default="7623_F2")
    ap.add_argument("--frames", type=int, default=40)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--out", type=Path, default=Path("demo/device_data.json"))
    args = ap.parse_args()

    src = paths.PROCESSED_DIR / "deepfish" / "images" / "val"
    lab_root = paths.PROCESSED_DIR / "deepfish" / "labels" / "val"
    files = sorted(src.glob(f"{args.clip}_f*.jpg"))[::args.stride][:args.frames]
    if not files:
        raise SystemExit(f"No frames for clip {args.clip!r} under {paths.rel(src)}")

    ck = sorted(paths.MODELS_ROOT.glob(f"*-{args.imgsz}-s0.pt"))
    if not ck:
        raise SystemExit(f"No {args.imgsz}px seed-0 checkpoint in {paths.rel(paths.MODELS_ROOT)}/")

    device = resolve_device("auto")
    from ultralytics import YOLO
    model = YOLO(str(ck[0]))
    print(f"clip   : {args.clip}  ({len(files)} frames, every {args.stride})")
    print(f"model  : {ck[0].name} @ {args.imgsz}px on {device}")

    # Warm up before timing anything.
    for _ in range(8):
        model.predict(str(files[0]), imgsz=args.imgsz, conf=CONF,
                      device=device.ultralytics, verbose=False)

    frames, e2e, gpu_only = [], [], []
    for f in files:
        t0 = time.perf_counter()
        r = model.predict(str(f), imgsz=args.imgsz, conf=CONF,
                          device=device.ultralytics, verbose=False)[0]
        e2e.append((time.perf_counter() - t0) * 1000)
        # Ultralytics reports its own per-stage split; `inference` is the GPU
        # forward pass alone, without decode, letterboxing or NMS.
        gpu_only.append(float(r.speed.get("inference", 0.0)))

        gt, _ = read_label(lab_root / f"{f.stem}.txt")
        frames.append({
            "name": f.stem,
            "boxes": [[round(v, 1) for v in b] for b in r.boxes.xyxy.tolist()],
            "confs": [round(float(c), 2) for c in r.boxes.conf.tolist()],
            "gt_count": len(gt),
            "jpeg_b64": encode(f, WEB_WIDTH),
        })

    counts = [len(fr["boxes"]) for fr in frames]
    payload = {
        "built_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "clip": args.clip, "imgsz": args.imgsz, "checkpoint": ck[0].name,
        "conf": CONF, "device": str(device), "web_width": WEB_WIDTH,
        "native_width": 1920,
        "timing_desktop": {
            "end_to_end_median_ms": round(st.median(e2e), 2),
            "gpu_only_median_ms": round(st.median(gpu_only), 2),
            "overhead_share": round(1 - st.median(gpu_only) / st.median(e2e), 3),
            "n": len(e2e),
        },
        "maxn": max(counts), "mean_count": round(st.mean(counts), 2),
        "empty_frames": sum(1 for c in counts if c == 0),
        "frames": frames,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload), encoding="utf-8")

    t = payload["timing_desktop"]
    print(f"\nMaxN {payload['maxn']}  mean {payload['mean_count']}  "
          f"empty {payload['empty_frames']}/{len(frames)}")
    print(f"end-to-end {t['end_to_end_median_ms']} ms | "
          f"GPU forward only {t['gpu_only_median_ms']} ms | "
          f"overhead {t['overhead_share']*100:.0f}% of the frame")
    print(f"written: {args.out}  ({args.out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
