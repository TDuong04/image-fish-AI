#!/usr/bin/env python
"""Export trained checkpoints to ONNX at the shapes they were EVALUATED at, and prove parity.

    python scripts/export_onnx.py
    python scripts/export_onnx.py --models models/20260923-yolov8n-960-s2.pt --frames 12

Two decisions in here are easy to get wrong.

1. **Shape.** Ultralytics validates and predicts a 1920x1080 frame at imgsz 960 as
   544x960 (and imgsz 640 as 384x640), because it letterboxes to the smallest
   stride-32 rectangle rather than a square. Every accuracy figure in this project
   was measured at those shapes. A static 960x960 engine therefore runs 1.76x the
   pixels of the geometry that was evaluated, so its latency would describe a
   different workload than its accuracy does. This exports the evaluated shapes.

2. **Parity before speed.** A fast engine that detects the wrong things is worse than
   a slow correct one. After export this runs real frames through PyTorch and
   ONNX Runtime and compares both the raw output tensors and the detections that
   survive thresholding and NMS. The preprocessed inputs and the PyTorch outputs are
   saved so the same frames can be pushed through the TensorRT engine on the Jetson
   and compared against exactly the same reference (scripts/engine_parity.py).

NMS is deliberately NOT baked into the graph: where NMS runs is its own measured
decision (FD-034), and on-graph NMS is the commonest cause of TensorRT conversion
failure on Jetson.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from fish import paths  # noqa: E402
from fish.parity import CONF, NMS_IOU, compare_detections, nms, sha256, verdict  # noqa: E402

OPSET = 17


def evaluated_shape(imgsz: int) -> tuple[int, int]:
    """(H, W) that Ultralytics actually feeds the network for a 1920x1080 frame."""
    from ultralytics.data.augment import LetterBox
    out = LetterBox((imgsz, imgsz), auto=True, stride=32)(image=np.zeros((1080, 1920, 3), np.uint8))
    return int(out.shape[0]), int(out.shape[1])


def preprocess(img_bgr: np.ndarray, hw: tuple[int, int]) -> np.ndarray:
    """Letterbox to exactly (H, W), BGR->RGB, /255, NCHW float32. Matches Ultralytics."""
    from ultralytics.data.augment import LetterBox
    lb = LetterBox(hw, auto=False, stride=32)(image=img_bgr)
    assert lb.shape[:2] == hw, f"letterbox gave {lb.shape[:2]}, wanted {hw}"
    x = np.ascontiguousarray(lb[:, :, ::-1].transpose(2, 0, 1), dtype=np.float32) / 255.0
    return x[None]


def pick_frames(n: int) -> list[Path]:
    """Deterministic and varied: mostly DeepFish val, a few crowded OzFish frames."""
    deep = sorted((paths.PROCESSED_DIR / "deepfish" / "images" / "val").glob("*.jpg"))
    oz = sorted((paths.PROCESSED_DIR / "ozfish_eval" / "images" / "val").glob("*.jpg"))
    n_oz = max(1, n // 3) if oz else 0
    n_deep = n - n_oz
    pick = []
    if deep:
        step = max(1, len(deep) // n_deep)
        pick += deep[::step][:n_deep]
    if oz:
        step = max(1, len(oz) // n_oz)
        pick += oz[::step][:n_oz]
    return pick


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None,
                    help="checkpoints (default: the seed-2 checkpoint at each resolution)")
    ap.add_argument("--frames", type=int, default=12)
    ap.add_argument("--opset", type=int, default=OPSET)
    args = ap.parse_args()

    models = ([Path(m) for m in args.models] if args.models
              else [paths.MODELS_ROOT / "20260922-yolov8n-640-s2.pt",
                    paths.MODELS_ROOT / "20260923-yolov8n-960-s2.pt"])
    for m in models:
        if not m.exists():
            raise SystemExit(f"{m} not found. Run: python scripts/fetch_models.py")

    frames = pick_frames(args.frames)
    if not frames:
        raise SystemExit("No validation frames found. Run: python scripts/get_data.py")
    print(f"parity frames: {len(frames)} "
          f"({sum('ozfish_eval' in str(f) for f in frames)} OzFish, "
          f"{sum('ozfish_eval' not in str(f) for f in frames)} DeepFish)")

    import onnxruntime as ort
    import torch
    import ultralytics
    from ultralytics import YOLO

    out_dir = paths.MODELS_ROOT / "onnx"
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    failures = 0

    for pt in models:
        m = re.search(r"-(\d+)-s(\d+)", pt.stem)
        imgsz = int(m.group(1))
        hw = evaluated_shape(imgsz)
        tag = f"{pt.stem}-{hw[0]}x{hw[1]}"
        print(f"\n=== {pt.name}  imgsz {imgsz}  ->  static {hw[0]}x{hw[1]} (HxW) ===")

        yolo = YOLO(str(pt))
        produced = Path(yolo.export(format="onnx", imgsz=list(hw), opset=args.opset,
                                    simplify=False, dynamic=False, batch=1, half=False,
                                    device="cpu"))
        dest = out_dir / f"{tag}.onnx"
        shutil.move(str(produced), dest)
        print(f"  exported {dest.name}  ({dest.stat().st_size/1e6:.1f} MB)  sha {sha256(dest)[:12]}")

        sess = ort.InferenceSession(str(dest), providers=["CPUExecutionProvider"])
        in_name = sess.get_inputs()[0].name
        in_shape = sess.get_inputs()[0].shape
        assert list(in_shape) == [1, 3, hw[0], hw[1]], f"exported shape {in_shape}"

        net = YOLO(str(pt)).model.float().eval().cpu()
        inputs_dir = out_dir / "parity" / tag
        inputs_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, f in enumerate(frames):
            img = cv2.imread(str(f))
            x = preprocess(img, hw)
            with torch.no_grad():
                t_out = net(torch.from_numpy(x))
            t_raw = (t_out[0] if isinstance(t_out, (list, tuple)) else t_out).numpy()
            o_raw = sess.run(None, {in_name: x})[0]

            x.tofile(inputs_dir / f"frame_{i:02d}.bin")
            np.save(inputs_dir / f"torch_{i:02d}.npy", t_raw)

            d_t, d_o = nms(t_raw), nms(o_raw)
            cmp_ = compare_detections(d_t, d_o)
            rows.append({"frame": f.name, "max_abs_raw": float(np.abs(t_raw - o_raw).max()),
                         "torch_dets": len(d_t), **cmp_})

        worst = max(r["max_abs_raw"] for r in rows)
        v = verdict(rows)
        ok = worst < 1e-2 and v["pass"]
        failures += 0 if ok else 1
        print(f"  parity vs PyTorch on {len(rows)} real frames:")
        print(f"    worst raw-tensor diff     : {worst:.2e}   (box coords are in pixels)")
        print(f"    confident misses / extras : {v['unmatched_ref']} / {v['extra']}"
              f"   (borderline, not counted: {v['borderline']})")
        print(f"    worst matched box IoU     : {v['worst_iou']}")
        print(f"    => {'PASS' if ok else 'FAIL -- do not build an engine from this: ' + '; '.join(v['reasons'])}")

        (inputs_dir / "frames.json").write_text(json.dumps(
            {"model": pt.name, "hw": list(hw), "frames": [f.name for f in frames],
             "conf": CONF, "nms_iou": NMS_IOU, "input_name": in_name,
             "output_shape_torch": list(t_raw.shape)}, indent=2), encoding="utf-8")

        manifest.append({"onnx": dest.name, "sha256": sha256(dest), "source_pt": pt.name,
                         "source_pt_sha256": sha256(pt), "imgsz": imgsz, "hw": list(hw),
                         "opset": args.opset, "parity_pass": ok, "parity_frames": len(rows),
                         "worst_raw_diff": worst, "unmatched": v["unmatched_ref"],
                         "torch": torch.__version__, "ultralytics": ultralytics.__version__,
                         "onnxruntime": ort.__version__})

    meta = {"built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "nms_in_graph": False, "models": manifest}
    (paths.DOCS_DIR / "onnx_manifest.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"\nwritten: docs/onnx_manifest.json   inputs/refs: {paths.rel(out_dir / 'parity')}/")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
