#!/usr/bin/env python
"""Compare a TensorRT engine's outputs against the PyTorch reference (FD-041).

    python scripts/engine_parity.py --parity-dir models/onnx/parity/<tag> --trt-dir <dir-with-trt_NN.npy>
    python scripts/engine_parity.py --tag 20260923-yolov8n-960-s2-544x960 --label fp16

Workflow:
    1. scripts/export_onnx.py   -> ONNX, preprocessed frames, PyTorch reference outputs
    2. (board) trtexec builds the engine; edge/engine_dump.py runs it on the same frames
    3. copy the trt_NN.npy files back here
    4. this script compares the engine with PyTorch on exactly the same inputs

A fast engine that detects the wrong things is worse than a slow correct one, so this
runs before any latency number is treated as meaningful. It uses the same comparison as
the ONNX stage (src/fish/parity.py), so a pass at each stage means the same thing.

FP16 will not match FP32 bit for bit. Detections whose confidence sits within a few
hundredths of the threshold can legitimately cross it, so those are reported as
borderline and not counted against the engine; a confident detection that goes missing
or appears from nowhere is a real disagreement and fails the check.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from fish import paths  # noqa: E402
from fish.parity import compare_detections, nms, verdict  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tag", help="parity set name under models/onnx/parity/")
    ap.add_argument("--parity-dir", type=Path)
    ap.add_argument("--trt-dir", type=Path, help="dir of trt_NN.npy (default: <parity-dir>/trt)")
    ap.add_argument("--label", default="fp16", help="precision label recorded in the result")
    args = ap.parse_args()

    pdir = args.parity_dir or (paths.MODELS_ROOT / "onnx" / "parity" / (args.tag or ""))
    if not (pdir / "frames.json").exists():
        raise SystemExit(f"No parity set at {pdir}. Run scripts/export_onnx.py first.")
    tdir = args.trt_dir or (pdir / "trt")
    meta = json.loads((pdir / "frames.json").read_text(encoding="utf-8"))

    rows, raw_diffs = [], []
    for i, name in enumerate(meta["frames"]):
        t_path, e_path = pdir / f"torch_{i:02d}.npy", tdir / f"trt_{i:02d}.npy"
        if not e_path.exists():
            raise SystemExit(f"Missing engine output {e_path}. Did edge/engine_dump.py finish?")
        ref, got = np.load(t_path), np.load(e_path)
        if ref.shape != got.shape:
            raise SystemExit(f"{name}: shape {got.shape} from the engine, {ref.shape} from PyTorch. "
                             f"The engine was built for a different input shape.")
        raw_diffs.append(float(np.abs(ref - got).max()))
        cmp_ = compare_detections(nms(ref), nms(got))
        rows.append({"frame": name, "max_abs_raw": raw_diffs[-1], **cmp_})
        print(f"  {name[:34]:34s} ref {cmp_['ref']:3d} got {cmp_['got']:3d}  "
              f"matched {cmp_['matched']:3d}  raw diff {raw_diffs[-1]:.3f}")

    v = verdict(rows)
    print(f"\n  model          : {meta['model']}  ({meta['hw'][0]}x{meta['hw'][1]}, {args.label})")
    print(f"  frames         : {v['frames']}")
    print(f"  confident miss : {v['unmatched_ref']}    confident extra: {v['extra']}"
          f"    borderline (not counted): {v['borderline']}")
    print(f"  worst IoU      : {v['worst_iou']}    worst conf drift: {v['worst_dconf']}")
    print(f"  worst raw diff : {max(raw_diffs):.3f}  (box coords in pixels, scores 0-1)")
    print(f"  => {'PASS' if v['pass'] else 'FAIL: ' + '; '.join(v['reasons'])}")

    out = pdir / f"engine_parity_{args.label}.json"
    out.write_text(json.dumps({"when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                               "model": meta["model"], "hw": meta["hw"], "label": args.label,
                               "verdict": v, "worst_raw_diff": max(raw_diffs), "rows": rows},
                              indent=2), encoding="utf-8")
    print(f"  written: {paths.rel(out)}")
    return 0 if v["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
