#!/usr/bin/env python
"""Does the detector depend on colour? Measure it over the full validation split.

    python scripts/experiment_colour.py
    python scripts/experiment_colour.py --models models/*-960-*.pt

A single-frame probe suggested the model leans heavily on colour: removing it
dropped that frame's confidence from 0.71 to 0.03. One frame is not a finding,
so this runs the same perturbations over all 1,339 validation frames and reports
real mAP for each.

Why it matters: underwater colour shifts enormously with depth, turbidity and
time of day -- red is gone within a few metres. A detector that depends on hue
is fragile exactly where it has to be robust, and that would be one plausible
explanation for why both configurations lost over half their performance on
OzFish footage.

The perturbations only change appearance, never geometry, so the original labels
stay valid and are hard-linked rather than copied.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics as st
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402


def greyscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.COLOR_GRAY2BGR)


def hue_shift(img: np.ndarray, deg: int) -> np.ndarray:
    h = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # OpenCV hue is 0-179, so a 180-degree rotation is +90 here.
    h[:, :, 0] = (h[:, :, 0].astype(int) + deg // 2) % 180
    return cv2.cvtColor(h, cv2.COLOR_HSV2BGR)


def desaturate(img: np.ndarray, keep: float) -> np.ndarray:
    return cv2.addWeighted(img, keep, greyscale(img), 1 - keep, 0)


VARIANTS = {
    "original": lambda im: im,
    "desat50": lambda im: desaturate(im, 0.5),
    "greyscale": greyscale,
    "hue90": lambda im: hue_shift(im, 90),
    "hue180": lambda im: hue_shift(im, 180),
}


def build_variant(name: str, src_root: Path, out_root: Path) -> Path:
    """Write a perturbed copy of the val split. Labels are hard-linked."""
    img_out = out_root / "images" / "val"
    lbl_out = out_root / "labels" / "val"
    if (img_out).exists() and len(list(img_out.glob("*.jpg"))):
        return out_root
    img_out.mkdir(parents=True, exist_ok=True)
    lbl_out.mkdir(parents=True, exist_ok=True)

    fn = VARIANTS[name]
    srcs = sorted((src_root / "images" / "val").glob("*.jpg"))
    for i, s in enumerate(srcs):
        if i % 300 == 0:
            print(f"    {name}: {i}/{len(srcs)}", flush=True)
        im = cv2.imread(str(s))
        cv2.imwrite(str(img_out / s.name), fn(im), [cv2.IMWRITE_JPEG_QUALITY, 92])
        lab_src = src_root / "labels" / "val" / f"{s.stem}.txt"
        lab_dst = lbl_out / f"{s.stem}.txt"
        if lab_src.exists() and not lab_dst.exists():
            try:
                os.link(lab_src, lab_dst)
            except OSError:
                shutil.copy2(lab_src, lab_dst)
    return out_root


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--variants", nargs="*", default=list(VARIANTS))
    args = ap.parse_args()

    models = ([Path(m) for m in args.models] if args.models
              else sorted(paths.MODELS_ROOT.glob("*.pt")))
    models = [m for m in models if m.exists()]
    if not models:
        raise SystemExit(f"No checkpoints in {paths.rel(paths.MODELS_ROOT)}/")

    src_root = paths.PROCESSED_DIR / "deepfish"
    if not src_root.exists():
        raise SystemExit("Run scripts/prepare_dataset.py deepfish first.")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    out_dir = paths.RUNS_ROOT / f"exp-colour-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device("auto")
    print(f"device  : {device}")
    print(f"models  : {len(models)}   variants: {', '.join(args.variants)}\n")

    # Materialise each perturbed split once, then reuse across all models.
    data_yamls = {}
    for v in args.variants:
        root = (src_root if v == "original"
                else build_variant(v, src_root, paths.INTERIM_DIR / f"colour_{v}"))
        spec = {"path": str(root.resolve()), "train": "images/val",
                "val": "images/val", "names": {0: "fish"}}
        p = out_dir / f"data-{v}.yaml"
        p.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
        data_yamls[v] = p
        print(f"  [{v}] ready")

    from ultralytics import YOLO
    import re
    rows = []
    print()
    for mp in models:
        m = re.search(r"yolov8n-(\d+)-s(\d+)", mp.stem)
        if not m:
            continue
        imgsz, seed = int(m.group(1)), int(m.group(2))
        model = YOLO(str(mp))
        for v in args.variants:
            r = model.val(data=str(data_yamls[v]), split="val", imgsz=imgsz,
                          conf=0.001, iou=0.7, device=device.ultralytics,
                          plots=False, verbose=False,
                          project=str(out_dir), name=f"{mp.stem}-{v}", exist_ok=True)
            rows.append({"model": mp.stem, "imgsz": imgsz, "seed": seed, "variant": v,
                         "map50": float(r.box.map50), "map50_95": float(r.box.map),
                         "recall": float(r.box.mr)})
            print(f"  {mp.stem:30s} {v:10s} mAP50 {rows[-1]['map50']:.4f}")

    # Group by resolution and variant so seeds become error bars.
    print(f"\n{'='*66}\n  mAP@50 by input size and perturbation (3 seeds each)\n{'='*66}")
    summary = {}
    for imgsz in sorted({r["imgsz"] for r in rows}):
        base = None
        print(f"\n  {imgsz}px")
        for v in args.variants:
            vals = [r["map50"] for r in rows if r["imgsz"] == imgsz and r["variant"] == v]
            if not vals:
                continue
            mean = st.mean(vals)
            sd = st.stdev(vals) if len(vals) > 1 else 0.0
            if v == "original":
                base = mean
            drop = "" if base is None or v == "original" else \
                f"   {(mean-base)/base*100:+6.1f}% vs original"
            summary[f"{imgsz}-{v}"] = {"mean": mean, "sd": sd, "n": len(vals)}
            print(f"    {v:10s} {mean:.4f} +- {sd:.4f}  (n={len(vals)}){drop}")

    (out_dir / "experiment.json").write_text(json.dumps(
        {"when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
         "device": str(device), "conf": 0.001, "iou": 0.7,
         "note": "perturbations alter appearance only; geometry and labels unchanged",
         "runs": rows, "summary": summary}, indent=2), encoding="utf-8")
    print(f"\n  written: {paths.rel(out_dir / 'experiment.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
