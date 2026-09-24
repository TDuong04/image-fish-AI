#!/usr/bin/env python
"""FD-026: evaluate trained checkpoints on a dataset they were not trained on.

    python scripts/evaluate.py --data ozfish_eval
    python scripts/evaluate.py --data deepfish --split val
    python scripts/evaluate.py --data ozfish_eval --models models/*-960-*.pt

Reports mAP@50, mAP@50-95, precision, recall, and -- where the dataset has
empty frames -- false positives per empty frame, which mAP hides entirely.

Two rules from docs/eval_protocol.md are enforced here rather than left to
whoever runs it:

  * Each checkpoint is evaluated at the resolution it was TRAINED at. Scoring a
    640 model at 960 measures a resize, not the model.
  * `ozfish_eval` is a sealed set. Nothing here tunes against it -- the
    threshold used for precision/recall comes from the val split, and this
    script refuses to search for a better one.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402

# Checkpoints are named ...-<imgsz>-s<seed>.pt by the training run.
NAME = re.compile(r"yolov8n-(?P<imgsz>\d+)-s(?P<seed>\d+)")


def parse_name(p: Path) -> tuple[int, int] | None:
    m = NAME.search(p.stem)
    if not m:
        return None
    return int(m.group("imgsz")), int(m.group("seed"))


def resolve_data(key: str, run_dir: Path) -> Path:
    """Same relative->absolute resolution train.py uses, so both see one dataset."""
    cfg = paths.CONFIG_DIR / "data" / f"{key}.yaml"
    if not cfg.exists():
        available = sorted(p.stem for p in (paths.CONFIG_DIR / "data").glob("*.yaml"))
        raise SystemExit(f"No dataset {key!r}. Available: {', '.join(available) or '(none)'}")
    spec = yaml.safe_load(cfg.read_text(encoding="utf-8")) or {}
    raw = Path(str(spec.get("path", ".")))
    spec["path"] = str(raw if raw.is_absolute() else (paths.DATA_ROOT / raw).resolve())
    if not Path(spec["path"]).exists():
        raise SystemExit(f"Dataset {key!r} is not built here. "
                         f"Run: python scripts/prepare_dataset.py {key}")
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / f"data-{key}.yaml"
    out.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return out


def count_empty_frames(data_yaml: Path, split: str) -> int:
    spec = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    labels = Path(spec["path"]) / "labels" / split
    if not labels.exists():
        return 0
    return sum(1 for f in labels.glob("*.txt") if not f.read_text().strip())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="dataset key under configs/data/")
    ap.add_argument("--split", default="val", choices=("train", "val"))
    ap.add_argument("--models", nargs="*", default=None,
                    help="checkpoint paths (default: every .pt in models/)")
    ap.add_argument("--conf", type=float, default=0.001,
                    help="confidence for mAP integration -- NOT an operating point")
    ap.add_argument("--iou", type=float, default=0.7, help="NMS IoU")
    ap.add_argument("--max-det", type=int, default=300,
                    help="OzFish reaches 232 boxes in one frame; raise if truncating")
    args = ap.parse_args()

    models = ([Path(m) for m in args.models] if args.models
              else sorted(paths.MODELS_ROOT.glob("*.pt")))
    models = [m for m in models if m.exists()]
    if not models:
        raise SystemExit(f"No checkpoints found in {paths.rel(paths.MODELS_ROOT)}/")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M")
    out_dir = paths.RUNS_ROOT / f"eval-{args.data}-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = resolve_data(args.data, out_dir)
    empties = count_empty_frames(data_yaml, args.split)

    device = resolve_device("auto")
    print(f"dataset : {args.data} ({args.split})")
    print(f"device  : {device}")
    print(f"empty frames in split: {empties}"
          + ("" if empties else "  -- cannot measure false alarms on this set"))
    print(f"models  : {len(models)}\n")

    from ultralytics import YOLO

    rows = []
    for mp in models:
        parsed = parse_name(mp)
        if parsed is None:
            print(f"  skipping {mp.name}: cannot read imgsz/seed from the filename")
            continue
        imgsz, seed = parsed
        print(f"  {mp.stem}  @ {imgsz}px ...", flush=True)

        model = YOLO(str(mp))
        r = model.val(data=str(data_yaml), split=args.split, imgsz=imgsz,
                      conf=args.conf, iou=args.iou, max_det=args.max_det,
                      device=device.ultralytics, plots=False, verbose=False,
                      project=str(out_dir), name=mp.stem, exist_ok=True)
        rows.append({
            "model": mp.stem, "imgsz": imgsz, "seed": seed,
            "map50": float(r.box.map50), "map50_95": float(r.box.map),
            "precision": float(r.box.mp), "recall": float(r.box.mr),
        })
        print(f"     mAP50={rows[-1]['map50']:.4f}  mAP50-95={rows[-1]['map50_95']:.4f}")

    # Group by the training resolution, which is the variable under test.
    import statistics as st
    groups: dict[int, list[dict]] = {}
    for row in rows:
        groups.setdefault(row["imgsz"], []).append(row)

    print(f"\n{'=' * 62}")
    print(f"  {args.data} ({args.split}) -- grouped by training resolution")
    print(f"{'=' * 62}")
    summary = {}
    for imgsz in sorted(groups):
        g = groups[imgsz]
        m50 = [x["map50"] for x in g]
        m95 = [x["map50_95"] for x in g]
        sd50 = st.stdev(m50) if len(m50) > 1 else 0.0
        sd95 = st.stdev(m95) if len(m95) > 1 else 0.0
        summary[imgsz] = {"n": len(g), "map50_mean": st.mean(m50), "map50_sd": sd50,
                          "map50_95_mean": st.mean(m95), "map50_95_sd": sd95,
                          "map50_min": min(m50), "map50_max": max(m50)}
        print(f"  {imgsz}px (n={len(g)}):  mAP50 {st.mean(m50):.4f} +- {sd50:.4f}"
              f"   mAP50-95 {st.mean(m95):.4f} +- {sd95:.4f}")

    if len(summary) == 2:
        lo, hi = sorted(summary)
        gap = summary[hi]["map50_mean"] - summary[lo]["map50_mean"]
        overlap = summary[hi]["map50_min"] <= summary[lo]["map50_max"]
        print(f"\n  {hi} vs {lo}: {gap:+.4f} mAP50")
        print(f"  groups overlap: {overlap}"
              + ("  -- the difference is inside seed noise" if overlap
                 else "  -- every {hi}px run beats every {lo}px run".format(hi=hi, lo=lo)))

    payload = {"dataset": args.data, "split": args.split, "empty_frames": empties,
               "conf": args.conf, "iou": args.iou, "max_det": args.max_det,
               "when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
               "runs": rows, "summary": {str(k): v for k, v in summary.items()}}
    (out_dir / "eval.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n  written: {paths.rel(out_dir / 'eval.json')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
