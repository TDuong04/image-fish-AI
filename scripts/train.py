#!/usr/bin/env python
"""Train a fish detector. One entrypoint, every machine.

    python scripts/train.py --config smoke                  # 3 epochs, any machine
    python scripts/train.py --config yolov8n_640            # FD-020 baseline, all seeds
    python scripts/train.py --config yolov8n_640 --seed 0   # one seed only
    python scripts/train.py --config yolov8n_640 --dry-run  # resolve settings, train nothing

Writes runs/<UTC-date>-<name>-s<seed>/ containing config.yaml, metrics.json,
env.txt and env.json, so any number can be traced back to the exact config,
commit and machine that produced it (CLAUDE.md, "Reproducibility").

The `if __name__ == "__main__"` guard at the bottom is required: Windows and
macOS spawn rather than fork DataLoader workers, and without it this file is
re-imported in every worker process.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from fish import env as fenv  # noqa: E402
from fish import paths  # noqa: E402
from fish.device import resolve_device, suggested_batch  # noqa: E402


def load_train_config(name: str) -> dict:
    path = paths.CONFIG_DIR / "train" / f"{name}.yaml"
    if not path.exists():
        available = sorted(p.stem for p in (paths.CONFIG_DIR / "train").glob("*.yaml"))
        raise SystemExit(
            f"No training config {name!r} at {paths.rel(path)}.\n"
            f"Available: {', '.join(available) or '(none)'}"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def resolve_dataset(key: str) -> Path:
    """Map a dataset key to its Ultralytics data.yaml, with an actionable error."""
    path = paths.CONFIG_DIR / "data" / f"{key}.yaml"
    if not path.exists():
        raise SystemExit(
            f"\nDataset config {key!r} not found at {paths.rel(path)}.\n\n"
            f"The dataset is not set up on this machine yet. See docs/TRAINING.md\n"
            f"step 4, and ticket FD-011 (dataset conversion) in docs/tickets/BACKLOG.md.\n"
            f"Nothing has been downloaded -- this is expected on a fresh checkout."
        )
    return path


def materialise_dataset(cfg_path: Path, run_dir: Path) -> Path:
    """Resolve the committed dataset config against this machine's DATA_ROOT.

    Committed configs store `path` relative to DATA_ROOT so they are portable.
    Ultralytics needs an absolute path, so write a resolved copy into the run
    directory -- which also records exactly which data the run consumed.
    """
    spec = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
    raw = Path(str(spec.get("path", ".")))
    spec["path"] = str(raw if raw.is_absolute() else (paths.DATA_ROOT / raw).resolve())

    root = Path(spec["path"])
    if not root.exists():
        raise SystemExit(
            f"\nDataset {cfg_path.stem!r} points at {root}, which does not exist "
            f"on this machine.\nRun: python scripts/prepare_dataset.py {cfg_path.stem}"
        )
    run_dir.mkdir(parents=True, exist_ok=True)
    out = run_dir / "data.yaml"
    out.write_text(yaml.safe_dump(spec, sort_keys=False), encoding="utf-8")
    return out


def set_seed(seed: int) -> None:
    random.seed(seed)
    try:
        import numpy as np
        np.random.seed(seed)
    except ImportError:
        pass
    import torch
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one(cfg: dict, profile: dict, seed: int, dry_run: bool,
              resume: bool = False) -> Path | None:
    # Checked before device resolution so the message names the real reason
    # rather than failing on a CUDA probe first.
    if profile.get("training_allowed") is False and not dry_run:
        raise SystemExit(
            f"\nProfile {fenv.profile_name()!r} is inference-only "
            f"({profile.get('description')}).\nRefusing to train. Set FISH_ENV to a "
            f"training machine profile -- see configs/env/."
        )

    try:
        device = resolve_device(cfg.get("device") or profile.get("device", "auto"))
    except RuntimeError as exc:
        raise SystemExit(f"\n{exc}") from None

    # $FISH_BATCH lets a supervisor retry an OOM with a smaller batch without
    # rewriting the committed config (see scripts/supervise.py).
    batch_env = os.environ.get("FISH_BATCH")
    if batch_env:
        batch = int(batch_env)
        print(f"  (batch overridden to {batch} by $FISH_BATCH)")
    else:
        batch = cfg.get("batch") or profile.get("batch")
        if batch is None:
            batch = suggested_batch(device, cfg.get("imgsz", 640), Path(cfg["model"]).stem)
    workers = cfg.get("workers")
    if workers is None:
        workers = profile.get("workers", 4)
    amp = cfg.get("amp")
    if amp is None:
        amp = profile.get("amp", True)

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_name = f"{stamp}-{cfg['name']}-s{seed}"
    run_dir = paths.RUNS_ROOT / run_name

    resolved = {
        **cfg,
        "seed": seed,
        "batch": batch,
        "workers": workers,
        "amp": amp,
        "device": device.ultralytics,
        "device_resolved": str(device),
        "fish_env": fenv.profile_name(),
        "run_dir": paths.rel(run_dir),
    }

    print(f"\n--- {run_name} ---")
    # .get(), not [] -- these keys are optional in a config and defaulted at the
    # point of use, so indexing them here turned a missing key into a KeyError
    # traceback before any real validation ran.
    for key in ("model", "data", "imgsz", "epochs", "patience", "batch", "workers", "amp"):
        print(f"  {key:10s}: {resolved.get(key, '(default)')}")
    print(f"  {'device':10s}: {device}")

    if dry_run:
        print("  (dry run -- nothing trained)")
        return None

    data_cfg = resolve_dataset(cfg["data"])
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    data_yaml = materialise_dataset(data_cfg, run_dir)
    (run_dir / "config.yaml").write_text(
        yaml.safe_dump(resolved, sort_keys=False), encoding="utf-8"
    )
    fenv.write_env_file(run_dir)

    from ultralytics import YOLO

    # Resume picks up from the interrupted run's own last.pt, which carries the
    # optimizer and epoch counter. Ultralytics rejects every other argument when
    # resuming -- they are restored from the checkpoint -- so the call shapes
    # differ and must not be merged.
    last = run_dir / "weights" / "last.pt"
    if resume and last.exists():
        print(f"  resuming from {paths.rel(last)}")
        model = YOLO(str(last))
        results = model.train(resume=True)
    else:
        if resume:
            print("  (--resume asked, but no last.pt yet -- starting fresh)")
        model = YOLO(cfg["model"])
        results = model.train(
            data=str(data_yaml),
            epochs=cfg.get("epochs", 300),
            patience=cfg.get("patience", 50),
            imgsz=cfg.get("imgsz", 640),
            batch=batch,
            workers=workers,
            amp=amp,
            device=device.ultralytics,
            seed=seed,
            deterministic=True,
            close_mosaic=cfg.get("close_mosaic", 10),
            project=str(paths.RUNS_ROOT),
            name=run_name,
            exist_ok=True,
            val=True,
            **(cfg.get("extra") or {}),
        )

    metrics = {}
    box = getattr(getattr(results, "box", None), "__dict__", {})
    try:
        metrics = {
            "map50": float(results.box.map50),
            "map50_95": float(results.box.map),
            "precision": float(results.box.mp),
            "recall": float(results.box.mr),
        }
    except AttributeError:
        metrics = {"raw": str(box)[:2000]}
    metrics["seed"] = seed
    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\n  metrics: {json.dumps(metrics)}")
    print(f"  written: {paths.rel(run_dir)}")
    return run_dir


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="yolov8n_640", help="name under configs/train/")
    ap.add_argument("--seed", type=int, default=None,
                    help="run a single seed instead of every seed in the config")
    ap.add_argument("--dry-run", action="store_true",
                    help="resolve and print settings without training")
    ap.add_argument("--resume", action="store_true",
                    help="continue an interrupted run from its last.pt instead of restarting")
    args = ap.parse_args()

    try:
        profile = fenv.load_profile()
    except FileNotFoundError as exc:
        raise SystemExit(f"\n{exc}") from None
    fenv.apply_profile(profile)
    paths.ensure_dirs()

    cfg = load_train_config(args.config)
    seeds = [args.seed] if args.seed is not None else cfg.get("seeds", [0])

    print(f"config  : {args.config}")
    print(f"profile : {fenv.profile_name()}")
    print(f"seeds   : {seeds}")
    if len(seeds) == 1 and args.seed is None:
        print("NOTE: a single seed does not support a comparison between models (FD-103).")

    run_dirs = [d for s in seeds
                if (d := train_one(cfg, profile, s, args.dry_run, args.resume))]

    if len(run_dirs) > 1:
        maps = []
        for d in run_dirs:
            try:
                maps.append(json.loads((d / "metrics.json").read_text())["map50"])
            except (OSError, KeyError, json.JSONDecodeError):
                pass
        if maps:
            mean = sum(maps) / len(maps)
            print(f"\n=== {len(maps)} seeds: mAP@50 mean {mean:.4f}, "
                  f"range {min(maps):.4f}-{max(maps):.4f} (spread {max(maps)-min(maps):.4f}) ===")
            print("Report mean +- range, never the best seed (FD-103).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
