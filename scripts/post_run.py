#!/usr/bin/env python
"""After a training run: run the error analysis and the held-out OzFish score on its best checkpoint.

    python scripts/post_run.py --slug yolov8s-960-kakadu-ozfish-s1 --imgsz 960

Finds the newest finished run folder `runs/<date>-<slug>` (one with `metrics.json` and `weights/best.pt`) and runs
`scripts/error_analysis.py` (DeepFish validation only) and `scripts/eval_ozfish_val.py` (held-out OzFish deployments, never
the sealed set) on it, writing `runs/analysis-<date>-<run>/analysis.json` and `ozfish_val.txt`. It exists so the queue can
produce the numbers on the training machine without anyone fetching a checkpoint first. Exits non-zero if a step fails.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish import paths  # noqa: E402


def find_run(runs_root: Path, slug: str) -> Path | None:
    """Newest folder named `<date>-<slug>` that finished (has metrics.json and weights/best.pt)."""
    done = [d for d in runs_root.glob(f"*-{slug}")
            if d.is_dir() and (d / "metrics.json").exists() and (d / "weights" / "best.pt").exists()]
    return sorted(done, key=lambda d: d.name)[-1] if done else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--slug", required=True, help="run name without the date, for example yolov8s-960-kakadu-ozfish-s1")
    ap.add_argument("--imgsz", type=int, required=True)
    args = ap.parse_args()

    run = find_run(paths.RUNS_ROOT, args.slug)
    if run is None:
        sys.exit(f"no finished run matching runs/*-{args.slug} (needs metrics.json and weights/best.pt)")
    out = paths.RUNS_ROOT / f"analysis-{datetime.now().strftime('%Y%m%d')}-{run.name}"
    out.mkdir(parents=True, exist_ok=True)
    scripts = Path(__file__).resolve().parent
    best = run / "weights" / "best.pt"

    r1 = subprocess.run([sys.executable, str(scripts / "error_analysis.py"), "--model", str(best), "--imgsz", str(args.imgsz),
                         "--worst", "0", "--out", str(out)])
    r2 = subprocess.run([sys.executable, str(scripts / "eval_ozfish_val.py"), str(best), "--imgsz", str(args.imgsz)],
                        capture_output=True, text=True)
    (out / "ozfish_val.txt").write_text(r2.stdout + r2.stderr, encoding="utf-8")
    print(f"{run.name}: analysis exit {r1.returncode}, held-out OzFish exit {r2.returncode}; wrote {paths.rel(out)}")
    return 0 if r1.returncode == 0 and r2.returncode == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
