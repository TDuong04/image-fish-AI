#!/usr/bin/env python
"""Entrypoint for an unattended Kaggle training session.

The notebook is a thin shell around this file so the logic stays in git, gets
tested, and can be fixed without editing a notebook by hand.

    python kaggle/kaggle_train.py --config yolov8n_640 --seeds 0 1 2

What it handles that a plain `supervise.py` call does not:

  * **The session wall clock.** Kaggle kills a GPU session at ~12h. Anything
    not written to /kaggle/working by then is gone, so training stops itself
    with margin and the last checkpoint survives.
  * **Resuming across sessions.** Kaggle runs are stateless, but a previous
    run's output can be attached as an input dataset. If one is attached, its
    checkpoints are copied in first, so session N+1 continues session N rather
    than restarting from epoch 0.
  * **Reporting.** Writes a single kaggle_report.json summarising what ran,
    what finished and what broke, so the next session (or an agent) reads
    structured state instead of scrolling 12 hours of logs.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Kaggle's writable, persisted output directory. Everything that must survive
# the session goes here; /kaggle/temp does not.
WORKING = Path("/kaggle/working")
INPUT = Path("/kaggle/input")

IS_KAGGLE = WORKING.exists()
REPO_DIR = WORKING / "image-fish-AI" if IS_KAGGLE else Path(__file__).resolve().parents[1]

# Kaggle GPU sessions are cut at 12h. Stop with real margin: the final
# validation pass and the copy-out both need time.
DEFAULT_BUDGET_H = 10.5


def sh(cmd: list[str], cwd: Path | None = None, check: bool = True) -> int:
    print(f"$ {' '.join(str(c) for c in cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=cwd)
    if check and result.returncode != 0:
        raise SystemExit(f"Command failed ({result.returncode}): {' '.join(cmd)}")
    return result.returncode


def clone_repo(repo_url: str, branch: str = "main") -> Path:
    """Clone or update the project inside /kaggle/working."""
    if REPO_DIR.exists() and (REPO_DIR / ".git").exists():
        print(f"repo already present at {REPO_DIR}; fetching latest")
        sh(["git", "fetch", "--depth", "1", "origin", branch], cwd=REPO_DIR, check=False)
        sh(["git", "reset", "--hard", f"origin/{branch}"], cwd=REPO_DIR, check=False)
    else:
        sh(["git", "clone", "--depth", "1", "--branch", branch, repo_url, str(REPO_DIR)])
    return REPO_DIR


def restore_previous_checkpoints() -> int:
    """Copy checkpoints from an attached previous-run dataset into runs/.

    Kaggle sessions are stateless. Attaching the previous notebook version's
    output as an input dataset is the supported way to carry state forward;
    without this every session would restart at epoch 0 and 12 hours would buy
    nothing.
    """
    if not INPUT.exists():
        return 0
    restored = 0
    dest_root = REPO_DIR / "runs"
    for candidate in INPUT.glob("*/runs"):
        for run in candidate.iterdir():
            if not run.is_dir():
                continue
            dest = dest_root / run.name
            if dest.exists():
                continue
            shutil.copytree(run, dest)
            restored += 1
            print(f"  restored previous run: {run.name}")
    if restored:
        print(f"restored {restored} run director{'y' if restored == 1 else 'ies'} "
              f"-- training will resume rather than restart")
    return restored


def gpu_report() -> dict:
    info: dict = {}
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=30)
        info["gpu"] = out.stdout.strip() or "none"
    except Exception as exc:
        info["gpu"] = f"unavailable: {exc}"
    return info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="yolov8n_640")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--repo", default="https://github.com/TDuong04/image-fish-AI.git",
                    help="clone URL; for a private repo the token is injected by the notebook")
    ap.add_argument("--branch", default="main")
    ap.add_argument("--max-hours", type=float, default=DEFAULT_BUDGET_H)
    ap.add_argument("--skip-clone", action="store_true",
                    help="use the repo already present (local testing)")
    args = ap.parse_args()

    started = datetime.now(timezone.utc)
    report: dict = {
        "started_utc": started.isoformat(timespec="seconds"),
        "config": args.config, "seeds": args.seeds,
        "on_kaggle": IS_KAGGLE, **gpu_report(),
    }
    print(f"=== Kaggle training session ===")
    print(f"  config  : {args.config}")
    print(f"  seeds   : {args.seeds}")
    print(f"  budget  : {args.max_hours} h")
    print(f"  gpu     : {report['gpu']}")

    if not args.skip_clone:
        clone_repo(args.repo, args.branch)
    repo = REPO_DIR
    print(f"  repo    : {repo}")

    # Keep datasets and caches inside the persisted working dir, so a restored
    # session does not re-download 1.2 GB it already has.
    os.environ.setdefault("FISH_DATA_ROOT", str(repo / "data"))
    os.environ.setdefault("FISH_ENV", "colab")   # hosted-notebook profile

    restored = restore_previous_checkpoints()
    report["restored_runs"] = restored

    # Snapshot what the clone (and any restored checkpoints) already carried, so
    # the final report can distinguish inherited results from this session's.
    # Keyed by name AND content: run directories are named by UTC date, so a
    # session can legitimately write to the same name as an inherited run. A
    # name-only check silently discarded that session's real result.
    runs_dir = repo / "runs"
    pre_existing_runs = {}
    for d in runs_dir.glob("*"):
        m = d / "metrics.json"
        if d.is_dir() and m.exists():
            pre_existing_runs[d.name] = m.read_bytes()
    if pre_existing_runs:
        print(f"inherited {len(pre_existing_runs)} run(s) with metrics from the clone; "
              f"excluded from this session's report unless this session rewrites them")

    py = sys.executable
    # check=True: a failed install must stop the session here. Running it with
    # check=False let bootstrap's "refusing to install into the system Python"
    # pass silently, and training then died on `import ultralytics` after the
    # data download had already run.
    sh([py, str(repo / "scripts" / "bootstrap.py"), "--allow-system"], cwd=repo)

    # Prove the install actually took before spending GPU hours on it.
    probe = subprocess.run(
        [py, "-c", "import ultralytics, torch; "
                   "print(ultralytics.__version__, torch.__version__, torch.cuda.is_available())"],
        capture_output=True, text=True, cwd=repo)
    if probe.returncode != 0:
        report["outcome"] = "setup_failed"
        report["setup_error"] = (probe.stdout + probe.stderr)[-1500:]
        out = (WORKING if IS_KAGGLE else repo) / "kaggle_report.json"
        out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print("SETUP FAILED -- dependencies are not importable:")
        print(probe.stdout + probe.stderr)
        return 1
    print(f"deps ok: ultralytics/torch/cuda = {probe.stdout.strip()}")

    sh([py, str(repo / "scripts" / "check_env.py")], cwd=repo, check=False)
    sh([py, str(repo / "scripts" / "get_data.py")], cwd=repo)

    code = sh([py, str(repo / "scripts" / "supervise.py"),
               "--config", args.config,
               "--seeds", *[str(s) for s in args.seeds],
               "--max-hours", str(args.max_hours),
               "--resume"],
              cwd=repo, check=False)

    outcome = {0: "finished", 1: "failed", 2: "budget_reached"}.get(code, f"exit_{code}")
    report["outcome"] = outcome
    report["exit_code"] = code
    report["ended_utc"] = datetime.now(timezone.utc).isoformat(timespec="seconds")

    # Pull the supervisor's structured state into the report so the next session
    # needs to read exactly one file.
    sup = repo / "runs" / "_supervisor"
    if (sup / "summary.json").exists():
        report["supervisor"] = json.loads((sup / "summary.json").read_text())
    report["failures"] = [json.loads(f.read_text()) for f in sorted(sup.glob("*-failure.json"))]

    # Only runs produced by THIS session. The repo commits run provenance, so a
    # clone arrives carrying previous machines' metrics.json -- reporting those
    # made a failed session look like it had results.
    results = {}
    for m in sorted((repo / "runs").glob("*/metrics.json")):
        raw = m.read_bytes()
        if pre_existing_runs.get(m.parent.name) == raw:
            continue  # unchanged since the clone -- genuinely inherited
        try:
            results[m.parent.name] = json.loads(raw.decode("utf-8"))
        except (json.JSONDecodeError, UnicodeDecodeError):
            pass
    report["metrics"] = results
    report["inherited_runs_ignored"] = sorted(
        n for n in pre_existing_runs if n not in results)

    out = (WORKING if IS_KAGGLE else repo) / "kaggle_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"\n=== session {outcome} ===")
    for name, m in results.items():
        if "map50" in m:
            print(f"  {name}: mAP50={m['map50']:.4f} mAP50-95={m['map50_95']:.4f}")
    if report["failures"]:
        print(f"  {len(report['failures'])} failure(s) recorded -- see kaggle_report.json")
    if outcome == "budget_reached":
        print("  Budget reached with work left. Attach THIS version's output as an")
        print("  input dataset to the next run, and it will resume from here.")
    print(f"  report: {out}")
    return 0 if code in (0, 2) else 1


if __name__ == "__main__":
    raise SystemExit(main())
