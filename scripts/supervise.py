#!/usr/bin/env python
"""Run training unattended: survive crashes, resume, and report why it stopped.

    python scripts/supervise.py --config yolov8n_640 --seed 0
    python scripts/supervise.py --config yolov8n_640 --max-hours 11
    python scripts/supervise.py --config yolov8n_640 --seeds 0 1 2

Training on a free cloud runtime fails for boring reasons -- an OOM on a
slightly unlucky batch, a transient download, the session hitting its wall
clock. None of those need a human, and none of them should cost a night of
GPU time.

This wraps scripts/train.py and:

  * resumes from the run's own last.pt instead of restarting from epoch 0;
  * classifies the failure and only retries the ones retrying can fix
    (an OOM gets a smaller batch; a bad config is reported and stops);
  * stops itself *before* the platform's wall clock kills it, so the last
    checkpoint is written rather than lost;
  * leaves status.json / failure.json behind, so a supervising agent reads
    structured state instead of parsing scrollback.

Exit codes: 0 finished, 1 gave up (see failure.json), 2 stopped on the time
budget with work still to do (re-run to continue).
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish import paths  # noqa: E402

TRAIN = Path(__file__).parent / "train.py"

# Ordered: first match wins, so put the specific patterns above the generic.
FAILURE_PATTERNS = [
    ("oom", re.compile(
        r"CUDA out of memory|CUBLAS_STATUS_ALLOC_FAILED|torch\.cuda\.OutOfMemoryError", re.I)),
    ("dataloader_worker", re.compile(
        r"DataLoader worker.*(killed|exited)|worker.*SIGKILL", re.I)),
    ("cuda_unavailable", re.compile(
        r"cuda.*not available|no CUDA-capable device|CUDA driver", re.I)),
    ("disk_full", re.compile(r"No space left on device|OSError:.*28", re.I)),
    ("corrupt_data", re.compile(r"corrupt|truncated|cannot identify image file", re.I)),
    ("missing_dataset", re.compile(r"does not exist on this machine|Dataset .* not found", re.I)),
    ("bad_config", re.compile(r"No training config|No machine profile|unrecognized argument", re.I)),
    ("network", re.compile(r"ConnectionError|Temporary failure in name resolution|timed out", re.I)),
    # Last resort: an unhandled exception is a code/config bug, not a flake.
    # Labelling it distinguishes "our bug" from "the platform hiccupped", which
    # is the difference between an agent patching code and a retry loop.
    ("code_error", re.compile(
        r"Traceback \(most recent call last\)|KeyError|AttributeError|TypeError|"
        r"ImportError|ModuleNotFoundError|ValueError", re.I)),
]

# What a retry can actually fix. Everything else needs a human or an agent.
RETRYABLE = {"oom", "dataloader_worker", "network", "corrupt_data"}


def classify(log: str) -> str:
    for name, pattern in FAILURE_PATTERNS:
        if pattern.search(log):
            return name
    return "unknown"


def tail(text: str, lines: int = 40) -> str:
    return "\n".join(text.splitlines()[-lines:])


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def run_once(config: str, seed: int, resume: bool, extra_env: dict | None,
             log_path: Path) -> tuple[int, str]:
    """One training attempt. Streams to console and captures for classification."""
    cmd = [sys.executable, str(TRAIN), "--config", config, "--seed", str(seed)]
    if resume:
        cmd.append("--resume")

    import os
    env = {**os.environ, **(extra_env or {})}

    # Ultralytics prints box-drawing characters and emoji. A Windows console
    # defaults to cp1252 and raises UnicodeEncodeError on them, which would kill
    # the supervisor over cosmetics -- so force UTF-8 on the child and never let
    # an encoding error escape the echo loop.
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUTF8", "1")

    print(f"\n{'=' * 62}\n  $ {' '.join(cmd)}\n{'=' * 62}", flush=True)
    captured: list[str] = []
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, encoding="utf-8", errors="replace", env=env,
                            bufsize=1)
    assert proc.stdout is not None
    for line in proc.stdout:
        captured.append(line)
        try:
            sys.stdout.write(line)
        except UnicodeEncodeError:
            sys.stdout.write(line.encode("ascii", "replace").decode("ascii"))
        sys.stdout.flush()
    proc.wait()

    log = "".join(captured)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text(log, encoding="utf-8")
    return proc.returncode, log


def supervise_seed(config: str, seed: int, args, deadline: float | None) -> str:
    """Drive one seed to completion. Returns 'done' | 'failed' | 'timeout'."""
    state_dir = paths.RUNS_ROOT / "_supervisor"
    stem = f"{config}-s{seed}"
    batch_override: int | None = None
    resume = args.resume

    for attempt in range(1, args.max_retries + 2):
        if deadline and time.time() > deadline:
            print(f"\n[supervisor] time budget reached before attempt {attempt}; stopping cleanly.")
            return "timeout"

        extra_env = {}
        if batch_override:
            # train.py reads batch from config; this env var is the documented
            # override hook so a retry need not rewrite the committed config.
            extra_env["FISH_BATCH"] = str(batch_override)

        started = time.time()
        code, log = run_once(config, seed, resume, extra_env,
                             state_dir / f"{stem}-attempt{attempt}.log")
        elapsed = (time.time() - started) / 60

        if code == 0:
            print(f"\n[supervisor] {stem} finished in {elapsed:.0f} min "
                  f"(attempt {attempt}).")
            write_json(state_dir / f"{stem}-status.json", {
                "config": config, "seed": seed, "result": "done",
                "attempts": attempt, "minutes": round(elapsed, 1),
                "finished_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            })
            return "done"

        kind = classify(log)
        print(f"\n[supervisor] attempt {attempt} failed after {elapsed:.0f} min "
              f"-- exit {code}, classified as '{kind}'.")

        failure = {
            "config": config, "seed": seed, "attempt": attempt,
            "exit_code": code, "failure_kind": kind,
            "retryable": kind in RETRYABLE,
            "minutes": round(elapsed, 1),
            "when_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "log_tail": tail(log, 40),
            "log_file": paths.rel(state_dir / f"{stem}-attempt{attempt}.log"),
        }
        write_json(state_dir / f"{stem}-failure.json", failure)

        if kind not in RETRYABLE:
            print(f"[supervisor] '{kind}' is not something a retry fixes. Stopping.")
            print(f"[supervisor] see {paths.rel(state_dir / f'{stem}-failure.json')}")
            return "failed"

        if attempt > args.max_retries:
            print(f"[supervisor] out of retries ({args.max_retries}).")
            return "failed"

        # An OOM is the one failure a retry can genuinely fix, by asking for less.
        if kind == "oom":
            current = batch_override or _config_batch(config) or 16
            batch_override = max(2, current // 2)
            print(f"[supervisor] halving batch to {batch_override} and resuming.")
        else:
            print("[supervisor] transient failure; resuming from last checkpoint.")

        resume = True  # anything after attempt 1 continues rather than restarts
        time.sleep(min(30 * attempt, 120))

    return "failed"


def _config_batch(config: str) -> int | None:
    import yaml
    path = paths.CONFIG_DIR / "train" / f"{config}.yaml"
    if not path.exists():
        return None
    return (yaml.safe_load(path.read_text(encoding="utf-8")) or {}).get("batch")


def main() -> int:
    # Same reason as in run_once: the supervisor's own prints must not die on a
    # character the console cannot represent.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True, help="name under configs/train/")
    ap.add_argument("--seed", type=int, default=None, help="a single seed")
    ap.add_argument("--seeds", type=int, nargs="+", default=None,
                    help="several seeds, run in order")
    ap.add_argument("--max-retries", type=int, default=3,
                    help="retries per seed for retryable failures (default 3)")
    ap.add_argument("--max-hours", type=float, default=None,
                    help="stop cleanly before this many hours elapse -- set it below the "
                         "platform's session limit (Kaggle kills at 12h)")
    ap.add_argument("--resume", action="store_true",
                    help="resume the first attempt too, not just retries")
    args = ap.parse_args()

    if args.seed is not None and args.seeds:
        raise SystemExit("Use --seed or --seeds, not both.")
    seeds = args.seeds or ([args.seed] if args.seed is not None else None)
    if seeds is None:
        import yaml
        cfg_path = paths.CONFIG_DIR / "train" / f"{args.config}.yaml"
        if not cfg_path.exists():
            raise SystemExit(f"No training config {args.config!r}.")
        seeds = (yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}).get("seeds", [0])

    deadline = time.time() + args.max_hours * 3600 if args.max_hours else None
    state_dir = paths.RUNS_ROOT / "_supervisor"

    print(f"[supervisor] config={args.config} seeds={seeds} "
          f"max_retries={args.max_retries} "
          f"budget={f'{args.max_hours}h' if args.max_hours else 'none'}")

    results: dict[str, str] = {}
    for seed in seeds:
        outcome = supervise_seed(args.config, seed, args, deadline)
        results[f"seed{seed}"] = outcome
        write_json(state_dir / "summary.json", {
            "config": args.config, "seeds": seeds, "results": results,
            "updated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        })
        if outcome == "timeout":
            print("\n[supervisor] stopped on the time budget. Re-run with --resume "
                  "to continue where this left off.")
            return 2
        if outcome == "failed":
            print(f"\n[supervisor] seed {seed} could not be recovered; not starting "
                  f"the remaining seeds.")
            return 1

    print(f"\n[supervisor] all seeds finished: {results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
