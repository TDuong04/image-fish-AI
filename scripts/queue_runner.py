#!/usr/bin/env python
"""Run a plan of steps one after another, unattended, and record what happened.

    python scripts/queue_runner.py configs/overnight_plan.yaml

A plan is a YAML list of steps. Each step has a `name` and a `cmd` (a list; the word `{python}`
becomes the interpreter running this script) and may have:

    wait_files:    paths (globs ok, relative to the repo) that must all exist before it starts
    wait_idle:     substrings; wait until no process command line contains any of them ...
    idle_seconds:  ... continuously for this long (default 120; bridges gaps between queued jobs)
    skip_if:       a glob; if it matches, the step is marked done without running (makes reruns safe)
    stop_on_failure: stop the whole queue if this step fails (default: carry on with the next step)
    env:           extra environment variables for the command

The plan file is re-read before every step, so steps can be added or corrected while the queue
runs. State goes to `<DATA_ROOT>/.cache/queue/status.json` and each step's output to
`<DATA_ROOT>/.cache/queue/<name>.log`. A step already `done` in the status file is not rerun, so
restarting the runner resumes the plan. Needs Python 3.10+; `wait_idle` is Linux-only (it reads /proc). Patterns are matched relative to the
repo root, so a checkout path containing `[` or `]` does not break them.
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from fish import paths  # noqa: E402

QUEUE_DIR = paths.DATA_ROOT / ".cache" / "queue"
STATUS = QUEUE_DIR / "status.json"


def load_status() -> dict:
    return json.loads(STATUS.read_text()) if STATUS.exists() else {"steps": {}}


def save_status(status: dict) -> None:
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    tmp = STATUS.with_suffix(".tmp")
    tmp.write_text(json.dumps(status, indent=2))
    tmp.replace(STATUS)


def matching_processes(needles: list[str]) -> list[int]:
    found = []
    for pid_dir in Path("/proc").glob("[0-9]*"):
        try:
            cmd = (pid_dir / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="ignore")
        except OSError:
            continue
        if int(pid_dir.name) != os.getpid() and any(n in cmd for n in needles) and "queue_runner" not in cmd:
            found.append(int(pid_dir.name))
    return found


def wait_for(step: dict, status: dict, poll: float) -> None:
    name = step["name"]
    for pattern in step.get("wait_files", []):
        while not glob.glob(pattern, root_dir=str(paths.REPO_ROOT)):
            status["steps"][name] = {"state": "waiting", "for": pattern}
            save_status(status)
            time.sleep(poll)
    needles = step.get("wait_idle", [])
    if needles:
        quiet_since = None
        need = float(step.get("idle_seconds", 120))
        while True:
            busy = matching_processes(needles)
            if busy:
                quiet_since = None
                status["steps"][name] = {"state": "waiting", "for": f"idle of {needles} (pids {busy[:3]})"}
                save_status(status)
            else:
                quiet_since = quiet_since or time.time()
                if time.time() - quiet_since >= need:
                    return
            time.sleep(poll)


def run_step(step: dict, status: dict) -> int:
    name = step["name"]
    cmd = [sys.executable if c == "{python}" else str(c) for c in step["cmd"]]
    env = {**os.environ, **{k: str(v) for k, v in (step.get("env") or {}).items()}}
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    status["steps"][name] = {"state": "running", "started": time.strftime("%Y-%m-%dT%H:%M:%S"), "cmd": cmd}
    save_status(status)
    with open(QUEUE_DIR / f"{name}.log", "ab") as log:
        log.write(f"\n=== {time.strftime('%F %T')} {' '.join(cmd)}\n".encode())
        log.flush()
        code = subprocess.call(cmd, cwd=str(paths.REPO_ROOT), env=env, stdout=log, stderr=subprocess.STDOUT)
    status["steps"][name].update({"state": "done" if code == 0 else "failed", "exit": code,
                                  "ended": time.strftime("%Y-%m-%dT%H:%M:%S")})
    save_status(status)
    return code


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("plan", type=Path)
    ap.add_argument("--poll", type=float, default=30.0, help="seconds between checks while waiting")
    args = ap.parse_args()
    if not args.plan.exists():
        sys.exit(f"plan not found: {args.plan}")

    status = load_status()
    status["runner_pid"], status["runner_started"] = os.getpid(), time.strftime("%Y-%m-%dT%H:%M:%S")
    save_status(status)
    while True:
        plan = yaml.safe_load(args.plan.read_text(encoding="utf-8")) or {}
        pending = [s for s in plan.get("steps", []) if status["steps"].get(s["name"], {}).get("state") not in ("done", "failed")]
        if not pending:
            status["finished"] = time.strftime("%Y-%m-%dT%H:%M:%S")
            save_status(status)
            print("queue finished")
            return 0
        step = pending[0]
        name = step["name"]
        skip = step.get("skip_if")
        if skip and glob.glob(skip, root_dir=str(paths.REPO_ROOT)):
            status["steps"][name] = {"state": "done", "exit": 0, "note": f"skipped, {skip} exists"}
            save_status(status)
            continue
        wait_for(step, status, args.poll)
        code = run_step(step, status)
        if code != 0 and step.get("stop_on_failure"):
            print(f"step {name} failed with {code}; stopping as requested")
            return code


if __name__ == "__main__":
    sys.exit(main())
