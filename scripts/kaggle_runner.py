#!/usr/bin/env python
"""Drive Kaggle training sessions from here: push, poll, fetch, decide.

    python scripts/kaggle_runner.py status          # what is Kaggle doing now?
    python scripts/kaggle_runner.py push            # start a session
    python scripts/kaggle_runner.py fetch           # pull the finished output
    python scripts/kaggle_runner.py cycle           # do whichever of those is due

`cycle` is the one an automation loop calls. It is deliberately idempotent and
cheap: if a kernel is still running it does nothing and says so, so calling it
every 30 minutes costs one API request and never disturbs a live session.

Kaggle sessions are stateless and capped at ~12h, so a full multi-seed plan
spans several sessions. `cycle` walks a plan in configs/kaggle_plan.yaml,
attaching the previous session's output to the next so training resumes rather
than restarting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import yaml  # noqa: E402

from fish import paths  # noqa: E402

KERNEL_DIR = paths.REPO_ROOT / "kaggle"
NOTEBOOK = KERNEL_DIR / "fish-detection-train.ipynb"
STATE = paths.RUNS_ROOT / "_kaggle"

# Kaggle's own status vocabulary.
BUSY = {"queued", "running"}
DONE = {"complete"}
BAD = {"error", "cancelrequested", "cancelacknowledged"}


def kaggle(*args: str, check: bool = True) -> tuple[int, str]:
    cmd = [sys.executable, "-m", "kaggle", *args]
    proc = subprocess.run(cmd, capture_output=True, text=True,
                          encoding="utf-8", errors="replace")
    out = (proc.stdout or "") + (proc.stderr or "")
    if check and proc.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed:\n{out}")
    return proc.returncode, out


def load_plan() -> dict:
    path = paths.CONFIG_DIR / "kaggle_plan.yaml"
    if not path.exists():
        raise SystemExit(
            f"No plan at {paths.rel(path)}.\n"
            f"Run: python scripts/kaggle_runner.py init"
        )
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


def save_plan(plan: dict) -> None:
    path = paths.CONFIG_DIR / "kaggle_plan.yaml"
    path.write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")


def kernel_slug(plan: dict) -> str:
    return f"{plan['username']}/{plan['kernel']}"


def write_metadata(plan: dict, step: dict) -> Path:
    """kernel-metadata.json is how `kaggle kernels push` is configured."""
    meta = {
        "id": kernel_slug(plan),
        "title": plan["kernel"],
        "code_file": NOTEBOOK.name,
        "language": "python",
        "kernel_type": "notebook",
        "is_private": plan.get("private", False),
        "enable_gpu": True,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        # Attaching the previous session's output is what makes a multi-session
        # run continue instead of restarting from epoch 0.
        "kernel_sources": ([kernel_slug(plan)] if step.get("resume_from_previous") else []),
    }
    path = KERNEL_DIR / "kernel-metadata.json"
    path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    return path


def patch_notebook(step: dict) -> None:
    """Rewrite the notebook's config cell to match the current plan step.

    The notebook is the unit Kaggle executes, so the settings have to live
    inside it. Editing the single config cell keeps the rest under git.
    """
    nb = json.loads(NOTEBOOK.read_text(encoding="utf-8"))
    for cell in nb["cells"]:
        src = "".join(cell["source"])
        if cell["cell_type"] == "code" and src.lstrip().startswith("CONFIG"):
            seeds = step.get("seeds", [0])
            cell["source"] = [
                f'CONFIG     = "{step["config"]}"\n',
                f"SEEDS      = {seeds}\n",
                f'MAX_HOURS  = {step.get("max_hours", 10.5)}\n',
                "\n",
                'REPO_OWNER = "TDuong04"\n',
                'REPO_NAME  = "image-fish-AI"\n',
                'BRANCH     = "main"',
            ]
            NOTEBOOK.write_text(json.dumps(nb, indent=1), encoding="utf-8")
            return
    raise SystemExit("Could not find the CONFIG cell in the notebook.")


def get_status(plan: dict) -> dict:
    code, out = kaggle("kernels", "status", kernel_slug(plan), check=False)
    text = out.strip()
    low = text.lower()

    # Check auth failures FIRST. A permission error is not "no kernel yet" --
    # conflating them would make `cycle` push a new version every tick while
    # credentials are broken.
    if any(s in low for s in ("authentication required", "permission",
                              "denied", "401", "403", "unauthorized")):
        return {"status": "auth_error", "raw": text[:500]}

    if "not found" in low or "404" in low:
        return {"status": "absent", "raw": text[:500]}

    for word in ("complete", "running", "queued", "error", "cancelrequested"):
        if word in low:
            return {"status": word, "raw": text[:500]}

    return {"status": "unknown", "raw": text[:500]}


def cmd_init(args) -> int:
    plan = {
        "username": args.username,
        "kernel": args.kernel,
        "private": False,
        "current": 0,
        "steps": [
            {"config": "smoke", "seeds": [0], "max_hours": 1.0,
             "note": "pipeline check on Kaggle hardware before spending real hours"},
            {"config": "yolov8n_640", "seeds": [0], "max_hours": 10.5},
            {"config": "yolov8n_640", "seeds": [1], "max_hours": 10.5},
            {"config": "yolov8n_640", "seeds": [2], "max_hours": 10.5},
            {"config": "yolov8n_960", "seeds": [0], "max_hours": 10.5},
            {"config": "yolov8n_960", "seeds": [1], "max_hours": 10.5},
            {"config": "yolov8n_960", "seeds": [2], "max_hours": 10.5},
        ],
    }
    save_plan(plan)
    print(f"Wrote {paths.rel(paths.CONFIG_DIR / 'kaggle_plan.yaml')} "
          f"with {len(plan['steps'])} steps.")
    print("Edit it freely -- 'cycle' walks it in order.")
    return 0


def cmd_status(args) -> int:
    plan = load_plan()
    st = get_status(plan)
    step_i = plan.get("current", 0)
    steps = plan["steps"]
    print(f"kernel : {kernel_slug(plan)}")
    print(f"status : {st['status']}")
    print(f"step   : {step_i + 1}/{len(steps)}"
          + (f"  ({steps[step_i]['config']} seed {steps[step_i].get('seeds')})"
             if step_i < len(steps) else "  (plan complete)"))
    if st["status"] == "unknown":
        print(f"raw    : {st['raw'][:200]}")
    return 0


def cmd_push(args) -> int:
    plan = load_plan()
    step_i = plan.get("current", 0)
    if step_i >= len(plan["steps"]):
        print("Plan complete -- nothing to push.")
        return 0
    step = plan["steps"][step_i]
    # Every step after the first continues the same kernel, so its output is
    # attached as a source and checkpoints carry forward.
    step.setdefault("resume_from_previous", step_i > 0)

    patch_notebook(step)
    write_metadata(plan, step)
    print(f"pushing step {step_i + 1}/{len(plan['steps'])}: "
          f"{step['config']} seeds={step.get('seeds')} "
          f"(resume={step['resume_from_previous']})")
    code, out = kaggle("kernels", "push", "-p", str(KERNEL_DIR), check=False)
    print(out.strip()[:500])
    if code != 0:
        return 1

    STATE.mkdir(parents=True, exist_ok=True)
    (STATE / "last_push.json").write_text(json.dumps({
        "step": step_i, "config": step["config"], "seeds": step.get("seeds"),
        "pushed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }, indent=2), encoding="utf-8")
    return 0


def cmd_fetch(args) -> int:
    plan = load_plan()
    STATE.mkdir(parents=True, exist_ok=True)
    out_dir = STATE / f"output-step{plan.get('current', 0)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    code, out = kaggle("kernels", "output", kernel_slug(plan),
                       "-p", str(out_dir), check=False)
    print(out.strip()[:400])

    report = out_dir / "kaggle_report.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        print(f"\noutcome : {data.get('outcome')}")
        for name, m in (data.get("metrics") or {}).items():
            if "map50" in m:
                print(f"  {name}: mAP50={m['map50']:.4f} mAP50-95={m['map50_95']:.4f}")
        for f in data.get("failures") or []:
            print(f"  FAILURE {f['failure_kind']} retryable={f['retryable']} "
                  f"seed={f['seed']}")
            print(f"    {f['log_tail'].splitlines()[-1][:160]}" if f.get("log_tail") else "")
        return 0
    print("No kaggle_report.json in the output -- the session did not finish cleanly.")
    return 1


def cmd_cycle(args) -> int:
    """One supervision tick. Safe to call on a timer."""
    plan = load_plan()
    step_i = plan.get("current", 0)
    steps = plan["steps"]

    if step_i >= len(steps):
        print("PLAN COMPLETE -- every step finished. Nothing to do.")
        return 0

    st = get_status(plan)
    print(f"[{datetime.now(timezone.utc).strftime('%H:%M')}Z] "
          f"step {step_i + 1}/{len(steps)} ({steps[step_i]['config']}) "
          f"-- kernel status: {st['status']}")

    if st["status"] == "auth_error":
        print("ACTION: none -- Kaggle credentials are not working.")
        print("  Fix with: python -m kaggle auth login")
        print(f"  {st['raw'][:200]}")
        return 1

    if st["status"] in BUSY:
        print("ACTION: none. Kernel is still working; check again later.")
        return 0

    if st["status"] == "absent":
        print("ACTION: no kernel yet -- pushing the first version.")
        return cmd_push(args)

    if st["status"] in DONE:
        print("ACTION: session finished -- fetching output.")
        rc = cmd_fetch(args)
        out_dir = STATE / f"output-step{step_i}"
        report = out_dir / "kaggle_report.json"
        outcome = None
        if report.exists():
            outcome = json.loads(report.read_text(encoding="utf-8")).get("outcome")

        if outcome == "finished":
            plan["current"] = step_i + 1
            save_plan(plan)
            print(f"STEP {step_i + 1} DONE -> advancing to step {step_i + 2}.")
            if plan["current"] < len(steps):
                return cmd_push(args)
            print("PLAN COMPLETE.")
            return 0

        if outcome == "budget_reached":
            print("Budget reached with work left -- re-pushing the SAME step to continue.")
            steps[step_i]["resume_from_previous"] = True
            save_plan(plan)
            return cmd_push(args)

        print(f"Session ended with outcome={outcome!r}. "
              f"NEEDS A DECISION -- inspect the failures above before re-pushing.")
        return 1

    if st["status"] in BAD:
        print("ACTION: kernel errored. Fetching output so the cause is readable.")
        cmd_fetch(args)
        print("NEEDS A DECISION -- do not blindly re-push; diagnose first.")
        return 1

    print(f"Unrecognised status {st['status']!r}; taking no action.")
    print(st["raw"][:300])
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="write configs/kaggle_plan.yaml")
    p_init.add_argument("--username", required=True)
    p_init.add_argument("--kernel", default="fish-detection-train")
    p_init.set_defaults(func=cmd_init)

    for name, fn, helptext in (
        ("status", cmd_status, "print kernel and plan status"),
        ("push", cmd_push, "push the current plan step"),
        ("fetch", cmd_fetch, "download the last session output"),
        ("cycle", cmd_cycle, "one supervision tick -- what an automation loop calls"),
    ):
        p = sub.add_parser(name, help=helptext)
        p.set_defaults(func=fn)

    args = ap.parse_args()
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
