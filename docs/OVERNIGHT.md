# Overnight runbook

How the unattended run works and what a scheduled check (a person or an assistant) does each time.
Written 2026-10-04. Update it when the plan changes.

## What runs where

| Where | What | How to see it |
|---|---|---|
| GPU server `ai-gpu` (RTX 5060 8 GB) | `scripts/queue_runner.py configs/overnight_plan.yaml`, detached | `data/.cache/queue/status.json`, per-step logs `data/.cache/queue/<step>.log` (relative to the repo on the server) |
| The same | Training runs, one at a time, in `runs/<date>-<name>-s<seed>/` | `results.csv` (epoch, mAP), `metrics.json` exists only when a run finished |
| Your computer | Nothing needs to stay on. Launching a shell from the assistant may be stopped when the computer is short of memory; the server work is detached and survives that. | |

The plan, in order: resume the crashed YOLOv8s run, YOLOv8m, YOLO11s (all 960 px, seed 0), then the Kakadu
comparison: `yolov8n_960_kakadu` against `yolov8n_960` controls on seeds 0, 1, 2 (seed 0 control already done).

## A check, step by step

1. Reach the server (`ssh ai-gpu`, add `-o ClearAllForwardings=yes`). The tunnel drops now and then: retry up to
   3 times before calling it down.
2. Read `data/.cache/queue/status.json` and `ps` (is the runner alive, is a `train.py` running).
3. For each run folder: epochs done and the latest and best mAP from `results.csv`; whether `metrics.json` exists.
4. A run that finished: copy `args.yaml config.yaml data.yaml env.json env.txt metrics.json results.csv` into `runs/`
   here, `weights/best.pt` into `models/<run>.pt` (git-ignored), run `scripts/error_analysis.py --imgsz 960 --worst 0
   --out runs/analysis-<date>-<run>`, then document it in `docs/PROGRESS.md` as a result entry (see CLAUDE.md:
   metric, split, n seeds, spread, run folder, limits). Commit locally.
5. A step `failed`, or the runner is gone: read the step log, find the cause, fix it (below), requeue.
6. Report plainly, including failures. Say what was done and what was not.

## Repairing during the night

Allowed without asking: fix code and configs in this repo, add or reorder steps in `configs/overnight_plan.yaml`
(the runner re-reads it between steps; copy it to the server), restart a dead runner (`status.json` keeps finished
steps), resume a crashed run with `scripts/train.py --config <name> --seed <n> --resume`, rebuild a deleted
processed dataset with `scripts/get_data.py` (idempotent, rebuilds from `data/raw` and `data/interim`), rerun
`scripts/prepare_kakadu.py`, commit locally.

Kill a process only by PID found from `ps` or `status.json`, never `pkill -f` (it matches the ssh command itself
and kills the session).

**Do not, without being asked:** push to any remote; delete datasets, runs or archives; touch the Jetson; use or
analyse the sealed OzFish set for tuning; run anything that sends data to a new external service; change the
evaluation protocol (`docs/eval_protocol.md`); claim a comparison from fewer than 3 seeds.

## Stop and report instead of repairing when

- the same step fails twice after a fix;
- disk on the server is under 50 GB free;
- data was deleted or changed by someone else: restore from raw if that is safe, then say so;
- a result looks too good (more than 0.05 mAP@50 above the control) or too bad: check the split and the
  evaluation before reporting it, and say what was checked.

## Known hazards

- **Someone else works on the server.** On 2026-10-04 `data/processed/` was emptied while a run was training and
  the run crashed (PROGRESS section 7). Rebuilt from raw with the identical split (verified against the workstation copy).
- **The path to Zenodo from the server is slow** (about 45 KB/s) and the upload from the workstation is slow
  (about 60 KB/s). Do not plan on moving more than a few hundred MB.
- **Housekeeping on the server (2026-10-05).** Unused files were archived with `scripts/archive_paths.py` into `data/archive/<timestamp>/` (moved, not deleted; each folder has a `MANIFEST.json`,
  and `--restore <manifest> --yes` undoes it): our redundant OzFish copy and the leaky random-split diagnostic dataset (`20261005-173254`), the 397 loose Kakadu PNGs and a JSON that were byte-size-identical
  copies of members of `kakadu_training_data.zip`, and our abandoned upload and download leftovers (`20261005-173255`, with `MANIFEST_loose_copies.json` rebuilt after a manifest overwrite).
  **`data/raw/fixed-archive.zip` was left in place:** it is the same size as the Kakadu zip but a different checksum, it is not ours, and nothing uses it; ask before touching it.
  The other project in `~/workspace` (`RL-agent-iot`) is not ours and was not touched.
- **A held-out split inside Kakadu leaks** (PROGRESS F13). Kakadu is training data only; judge on the DeepFish
  val split (comparable with every run) and, at the end, once, on the sealed OzFish set.
