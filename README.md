# Fish Detection for Jetson Orin Nano

Training a model that finds fish in underwater video, small enough to run on a
Jetson Orin Nano.

---

## Run it — five commands

Copy and paste these in order. **You do not need to read any code.**

### 1. Get the project

```bash
git clone https://git.grifufu.io.vn/IoT-Group/image-fish-AI.git
cd image-fish-AI
```

### 2. Make a Python environment

You need **Python 3.10 or 3.11**. Check with `python --version`. If yours is
different, install 3.11 — you don't have to remove the one you have.

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# Mac / Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

You should now see `(.venv)` at the start of your command prompt. If you close the
terminal, run the `activate` line again before anything else.

### 3. Install everything

```bash
python scripts/bootstrap.py
```

Takes about 5 minutes. It works out what kind of computer you have and installs the
right version of everything. **Don't install PyTorch yourself** — on Windows the
normal way silently installs a version that ignores your graphics card and trains
100× slower, with no error message.

### 4. Check it worked

```bash
python scripts/check_env.py
```

You want the last line to say:

```
READY -- this machine can train.
```

If it doesn't, it tells you what's wrong and what to type. Don't continue until it
says READY.

### 5. Get the data

```bash
python scripts/get_data.py
```

Downloads about 1.2 GB and prepares it. Takes 5–15 minutes. **Safe to run again** —
if it gets interrupted, just run it a second time and it picks up where it stopped.

### 6. Train

```bash
python scripts/train.py --config smoke
```

A 2-minute test run on a tiny slice of the data. It proves everything works. The
accuracy will be terrible — that's expected, it only trains for 3 epochs.

Then the real thing:

```bash
python scripts/train.py --config yolov8n_640
```

**This takes several hours** (~2.5 hours per seed, and it runs 3 seeds). Leave it
running. Your results appear in a new folder under `runs/`.

---

## Did it work?

While training runs, open a second terminal and look at:

```bash
runs/<date>-yolov8n-640-s0/results.csv
```

The `metrics/mAP50(B)` column is the score. Higher is better, 1.0 is perfect. It
starts near 0 and climbs. Anything above ~0.5 is a reasonable detector.

When a run finishes it writes `metrics.json` with the final numbers. **If there is
no `metrics.json`, the run did not finish** and its numbers should not be used.

---

## Common problems

| It says | Do this |
|---|---|
| `Python 3.x is not supported` | You're on the wrong Python. Redo step 2 with `py -3.11`. |
| `Refusing to install into the system Python` | You skipped `activate`. Redo step 2. |
| `no GPU available ... CPU-only build` | Copy the `pip install` command it prints and run it. |
| `Dataset 'deepfish' ... does not exist` | Run `python scripts/get_data.py`. |
| `Profile 'jetson' is inference-only` | Correct behaviour — the Jetson runs models, it doesn't train them. |
| Download failed halfway | Run `python scripts/get_data.py` again. |

---

## What is actually going on

**The goal.** Find fish in underwater video with a box around each one, using a
model small enough to run on a Jetson Orin Nano — a credit-card-sized computer that
can sit on a boat or a buoy instead of in a data centre.

**The data.** Two sets, both already downloaded by step 5:

| | Images | What it is |
|---|---|---|
| DeepFish | 6517 | Queensland reef habitats. About a third have no fish at all — that matters, because a detector that cries wolf on empty water is useless. This is what we train on. |
| OzFish | 352 | Baited camera stations. Crowded scenes, up to 232 fish in one frame. **Never trained on** — kept sealed so we can honestly test on something the model has never seen. |

**The hard part.** The fish are small. At the image size everyone normally uses
(640 pixels), about 14% of the fish in DeepFish — and 35% in OzFish — end up smaller
than 16 pixels across. That is roughly the smallest thing this kind of model can
physically represent. So the model isn't just bad at those fish; it largely cannot
see them.

The fix is to feed the model bigger images. The problem is that bigger images are
exactly what a small device like the Jetson can least afford. **Finding the best
trade-off between those two is the point of this project.**

**How we avoid fooling ourselves.** Video frames that come one after another look
almost identical. If you split them randomly into "training" and "testing" piles,
the model sees nearly the same picture in both, and its test score looks great while
meaning nothing. So the data is split by **filming location** — all footage from one
site goes entirely into one pile or the other, never both. The code checks this and
refuses to continue if it's violated.

---

## Where things are

| Folder | What's in it |
|---|---|
| `scripts/` | The commands you run |
| `configs/` | Settings — what to train, on what data, on which machine |
| `docs/` | Explanations (below) |
| `runs/` | Results, one folder per training run |
| `src/fish/` | Shared code |
| `tests/` | Automatic checks (`python -m pytest`) |

| Document | Read it when |
|---|---|
| `docs/TRAINING.md` | You want the detailed version of this page |
| `docs/data_audit.md` | You want to know what's actually in the data |
| `docs/eval_protocol.md` | **Before quoting any number to anyone** |
| `docs/tickets/BACKLOG.md` | You want to know what's done and what's next |
| `CLAUDE.md` | You're changing how the project works |

---

## Status

**Working:** environment setup, data pipeline, training, evaluation. Verified on
Windows 11 with an RTX 4060.

**Not built yet:** everything on the Jetson itself — converting the model to run on
the board, measuring its real speed, and the field pipeline. That's epics E3 and E4
in the backlog.

**No trained model yet.** A baseline run was started and stopped early on purpose;
see `runs/20260827-yolov8n-640-s0/INTERRUPTED.md`. Nothing in this repo should be
quoted as a result until a run finishes and writes `metrics.json`.
