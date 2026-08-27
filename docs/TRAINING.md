# Training — setup and run

For anyone on the team, on Windows, Linux or macOS. Roughly 20 minutes to a
working environment, plus dataset time.

> **Status:** steps 1–3 and 5 work today. **Step 4 (dataset) is not done yet** —
> nothing has been downloaded and `configs/data/` does not exist. Until FD-010/FD-011
> land, `train.py` stops with an explanatory message instead of training. That is
> expected on a fresh checkout, not a broken install.

---

## 0. What you need

| | Requirement | Notes |
|---|---|---|
| Python | **3.10 or 3.11** | Not 3.12+, not 3.9. Ultralytics and the CUDA torch wheels don't publish for them. |
| GPU | NVIDIA, CUDA 12.1+ driver, ≥6 GB VRAM | Apple Silicon works via MPS but is slow. CPU is smoke-test only. |
| Disk | ≥50 GB free wherever the data lives | Not necessarily your system drive — see step 2. |
| Git | any | |

Check your Python before anything else:

```bash
python --version
```

If it isn't 3.10/3.11, install one. You do **not** need to change your system
Python — just point the venv at the right one (step 1).

---

## 1. Clone and create the environment

```bash
git clone <repo-url> fish-detection
cd fish-detection
```

Create a venv **with Python 3.10/3.11 specifically**:

```bash
# Windows
py -3.10 -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.10 -m venv .venv
source .venv/bin/activate
```

Your prompt should now show `(.venv)`. Then:

```bash
python scripts/bootstrap.py
```

This detects your OS, architecture and CUDA driver version and installs the
**correct** torch wheel for your machine, then the rest of the training deps.

Why not just `pip install torch`: on Windows that silently installs a **CPU-only**
build. You get no error, no warning — just training that is ~100× slower than it
should be. `bootstrap.py` exists to prevent exactly that.

Preview without installing anything:

```bash
python scripts/bootstrap.py --dry-run
```

---

## 2. Tell it where the data lives *(optional, but read this)*

By default, data goes in `<repo>/data/`. If your repo is on a small drive, put it
elsewhere:

```bash
# Windows (PowerShell, persists across sessions)
setx FISH_DATA_ROOT "E:\fish-data"

# Linux / macOS (add to ~/.bashrc or ~/.zshrc)
export FISH_DATA_ROOT=/mnt/big/fish-data
```

Model and dataset caches follow `FISH_DATA_ROOT` automatically, so nothing
multi-GB lands on your system drive.

**Never hardcode a path in a script or config.** Everything resolves from the repo
root or this variable — that's what lets the same code run on your laptop, on
Colab and on the Jetson.

---

## 3. Verify the machine

```bash
python scripts/check_env.py
```

Prints a report and **exits non-zero if this machine can't train**. Run it after
setup and any time something behaves oddly. Sample output on a machine that isn't
ready:

```
=== PyTorch ===
  torch    : 2.12.1+cpu
  [warn] no GPU available -- torch reports CPU only. The installed wheel is a
         CPU-only build. Reinstall with:
         pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

NOT READY -- 2 blocking problem(s): python-too-new, no-ultralytics
```

Before a real training run, make a missing GPU a hard failure:

```bash
python scripts/check_env.py --require-gpu
```

### Machine profiles

Per-machine settings (device, workers, batch, cache locations) live in
`configs/env/`. Pick one with `FISH_ENV`:

| Profile | For |
|---|---|
| `workstation` *(default)* | x86 desktop/laptop with an NVIDIA GPU |
| `colab` | Colab / Kaggle notebooks — ephemeral, set `FISH_DATA_ROOT` to mounted storage |
| `cpu` | No GPU. Smoke tests only |
| `jetson` | Orin Nano. **Inference only** — `train.py` refuses to run here |

```bash
FISH_ENV=colab python scripts/train.py --config yolov8n_640
```

Adding your machine means **copying a YAML file**, not editing code.

---

## 4. Get the dataset — **NOT DONE YET**

`configs/data/<key>.yaml` doesn't exist yet, so training stops here with:

```
Dataset config 'ozfish' not found at configs/data/ozfish.yaml.
The dataset is not set up on this machine yet. See docs/TRAINING.md step 4,
and ticket FD-011 in docs/tickets/BACKLOG.md.
```

That message is the script working correctly, not failing.

**Why it's blocked:** OzFish is the only real box-labelled source, and acquisition
is unresolved — see **FD-113** and **FD-010** in `docs/tickets/BACKLOG.md`. Short
version: the official Pawsey links can't be resolved with `curl` (JS portal with a
catch-all route — every path returns 200, including nonsense ones), and the
HuggingFace "mirror" turned out to be a 74 MB model file, not a dataset.

When it lands, this step becomes: run the fetch script, run the converter, run the
leakage check, and a `configs/data/ozfish.yaml` appears. Nobody should be
hand-placing files.

---

## 5. Train

```bash
# Smoke test first -- minutes, works on any machine including CPU
python scripts/train.py --config smoke

# The FD-020 baseline: 3 seeds
python scripts/train.py --config yolov8n_640

# One seed
python scripts/train.py --config yolov8n_640 --seed 0

# Resolve and print every setting, train nothing
python scripts/train.py --config yolov8n_640 --dry-run
```

**Always run `--config smoke` after changing anything in the pipeline.** It catches
the shape/path/format errors that otherwise surface an hour into a real run.

### What you get

Each run writes `runs/<UTC-date>-<name>-s<seed>/`:

| File | Contents |
|---|---|
| `config.yaml` | Every resolved setting, including ones you didn't pass |
| `metrics.json` | mAP@50, mAP@50-95, precision, recall |
| `env.txt` / `env.json` | Platform, Python, torch, CUDA, GPU, **git SHA**, profile |

The git SHA is why `runs/**/config.yaml`, `metrics.json` and `env.txt` are tracked
in git while weights are not: any number in the report has to be traceable to the
commit and machine that produced it.

### Multiple seeds — not optional

`yolov8n_640` runs seeds `[0, 1, 2]` and prints:

```
=== 3 seeds: mAP@50 mean 0.7412, range 0.7301-0.7550 (spread 0.0249) ===
Report mean +- range, never the best seed (FD-103).
```

Seed-to-seed variance on a dataset this size routinely reaches ±1–2 mAP points —
often larger than the difference you're trying to measure. **A single run cannot
support a comparison between two models.** If you're about to claim model A beats
model B, you need ≥3 seeds each and a spread smaller than the gap.

### Batch size

Leave `batch: null` and it's derived from your detected VRAM — deliberately
conservative so your first run doesn't OOM. Override per machine in
`configs/env/<profile>.yaml`, or per experiment in `configs/train/<config>.yaml`.

---

## Troubleshooting

**`Python 3.x is not supported`** — you're on the wrong interpreter. Recreate the
venv with `py -3.10` / `python3.10` explicitly. Having 3.12+ as your system Python
is fine; the venv just has to point elsewhere.

**`no GPU available -- the installed wheel is a CPU-only build`** — reinstall torch
from the CUDA index (`check_env.py` prints the exact command), or re-run
`bootstrap.py` inside the venv.

**Refusing to install into the system Python** — you didn't activate the venv.
`(.venv)` should be in your prompt.

**`No machine profile 'x'`** — bad `FISH_ENV`. The error lists valid names.

**`Profile 'jetson' is inference-only`** — working as intended. The Jetson is for
inference; training deps don't install on ARM64+JetPack anyway.

**Windows: DataLoader workers hang or spawn repeatedly** — set `workers: 0` in your
env profile. Windows spawns rather than forks.

**Shell script fails on the Jetson with `\r: command not found`** — a CRLF file got
committed. `.gitattributes` should prevent this; re-clone or run `dos2unix`.

---

## Not covered here

Export to ONNX/TensorRT and anything on the Orin Nano — see epics **E3** and **E4**
in `docs/tickets/BACKLOG.md`. TensorRT engines are **not portable**: they're built
on the target board, never copied from a desktop.

Project constraints, measurement rules and reporting standards: **`CLAUDE.md`**.
Read it before you report a number to anyone.
