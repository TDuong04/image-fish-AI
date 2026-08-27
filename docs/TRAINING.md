# Training — setup and run

For anyone on the team, on Windows, Linux or macOS. About 20 minutes to a working
environment, then one command to build the dataset and one to train.

> **Status:** the full path works end to end — environment, dataset, training,
> evaluation. Verified on Windows 11 + RTX 4060 (torch 2.6.0+cu124, ultralytics
> 8.4.130). The Jetson half (E3/E4) is not built yet.

---

## 0. What you need

| | Requirement | Notes |
|---|---|---|
| Python | **3.10 or 3.11** | Not 3.12+, not 3.9. Ultralytics and CUDA torch wheels don't publish for them. |
| GPU | NVIDIA, CUDA 12.1+ driver, ≥6 GB VRAM | Apple Silicon works via MPS but is slow. CPU is smoke-test only. |
| Disk | ~5 GB | 1.5 GB downloads + hard-linked datasets (no duplication). |

```bash
python --version     # must be 3.10 or 3.11
```

If it isn't, install one — you don't need to change your system Python, just point
the venv at the right interpreter (step 1).

---

## 1. Clone and bootstrap

```bash
git clone https://git.grifufu.io.vn/IoT-Group/image-fish-AI.git
cd image-fish-AI
```

```bash
# Windows
py -3.11 -m venv .venv
.venv\Scripts\activate

# Linux / macOS
python3.11 -m venv .venv
source .venv/bin/activate
```

```bash
python scripts/bootstrap.py
```

Detects your OS, architecture and CUDA driver, then installs the **correct** torch
wheel. This matters: on Windows a plain `pip install torch` silently installs a
**CPU-only** build — no error, no warning, just training that is ~100× slower than
it should be.

Preview without installing: `python scripts/bootstrap.py --dry-run`

---

## 2. Optional: put data on another drive

Defaults to `<repo>/data/`. To move it:

```bash
setx FISH_DATA_ROOT "E:\fish-data"          # Windows
export FISH_DATA_ROOT=/mnt/big/fish-data    # Linux / macOS
```

Model and dataset caches follow it automatically, so nothing multi-GB lands on your
system drive. **Never hardcode a path in a script or config** — that's what lets the
same code run on your laptop, on Colab and on the Jetson. `pytest` enforces it.

---

## 3. Verify the machine

```bash
python scripts/check_env.py
```

Exits non-zero if this machine can't train. A healthy report ends with:

```
  [ ok ] GPU available: NVIDIA GeForce RTX 4060 Laptop GPU
  [ ok ] resolved device: cuda (NVIDIA GeForce RTX 4060 Laptop GPU, 8.0 GB)
  [ ok ] 512x512 matmul on GPU succeeded
READY -- this machine can train.
```

Before a real run: `python scripts/check_env.py --require-gpu`

### Machine profiles

Per-machine settings live in `configs/env/`. Select with `FISH_ENV`:

| Profile | For |
|---|---|
| `workstation` *(default)* | x86 desktop/laptop with an NVIDIA GPU |
| `colab` | Colab / Kaggle — ephemeral, set `FISH_DATA_ROOT` to mounted storage |
| `cpu` | No GPU. Smoke tests only |
| `jetson` | Orin Nano. **Inference only** — `train.py` refuses to run here |

Adding your machine means copying a YAML file, not editing code.

---

## 4. Get the data

Two datasets, both from the YOLO-Fish authors' Google Drive. Roughly 1.3 GB.

```bash
mkdir -p data/raw && cd data/raw
python -m gdown 10Pr4lLeSGTfkjA40ReGSC8H3a9onfMZ0 -O deepfish_darknet.zip   # 1.09 GB
python -m gdown 1C_7l2YFc5fXt1DMsuVZPDFKTJLm0syX3 -O ozfish_test.zip        # 60 MB
cd ../..

python -c "import zipfile; zipfile.ZipFile('data/raw/deepfish_darknet.zip').extractall('data/interim/deepfish_darknet')"
python -c "import zipfile; zipfile.ZipFile('data/raw/ozfish_test.zip').extractall('data/interim/ozfish_test')"
```

Then build the datasets:

```bash
python scripts/prepare_dataset.py deepfish --group site
python scripts/prepare_dataset.py ozfish_test --all-val --name ozfish_eval
python scripts/prepare_dataset.py deepfish --group site --max-per-group 12 --name smoke
```

| Dataset | Role | Images | Boxes | Empty |
|---|---|---|---|---|
| `deepfish` | train + val | 5178 / 1339 | 14078 / 1385 | ~30% |
| `ozfish_eval` | **sealed** cross-dataset test | 352 | 8329 | 0% |
| `smoke` | pipeline test | 168 / 72 | 398 / 67 | ~27% |

**Why `--group site` is not optional.** Frames come from continuous video —
DeepFish clips run to 462 consecutive frames of the same scene. A random per-image
split would put ~73 near-identical frames of one clip on both sides of the boundary
and inflate mAP by an unknown margin. Splits are made over whole capture sites, and
the script **asserts** no site appears in both. It fails rather than warns.

Images are hard-linked, not copied, so this costs no meaningful disk.

Inspect any dataset:

```bash
python scripts/audit_dataset.py data/processed/deepfish/images/train
```

---

## 5. Train

```bash
python scripts/train.py --config smoke            # ~2 min, verifies the pipeline
python scripts/train.py --config yolov8n_640      # baseline, 3 seeds
python scripts/train.py --config yolov8n_960      # high-res comparison
python scripts/train.py --config yolov8n_640 --seed 0        # single seed
python scripts/train.py --config yolov8n_640 --dry-run       # settings only
```

**Always run `--config smoke` after touching the pipeline.** It catches the
shape/path/format errors that otherwise surface an hour into a real run.

A full 640 run is ~140 s/epoch on an RTX 4060, up to 150 epochs with `patience=30`
— budget several hours per seed.

### What you get

`runs/<UTC-date>-<name>-s<seed>/`:

| File | Contents |
|---|---|
| `config.yaml` | every resolved setting |
| `data.yaml` | the exact dataset consumed, resolved for this machine |
| `metrics.json` | mAP@50, mAP@50-95, precision, recall |
| `env.txt` / `env.json` | platform, torch, CUDA, GPU, **git SHA**, profile |
| `weights/best.pt` | selected by val mAP, not last epoch |

`runs/**/config.yaml`, `metrics.json` and `env.txt` are tracked in git; weights are
not. Any number in the report must be traceable to a commit and a machine.

### Seeds are not optional

`yolov8n_640` runs seeds `[0, 1, 2]` and prints:

```
=== 3 seeds: mAP@50 mean 0.7412, range 0.7301-0.7550 (spread 0.0249) ===
Report mean +- range, never the best seed (FD-103).
```

Seed-to-seed variance here routinely reaches ±1–2 mAP points — often larger than
the difference you are trying to measure. **A single run cannot support a
comparison between two models.**

---

## 6. Before you trust a number

Read `docs/eval_protocol.md`. It was written before any real training run, and it
fixes the metrics, the operating threshold, the sealed test set and the seed rules
in advance. The short version:

- `ozfish_eval` is **sealed** — no model selection, no threshold tuning, no
  quantisation calibration. Read once per reported model.
- mAP is threshold-free; the operating threshold is chosen on DeepFish val.
- False-positives-per-empty-frame is reported on DeepFish val only — `ozfish_eval`
  has no empty frames and cannot measure it.
- DeepFish and OzFish are sharply different distributions (2.4 vs 23.7 boxes per
  image). Cross-dataset results measure a large domain shift, not like-for-like
  generalisation.

---

## Tests

```bash
python -m pytest
```

55 checks covering frame grouping, label parsing, and the portability rules (no
absolute paths, no hardcoded `cuda`, `__main__` guards). Run before pushing — these
already caught one real bug where dataset configs were being written with an
absolute path to one developer's drive.

---

## Troubleshooting

**`Python 3.x is not supported`** — wrong interpreter. Recreate the venv with
`py -3.11` / `python3.11` explicitly.

**`no GPU available -- the installed wheel is a CPU-only build`** — reinstall torch
from the CUDA index (`check_env.py` prints the command), inside the venv.

**`Refusing to install into the system Python`** — venv not activated. `(.venv)`
should be in your prompt.

**`Dataset 'x' points at ... which does not exist`** — run
`python scripts/prepare_dataset.py x`.

**`Profile 'jetson' is inference-only`** — working as intended.

**Windows: DataLoader workers hang** — set `workers: 0` in your env profile.
Windows spawns rather than forks.

**Shell script fails on the Jetson with `\r: command not found`** — a CRLF file got
committed. `.gitattributes` prevents this; re-clone or run `dos2unix`.

---

## Not covered here

ONNX/TensorRT export and anything on the Orin Nano — epics **E3** and **E4** in
`docs/tickets/BACKLOG.md`. TensorRT engines are **not portable**: they are built on
the target board, never copied from a desktop.

Project constraints and reporting standards: **`CLAUDE.md`**. Read it before
reporting a number to anyone.
