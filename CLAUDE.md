# CLAUDE.md — Fish Detection on Jetson Orin Nano

## What this project is

Train a fish **object detector** (bounding boxes, underwater/BRUVS footage) that runs on a
**Jetson Orin Nano (8GB module, in hand)**, and write an academic report whose performance claims are
validated on the real board — not extrapolated from desktop numbers.

Two deliverables, both required:
1. **Report** — reproducible experiments, honest baselines, cross-dataset evaluation.
2. **Device** — a working Orin Nano inference pipeline with measured, reproducible FPS.

Deployment mode: **offline batch on recorded video first**, real-time camera as a stretch goal.

## Hard constraints (do not design around these being false)

### Target hardware
- **Jetson Orin Nano 8GB**, Ampere, 6-core Cortex-A78AE, passively cooled (no fan). The facts
  that are still unknown — Super mode, JetPack/L4T, TensorRT, storage — are listed in
  `docs/hardware.md`. Run `edge/probe_board.py` on the board and record them before writing any
  export code: the JetPack version pins TensorRT, which pins the usable ONNX opset.
- **Memory is unified.** *All* Jetsons share one pool between CPU and GPU — the 8GB module is
  not "8GB of dedicated VRAM", and JetPack plus a desktop session consumes ~1.5–2 GB before your
  process starts. Even so, YOLOv8n is ~6 MB in FP16, so memory is not the constraint here and
  INT8 is a research question about accuracy cost, not a deployment necessity.
- **No fan means thermal throttling is the main threat to measurement integrity.** A burst
  benchmark overstates a passively cooled module. Sustained (30+ min) numbers are the real ones,
  and ambient temperature, clocks and temperature are logged with every one.
- **No DLA and no NVENC.** Orin Nano is the Orin module without the hardware video encoder, and
  without a DLA (Orin NX and AGX Orin have both). Decode is accelerated (NVDEC); *encoding*
  annotated video is CPU x264 and will dominate any pipeline that does it. Everything infers on
  the one 1024-core Ampere GPU — nothing to offload to.
- **Every FPS number MUST cite its `nvpmodel` mode** and `jetson_clocks` state. Do not assert a
  magnitude for the mode-to-mode swing before measuring it. The modes this board offers depend
  on whether Super (JetPack 6.2+) is present, which `edge/probe_board.py` reports from
  `/etc/nvpmodel.conf`. Measure the swing on the board (FD-044); do not inherit it from a memo.
- **TensorRT engines are not portable.** They are specific to GPU arch + TensorRT version +
  driver. You cannot build a `.engine` on the x86 dev box and ship it. Build ONNX on x86,
  build the engine **on the Orin**.

### Portability — code must run on any machine, not just this one

This project already spans **three platforms**: a Windows workstation, a hosted Linux GPU
notebook (Kaggle/Colab), and an ARM64 Jetson. Code that only runs on the author's laptop is a
defect, not a shortcut. Treat portability as a functional requirement.

**Non-negotiable rules:**

- **No absolute paths in code, configs, or docs.** No `D:\`, no `/home/<user>/`, no
  `/kaggle/input/`. Every path derives from the repo root or from an environment variable.
- **One path module.** `src/fish/paths.py` resolves the repo root from `__file__` and exposes
  `DATA_ROOT`, `RUNS_ROOT`, `MODELS_ROOT`. `DATA_ROOT` reads `$FISH_DATA_ROOT` and falls back to
  `<repo>/data`. Nothing else computes a path from scratch.
- **`pathlib.Path` only.** Never string-concatenate paths, never hardcode `/` or `\`, never
  `os.path.join` with a literal separator.
- **Never hardcode a device.** `torch.device("cuda")` fails on a CPU-only box and on Apple
  silicon. Resolve once: `cuda` → `mps` → `cpu`, and log which was chosen.
- **Never assume a GPU exists, or how much VRAM it has.** Batch size, image size, worker count
  and AMP come from a config, with a documented default that fits ~6 GB VRAM. Auto-scale or
  fail loudly with a clear message — do not silently OOM.
- **Python entrypoints, not shell scripts.** No `.ps1`/`.bat` as the only way to do something,
  no bash-only pipelines in the critical path. If a shell wrapper is convenient, it must be a
  thin call into a Python entrypoint that works standalone.
- **Guard multiprocessing.** Windows and macOS use `spawn`, not `fork`. Any script using
  DataLoader workers needs an `if __name__ == "__main__":` guard or it will fork-bomb on
  Windows. Default `num_workers` must be safe cross-platform.
- **Case sensitivity.** Linux is case-sensitive; Windows and macOS are usually not. A path that
  works here can 404 on the Jetson. Keep all data filenames lowercase, no spaces.
- **Line endings.** Commit a `.gitattributes` (`* text=auto`, `*.sh text eol=lf`) or shell
  scripts written on Windows will fail on the Jetson with `\r: command not found`.
- **Three requirement sets, not one.** `requirements-train.txt` (x86 + CUDA),
  `requirements-edge.txt` (Jetson), `requirements-dev.txt`. They cannot be merged — see below.
- **A machine profile, not machine assumptions.** Per-platform settings live in
  `configs/env/{workstation,colab,jetson}.yaml`, selected by `$FISH_ENV`. Adding a fourth
  machine means adding a fourth YAML file and changing no code.

**Jetson-specific packaging trap:** `pip install torch` does **not** give you a working
GPU torch on a Jetson. ARM64 + JetPack requires NVIDIA's prebuilt wheels (or the `dusty-nv`
containers) matched to the exact JetPack version. `requirements-train.txt` will never install
on the board. This is why the edge code in `edge/` must not import training-only dependencies.

**Every script must self-check and fail with a readable message**, not a stack trace: missing
dataset, no GPU when one is required, insufficient free disk, wrong Python version.

### Known-good environments

Recorded in `docs/environments.md`, not assumed in code. Current entries:

| Profile | Platform | GPU | Notes |
|---|---|---|---|
| `workstation` | Windows 11 | RTX 4060 Laptop, 8 GB | Fine for YOLOv8n/s/m @640. Not enough for native 1080p training. |
| `colab` / `kaggle` | Linux x86 | T4 16 GB | Used for the earlier LCFCN run. Ephemeral — nothing persists. |
| `jetson` | Linux ARM64 | Orin Nano | Inference only. Never train here. |

**Workstation state (verified):** Python 3.11 venv at `.venv/`, torch 2.6.0+cu124 with
`cuda.is_available() == True` on the RTX 4060, ultralytics 8.4.130. The system Python is 3.14,
which Ultralytics and CUDA torch wheels do not support — never run the project outside the venv.
A plain `pip install torch` on Windows installs a **CPU-only build** without any error, so always
install through `scripts/bootstrap.py` and confirm with `scripts/check_env.py`.

### Disk
- Full CFC v1.1 is ~132 GB and is **out of scope** (sonar modality, and it does not fit).
- Assume disk is scarce on every machine. Scripts check free space before any large download
  and refuse rather than filling the volume mid-run.
- On the current workstation the system drive has ~7 GB free while the data drive has ~182 GB —
  so caches must not default to the system drive. Set `HF_HOME`, `TORCH_HOME`, `ULTRALYTICS_DIR`
  and the pip cache via the env profile, never by editing code.

### Paths on this workstation (tooling gotcha, not a code concern)
The current checkout sits at a path containing `[`, `]` and a space. PowerShell treats `[` as a
wildcard, so `Test-Path`/`Remove-Item`/`Get-ChildItem` need **`-LiteralPath`**. This is a
reason to keep paths out of scripts entirely — it is not a rule for the code to encode.

### Licensing — decide this BEFORE training (FD-008), not after
- Ultralytics YOLOv8/v11 is **AGPL-3.0**. YOLO-Fish is **GPL-3.0**.
- **The AGPL reaches the COCO-pretrained `yolov8n.pt` weights** you initialise from, so the
  resulting checkpoint is encumbered regardless of what code ships.
- The trigger is **conveying**, not selling. The question is not "will it be sold" but
  **"will any copy of this software or these weights ever leave the institution?"** Handing a
  unit to a partner fisheries agency is conveying.
- If a non-copyleft path is needed: **YOLOX-nano or NanoDet-Plus**. **Not Ultralytics' `RTDETR`**
  — that implementation is AGPL like the rest of the repo; only upstream `lyuwenyu/RT-DETR` is
  Apache. RT-DETR is also a poor Orin candidate (deformable attention has a weak TensorRT plugin
  story on Jetson).
- OzFish is CC BY 3.0 AU — attribution required, and **the converted YOLO labels are a
  derivative work**. A third-party mirror breaks the attribution provenance chain CC BY needs.
  DeepFish and CFC are MIT.

## Data reality

| Source | Boxes? | On disk | Notes |
|---|---|---|---|
| DeepFish | **Yes** — darknet export, 6,517 images, 15,463 boxes | **Yes** | From the YOLO-Fish authors' Drive archive. Training + validation, split by site. The original FishLoc split is points only. |
| OzFish | **Yes, single-class** | **Test set only** (352 images, 8,329 boxes) | Sealed cross-dataset test. The full ~45k-box set is still not acquired. |
| CFC | Yes, sonar | Labels yes (~206 MB), images no | **Sonar, not optical.** Different modality — do not mix into an optical training set |
| YOLO-Fish | Weights only | 10 darknet `.weights` (~2.37 GB) | Baseline comparison point; darknet, not PyTorch |

**The most important data fact:** DeepFish and the OzFish test set came from the YOLO-Fish authors'
Google Drive archives (IDs were in `fish-research/README.md` all along), which removed the OzFish
acquisition blocker for training. The full OzFish training set is still unacquired. Two earlier
claims about it were wrong and are recorded in `docs/PROGRESS.md` §6: the HuggingFace "mirror" is a
3-file *model* repo, not a dataset, and "Pawsey URLs are 404" is unsupported — every FDFML path
returns HTTP 200, but so does a nonsense path (JS portal with a catch-all route), so **curl can
prove neither presence nor absence.**

**Frames from one video are near-duplicates.** DeepFish clips run to 462 consecutive frames. Splits
are made over whole capture *sites*, never per image, and `prepare_dataset.py` asserts no site
appears on both sides. If a new dataset's filenames match no pattern in `src/fish/datasets.py`,
every frame becomes its own group and the leakage check passes while protecting nothing — verify
the grouping yields far fewer groups than frames before trusting any split.

**OzFish boxes are already single-class.** Per `fish-research/ozfish/README.md` line 57: the box
annotations are *"fish/no-fish only and have no species/genus/family labels."* The 507-species
figure belongs to the ~80k **crops**, a different artefact. There is no species collapse to
perform. (`report_sections.md` §2.2 was corrected; the file as a whole is still stale.)

**Sonar ≠ optical.** CFC is sonar imagery. It is a valid separate benchmark and the right
data for domain-adaptation work, but it is not extra training data for an optical detector.

## Rules of engagement

### Measurement
- Every latency number is **end-to-end**: decode → preprocess → inference → NMS → postprocess.
  GPU-kernel-only timing is not a deployment result and must never be reported as one.
- Discard warm-up iterations. Report median and p95, not just mean.
- The chain PyTorch FP32 → ONNX → TensorRT FP16 → INT8 is a **parity check** — run it to catch
  silent export corruption. It is *not* a results-table axis: TensorRT FP32 on an Orin Nano is
  not a configuration anyone deploys, and measuring it burns board time on a cell no one reads.
  The accuracy lost to **quantization** is a core finding; FP32-on-device is not.
- **Single runs do not support comparisons.** Seed-to-seed mAP variance on a dataset this size
  routinely exceeds the differences being decided. Any comparison that drives a decision needs
  ≥3 seeds, reported as mean ± range, against a minimum detectable difference stated in advance.
- Frames from one video are **not independent samples**. Per-frame mAP over adjacent frames
  overstates confidence; say so, and prefer clip-level metrics where the claim allows.
- Cross-dataset evaluation (train on A, test on B) is the honest generality metric. In-domain
  mAP alone overstates field performance.
- BRUVS footage is mostly empty frames. Report false-positive rate on empty frames explicitly;
  mAP hides this.

### Reproducibility
- Pin every seed. Log the git SHA, config, and environment with every run.
- Never report a number that is not regenerable from a committed config.
- If a number comes from a Kaggle/Colab session that no longer exists, mark it clearly as
  unreproduced rather than quietly citing it.

### Honesty
- Do not compare our numbers to published numbers computed on a different split, image size,
  or metric definition without saying so in the same sentence.
- A model trained for 10 epochs is not a converged result. Label it as what it is.
- If a claim can't be validated on the actual board, the report says so.
- **Never inherit a hardware number from a memo, a blog, or this file.** Power-mode swings, FPS,
  TOPS and quantization deltas get measured on the board and cited to the run that produced them.

## Project state

The authoritative, current account is **`docs/PROGRESS.md`** — what was built, every result,
every correction, every open item. Read it before starting work; do not rely on this file for
status, because this file holds rules and constraints, not results.

Headline, as of 2026-10-04: six converged YOLOv8n runs (3 seeds x 640/960 px) on DeepFish;
960 beats 640 by +0.087 mAP@50 in-domain but only +0.013 on the sealed OzFish set; the detector
tolerates missing colour (−7 to −9%) but not wrong colour (up to −49%). **No Jetson Orin Nano
exists in this project, so there is no device measurement of any kind.**

Still-stale artefacts: `report_sections.md` describes the older LCFCN counting experiment
(val MAE 0.3375 @ 10 epochs, Kaggle T4) — a different task, not the baseline for this work.
`fish-research/` holds 5 cloned repos and ~4 GB of labels and weights, git-ignored.

## Conventions

- Datasets → `$FISH_DATA_ROOT` (default `<repo>/data`), never inside a cloned repo, never an
  absolute path written into a file.
- Experiment outputs → `runs/<YYYYMMDD>-<slug>/` with `config.yaml`, `metrics.json`, `env.txt`.
  `env.txt` records platform, Python, torch, CUDA/JetPack, GPU name and the `$FISH_ENV` profile,
  so a result can always be traced to the machine that produced it.
- Tickets → `docs/tickets/BACKLOG.md`, IDs `FD-###`. Update status in place; don't fork copies.
- Team setup + training instructions → `docs/TRAINING.md`. Keep it truthful about what is
  runnable today; a teammate hitting an undocumented wall is a bug in that file.
- Python 3.10/3.11 venv at `.venv/`; pins in `requirements-{train,edge,dev}.txt`.
- Shared, platform-independent code lives in `src/fish/`. Training entrypoints in `scripts/`.
- Jetson-side code lives in `edge/` and must not import training-only dependencies
  (no `ultralytics`, no `torch` at inference time — TensorRT + numpy + the decode path only).

## Documentation is part of the work

**A finding or a code change is not finished until it is in the docs.** This is a rule, not a
courtesy. Results that live only in a chat, a terminal scrollback or a Kaggle output store are
lost, and this project has already been bitten by exactly that (see `docs/PROGRESS.md` §7).

Update the docs **in the same change** as the work, and commit them together:

| You did this | Update this, in the same commit |
|---|---|
| Produced or revised any **result** (a metric, an experiment, a measurement) | `docs/PROGRESS.md` §4/§5 — the number, how it was measured, how many seeds, the committed run it came from |
| Changed **code** (a script, a module, a config, a pipeline stage) | `docs/PROGRESS.md` §2 — what it is and what it does; plus the doc that explains how to use it |
| Changed **how to set up, train or run** | `docs/TRAINING.md` and `README.md` |
| Changed **metrics, thresholds, splits or the sealed-set rule** | `docs/eval_protocol.md` — and say in `PROGRESS.md` that the contract changed and why |
| Changed or added **data** | `docs/data_audit.md` and `PROGRESS.md` §3 |
| Finished, started or reshaped a **ticket** | `docs/tickets/BACKLOG.md` — status in place |
| Found a bug | `docs/PROGRESS.md` §7 — the bug, its effect, and how it was caught |
| Added an **open question or a known gap** | `docs/PROGRESS.md` §8 |

**When a finding is overturned, record the correction — do not quietly overwrite it.** Add a row to
`PROGRESS.md` §6 stating the earlier claim, what replaced it, and what exposed it. A correction
deleted is a mistake someone repeats.

**What a result entry must contain.** The number alone is not a result. Include: the metric and
split, `n` (seeds or samples), the spread, the committed run directory, and anything that limits
the claim. If you cannot name the run a number came from, it does not go in the docs as a result.

**Before saying a task is done**, check the table above against what you changed and confirm each
row was handled. If a row applied and you skipped it, the task is not done — say so rather than
reporting completion.

**Keep claims calibrated to the evidence.** Do not write "confirmed", "converged" or "robust" into
a doc unless the evidence in the same entry supports the word. One frame, one seed or one dataset
supports "observed", not "found".

## Before calling any script done

Ask: *would this run unchanged on a fresh Linux box with a different GPU and no dataset yet?*
If it needs a path edited, a drive letter, a hand-set device, or a shell only this machine has,
it is not done.

## When working here

- Check `docs/tickets/` before starting; work the ticket, update its status.
- Don't download a multi-GB dataset without confirming free disk first and asking.
- Don't claim a training run succeeded without showing the metric output.
- Update the docs in the same change as the work — see "Documentation is part of the work".
- Check `docs/PROGRESS.md` before starting, so you do not redo a finished experiment or repeat a
  claim already recorded as wrong in its corrections table.
- Don't write TensorRT/JetPack-version-specific code before the board's JetPack version is
  confirmed and recorded in `docs/hardware.md`.
