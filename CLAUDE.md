# CLAUDE.md — Fish Detection on Jetson Orin Nano

## What this project is

Train a fish **object detector** (bounding boxes, underwater/BRUVS footage) that runs on a
**Jetson Orin Nano (4GB or 8GB)**, and write an academic report whose performance claims are
validated on the real board — not extrapolated from desktop numbers.

Two deliverables, both required:
1. **Report** — reproducible experiments, honest baselines, cross-dataset evaluation.
2. **Device** — a working Orin Nano inference pipeline with measured, reproducible FPS.

Deployment mode: **offline batch on recorded video first**, real-time camera as a stretch goal.

## Hard constraints (do not design around these being false)

### Target hardware
- **Jetson Orin Nano**, Ampere, JetPack 5.1.x or 6.x. Confirm the exact JetPack/L4T version
  before writing any export code — it pins TensorRT, which pins the usable ONNX opset.
- **4GB vs 8GB matters.** *All* Jetsons use unified memory — the 8GB module is not "8GB of
  dedicated VRAM." The difference is quantity, not architecture, and JetPack plus a desktop
  session consumes ~1.5–2 GB before your process starts. Assume 4GB until confirmed.
- **No DLA and no NVENC.** Orin Nano is the Orin module without the hardware video encoder, and
  without a DLA (Orin NX and AGX Orin have both). Decode is accelerated (NVDEC); *encoding*
  annotated video is CPU x264 and will dominate any pipeline that does it. Everything infers on
  the one 1024-core Ampere GPU — nothing to offload to.
- **Every FPS number MUST cite its `nvpmodel` mode** and `jetson_clocks` state. Do not assert a
  magnitude for the mode-to-mode swing before measuring it: the 4GB module's modes are 7W/10W
  (a 1.43× envelope), and the large swings quoted online are 8GB-under-JetPack-6.2-Super
  figures. Measure it on the board (FD-044); do not inherit it from a memo.
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

**Current workstation state — fix before training (see FD-002):** installed torch is a
**CPU-only build** (`2.12.1+cpu`, `cuda.is_available() == False`), so training would silently
run on CPU; and system Python is **3.14**, which Ultralytics and CUDA torch wheels do not
support. Use a dedicated Python 3.10/3.11 environment.

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
| DeepFish | Main split is points; **box derivatives exist** | No | FishLoc (3,200) is points only. Seg split (620) yields boxes from masks — but check `deepfish_darknet.zip` (gdown ID already in `fish-research/README.md`) first, and the ~4.5k-image detection set cited in the literature |
| OzFish | **Yes, ~45k boxes, already single-class** | No | Acquisition unresolved — see below |
| CFC | Yes, sonar | Labels yes (~206 MB), images no | **Sonar, not optical.** Different modality — do not mix into an optical training set |
| YOLO-Fish | Weights only | 10 darknet `.weights` (~2.37 GB) | Baseline comparison point; darknet, not PyTorch |

**The single most important data fact:** OzFish is the only real box source, **and it is not yet
acquired.** Everything in E2–E5 blocks on it. The previously-recorded acquisition routes do not
hold up: the HuggingFace "mirror" is a 3-file *model* repo (a 74.7 MB `.pt`), not a dataset; and
"Pawsey URLs are 404" is unsupported — every FDFML path returns HTTP 200, but so does a
nonsense path, serving a 1.7 KB HTML shell. It is a JS portal with a catch-all route, so **curl
can prove neither presence nor absence.** Someone must open it in a real browser. See FD-113.

**OzFish boxes are already single-class.** Per `fish-research/ozfish/README.md` line 57: the box
annotations are *"fish/no-fish only and have no species/genus/family labels."* The 507-species
figure belongs to the ~80k **crops**, a different artefact. There is no species collapse to
perform — and `report_sections.md` §2.2 currently states this incorrectly.

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

## Prior work in this repo (state as of 2026-08-27)

- `report_sections.md` — draft Related Work + Results. Describes an **LCFCN point-supervised
  counting** run (val MAE 0.3375 @ 10 epochs, Kaggle T4). That is a *counting* experiment on a
  different task from this project's detection goal. It is prior context and possibly a report
  section, **not** the baseline for the Jetson detector.
- `fish-research/` — 5 cloned repos + downloaded label archives and weights (~4 GB).
- `fish_download_links.md`, `fish-research/README.md` — provisioning notes and download URLs.
- **The project is not a git repository.** Nothing above is version-controlled.
- **Neither the board nor the dataset is in hand.** Both are on the critical path (FD-113,
  FD-118). Until they are, prioritise work that needs neither: E6 report tickets, the portability
  layer, and the desktop-TensorRT dry-run track (FD-119) — the RTX 4060 is also Ampere and runs
  TensorRT, so the whole export/quantise/benchmark chain can be debugged before the Orin arrives.

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

## Before calling any script done

Ask: *would this run unchanged on a fresh Linux box with a different GPU and no dataset yet?*
If it needs a path edited, a drive letter, a hand-set device, or a shell only this machine has,
it is not done.

## When working here

- Check `docs/tickets/` before starting; work the ticket, update its status.
- Don't download a multi-GB dataset without confirming free disk first and asking.
- Don't claim a training run succeeded without showing the metric output.
- Don't write TensorRT/JetPack-version-specific code before the board's JetPack version is
  confirmed and recorded in `docs/hardware.md`.
