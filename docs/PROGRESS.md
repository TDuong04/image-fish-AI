# Project progress and findings

The running record of what was built, what was measured, what was found, and what
turned out to be wrong. **This file is kept current by rule** — see "Documentation is
part of the work" in `CLAUDE.md`. If a result or a code change is not reflected here,
the work is not finished.

Last updated: 2026-10-04.

---

## 1. Where the project stands

| Area | State |
|---|---|
| Environment | Working. Python 3.11 venv, torch 2.6.0+cu124, ultralytics 8.4.130, RTX 4060 Laptop 8 GB. |
| Data | **In hand.** DeepFish (train/val), Kakadu (extra train only, F13) and an OzFish test set (sealed). The OzFish training set is reachable at a direct link (about 2.9 GB, 1,800 frames, 45k boxes) and not yet downloaded. |
| Training | **Planned grid done** (6 runs: 3 seeds x {640, 960} px), then an improvement programme (F12 to F18): the strongest result is adding Kakadu training data, **mean mAP@50 0.698 against 0.473 for a same-machine control, 3 seeds each, on the DeepFish val split (F18)**. Not yet confirmed on the sealed OzFish set. Larger models and stronger augmentation help on one seed each (F14, F15), and one seed of the combined recipe (Kakadu + OzFish data + augmentation) reaches 0.755 mAP@50 on the same split (F22, a screen); the split test (F23) attributes that gain to the OzFish data, not the augmentation (0.764 with default augmentation, 0.698 for augmentation alone, one seed each); on three seeds the OzFish-data recipe reaches 0.750 mAP@50 (F24), and YOLOv8s on the same data 0.821 (one seed, Orin latency unmeasured). |
| Evaluation | In-domain, cross-dataset (sealed OzFish), size-stratified recall, and a colour-perturbation study. |
| Weights | Published as GitHub release `v0.1.0-models`, fetched and checksum-verified by `scripts/fetch_models.py`. |
| Demos | Three shareable pages and a runnable Gradio app. |
| Automation | Crash-recovering supervisor and a Kaggle session runner. |
| **Jetson Orin Nano** | **In hand and probed:** 8 GB Super dev kit, L4T 36.4.7, TensorRT 10.3.0, active fan, SD-card boot with ~8 GB free (`docs/hardware.md`). **Two FP16 engines built on the board at the shapes the accuracy was evaluated at; both match PyTorch on real frames, and GPU kernel time is measured (2.56 ms at 384x640, 4.37 ms at 544x960, a 20 s burst).** **End to end, a serial pipeline takes about 33 ms per frame at 640 and 41 ms at 960 (about 30 and 24 fps), and a web app runs on the board** (`edge/app.py`). **A 30-minute sustained run held about 34 fps at 960 with no throttling (F9, one run, 25 W mode, ambient 26 C assumed).** **YOLOv8s on the combined data takes about 50 ms per frame (19.9 fps serial) against 41 ms for YOLOv8n (F28, 25 W, 2026-10-07).** **A 30-minute run of YOLOv8s held 24.4 fps with no throttling, plateau about 58 C (F29, one run, 25 W, ambient not measured).** **MAXN_SUPER gave no speed-up of the network; the video frame rate was unchanged (F31, 10-minute runs).** Not yet measured: hardware video decode, INT8, on-device mAP, the 15 W and 7 W modes. The deployment half of the thesis is only partly answered. |
| Report | Draft started 2026-10-06 in `docs/report/` (`paper.md`, `references.bib`, `references.md`, `README.md`); sections that need F27, YOLOv8s on the Orin and the sealed test are marked pending. `docs/RESEARCH.md` (2026-10-06) is the narrative of the research and methodology and the suggested next steps; `report_sections.md` still describes the older LCFCN *counting* experiment, not this work, and the final report prose is not yet written. |

---

## 2. What was built

### Data
- `scripts/get_data.py` — one command: download, extract, prepare. Idempotent.
- `scripts/prepare_dataset.py` — builds Ultralytics datasets with a **site-level split** and a hard
  assertion that no site appears in both train and val.
- `src/fish/datasets.py` — frame grouping (filename → source clip → capture site) and label parsing
  that rejects malformed boxes instead of dropping them.
- `scripts/audit_dataset.py` — box-size and density statistics at candidate input resolutions.

### Training and automation
- `scripts/train.py` — single entrypoint; `--resume`; `$FISH_BATCH` override; records config, env and
  git SHA per run.
- `scripts/supervise.py` — retries only failures a retry can fix (OOM halves the batch), stops on the
  rest, writes structured failure JSON.
- `scripts/kaggle_runner.py`, `kaggle/` — drives multi-session training on Kaggle from a plan file.
- `scripts/bootstrap.py`, `check_env.py` — right torch wheel per machine; diagnose a machine that cannot train.
  Picks `cu128` for drivers reporting CUDA 12.8 or newer: RTX 50-series (Blackwell) GPUs have no kernels in older
  wheels. Also reads the `CUDA UMD Version:` header label newer drivers print (`tests/test_bootstrap.py`).
- `scripts/error_analysis.py` — Step 0 of the improvement plan: splits every ground-truth box and prediction on the
  DeepFish **validation** split (refuses `ozfish_eval`) into outcomes at IoU 0.5 — missed fish (no prediction /
  found but under the threshold / badly placed), false alarms (duplicate / badly placed / background), recall by box
  size, and a precision-recall-F1 sweep over thresholds — and draws the worst frames. Writes
  `runs/analysis-<date>-<model>/analysis.json`.
- `scripts/label_audit.py` — FD-105. Draws a seeded random sample of missed fish (M##), found fish (F##, the control)
  and background false alarms (B##) as zoomed contact sheets for a person to review; verdicts are recorded in
  `verdicts.json` and `--summarise` turns them into counts. Validation split only; refuses `ozfish_eval`.
- `configs/train/yolov8n_960_aug.yaml`, `yolov8n_960_p2.yaml` — FD-129 small-fish screens at 960: stronger scale and hue
  augmentation (bundled, so a win does not say which half helped), and a stride-4 (P2) head initialised from COCO
  weights wherever the layer shapes match. **Running, no result yet** (see section 8).
- `configs/train/yolov8s_960.yaml`, `yolov8m_960.yaml`, `yolo11s_960.yaml` — FD-134 larger-model screens: default
  augmentation, 960 px, patience 30, everything else as `yolov8n_960`. Queued on `ai-gpu` behind the FD-129 runs, seed 0
  only; auto batch is 6, 3 and 5. **No result yet.**
- `scripts/fetch_url.py` — resumable multi-connection download with Range requests and MD5 verification, stdlib
  only. Added because Zenodo gave one connection about 82 KB/s (the 5.9 GB Kakadu zip would take about 20 hours);
  it refuses without free disk and refuses a server that ignores Range. Progress is kept in `<file>.parts`.
- `src/fish/zip64.py` — reads zip members compressed with Deflate64 (method 9), which Python's `zipfile` cannot;
  every member is checked against the archive's CRC-32 and size. Needs `inflate64`.
- `scripts/prepare_kakadu.py` — FD-135. Converts the Kakadu archive to a single-class YOLO dataset: merges 23 species
  into `fish`, groups frames into scenes by picture similarity (the data has no video or site field), keeps at most 15
  frames per group, and writes **train only** by default. Saves the grouping to `splits/kakadu_groups.json`. See F13.
- `configs/data/kakadu.yaml`, `configs/data/deepfish_kakadu.yaml`, `configs/train/yolov8n_960_kakadu.yaml` —
  DeepFish train + Kakadu train, validated on the DeepFish val split so numbers stay comparable. **Not run yet.**
- `scripts/queue_runner.py`, `configs/overnight_plan.yaml` — runs a YAML plan of steps unattended: waits for files or
  for the GPU to go quiet, skips finished work, records state in `data/.cache/queue/status.json`, keeps going after a
  failed step, and re-reads the plan between steps so it can be fixed while running. How a check works:
  `docs/OVERNIGHT.md`.
- `scripts/upload_parts.py`, `scripts/assemble_parts.py` — resumable upload of a folder of parts over ssh with a
  checksum check, and the matching reassembly. Built for the 60 KB/s link; the full archive then reached the server
  another way, so they are unused for now.
- `scripts/eval_positive_only.py` — re-scores checkpoints on the validation images that contain at least one fish (875 of 1,339), to test how much of the gap to published DeepFish numbers the empty-frame convention explains. Refuses `ozfish_eval`; workers off, so safe on Windows. See F19.
- `scripts/audit_ozfish_overlap.py` — compares the sealed OzFish test names with the 1,758 labelled OzFish frames by video, frame and stereo deployment, and reports how many frames remain usable for training (794). Reads file names only. See `docs/data_audit.md`.
- `scripts/prepare_ozfish_train.py` — converts the OzFish labelled frames (SageMaker Ground Truth boxes) to a one-class YOLO set at `data/processed/ozfish_train`, keeps only frames of deployments absent from the sealed test, drops frames within thumbnail distance 4.0 of a test frame, and writes a leak report. `--report-only` writes nothing. Result: 779 images, 17,564 boxes (`docs/data_audit.md`).
- `scripts/prepare_random_split.py`, `configs/train/yolov8n_{640,960}_randomsplit.yaml`, `configs/plan_randomsplit.yaml` — a **deliberately leaky** random per-image split of the 4,505 fish-containing DeepFish images (15,463 boxes: the published set's size), used only as a diagnostic of how much of the gap to published numbers is the split. Nothing trained on it may be reported as a result.
- `scripts/eval_ozfish_val.py` — scores checkpoints on the held-out OzFish deployments (`ozfish_train/images/val`); refuses `ozfish_eval`. `configs/data/deepfish_kakadu_ozfish.yaml`, `configs/train/yolov8n_960_all.yaml`, `configs/plan_allin.yaml` — the combined-recipe screen (DeepFish + Kakadu + OzFish train, stronger augmentation, one seed, validated on DeepFish val); queued, no result yet.
- `configs/train/yolov8n_960_kakadu_ozfish.yaml`, `yolov8n_960_kakadu_aug.yaml`, `configs/plan_split.yaml` — the split test of the combined recipe (F22): Kakadu + OzFish data with default augmentation, and Kakadu data with the stronger augmentation, one seed each, so the two changes of F22 can be told apart. **Queued 2026-10-05, no result yet.**
- `scripts/archive_paths.py` — archives files we no longer use on a shared machine: moves (a rename on the same disk), never deletes or compresses, writes a `MANIFEST.json` that can undo every move, defaults to a dry run, and refuses the datasets, raw archives and interim sources the project depends on and paths outside `DATA_ROOT` without `--allow-outside`. Two runs in the same second get separate folders (a first version let the second overwrite the first's manifest; see section 7).
- `configs/train/yolov8s_960_kakadu_ozfish.yaml`, `configs/plan_confirm.yaml` — the 3-seed confirmation of the best recipe (YOLOv8n on DeepFish + Kakadu + OzFish train, default augmentation: seeds 1 and 2) and one seed of YOLOv8s on the same data. **Queued 2026-10-05, no result yet.**
- `scripts/eval_tiled.py` — tiled inference test: runs a checkpoint on overlapping crops plus the whole frame, merges with NMS, and scores every variant with one AP routine on the DeepFish val split (refuses the sealed OzFish set); output `tiled.json` (F25).
- `configs/train/yolov8s_1280_kakadu_ozfish.yaml`, `configs/data/deepfish_kakadu_ozfish_1280.yaml`, `configs/plan_hires.yaml` — the 1280 px screen (queued 2026-10-06; result in F27: no gain in small-fish recall): Kakadu and OzFish re-cut at 1280 px wide (same frames, groups and split: 7,318, 622 train and 157 held-out, checked) because the existing copies are 960 px wide and a 1280 run would otherwise be fed stretched images; YOLOv8s, one seed, automatic batch 3.
- `tests/test_report.py` — keeps the report draft traceable: every finding `paper.md` cites exists in this file (except those listed as pending) and every bibliography key has a row in `docs/report/references.md`.
- `scripts/export_onnx.py` (changed 2026-10-06) — the checkpoint-name pattern now accepts names such as `yolov8s-960-kakadu-ozfish-s0`; it only matched `yolov8n-960-s2` before. Used to export the two best models for the Nano: `models/onnx/20261005-yolov8n-960-kakadu-ozfish-s0-544x960.onnx` (parity PASS) and `...yolov8s-960-kakadu-ozfish-s0-544x960.onnx` (**strict raw-tensor gate FAILED by 0.0006 px**: 0.0106 against 0.01; detections identical to PyTorch on 12 frames, worst IoU 0.99998; gate not relaxed; not yet built or timed on the board).
- `scripts/post_run.py`, `configs/plan_v8s_seeds.yaml`, `tests/test_post_run.py` — the overnight queue of 2026-10-08: two more seeds of YOLOv8s on the combined data (seeds 1 and 2 of `yolov8s_960_kakadu_ozfish`, batch 5 as in F24), each followed by `post_run.py`, which finds the newest finished run folder and runs the error analysis (DeepFish val) and the held-out OzFish score on the server so no checkpoint has to be fetched first. **Queued, no result yet.**
- `configs/env/aigpu.yaml` — profile for the shared remote server (RTX 5060 8 GB, headless container). Selected
  with `FISH_ENV=aigpu`. Verified `check_env.py --require-gpu` READY there with torch 2.11.0+cu128,
  ultralytics 8.4.172 (the workstation has 8.4.130; a cross-machine comparison must note this).

### Evaluation and analysis
- `scripts/evaluate.py` — scores checkpoints on any dataset, each **at the resolution it was trained at**.
- `scripts/experiment_colour.py` — colour-perturbation study over the full validation split.
- `docs/eval_protocol.md` — metrics, thresholds, sealed set and seed rules, fixed before results existed.

### Distribution and demos
- `scripts/fetch_models.py`, `docs/model_manifest.json` — verified weight download.
- `demo/app.py` — Gradio app: photo (640 vs 960 side by side) and video (annotated clip + MaxN).
- `scripts/build_demo.py`, `scripts/build_device_demo.py` — data behind the two comparison pages.

### Device
- `edge/probe_board.py` — run on the Jetson; reports module, L4T, TensorRT, power modes (including
  whether Super is present), memory, storage, temperatures and the fan tachometer. Standard library
  only, Python 3.8-safe, needs no sudo, and can be streamed over SSH without installing anything.
  Validated on the real board on 2026-10-04; output committed as `docs/board_probe_20261004.json`.
- `docs/hardware.md` — what is confirmed, what is documented but unverified, and what is unknown.
- `scripts/export_onnx.py` — exports checkpoints to static-shape ONNX **at the shapes they were evaluated at**
  (384x640 and 544x960, not square), then proves parity against PyTorch on real frames and saves the
  preprocessed inputs and PyTorch reference outputs for the engine check. NMS is deliberately not in the graph.
- `edge/build_engines.py` — runs on the board; builds FP16 TensorRT engines one at a time with `trtexec`
  (identical flags for every model). Its commands were checked against the real build log.
- `edge/engine_dump.py` — runs on the board; feeds saved frames through an engine via `trtexec` and writes the raw
  outputs. Uses `trtexec` rather than a Python runner so the engine is the only thing under test.
- `scripts/engine_parity.py` — compares the engine's outputs with the PyTorch reference on identical inputs.
- `src/fish/parity.py` — the shared comparison (used by the ONNX and engine checks). Detections within 0.03 of
  the confidence threshold are reported as *borderline* and not held against an FP16 engine; a confident
  detection that goes missing or appears from nowhere fails the check.
- `edge/trt_runner.py` — runs on the board; the real on-device inference path (letterbox, host-to-device, GPU
  forward, device-to-host, threshold + NMS, map back to the frame) with a per-stage timing breakdown. Needs only
  what the board has (TensorRT, numpy, OpenCV) plus `cuda-python`; no torch, no Ultralytics. Its letterbox is
  bit-identical to Ultralytics' on 14 source/target size combinations and its decode selects the same detections
  as Ultralytics' NMS (`tests/test_edge_runner.py`). It reproduces `trtexec` outputs to within 5e-4, which is
  `trtexec`'s own text rounding.
- `edge/app.py` — the web app for the board: photo tab (640 vs 960, median of 7 warm runs, stage timings) and
  video tab (MaxN, per-frame CSV, the frame with the most fish). It deliberately writes **no annotated video**,
  because the Orin Nano has no hardware encoder. Shows the board's power mode, temperature and fan next to every
  timing. Listens on `0.0.0.0:7860` by default (no login; `--host 127.0.0.1` plus an SSH tunnel keeps it private);
  setup is in `docs/hardware.md`.
- `edge/sustained.py` — runs on the board; loops one clip through the same decode, letterbox, GPU and NMS path as
  `edge/trt_runner.py` for N minutes and writes one JSON line per 10 s window: fps, total / decode / copy+GPU
  latency (median and p95), mean detections, all thermal zones, fan RPM, GPU/CPU clocks and input power from
  sysfs (no sudo). Unreadable fields are `null`, never guessed. It changes no power mode and locks no clocks; the
  state is recorded in the first line. Ambient temperature cannot be read and is added by hand.

### Safety net
- 223 pytest checks: frame grouping, label parsing, no absolute paths, no hardcoded `cuda`,
  `__main__` guards, and `tests/test_docs.py`, which fails if a committed result run or a script
  is not named in this file — the mechanical half of the documentation rule in `CLAUDE.md`.

---

## 3. Data

| Dataset | Role | Images | Boxes | Empty frames | Notes |
|---|---|---|---|---|---|
| DeepFish | train + val | 6,517 | 15,463 | 2,012 (30.9%) | 20 sites, 68 clips, median 73 consecutive frames per clip |
| OzFish test | **sealed** cross-dataset test | 352 | 8,329 | 0 | 297 clips, mean 23.7 boxes/image, max 232 |
| OzFish labelled frames | acquired 2026-10-05; converted to `ozfish_train` (train only), **not yet used for training** | 1,758 (779 kept) | 43,572 (17,564 kept) | 5 | contains the 352 sealed test frames; only the 794 frames outside the test deployments may be trained on; details in `docs/data_audit.md` |
| Kakadu | extra **train only** (added 2026-10-04) | 44,112 (7,318 kept) | 82,904 (11,695 kept) | 0 | 23 species merged to one class; no video/site field; a held-out split inside it leaks (F13); details in `docs/data_audit.md` |

Split: **5,178 train / 1,339 val** (14 / 6 sites; val sites `7398, 7623, 9862, 9870, 9892, 9898`).

Both datasets are class-agnostic (`fish`). Source: the YOLO-Fish authors' Google Drive archives.

**Why the split is by site.** DeepFish clips run to 462 consecutive near-identical frames. A random
per-image split would put frames of one clip on both sides and inflate mAP by an unknown margin.

**Box size drives the project.** Measured on 1920x1080 frames, scaled to the model input:

| Input | DeepFish median box | Under 16 px | OzFish median box | Under 16 px |
|---|---|---|---|---|
| 640 | 29.1 px | 14.4% | 22.4 px | 34.8% |
| 960 | 43.7 px | 2.3% | 33.5 px | 20.4% |

~16 px is roughly the floor a stride-8 detection head can represent.

---

## 4. Training

YOLOv8n (3.0M parameters), COCO-pretrained, single class, on Kaggle T4. `patience=30`,
`epochs=150` ceiling, Ultralytics default augmentation (`hsv_h=0.015`, `mosaic=1.0`,
`close_mosaic=10`). Seeds 0, 1, 2. Every run early-stopped on its own — none was cut off
by the time budget.

| Run | Epochs | mAP@50 | mAP@50-95 |
|---|---|---|---|
| 640 · seed 0 | 88 | 0.3946 | 0.2312 |
| 640 · seed 1 | 74 | 0.4026 | 0.2458 |
| 640 · seed 2 | 98 | 0.4497 | 0.2616 |
| **640 mean ± sd** | | **0.4156 ± 0.0298** | **0.2462 ± 0.0152** |
| 960 · seed 0 | 119 | 0.4876 | 0.3124 |
| 960 · seed 1 | 123 | 0.4967 | 0.3186 |
| 960 · seed 2 | 122 | 0.5220 | 0.3201 |
| **960 mean ± sd** | | **0.5021 ± 0.0178** | **0.3170 ± 0.0041** |

Committed run directories (config, metrics and epoch curve for each live under `runs/<name>/`):

| Run | Directory |
|---|---|
| 640 · seed 0 | `20260922-yolov8n-640-s0` |
| 640 · seed 1 | `20260922-yolov8n-640-s1` |
| 640 · seed 2 | `20260922-yolov8n-640-s2` |
| 960 · seed 0 | `20260922-yolov8n-960-s0` |
| 960 · seed 1 | `20260922-yolov8n-960-s1` |
| 960 · seed 2 | `20260923-yolov8n-960-s2` |

Local re-evaluation reproduces these to within 0.0006 (640: 0.4155, 960: 0.5027), which
validates the evaluation harness.

Checkpoints are selected by Ultralytics' fitness (`0.1·mAP@50 + 0.9·mAP@50-95`), not mAP@50.

---

## 5. Findings

### F1. Resolution helps in-domain — 960 beats 640 by +0.087 mAP@50 (21%)
Every 960 run beats every 640 run (worst 960 = 0.4876, best 640 = 0.4497). mAP@50-95 gain is
+0.071 (29% relative), larger than the mAP@50 gain — consistent with small objects losing
localisation first.

### F2. Most of that advantage does not survive a change of scene
Evaluated on the sealed OzFish set, each model at its training resolution:

| | In-domain mAP@50 | Cross-dataset mAP@50 | Lost |
|---|---|---|---|
| 640 | 0.4155 ± 0.030 | 0.2038 ± 0.008 | 51% |
| 960 | 0.5027 ± 0.018 | 0.2165 ± 0.004 | 57% |
| 960 advantage | +0.087 | **+0.013** | 85% of it gone |

The ordering holds cross-dataset (no group overlap) but the gain falls from ~21% to ~6%. Both
configurations lose over half their performance. Training is sparse reef footage (2.4 boxes/image);
test is crowded baited-camera footage (23.7 boxes/image).

### F3. The gain is concentrated in medium-sized fish
Recall over all 1,339 validation frames (conf 0.25, IoU 0.5), bucketed by box size *as seen at 640 input*:

| Box size | 640 recall | 960 recall |
|---|---|---|
| < 16 px (n=10) | 0.0% | 0.0% |
| 16–32 px (n=273) | 0.4% | 1.5% |
| **32–96 px (n=746)** | **36.7%** | **44.5%** |
| > 96 px (n=356) | 90.2% | 92.1% |

Below ~32 px both models essentially fail; above 96 px both essentially succeed. Extra resolution
helps where fish are findable but hard.

### F4. The detector tolerates missing colour but not wrong colour
Full validation split, 3 seeds each, mAP@50:

| Perturbation | 640 | vs original | 960 | vs original |
|---|---|---|---|---|
| original | 0.4155 | — | 0.5027 | — |
| desaturate 50% | 0.4050 | −2.5% | 0.4974 | −1.1% |
| greyscale | 0.3868 | −6.9% | 0.4569 | −9.1% |
| hue +90° | 0.2701 | −35.0% | 0.3612 | −28.2% |
| hue +180° | 0.2106 | −49.3% | 0.2548 | −49.3% |

Absent colour costs under 10%; contradictory colour costs up to 49%. Underwater hue shifts
systematically toward blue-green with depth, which is exactly this kind of perturbation. Training
used `hsv_h=0.015` (almost no hue augmentation). **Hypothesis, not yet tested:** stronger hue
augmentation is free at inference and may recover robustness more cheaply than resolution did.
It also does not yet establish that colour dependence *caused* the OzFish drop.

### F5. Desktop timing says almost nothing about the Orin
On the RTX 4060, 640 and 960 run in near-identical end-to-end time (~13 ms), because decode,
letterboxing and NMS dominate rather than the network. On one measured clip, 57% of each frame was
non-network overhead (GPU forward 10.3 ms of 23.9 ms end to end). On a board with a fraction of the
GPU those proportions would shift, so neither figure scales to a device number.

### F6. On the Orin Nano the GPU cost of 960 over 640 is small, because both run at the evaluated shapes
Engines built on the board (FP16, seed 2, L4T 36.4.7, TensorRT 10.3.0), 25 W mode, `jetson_clocks` not applied,
active fan, `trtexec --duration=20`. Source: `runs/device-20261004/`.

| Engine | Input (HxW) | GPU compute median | p95 | Throughput |
|---|---|---|---|---|
| 640 | 384x640 | **2.560 ms** | 2.567 ms | 390.3 qps |
| 960 | 544x960 | **4.371 ms** | 4.379 ms | 228.6 qps |
| 960, square (earlier report; 3 s run) | 960x960 | 7.098 ms | 8.238 ms | 135.6 qps |

960 costs **1.71x** the 640 engine on the GPU, although it carries 2.12x the pixels: part of each frame's cost
is fixed. Extrapolating by pixel count before measuring predicted 4.0 ms (measured 4.37) and 1.9 ms
(measured 2.56), so it under-predicted the smaller model by about a third.

**What this measures and what it does not.** GPU kernel time on a static input. There is no decode, letterbox,
NMS or postprocessing, so it is **not an end-to-end frame rate**. It is a 20 s burst: temperatures rose from
roughly 49 to 54 C during the 640 run and 54 to 57 C during the 960 run while the fan ramped from 1,627 to
2,280 RPM, and were still climbing, so sustained behaviour is unknown. (The temperatures are approximate: the
logging helper mishandled the board's decimal-comma locale and they are read from its error output.) One engine
per resolution, both from seed 2, the best seed of each group; latency does not depend on the seed, but any
on-device accuracy comparison must account for it.

**Why the shapes matter.** The first engine reported from the board was built for a 960x960 square. Ultralytics
evaluates a 1080p frame at 544x960 (and 640 at 384x640), so that engine ran 1.76x the pixels of the geometry
the accuracy numbers describe. Latency and accuracy must describe the same workload.

### F7. Both FP16 engines match PyTorch on real frames
12 frames each (8 DeepFish including empty-water frames, 4 OzFish including crowded ones), same preprocessed
input through PyTorch FP32 and the engine.

| Engine | Confident misses | Confident extras | Borderline | Worst matched IoU | Worst confidence drift |
|---|---|---|---|---|---|
| 640 · 384x640 | 0 | 0 | 0 | 0.993 | 0.006 |
| 960 · 544x960 | 0 | 0 | 0 | 0.949 | 0.017 |

The ONNX stage also passes (raw output within about 3e-3 pixels of PyTorch, identical detections). **Limits:**
12 frames, one seed per resolution, FP16 only. This is agreement with PyTorch on a small sample, **not** an mAP
measurement on the device. Raw tensor differences reach 3.1 px (640) and 15.2 px (960) on some frames, mostly in
frames with no or few detections; detection-level agreement is what is judged, and the raw figure is recorded
so it is not hidden.

### F8. End to end on the Orin Nano, the GPU kernel is under 11% of a frame
`edge/trt_runner.py --bench 200` on the board: the same two FP16 engines, 25 W mode, clocks not locked, fan on.
Median of 200 runs per case; logs in `runs/device-20261004/e2e_*.log`. Frame `7623_F2_f000101` (DeepFish, 2 fish).

| Stage | 640 · 384x640 | 960 · 544x960 |
|---|---|---|
| JPEG decode (CPU) | 17.3 ms | 17.5 ms |
| Letterbox + normalise (CPU) | 4.7 ms | 8.0 ms |
| Copy in + GPU forward + copy out | 9.8 ms | 14.4 ms |
| Threshold + NMS (CPU) | 0.85 ms | 0.91 ms |
| **Total, serial** | **32.8 ms** (p95 33.4) | **40.8 ms** (p95 44.2) |
| Frames per second, serial | 30.5 | 24.5 |
| *Kernel-only time from F6, for scale* | *2.56 ms* | *4.37 ms* |

A crowded OzFish frame (7 and 10 detections) totals 32.4 ms and 40.9 ms, within 0.5 ms of the above.

- **The GPU kernel is 8% (640) and 11% (960) of the frame.** The Arm cores and memory traffic dominate, which
  `trtexec` cannot show. A kernel-only number is not a frame rate (F6 said so; this puts a number on it).
- **960 costs +8.1 ms (+25%) per frame over 640 end to end**, not the 2.25x the pixel count suggests. About three
  quarters of that extra is CPU resize plus data movement (letterbox +3.3 ms, copy+GPU +4.6 ms of which only
  1.8 ms is the kernel), not convolution. NMS differs by 0.06 ms.
- **CPU NMS is cheap here (about 0.9 ms)**, which is not the bottleneck FD-034 worried about. That is on sparse
  output: the most detections tested were 10.
- **Copy in + GPU + copy out is 9.8 ms against a 2.56 ms kernel.** About 7 ms goes to host/device copies and
  synchronisation with ordinary (pageable) host memory. Pinned memory or normalising on the GPU would likely
  shrink it, but that is a hypothesis and has not been tried.

**Limits.** Serial: stages run one after another with no overlap, so this is a simple pipeline and not the best
the hardware can do. JPEG decode is not video decode; the app's video tab uses software H.264 decode (it
reported about 8.6 ms per frame on the sample clip) and the board's hardware decoder (NVDEC) is **not used**, so
real video throughput could be better. One image per case, warm in the page cache. Junction temperature stayed at
49 to 50 C because the GPU is idle most of each frame, so this is **not** a sustained-load test (see F9). Both engines are
from seed 2.

### F9. A 30-minute run at 960 held its frame rate; temperature plateaued at about 56 C
`edge/sustained.py` on the board: 960 engine (`20260923-yolov8n-960-s2-544x960-fp16.engine`) looping
`sample_clip.mp4`, serial pipeline with software H.264 decode, **nvpmodel 25W** (not MAXN_SUPER), clocks not
locked, fan on, governor `schedutil`. Started 07:20:24Z, ended 07:50:24Z, 179 ten-second windows. **Ambient 26 C,
assumed by the user, not measured.** The web app was running but idle. Log: `runs/device-20261004/sustained_960.jsonl`.

| | First 2 min | Last 2 min |
|---|---|---|
| Frames per second (mean of windows) | 34.16 | 33.99 |
| Total p50 / p95 per frame | 28.6 / 30.2 ms | 28.8 / 29.9 ms |
| Junction temperature (mean) | 51.7 C | 56.5 C |

- **Throughput did not decay.** Per-window fps: min 33.03, median 34.14, max 35.08; first and last 2 minutes
  differ by 0.5%, inside the window-to-window spread. No throttling is visible at this load and ambient.
- **Temperature climbed then flattened.** Junction 49.5 C at start, 56.9 C maximum, 56.6 C at the end; slope over
  the last 15 min is +0.03 C/min. The fan rose from 1566 to 2089 RPM to hold it.
- **Input power 6.5 to 7.1 W** (INA3221 channel 1; the channel mapping is assumed, not checked against a meter).
  CPU core 0 ran 730 to 1344 MHz under `schedutil`.
- **The GPU clock reading is not usable.** `gpu_mhz` read 306 in all 179 windows against a 918 max, while the GPU
  was demonstrably working. The sysfs node was either not the live clock or was sampled while idle; it is
  recorded as unexplained and **no claim about GPU clocks is made**. Throughput flat at constant temperature is the
  evidence offered, not clocks.
- **Not comparable stage-for-stage with F8.** 34 fps here against 24.5 fps at 960 in F8 is mostly decode: 8.4 ms
  H.264 video against 17.5 ms JPEG in F8, from different source frames. The GPU stage (copy + forward) is 14.2 ms
  here against 14.4 ms in F8.

**Limits.** One run, one clip, one engine (seed 2), 960 only, 25 W mode only, serial pipeline. Load is a light
duty cycle (the GPU idles for part of each frame), so a heavier pipeline, MAXN_SUPER or a warmer room may behave
differently. Mean detections per frame is about 1.5, so NMS load is trivial. Not repeated, so run-to-run spread is
unknown.

### F10. Missing fish, not wrong boxes or the threshold, is what limits the detector (descriptive, one seed per size)
`scripts/error_analysis.py` on all 1,339 DeepFish validation frames (1,385 labelled fish), checkpoints
`20260923-yolov8n-960-s2` and `20260922-yolov8n-640-s2`, operating threshold 0.25, IoU 0.5. Runs:
`runs/analysis-20261004-20260923-yolov8n-960-s2`, `runs/analysis-20261004-20260922-yolov8n-640-s2`.

| At conf 0.25 | 960 | 640 |
|---|---|---|
| Fish found / missed | 697 / 688 (recall 0.503) | 638 / 747 (recall 0.461) |
| Missed: no prediction at all | 523 | 543 |
| Missed: found, but under 0.25 | 161 | 195 |
| Missed: found, but badly placed | 4 | 9 |
| False alarms (total) | 365 | 443 |
| False alarms on background / duplicate / badly placed | 321 / 12 / 32 | 395 / 18 / 30 |
| False alarms on the 464 empty frames | 89 (0.19 per frame) | 158 (0.34 per frame) |
| Recall, boxes 16-32 px (n=273) | 3.3% | 1.1% |
| Recall, boxes 32-96 px (n=746) | 48.9% | 42.5% |
| Recall, boxes over 96 px (n=356) | 90.7% | 89.3% |

- **About three quarters of the misses (523 of 688 at 960) have no prediction at all**, so lowering the threshold
  recovers at most the 161 found-but-under-threshold ones. Even at conf 0.05 recall at 960 is only 0.542.
- **Placement is not the problem.** 4 of 688 misses at 960 are found but badly placed.
- **Nearly all the misses are small fish**: of 688 missed at 960, 264 are 16-32 px and 381 are 32-96 px, against
  33 over 96 px. This is close to F3's 3-seed means (for example 48.9% against 44.5% for 32-96 px at 960); one seed, so a few points of difference are expected. It is also the check that the script is correct.
- **Raising the threshold helps precision and costs little recall.** Best F1 at 960 is 0.60 at conf 0.5-0.6
  (precision 0.83, recall 0.47) against 0.57 at 0.25. A threshold change is a deployment choice, not a fix for
  missed fish. It was selected on this validation split; the sealed set was not touched.
- **Most false alarms are on background** (321 of 365 at 960). 960 has 44% fewer empty-frame false alarms than 640 (89 vs 158).
- **What the worst frames show (viewed: 2 of 24 drawn, so this is an observation and not a finding).** All 12 worst
  frames at 960 come from one clip (`7398_F1`, mangrove roots). Labelled fish there are small and camouflaged
  against the roots, and at least one labelled box shows almost no visible fish. Some "background" false alarms may
  be unlabelled fish. Label quality was audited afterwards; see F11 (most misses are visible fish).

**Limits.** One seed per resolution (seed 2), one validation split of near-duplicate video frames: the counts are
descriptive, not independent samples. Outcome categories depend on the thresholds in the script (IoU 0.5 match,
0.1 for "badly placed"). 'No prediction' means no prediction with IoU of at least 0.5 under the threshold and none of at
least 0.1 above it, with predictions kept down to conf 0.001.

### F11. The labels are mostly sound: the model misses fish that are visible (one reviewer, a sample)
`scripts/label_audit.py` on the 960 seed-2 checkpoint, validation split, conf 0.25, sample seed 0:
48 of the 688 missed fish, 24 of the 697 found fish as a control, and 36 of the 321 background false alarms.
Reviewed by one person (the assistant) on zoomed still crops; **a second reviewer has not checked these**.
Run: `runs/audit-labels-20260923-yolov8n-960-s2` (`manifest.json`, `verdicts.json`, `summary.json`, sheets).

| Sample | n | Verdicts |
|---|---|---|
| Missed fish | 48 | **32 plainly visible (67%)**, 12 ambiguous (25%), 4 nothing visible (8%) |
| Found fish (control) | 24 | 24 plainly visible |
| Background false alarms | 36 | 16 mangrove leaf, 12 empty water, 4 top-left-corner box, 2 rock or pole, **2 possible unlabelled fish** |

- **Two thirds of sampled misses are real, visible fish**, so bad labels do not explain most of the failure; the
  model is missing fish a person sees. With n=48 the 67% has a wide interval (roughly 52% to 79%).
- **Only 8% of misses show nothing**, and 25% are faint or translucent, so label noise is a real but minor ceiling.
- **Few false alarms are unlabelled fish** (2 of 36). The rest are look-alikes: 44% are mangrove leaves, which the
  model takes for fish, and a third are empty water. This is a **hard-negative** problem (add and weight leaf and
  empty-water frames), not a label problem.
- **Site 7398 dominates.** 22 of the 48 sampled misses come from clip `7398_F1`. On the whole validation split,
  43 confident boxes sit anchored at the top-left corner (about 376-445 x 112-132 px), all but 2 in site 7398 (388
  frames), 9 of them on empty frames. That is about 12% of the 365 false alarms at 960, and it is site-specific;
  the same preprocessing finds fish correctly elsewhere, so it is not a pipeline bug. Cause unknown.

**Limits.** One reviewer, one checkpoint, one split, 48/24/36 sampled boxes. Crops are zoomed 2-4x, which makes small
fish look easier than they are at the model's input, so "plainly visible in the crop" does not mean "findable at 960".
The sample is skewed to site 7398. Counts are descriptive and the proportions carry wide intervals.

### F12. A stride-4 (P2) head did not raise small-fish recall and scored lower on mAP@50-95 (one seed, a screen)
FD-129 screen B: `yolov8n-960-p2`, seed 0, on `ai-gpu` (RTX 5060 8 GB, Ultralytics 8.4.172, **batch 5**), COCO weights
loaded into every layer whose shape still matched. Early-stopped at epoch 80 (best epoch 50) after 1.42 h. Run:
`runs/20261004-yolov8n-960-p2-s0`. Compared with the Kaggle baseline `20260922-yolov8n-960-s0` (T4, Ultralytics 8.4.159,
batch 23), validation split.

| | P2 (seed 0) | Baseline 960 (seed 0) | Baseline 960 range, 3 seeds |
|---|---|---|---|
| mAP@50 | 0.5075 | 0.4876 | 0.4876 to 0.5220 |
| mAP@50-95 | 0.2830 | 0.3124 | 0.3124 to 0.3201 |
| Recall, 16-32 px boxes (n=273) at conf 0.25 | 3.7% (10 found) | 1.5% (4 found) | |
| Recall, 32-96 px boxes (n=746) | 45.3% | 44.5% | |
| Fish found / missed (1,385) | 673 / 712 | 664 / 721 | |
| False alarms, all / empty frames (464) | 364 / 115 | 659 / 207 | |

Error analysis: `runs/analysis-20261004-20261004-yolov8n-960-p2-s0`, `runs/analysis-20261004-20260922-yolov8n-960-s0`.

- **Not a win.** P2 moved mAP@50 inside the baseline's own seed range and mAP@50-95 **below all three baseline seeds**
  (0.283 against 0.312 to 0.320). The small-fish hypothesis behind P2 is not supported: boxes of 16-32 px went from 4
  found to 10 of 273, and 32-96 px recall is unchanged, so the failure on small fish is not fixed by a finer output grid.
- **The false-alarm drop is not attributable to P2.** The baseline's own seeds differ as much: seed 0 has 659 false
  alarms and seed 2 has 365 (F10). One seed each cannot separate the head from seed-to-seed noise.
- **Confounded.** Machine, Ultralytics version and batch size (5 against 23) all differ from the baseline, and the
  P2 layers start from scratch. The same-machine control (`yolov8n-960` seed 0, queued) is what separates "P2" from
  "this setup"; until it finishes the comparison above is indicative only.

**Limits.** One seed, one split of near-duplicate frames. The baseline's seed spread at 960 is 0.034 mAP@50 and 0.008
mAP@50-95, so only the mAP@50-95 gap is clearly outside it. Not tested on the sealed OzFish set and not timed on the
Orin (the P2 head adds compute at 960).

### F13. Kakadu cannot be split into train and val without leaking, so it is train-only data
Archive: Jansen et al. 2022, doi 10.5281/zenodo.7250921, CC BY 4.0, md5 verified (`c282e3f0...`). Contents: 44,112
frames (43,715 JPEG, 397 PNG, 1920x1080 except 12), 82,904 boxes, 23 species, COCO format, compressed with
Deflate64. Examined on the workstation copy; figures below are from the archive itself, not a summary.

- **No grouping information.** The COCO images carry only `id`, `width`, `height`, `file_name` (a bare number) and
  `license`. **Consecutive ids are not neighbouring frames**: thumbnail distance between consecutive ids is 26.6,
  the same as between random pairs (26.7). Yet near-duplicates are everywhere: among 6,000 frames the median
  nearest-neighbour distance is 1.5, and viewed pairs are the same static camera scene.
- **Similarity grouping works only partly.** Single-linkage clustering of mean-removed 48x27 grayscale thumbnails
  (L1) gives, over all 44,112 frames: threshold 4.0 -> 2,296 groups, largest 4.4%; 5.0 -> 1,454 groups, largest 9.8%;
  6.0 -> 988 groups, **largest 33% (unrelated scenes merging)**. 4.0 was chosen as the last safe value.
- **A split on those groups still leaks.** At 4.0 the closest val/train pairs (distance 4.02) were looked at and are
  the **same scene** (same plants and branches, small camera or lighting change). Val-to-train nearest distances
  (median 6.32, 1st percentile 4.26) match the distances between different train groups (6.42, 4.13), so the groups
  are fragments of larger scenes, and 14.9% of val frames have a train frame within 5.0. Looser thresholds merge
  unrelated scenes first, so a better threshold does not exist for this feature. A distance check is necessary but
  not sufficient; the check in `prepare_kakadu.py` passes while the split still leaks.
- **Consequence.** Kakadu is used as **training data only**, capped at 15 frames per group (the pipeline
  default), and no model is judged on a Kakadu split. Judges stay the DeepFish val split and the sealed OzFish set.
  Better scene identity (a learned image embedding, or the authors' deployment records) could allow a Kakadu test
  set later.
- **No empty frames**: every image has at least one fish, so Kakadu cannot measure false alarms.
- Box sizes (native 1080p): median 150 px, 10th percentile 78 px; at 640 input that is about 50 px and 26 px, mostly
  the 32-96 px range where F3 and F10 show the models fail.

- **Leak check against the DeepFish validation split (2026-10-04, while the first Kakadu run was training).** The run's early scores (best 0.721 mAP@50 at epoch 29, against 0.456 for the control) were far above any earlier run, so the split was checked before anything was reported. On the exact 7,318 frames trained on, using the same thumbnail distance: nearest Kakadu frame to any of the 1,339 val frames is 5.65, none under 4.0, and the two datasets share no file names. Viewed closest pairs are unrelated scenes. **No leak was found.** Kakadu frames are, however, closer to the val frames than DeepFish's own train frames are (median 13.6 against 15.7; minimum 5.65 against 9.83), because the DeepFish split is by site. A large gain on this val split is therefore partly 'training data that looks like the validation sites' and is not evidence of a general improvement; the sealed OzFish set, used once at the end, is the test of that.

**Limits.** Grouping validated by eye on 3 pairs and by distance statistics, not exhaustively. 23 species merged to
one class without checking whether any species labels cover non-fish objects.

### F14. Against a same-machine control, stronger augmentation and a larger model both help in-domain (one seed each)
All seed 0, 960 px, DeepFish validation split, conf 0.25 for the counts (1,385 labelled fish, 464 empty frames).
Server runs on `ai-gpu` (RTX 5060, Ultralytics 8.4.172); the Kaggle baseline is the earlier T4 run (Ultralytics 8.4.159, batch 23).
Run folders `runs/<name>/`, analyses `runs/analysis-20261004-<name>/`.

| Run | Batch | mAP@50 | mAP@50-95 | Fish found / missed | False alarms (empty frames) | Recall 16-32 / 32-96 px |
|---|---|---|---|---|---|---|
| Kaggle baseline `20260922-yolov8n-960-s0` | 23 | 0.4876 | 0.3124 | 664 / 721 | 659 (207) | 1.5% / 44.5% |
| **Control** `20261004-yolov8n-960-s0` (server) | 10 | 0.4562 | 0.2957 | 648 / 737 | 531 (150) | 2.6% / 42.2% |
| P2 head `20261004-yolov8n-960-p2-s0` | 5 | 0.5075 | 0.2830 | 673 / 712 | 364 (115) | 3.7% / 45.3% |
| Aug (scale 0.9, hue 0.05) `20261004-yolov8n-960-aug-s0` | 10 | 0.5427 | 0.3285 | 729 / 656 | 474 (152) | 6.2% / 50.7% |
| YOLOv8s `20261004-yolov8s-960-s0` | 5-6, resumed | 0.5846 | 0.3588 | 741 / 644 | 396 (80) | 5.9% / 53.9% |

- **The machine matters as much as the seed spread.** The same seed on the server (control) scores 0.031 mAP@50 and
  0.017 mAP@50-95 below the Kaggle run, under the whole Kaggle 3-seed range (0.4876 to 0.5220). Batch size, Ultralytics
  version and the GPU all differ. Comparisons across machines are not trustworthy; only the same-machine control is.
- **Aug vs control (same machine, same batch, same seed): +0.087 mAP@50, +0.033 mAP@50-95, 81 more fish found
  (648 to 729), 57 fewer false alarms.** Recall rises at 16-32 px (2.6% to 6.2%) and at 32-96 px (42.2% to 50.7%). This
  is the largest gain of the cheap changes, about 2.5 times the 0.034 Kaggle seed spread, but it is **one seed**
  and a **bundle** (scale and hue changed together), so it is a candidate and not a result.
- **YOLOv8s vs control: +0.128 mAP@50, +0.063 mAP@50-95, 93 more fish found, empty-frame false alarms 150 to 80.** The
  largest single-seed effect so far. Caveats: it is about 3 times the compute (Orin latency not measured), its batch
  differs (5-6 against 10), and it was interrupted at epoch 50 by the data deletion in section 7 and resumed from
  `last.pt`, early-stopping at epoch 66. Its epoch time in `results.csv` restarts at the resume and is not comparable.
- **Small fish stay hard for every variant.** Even the best run finds 6% of 16-32 px fish and 54% of 32-96 px fish.

**Limits.** n=1 seed per row; seed spread on this machine is unknown. Batch sizes differ between rows, so only
control, aug (both batch 10) are a clean pair. One validation split of near-duplicate frames. Not tested on the sealed
OzFish set. 3 seeds on the winning ideas are the next step (queued as the Kakadu comparison for the control).

### F15. YOLOv8m has the best strict score and far fewer false alarms, but finds no more fish than YOLOv8s (one seed)
`20261004-yolov8m-960-s0` on `ai-gpu` (RTX 5060, Ultralytics 8.4.172), seed 0, 960 px, **batch 3**, early-stopped at
epoch 107 (best epoch 77) after 4.96 h. Validation split, conf 0.25 for the counts. Analysis:
`runs/analysis-20261004-20261004-yolov8m-960-s0`. Compared with the same-machine control and YOLOv8s from F14.

| Run | mAP@50 | mAP@50-95 | Precision / recall (final eval) | Fish found (of 1,385) | False alarms (empty frames) | Recall 32-96 px |
|---|---|---|---|---|---|---|
| Control YOLOv8n (batch 10) | 0.4562 | 0.2957 | 0.569 / 0.460 | 648 | 531 (150) | 42.2% |
| YOLOv8s (batch 5-6, resumed) | 0.5846 | 0.3588 | 0.824 / 0.518 | 741 | 396 (80) | 53.9% |
| **YOLOv8m (batch 3)** | **0.5774** | **0.3914** | **0.921 / 0.515** | 714 | **224 (42)** | 51.5% |

- **Against the control: +0.121 mAP@50 and +0.096 mAP@50-95, 66 more fish found, 307 fewer false alarms.**
- **Against YOLOv8s: mAP@50 level (-0.007), mAP@50-95 higher (+0.033), precision higher, false alarms down from 396 to
  224, but 27 fewer fish found.** Going from small to medium buys cleaner, better-placed boxes, not more recall.
- **Small fish are still missed**: 6 of 273 at 16-32 px (2.2%), 51.5% at 32-96 px.
- **Cost.** It is by far the largest model so far (about 26 million parameters against 3 million, per Ultralytics'
  published figures) and the slowest to train (2.8 minutes an epoch against 0.6). Its speed on the Orin is **not
  measured**; a model that cannot hold the frame rate at 960 px does not count as a result for this project.

**Limits.** One seed. Batch sizes differ (3, 5-6, 10), so the gaps mix model size with batch size; the seed spread
on this machine is unknown. One validation split of near-duplicate frames. Not tested on the sealed OzFish set.

### F16. YOLO11s raises the scores by cutting false alarms, not by finding more fish (one seed)
`20261004-yolo11s-960-s0` on `ai-gpu` (RTX 5060, Ultralytics 8.4.172), seed 0, 960 px, **batch 5**, early-stopped at
epoch 73 (best epoch 43) after 1.53 h. Validation split, conf 0.25 for the counts. Analysis:
`runs/analysis-20261004-20261004-yolo11s-960-s0`. Compared with the same-machine control and the earlier models.

| Run | mAP@50 | mAP@50-95 | Precision / recall (final eval) | Fish found (of 1,385) | False alarms (empty frames) | Recall 32-96 px |
|---|---|---|---|---|---|---|
| Control YOLOv8n (batch 10) | 0.4562 | 0.2957 | 0.569 / 0.460 | 648 | 531 (150) | 42.2% |
| YOLOv8s (batch 5-6, resumed) | 0.5846 | 0.3588 | 0.824 / 0.518 | 741 | 396 (80) | 53.9% |
| YOLOv8m (batch 3) | 0.5774 | 0.3914 | 0.921 / 0.515 | 714 | 224 (42) | 51.5% |
| **YOLO11s (batch 5)** | **0.5172** | **0.3336** | **0.833 / 0.470** | **640** | **161 (40)** | **41.0%** |

- **Against the control: +0.061 mAP@50 and +0.038 mAP@50-95, but 8 fewer fish found (640 against 648) and 370 fewer
  false alarms (161 against 531).** The whole gain comes from fewer false alarms and better-placed boxes; recall at
  conf 0.25 is unchanged (46.2% against 46.8%) and 32-96 px recall is slightly lower (41.0% against 42.2%).
- **It is below YOLOv8s and YOLOv8m on every score here**, with the fewest false alarms of any run. At this seed and
  this recipe the newer architecture does not help recall over YOLOv8n, which is what the project needs most.
- **Small fish are still missed**: 3 of 273 at 16-32 px (1.1%).

**Limits.** One seed; batch sizes differ between rows (3, 5, 5-6, 10), so the gaps mix architecture with batch size;
seed spread on this machine unknown. One validation split of near-duplicate frames. Not tested on the sealed
OzFish set; Orin speed not measured. The mAP gain over the control is a precision effect and should not be read as the
detector finding more fish.

### F17. Adding Kakadu to the training data raises the DeepFish-val score a great deal and finds many more small fish (one seed)
`20261004-yolov8n-960-kakadu-s0`: YOLOv8n, seed 0, 960 px, **batch 10** (same as the control), trained on DeepFish
train (5,178) plus Kakadu (7,318 frames, one class, max 15 per scene group, 960 px wide) = 12,496 images, validated on
the unchanged DeepFish val split (1,339). `ai-gpu`, Ultralytics 8.4.172, early-stopped at epoch 89 (best epoch 59) after
2.12 h. Analysis: `runs/analysis-20261004-20261004-yolov8n-960-kakadu-s0`. Control: `20261004-yolov8n-960-s0`.

| | Control (DeepFish only) | **DeepFish + Kakadu** |
|---|---|---|
| mAP@50 | 0.4562 | **0.7088** |
| mAP@50-95 | 0.2957 | **0.4781** |
| Precision / recall (final eval) | 0.569 / 0.460 | 0.840 / 0.622 |
| Fish found (of 1,385) at conf 0.25 | 648 | **852** |
| False alarms (empty frames) | 531 (150) | **131** (70) |
| Recall at 16-32 px (n=273) | 2.6% | **19.8%** |
| Recall at 32-96 px (n=746) | 42.2% | **61.4%** |
| Recall above 96 px (n=356) | 91.6% | 95.5% |

- **Difference: +0.253 mAP@50 and +0.182 mAP@50-95, 204 more fish found, 400 fewer false alarms.** Every size
  bucket improves, and the 16-32 px fish, which no earlier variant found, go from 2.6% to 19.8%. This is about
  seven times the 0.034 mAP@50 seed spread measured on Kaggle and far above any other change tried.
- **Why this needs care.** (1) One seed; the seed spread on this machine is not known yet (the control runs for seeds 1 and
  2 are in the queue). (2) The leak check found no near-duplicates between Kakadu and the val frames (F13), but Kakadu
  frames look **more like the val frames than DeepFish's own train frames do** (median distance 13.6 against 15.7),
  because DeepFish is split by site. So much of the gain is "training data that resembles the validation sites"; it
  does not show the model generalises. (3) The comparison is on DeepFish val only. The sealed OzFish set, which
  Kakadu shares no scenes with, is the test of generalisation and has **not** been run (it is used once, at the end).
- **The confidence threshold is no longer well placed.** 313 of the 533 missed fish were found but scored under 0.25, and the
  best-F1 threshold on val is 0.2 (it was 0.4 to 0.6 for earlier runs). A lower operating threshold would recover
  fish; that is a deployment setting chosen on val, not on the sealed set.

- **Seed spread on this machine, update (control seed 1, `20261004-yolov8n-960-s1`, batch 10, 59 epochs):** 0.4970 mAP@50 and 0.2966 mAP@50-95, against seed 0's 0.4562 and 0.2957. So with two control seeds the mAP@50 spread is 0.041 (the Kaggle 3-seed figure was 0.034) and the mAP@50-95 spread is only 0.001. Seed 1 finds 687 fish (seed 0: 648) with 525 false alarms (531). The Kakadu seed-0 score is still 0.21 above the better control seed. Kakadu seeds 1 and 2 and control seed 2 are queued.

- **Kakadu seed 1 (`20261004-yolov8n-960-kakadu-s1`, batch 10, early-stopped at epoch 101, 2.42 h; analysis `runs/analysis-20261004-20261004-yolov8n-960-kakadu-s1`):** 0.6844 mAP@50, 0.4640 mAP@50-95, precision 0.776, recall 0.599. Found 839 fish (seed 0: 852), 155 false alarms (131), recall 14.3% at 16-32 px (19.8%) and 61.0% at 32-96 px (61.4%). So the Kakadu gain reproduces on a second seed: 0.684 against the matching control seed's 0.497 is +0.187 mAP@50. The 3-seed comparison follows when seed 2 of both arms finishes.

- **Control seed 2 (`20261005-yolov8n-960-s2`, batch 10, 100 epochs):** run folder recorded here; its numbers go into the 3-seed entry once Kakadu seed 2 finishes.

**Limits.** n=1 seed for both rows. Same machine and batch size for the pair, so that confound is removed. One
validation split of near-duplicate frames. Kakadu labels were merged from 23 species and only lightly audited (F13).
Not tested on the sealed OzFish set; Orin speed unchanged from YOLOv8n (same model).

### F18. Three seeds each: adding Kakadu lifts every seed far above every control seed on the DeepFish val split
Same machine (`ai-gpu`, RTX 5060, Ultralytics 8.4.172), same batch (10), same recipe, 960 px, YOLOv8n, seeds 0, 1, 2. The only
difference is the training data: DeepFish train (5,178) against DeepFish train plus Kakadu (7,318 frames kept, 12,496 in all).
Both arms validate on the unchanged DeepFish val split (1,339 images, 1,385 fish). Run folders: control `20261004-yolov8n-960-s0`,
`20261004-yolov8n-960-s1`, `20261005-yolov8n-960-s2`; Kakadu `20261004-yolov8n-960-kakadu-s0`, `20261004-yolov8n-960-kakadu-s1`,
`20261005-yolov8n-960-kakadu-s2`. Analyses in `runs/analysis-<date>-<run>/`.

| Seed | Control mAP@50 | Kakadu mAP@50 | Difference | Control mAP@50-95 | Kakadu mAP@50-95 | Difference |
|---|---|---|---|---|---|---|
| 0 | 0.4562 | 0.7088 | +0.2527 | 0.2957 | 0.4781 | +0.1824 |
| 1 | 0.4970 | 0.6844 | +0.1874 | 0.2966 | 0.4640 | +0.1674 |
| 2 | 0.4669 | 0.7002 | +0.2333 | 0.3106 | 0.4712 | +0.1606 |
| **Mean (range)** | **0.4733 (0.041)** | **0.6978 (0.024)** | **+0.2245** | **0.3010 (0.015)** | **0.4711 (0.014)** | **+0.1701** |

| At conf 0.25 | Control (3 seeds) | Kakadu (3 seeds) |
|---|---|---|
| Fish found, of 1,385 | 648, 687, 644 (mean 660) | 852, 839, 891 (mean 861) |
| False alarms (on the 464 empty frames) | 531, 525, 535 (150, 219, 186) | 131, 155, 254 (70, 47, 88) |
| Recall at 16-32 px (273 fish) | 2.6%, 2.9%, 2.6% | 19.8%, 14.3%, 24.9% (mean 19.7%) |
| Recall at 32-96 px (746 fish) | 42.2%, 47.0%, 42.1% | 61.4%, 61.0%, 64.9% (mean 62.4%) |
| Recall above 96 px (356 fish) | 91.6%, 92.1%, 90.7% | 95.5%, 96.9%, 94.9% |

- **The difference exceeds the seed spread by a wide margin.** The weakest Kakadu seed (0.6844) is 0.187 mAP@50 above the best
  control seed (0.4970), which is 4.6 times the control's own range (0.041); for mAP@50-95 the gap is 0.153 against a control range of 0.015.
  All three paired differences have the same sign and similar size.
- **It finds more fish, at every box size, and far fewer false alarms.** About 200 more fish (of 1,385), 350 fewer false alarms, and the
  16-32 px fish, which every control and earlier variant found only 1% to 6% of, go from 2.7% to about 20%. Seed-to-seed variation within the Kakadu arm is larger
  for the smallest fish (14% to 25%) and for false alarms (131 to 254) than within the control.
- **Fish under 16 px (10 of them) are almost never found** (1 of 10 in one Kakadu run, 0 in every other run), **and about 80% of 16-32 px fish are still missed.**

**What this does not show.** (1) Kakadu frames resemble the DeepFish val frames more than DeepFish's own train frames do (median
thumbnail distance 13.6 against 15.7, F13), so part of this gain is training data that looks like the validation sites; it is not
evidence that the model generalises to a new kind of footage. (2) The sealed OzFish set, the test of that, has **not** been run and
should be run once, on the final candidate. (3) Kakadu's labels were lightly checked, not audited (F13). (4) One model size (YOLOv8n),
one validation split of near-duplicate frames. (5) The augmentation and larger-model gains (F14, F15) were measured on one seed and
have not been combined with Kakadu or repeated. (6) The best operating threshold on val moved to 0.2-0.25 for Kakadu models (0.5 for
the controls); a deployment threshold must be chosen on val, not on the sealed set.

### F19. Empty frames explain less than 0.03 of mAP@50, so they are not why our scores sit far below published DeepFish numbers
Published DeepFish detection results (for example YOLOv8 at about 0.92 mAP@50) use 4,505 images with 15k boxes. Our export has 6,517
images, and 6,517 minus our 2,012 empty frames is exactly 4,505, so the published set is very probably the fish-containing images only.
Our validation split keeps its 464 empty frames, where any detection is a false alarm. `scripts/eval_positive_only.py` re-scores
the same checkpoints on the 875 validation images that contain fish (960 px, conf 0.001, IoU 0.7), single run each:

| Checkpoint | mAP@50, full val (1,339) | mAP@50, fish-only (875) | mAP@50-95, fish-only |
|---|---|---|---|
| Control seed 0 | 0.4562 | 0.4794 | 0.3078 |
| Control seed 1 | 0.4970 | 0.5147 | 0.3046 |
| Kakadu seed 0 | 0.7088 | 0.7351 | 0.4911 |
| YOLOv8s | 0.5846 | 0.5941 | 0.3635 |
| YOLOv8m | 0.5774 | 0.5835 | 0.3940 |

- **Dropping the empty frames adds 0.006 to 0.026 mAP@50.** The convention explains a small part of the gap, not the 0.4 to 0.45 between our
  controls (0.46 to 0.50) and 0.92.
- **The remaining candidates, none yet tested:** (1) the split. Published results I could read split frames at random, which
  puts near-identical frames of one clip on both sides; ours is by site, and val frames are far from train frames (nearest-neighbour
  distance median 15.7 against about 1 inside a clip, F13). (2) The val sites are hard: small, camouflaged fish (F10, F11).
  (3) A small, narrow training set (F18). The test that separates (1) from the rest is a model trained on DeepFish with a random
  per-image split, clearly labelled as inflated; it has **not** been run.
- **Not a sign of an evaluation fault**: the checkpoints reproduce their recorded scores and the full-val numbers here agree with `metrics.json`.

**Limits.** Single evaluations of saved checkpoints; the subset, not the models, changed. I could not open the DeepFish papers' own
split descriptions (one page refused access), so the "random split" point rests on the OzFish study I could read and on the standard
practice of that literature, not on those specific papers.

### F20. DIAGNOSTIC: on a random per-image split the same pipeline scores 0.98 mAP@50, so the split, not a weak model, explains our gap to published numbers
**These numbers are inflated by construction and are not results.** `scripts/prepare_random_split.py` builds a deliberately leaky split of the DeepFish
images that contain fish: **4,505 images and 15,463 boxes, the size of the published DeepFish detection set**, split at random per image (seed 0): 3,604 train
(12,405 boxes), 901 val (3,058 boxes). Near-identical frames of one clip land on both sides. Same model, code and recipe as our other runs
(YOLOv8n, COCO weights, patience 30), one seed, on `ai-gpu`.

| Run | Split | Val images | mAP@50 | mAP@50-95 | P / R | Epochs |
|---|---|---|---|---|---|---|
| `20261005-yolov8n-640-randomsplit-s0` (batch 24, 640 px) | **random per image (leaky)** | 901 | **0.9787** | **0.6447** | 0.953 / 0.941 | 150 (hit the cap), best 139 |
| `20261005-yolov8n-960-randomsplit-s0` (batch 10, 960 px) | **random per image (leaky)** | 901 | **0.9771** | **0.6534** | 0.951 / 0.953 | 150 (hit the cap), best 137 |
| Our controls (3 seeds, 960 px, batch 10, F18) | by site | 1,339 | 0.4733 mean (0.456 to 0.497) | 0.3010 mean | | 59 to 107 |
| Published DeepFish YOLOv8 results (secondary sources, split not verified) | not verified | not verified | about 0.92 | not given | | |

- **The 960 px diagnostic matches the controls' resolution and batch size and gives the same answer**: 0.9771 mAP@50 and 0.6534 mAP@50-95, 1.06 h, on the same machine. Resolution is not what separates the two settings.
- **The split accounts for the gap, and more.** With only the split changed (and the empty frames dropped, which F19 shows is worth under 0.03), the
  same pipeline reaches 0.979, above the published 0.92 and about 0.5 above our site-split score. It reached 0.915 by epoch 13. So the low
  site-split numbers are not evidence of a weak model or a faulty pipeline. They reflect how much of a random split's score comes from seeing the same scene in training.
- **What this does not show.** The published papers' own splits were not read (one page refused access), so "they used a random split" is an inference from the OzFish study I could
  read and from common practice, not a verified fact. The 960 diagnostic and the 960 controls share resolution, batch size and machine but differ in the split and in the validation set (901 fish-only
  images against 1,339, which F19 shows is worth under 0.03), so the comparison is a diagnostic and not a like-for-like pair; one seed each and one random split.
- **Consequence for the report.** Numbers from random-split DeepFish results should not be compared with ours, and a high published DeepFish score says little about performance on a new site.
  Our site-split score is the more honest estimate for a new deployment; the sealed OzFish set is the stricter one.

### F21. On OzFish deployments no model trained on, the Kakadu gain holds: about +0.10 mAP@50 (3 seeds each, a decision-only check)
`scripts/prepare_ozfish_train.py --val-frac 0.2` holds out whole OzFish deployments (stereo pairs together) from the 779 trainable frames:
**622 training images (14,072 boxes) and 157 held-out images (3,492 boxes) from 79 deployments**; none of the 352 sealed-test deployments is in either part.
`scripts/eval_ozfish_val.py` scores the existing checkpoints (960 px, conf 0.001, IoU 0.7, `ai-gpu`) on the 157 held-out images. None of these models
saw any OzFish frame. The sealed OzFish test set was **not** used.

| Training data | Seed 0 | Seed 1 | Seed 2 | Mean mAP@50 | Mean mAP@50-95 | Mean recall |
|---|---|---|---|---|---|---|
| DeepFish only (controls) | 0.2653 | 0.2720 | 0.2313 | **0.2562** | 0.1219 | 0.257 |
| DeepFish + Kakadu | 0.3483 | 0.3677 | 0.3548 | **0.3569** | 0.1848 | 0.317 |

- **The weakest Kakadu seed (0.3483) is above the best control seed (0.2720)**: a difference of +0.101 mAP@50 and +0.063 mAP@50-95 on average, with the spread inside each arm
  at most 0.041. So the benefit of Kakadu data is not confined to the DeepFish val sites; it carries to a different dataset of crowded BRUVS footage that neither arm trained on.
- **The absolute level is low** (0.26 to 0.37 mAP@50, recall 0.26 to 0.33): OzFish frames hold about 22 fish each, many only 6 to 12 px wide at native size,
  and none of these models saw any BRUVS footage in training. This is the baseline against which adding OzFish training data (`ozfish_train/images/train`) is judged.
- **Limits.** The held-out deployments are disjoint from the training deployments but one reef can span deployments, so numbers are for decisions only. 157 images; same
  seeds as F18; the OzFish-style crowded scenes differ a lot from DeepFish, so this says the gain transfers, not by how much to a new site. Not the sealed test.

### F22. One seed of the combined recipe (Kakadu + OzFish data + stronger augmentation) beats every Kakadu-only seed on DeepFish val, with more false alarms at the default threshold (a screen)
`20261005-yolov8n-960-all-s0`: YOLOv8n, seed 0, 960 px, **batch 10** (same machine and batch as the F18 arms), trained on DeepFish train (5,178) + Kakadu (7,318) + OzFish train (622) with
the augmentation of the F14 screen (scale 0.9, hue 0.05, saturation 0.8, value 0.5); validated on the unchanged DeepFish val split; early-stopped at epoch 99 (best epoch 69) after
2.52 h on `ai-gpu`. Analysis: `runs/analysis-20261005-20261005-yolov8n-960-all-s0`. **It changes two things at once** (the OzFish data and the augmentation), so it shows the additions
can stack, not which one helped. **One seed: a screen, not a result.**

| | Kakadu-only (3 seeds, F18) | **All-in (1 seed)** |
|---|---|---|
| DeepFish val mAP@50 | 0.7088, 0.6844, 0.7002 (mean 0.6978, range 0.024) | **0.7551** |
| DeepFish val mAP@50-95 | 0.4781, 0.4640, 0.4712 (mean 0.4711, range 0.014) | **0.5228** |
| Precision / recall (final eval) | 0.840 / 0.622, 0.776 / 0.599, 0.808 / 0.612 | 0.843 / 0.639 |
| Fish found, of 1,385 (conf 0.25) | 852, 839, 891 | **952** |
| False alarms (on the 464 empty frames) | 131, 155, 254 (70, 47, 88) | **508 (187)** |
| Recall at 16-32 px / 32-96 px / above 96 px | 19.8, 14.3, 24.9% / 61.4, 61.0, 64.9% / 95.5, 96.9, 94.9% | **27.5% / 70.8% / 98.0%** |
| Best-F1 confidence threshold on val | 0.2 to 0.25 | **0.6** |
| Held-out OzFish deployments, mAP@50 / mAP@50-95 (F21) | 0.3483, 0.3677, 0.3548 / 0.1805, 0.1904, 0.1836 | **0.7071 / 0.3868** (P 0.829, R 0.620) |

- **On DeepFish val it exceeds the Kakadu-only seed spread**: +0.057 mAP@50 over the mean and +0.046 over the best seed (the arm's range is 0.024, so 1.9 to 2.3 times it), and +0.052 mAP@50-95 over the mean
  (3.7 times that arm's 0.014 range). It finds about 90 more fish and more small ones (16-32 px: 27.5% against 14 to 25%).
- **It makes far more false alarms at the default threshold** (508 against 131 to 254, 187 on empty frames against 47 to 88), and the best operating threshold on val moved to 0.6. Its scores are
  higher in the way mAP rewards (ranking) while its raw detections at 0.25 are noisier, so a deployment threshold must be chosen for it (on val, not on the sealed set).
- **On the held-out OzFish deployments the score doubles (0.71 against 0.36)**, which is expected: it trained on OzFish deployments and Kakadu-only did not. This is **not** evidence of
  generalising to a new reef: the held-out deployments are disjoint from the training deployments, but one reef can span deployments, so the number is optimistic (F21).
- **Not separated:** the OzFish data and the augmentation (bundled); seed spread of this recipe (n=1); the benefit of OzFish on DeepFish val specifically (OzFish looks very different, so
  the DeepFish-val gain probably comes mostly from the augmentation, which is untested).

- **Split test, arm A finished (interim; the full comparison F23 follows when arm B finishes).** `20261005-yolov8n-960-kakadu-ozfish-s0` (DeepFish + Kakadu + OzFish train data, DEFAULT augmentation, batch 10, seed 0, early-stopped at epoch 65, best 35, 1.65 h; analysis `runs/analysis-20261005-20261005-yolov8n-960-kakadu-ozfish-s0`): DeepFish val **0.7640 mAP@50 and 0.5260 mAP@50-95**, 928 fish found, **155 false alarms** (57 on empty frames), recall 18.7% at 16-32 px and 70.4% at 32-96 px, best-F1 threshold 0.2; held-out OzFish deployments **0.6860 / 0.3738**. So OzFish data alone matches or beats the combined recipe on DeepFish val (0.764 against 0.755) with far fewer false alarms (155 against 508), and reaches most of the held-out OzFish score (0.686 against 0.707). The stronger augmentation added no clear gain on top; its own effect is arm B. One seed each, so a screen.

**Limits.** n=1 seed for the all-in arm, batch 10, one machine. DeepFish val resembles Kakadu (F13). The sealed OzFish test has **not** been run; training on OzFish changes what it measures
(`docs/data_audit.md`). The two-model plan (a separate generalisation model without OzFish) has not been approved.

### F23. Split test: the OzFish training data accounts for the gain of the combined recipe; the stronger augmentation adds nothing measurable (one seed per new arm)
Four training recipes on the same machine and batch (`ai-gpu`, batch 10, YOLOv8n, 960 px, validated on the DeepFish val split), differing in only the stated parts. Kakadu-only is the 3-seed arm of F18; the others are **one seed
each, so screens**. New runs: `20261005-yolov8n-960-kakadu-ozfish-s0` (arm A, OzFish train data, default augmentation), `20261005-yolov8n-960-kakadu-aug-s0` (arm B, stronger augmentation, no OzFish; 150 epochs, hit the cap, best 138, 3.59 h);
combined recipe `20261005-yolov8n-960-all-s0` (F22). Analyses in `runs/analysis-20261005-<run>/`.

| | Kakadu-only (3 seeds) | **A: + OzFish data** | **B: + strong augmentation** | Both (all-in) |
|---|---|---|---|---|
| DeepFish val mAP@50 | 0.6978 (0.6844 to 0.7088) | **0.7640** | 0.6976 | 0.7551 |
| DeepFish val mAP@50-95 | 0.4711 (0.4640 to 0.4781) | **0.5260** | 0.4732 | 0.5228 |
| Fish found, of 1,385 (conf 0.25) | 861 (839 to 891) | 928 | 822 | **952** |
| False alarms (on 464 empty frames) | 180 (131 to 254) (70, 47, 88) | **155 (57)** | 195 (91) | 508 (187) |
| Recall 16-32 px | 19.7% (14.3 to 24.9) | 18.7% | 14.6% | **27.5%** |
| Recall 32-96 px | 62.4% (61.0 to 64.9) | 70.4% | 59.0% | **70.8%** |
| Best-F1 confidence on val | 0.2 to 0.25 | 0.2 | 0.2 | **0.6** |
| Held-out OzFish deployments mAP@50 / mAP@50-95 | 0.3569 / 0.1848 (0.3483 to 0.3677) | 0.6860 / 0.3738 | 0.3655 / 0.1906 | **0.7071 / 0.3868** |

- **The OzFish data accounts for the gain.** Adding it alone gives +0.066 mAP@50 and +0.055 mAP@50-95 over the Kakadu-only mean on DeepFish val, 2.7 and 3.9 times that arm's seed range (0.024 and 0.014), and +0.055 over the best Kakadu-only seed. It is the
  only change that moves the held-out OzFish score (0.357 to 0.686).
- **The stronger augmentation alone does nothing measurable.** Arm B equals the Kakadu-only mean on DeepFish val (0.6976 against 0.6978; mAP@50-95 +0.002) and on held-out OzFish (0.3655 against 0.3569, within the arm's 0.019 range).
  On top of OzFish data it does not help either: the combined recipe is 0.009 below arm A on mAP@50 (inside the seed spread), and it makes **353 more false alarms at conf 0.25** (508 against 155) and pushes the best threshold to 0.6.
  Its one possible benefit, more fish at 16-32 px (27.5% against 18.7%), is one seed and inside the 10.6-point range seen across the Kakadu-only seeds.
- **Best of these by evidence: arm A** (DeepFish + Kakadu + OzFish train data, default augmentation): the highest DeepFish val mAP@50-95, few false alarms, and a normal 0.2 threshold.

- **Interim: confirmation seed 1 of arm A (`20261005-yolov8n-960-kakadu-ozfish-s1`, batch 10, 60 epochs, best 30; analysis `runs/analysis-20261005-20261005-yolov8n-960-kakadu-ozfish-s1`; the full 3-seed finding F24 follows when seed 2 and the YOLOv8s run finish):** DeepFish val **0.7454 mAP@50 and 0.5007 mAP@50-95** (seed 0: 0.7640 and 0.5260), 897 fish found (928), held-out OzFish deployments 0.6820 / 0.3684 (0.6860 / 0.3738), recall 11.7% at 16-32 px (18.7%) and 70.0% at 32-96 px (70.4%). The OzFish gain reproduces on a second seed (0.745 against the Kakadu-only range 0.684 to 0.709). **But its false alarms at conf 0.25 are 477 (179 on empty frames), against 155 (57) for seed 0, and its best-F1 threshold is 0.4 against 0.2.** So the low false-alarm count of arm A (and the comparison with the combined recipe's 508 in this finding) rested on one seed and does not hold: false alarms at the default threshold vary a lot from seed to seed (the Kakadu-only seeds ranged 131 to 254), and the threshold must be chosen per model on validation. The 'accounts for the gain' conclusion about OzFish data concerns mAP, not false alarms.

- **Interim: arm A on three seeds (confirmation seed 2 = `20261005-yolov8n-960-kakadu-ozfish-s2`, batch 10, 80 epochs, best 50; analysis `runs/analysis-20261005-20261005-yolov8n-960-kakadu-ozfish-s2`; the full finding F24 follows when the YOLOv8s run finishes).** Seed 2: DeepFish val 0.7402 / 0.5158, 906 fish found, 175 false alarms (80 on empty frames), recall 14.6% at 16-32 px, best-F1 threshold 0.25, held-out OzFish 0.6980 / 0.3814. **Three seeds of YOLOv8n + DeepFish + Kakadu + OzFish train (default augmentation):** DeepFish val mAP@50 0.7640, 0.7454, 0.7402 (mean 0.7499, range 0.024); mAP@50-95 0.5260, 0.5007, 0.5158 (mean 0.5142, range 0.025); fish found 928, 897, 906 (mean 910); false alarms 155, 477, 175 (mean 269); held-out OzFish mAP@50 0.6860, 0.6820, 0.6980 (mean 0.6887, range 0.016). Against the Kakadu-only arm (mean 0.6978 mAP@50, 0.4711 mAP@50-95, 861 fish, 3 seeds) that is +0.052 mAP@50, +0.043 mAP@50-95 and about 49 more fish found; **the weakest of the three seeds (0.7402) is above the best Kakadu-only seed (0.7088)**, by 1.3 times that arm's 0.024 range, so the OzFish effect on DeepFish val holds on three seeds but is smaller than the single-seed estimate (+0.066). On the held-out OzFish deployments the gain is large and consistent (0.689 against 0.357 Kakadu-only and 0.256 DeepFish-only), with the usual caveat that those deployments may share reefs with the training ones.

**Limits.** One seed for arms A and B and the combined recipe; the effects are large against the Kakadu-only spread (OzFish) or zero (augmentation), but each new arm has no seed spread of its own. Held-out OzFish deployments may share reefs with
training deployments, so those scores are optimistic. DeepFish val resembles Kakadu (F13), although OzFish train frames are far from it (nearest distance 7.15). The sealed OzFish test has **not** been run.

### F24. The OzFish effect holds on three seeds; YOLOv8s on the same data is the most accurate model so far (one seed, speed on the Orin unmeasured)
**Arm A on three seeds**: YOLOv8n, DeepFish + Kakadu + OzFish train data, default augmentation, `ai-gpu`, batch 10, 960 px, validated on DeepFish val; runs `20261005-yolov8n-960-kakadu-ozfish-s0`,
`20261005-yolov8n-960-kakadu-ozfish-s1`, `20261005-yolov8n-960-kakadu-ozfish-s2`. **YOLOv8s** on the same data and recipe: `20261005-yolov8s-960-kakadu-ozfish-s0` (seed 0, **batch 5**, early-stopped at epoch 71,
best 41, 3.54 h at 179 s per epoch; YOLOv8n took about 1.5 minutes per epoch). Analyses in `runs/analysis-<date>-<run>/`. Held-out OzFish = the 157 held-out deployments of F21.

| | DeepFish-only control (3 seeds) | Kakadu-only (3 seeds) | **Arm A: YOLOv8n + OzFish (3 seeds)** | **YOLOv8s + OzFish (1 seed)** |
|---|---|---|---|---|
| DeepFish val mAP@50 | 0.4733 (0.456 to 0.497) | 0.6978 (0.684 to 0.709) | **0.7499** (0.7402, 0.7454, 0.7640; range 0.024) | **0.8209** |
| DeepFish val mAP@50-95 | 0.3010 | 0.4711 (0.464 to 0.478) | **0.5142** (0.5007, 0.5158, 0.5260; range 0.025) | **0.5624** |
| Fish found, of 1,385 (conf 0.25) | 660 | 861 (839 to 891) | 910 (897, 906, 928) | **1,004** |
| False alarms at conf 0.25 (on empty frames) | 530 | 180 (131 to 254) | **269 (155, 175, 477)** | **122 (24)** |
| Recall 16-32 px / 32-96 px / above 96 px | 2.7% / 43.8% / 91.5% | 19.7% / 62.4% / 95.8% | 15.0% / 69.9% / 97.9% | **26.4% / 77.5% / 99.4%** |
| Best-F1 confidence on val | 0.5 | 0.2 to 0.25 | 0.2, 0.4, 0.25 | 0.25 |
| Held-out OzFish mAP@50 / mAP@50-95 | 0.2562 / 0.1219 | 0.3569 / 0.1848 | **0.6887 / 0.3745** (0.686, 0.682, 0.698) | **0.7242 / 0.3976** |

- **The OzFish effect holds on three seeds.** The weakest arm-A seed (0.7402) is above the best Kakadu-only seed (0.7088), by 1.3 times that arm's 0.024 range; the mean gain is **+0.052 mAP@50 and +0.043 mAP@50-95**, about 49 more fish
  found, and on the held-out OzFish deployments +0.33 mAP@50 (0.689 against 0.357). The single-seed estimate of +0.066 (F23) was a little high.
- **False alarms at conf 0.25 and the best threshold vary a lot between seeds** (arm A: 155, 175, 477; best-F1 threshold 0.2 to 0.4; the Kakadu-only seeds ranged 131 to 254). A model's operating threshold must be chosen on validation for
  that model, and no false-alarm figure from one seed should be quoted as a property of a recipe.
- **YOLOv8s on the same data is the best model by every accuracy measure here**: +0.071 mAP@50 and +0.048 mAP@50-95 over the arm-A mean (+0.057 over its best seed, 2.4 times arm A's 0.024 range), 94 more fish found, the fewest false alarms of
  the combined-data models (122, 24 on empty frames), 26.4% of 16-32 px fish and 77.5% of 32-96 px fish, and +0.036 mAP@50 on the held-out OzFish deployments (0.724).
- **What the larger model costs.** About 3 times the compute per image, 2 times the training time per epoch at half the batch size (5 against 10, so this comparison also mixes batch size), and an unmeasured latency on the Orin.
  **The Nano is offline, so YOLOv8s has not been timed.** In the end-to-end measurement of F8 the YOLOv8n GPU stage was about 14 ms of 41 ms per frame at 960 px with the CPU stages dominating, so a larger GPU stage may still be
  affordable; that is a hypothesis and not a measurement, and a model that cannot hold the frame rate does not count for this project.

**Best model by evidence.** On accuracy, YOLOv8s trained on DeepFish + Kakadu + OzFish train data with default augmentation (one seed). On what is confirmed **and** deployable, YOLOv8n trained on the same data (three seeds, speed on the
Orin already measured). The choice between them waits for YOLOv8s Orin timing and, ideally, more YOLOv8s seeds.

**Not shown.** Seed spread of the YOLOv8s arm (n=1) and any effect of its different batch size; Orin latency of YOLOv8s; generalisation to a new reef (the held-out OzFish deployments may share reefs with the training ones); DeepFish val resembles Kakadu (F13);
the **sealed OzFish test has not been run**, and training on OzFish changed what it measures; a bigger model with the OzFish data was not run for YOLOv8m or YOLO11s.

### F25. Tiled inference finds more small fish but adds more false alarms than fish, so the score falls (one checkpoint, DeepFish val)
**Setup.** `scripts/eval_tiled.py` on the YOLOv8s checkpoint of F24 (`20261005-yolov8s-960-kakadu-ozfish-s0`, no retraining), DeepFish val (1,339 frames of 1920x1080), imgsz 960 per tile, 20% tile overlap, NMS IoU 0.7, operating
conf 0.25. One AP routine scores every row, so the untiled row is a same-script control: it gives 0.8167 AP50 against Ultralytics' 0.8209 (the 101-point pooled AP is close to, not identical to, Ultralytics' mAP).
Output: `runs/tiled-20261006-20261005-yolov8s-960-kakadu-ozfish-s0/tiled.json`. Tile boxes touching an inner tile edge are dropped (a cut fish gives a partial box); "keepedges" rows keep them.

| Variant | AP50 | AP50-95 | Fish found (of 1,385) | False alarms (on empty frames) | Recall 16-32 px / 32-96 px |
|---|---|---|---|---|---|
| **Whole frame (control)** | **0.8167** | **0.5556** | 1,004 | **123 (24)** | 26.4% / 77.5% |
| 2x2 tiles only | 0.5963 | 0.3575 | 803 | 314 | 27.5% / 68.2% |
| 2x2 tiles + whole frame | 0.7906 | 0.5441 | 1,085 | 459 (98) | 37.4% / 84.3% |
| 3x3 tiles only | 0.3655 | 0.1920 | 507 | 444 | 26.0% / 49.9% |
| 3x3 tiles + whole frame | 0.7843 | 0.5488 | 1,080 | 611 (127) | 38.8% / 83.1% |
| 2x2 + whole frame, edge boxes kept | 0.7124 | 0.4876 | 1,086 | 910 | 37.4% / 84.5% |
| 3x3 + whole frame, edge boxes kept | 0.7040 | 0.5011 | 1,087 | 1,588 | 38.8% / 84.0% |

- **Tiling does find more fish where it was meant to.** Adding tiles to the whole frame finds 76 to 81 more fish (+8%), and recall on 16-32 px fish rises from 26% to 37 to 39% and on 32-96 px fish from 77% to 83 to 84%. Fish above 96 px are unchanged (99.4%) and fish under 16 px are still not found (0%).
- **It pays for that in false alarms, and the net score falls.** False alarms at conf 0.25 rise 3.7x to 5x (123 to 459 and 611), including on empty frames (24 to 98 and 127), and AP50 drops by 0.026 to 0.033 and AP50-95 by 0.007 to 0.011. AP is threshold-free, so a different confidence cut does not rescue it; no tiled variant beats the whole-frame control.
- **Tiles alone are much worse** (AP50 0.60 and 0.37): each tile is magnified so large fish are cut and look unlike the training scale, and the whole frame is needed to anchor them.
- **The edge-box rule matters a great deal** (false alarms 459 against 910 for 2x2 + whole frame) and was needed to get even this far; whether a smarter merge (for example accepting only small tile boxes, or a higher confidence floor for tile boxes) turns the +81 fish into a net gain was **not tried**, and choosing such a rule on this split would be tuning on the validation set, so it would need a fresh screen and a different split to confirm.
- **Cost.** 2x2 + whole frame sends 5 images through the network per frame and 3x3 + whole frame 10, so on the Orin the GPU stage alone would be about 5x to 10x. Not measured there (the Nano is offline); seconds per frame in the JSON come from the workstation and say nothing about the Orin.

**Verdict.** On this checkpoint and this split, plain tiled inference does not improve detection quality: it trades 8% more fish found for several times the false alarms, at 5x to 10x the compute. It is not worth deploying as it stands.
**Limits.** One checkpoint, one seed, one split of near-duplicate frames; fixed tiling and overlap with no sweep; DeepFish val has few fish under 32 px (the 16-32 px bucket is a few hundred boxes), so the recall gains are real but their size is imprecise; no result on OzFish (sealed or held-out).

### F26. A small-box merge for tiled inference also does not beat the whole frame (four rules fixed in advance, all reported)
**Setup.** The merge rule F25 left untried: keep a tile box only if the fish is small (sqrt of box area at most 32 px as seen at 640 input, the existing bucket scale), optionally also requiring confidence of at least 0.4 on tile boxes,
then merge with the whole-frame boxes by NMS. Four variants (2x2 and 3x3 tiling, with and without the 0.4 floor) were **fixed before the run and all are reported; none was chosen on this split**. Same checkpoint, split, imgsz and scoring as F25.
Output: `runs/tiled-20261006-smallbox-20261005-yolov8s-960-kakadu-ozfish-s0/tiled.json` (it also repeats the F25 rows).

| Variant | AP50 | AP50-95 | Fish found (of 1,385) | False alarms | Recall 16-32 px |
|---|---|---|---|---|---|
| **Whole frame (control)** | **0.8167** | **0.5556** | 1,004 | **123** | 26.4% |
| 2x2, small tile boxes only | 0.8014 | 0.5493 | 1,025 | 236 | 33.0% |
| 2x2, small tile boxes, tile conf >= 0.4 | 0.8120 | 0.5537 | 1,019 | 181 | 31.1% |
| 3x3, small tile boxes only | 0.7850 | 0.5420 | 1,041 | 363 | 35.2% |
| 3x3, small tile boxes, tile conf >= 0.4 | 0.8098 | 0.5540 | 1,034 | 223 | 34.4% |

- **Restricting tiles to small boxes recovers most of the loss but not all of it.** The best variant (2x2 with the floor) is 0.005 AP50 and 0.002 AP50-95 below the whole frame, while finding 15 more fish and adding 58 false alarms. No variant is above the control on either score, and the gaps are within what a single split of near-duplicate frames can hide, so the fair reading is "no gain", not "a small loss".
- **Small fish are the part tiling helps, and it is not enough.** 16-32 px recall rises from 26% to 31-35%, but the extra fish arrive with roughly as many extra false alarms.
- **Not tried:** training at a higher resolution or on tiles (the model has never seen magnified fish at tile scale), or a tuned rule on a second split. Any rule chosen after seeing this table would be tuned on this split and would need a fresh one to confirm.
- **Limits.** One checkpoint, one split, workstation, no Orin timing; the cost is still 5 to 10 images per frame.

**Verdict.** Tiled inference, with or without a small-box merge, does not improve detection on this benchmark. It is dropped as a route to better accuracy for now.

### F27. Training at 1280 px did not raise small-fish recall or the DeepFish score; the held-out OzFish score rose slightly (one seed, a screen)
**Setup.** YOLOv8s, DeepFish + Kakadu + OzFish train data with Kakadu and OzFish re-cut at 1280 px wide (same frames, groups and split; 7,318, 622 train and 157 held-out), trained and validated at imgsz 1280 (letterbox 736x1280),
default augmentation, seed 0, `ai-gpu`, **automatic batch 3** (the 960 run used 5), early-stopped at epoch 98 with the best at epoch 68, 312 s per epoch (the 960 run: 179 s), about 8.5 h in all. Run: `20261006-yolov8s-1280-kakadu-ozfish-s0`;
analysis `runs/analysis-20261006-20261006-yolov8s-1280-kakadu-ozfish-s0/analysis.json`. Compared with the YOLOv8s 960 run of F24 (one seed, batch 5).

| | YOLOv8s 960 (F24) | **YOLOv8s 1280** |
|---|---|---|
| DeepFish val mAP@50 / mAP@50-95 | **0.8209 / 0.5624** | 0.7852 / 0.5552 |
| Fish found (of 1,385) at conf 0.25 | 1,004 | **1,009** |
| False alarms at conf 0.25 (on empty frames) | **122 (24)** | 344 (146) |
| Best-F1 confidence on val | 0.25 | 0.5 |
| Recall 16-32 px / 32-96 px / above 96 px | **26.4%** / 77.5% / 99.4% | 24.5% / **78.8%** / 99.4% |
| Missed: no prediction / found under threshold | 105 / 271 | 162 / 206 |
| Held-out OzFish mAP@50 / mAP@50-95 | 0.7242 / 0.3976 | **0.7358 / 0.4142** |

- **The hypothesis was not supported.** The reason for trying 1280 was to give small fish more pixels. Recall on 16-32 px fish did not rise (24.5% against 26.4%, 67 against 72 of 273 fish) and fish under 16 px are still never found (0 of 10). Total fish found is the same (1,009 against 1,004).
- **DeepFish mAP@50 is lower by 0.036** (mAP@50-95 lower by 0.007), which is about the size of the observed seed spread for similar arms, so this is "not better", not a measured loss. It also mixes batch size (3 against 5).
- **False alarms at the default 0.25 threshold are 2.8 times higher** (344 against 122), but the best-F1 threshold moved to 0.5, so the 1280 model is less well calibrated at 0.25 rather than clearly worse at every threshold. At its own best threshold (0.5) it finds 921 fish with 134 false alarms, which is worse than the 960 model at 0.25 (1,004 and 122).
- **Held-out OzFish is slightly higher** (+0.012 mAP@50, +0.017 mAP@50-95), about the size of the seed ranges seen for the YOLOv8n OzFish arm (0.016 and 0.013, F24); it is one seed and the 157-frame set is crowded footage that may share reefs with training deployments, so it is not evidence of a gain.
- **Cost.** Training was 1.7 times slower per epoch and about 3 times the pixels at inference; Orin latency at 1280 px was not measured (the Nano is offline).

**Verdict.** Higher training resolution is not a route to better small-fish recall on this data, at least for a single seed of this model; the detector's difficulty with 16-32 px and smaller fish is not fixed by input size alone. The 960 px YOLOv8s of F24 stays the most accurate model by DeepFish validation.
**Limits.** One seed each, different batch sizes (3 against 5), one validation split of near-duplicate frames, the DeepFish validation set has few fish under 32 px (273 in 16-32 px, 10 under 16 px), and Kakadu/OzFish frames at 1280 are re-cuts of the same source footage, not new data.

### F28. YOLOv8s costs about 9 ms (22%) more per frame than YOLOv8n on the Orin Nano: 50.1 ms against 40.8 ms end to end at 960 px (serial pipeline, 25 W)
`~/fish-detection/trt_runner.py --bench 200` on the board (same runner and method as F8), FP16 engines built on the board with `trtexec` (TensorRT 10.3.0, L4T 36.4.7) at 544x960 from the ONNX files of F24: YOLOv8s (`20261005-yolov8s-960-kakadu-ozfish-s0`) and YOLOv8n
(`20261005-yolov8n-960-kakadu-ozfish-s0`), with the original DeepFish-only baseline engine (`20260923-yolov8n-960-s2`) re-run as a control. **nvpmodel 25W, `jetson_clocks` inactive, governor `schedutil`, fan on**, junction temperature 52.0 to 52.9 C before and after
(the board had been building engines just before, so this is a warm board, not a cold one). Ambient not recorded. Median of 200 runs per case, two frames (a DeepFish frame with 2 fish and a crowded OzFish frame), serial pipeline, JPEG decode (not video).
Logs, build logs and the run record: `runs/device-20261007/`. ONNX sha256 values are in `docs/onnx_manifest.json` and were checked on the board after the transfer; engine sha256: YOLOv8n `b200633d...`, YOLOv8s `163068af...`.

| Median per frame, 544x960 FP16 | Baseline YOLOv8n (control) | New YOLOv8n | **New YOLOv8s** |
|---|---|---|---|
| JPEG decode (CPU) | 17.47 ms | 17.43 ms | 17.51 ms |
| Letterbox + normalise (CPU) | 8.03 ms | 8.04 ms | 8.07 ms |
| Copy in + GPU forward + copy out | 14.49 ms (p95 17.38) | 14.33 ms (p95 17.50) | **23.59 ms (p95 23.71)** |
| Threshold + NMS (CPU) | 0.94 ms | 0.93 ms | 0.93 ms |
| **Total, serial** (DeepFish frame) | 41.02 ms (p95 44.02) | 40.79 ms (p95 43.67) | **50.13 ms (p95 50.57)** |
| Total, serial (crowded OzFish frame) | 40.98 ms | 40.90 ms | 50.24 ms (p95 50.85) |
| Frames per second, serial | 24.4 | 24.5 | **19.9** |

- **The control reproduces F8.** The baseline YOLOv8n total is 41.0 ms against 40.8 ms in F8 (4 Oct), so today's board state matches the earlier measurement. The new YOLOv8n costs the same as the baseline (they have the same architecture).
- **Only the GPU stage changes.** YOLOv8s adds about 9.2 ms there (23.6 against 14.3 ms, +65%) and nothing elsewhere, so the total rises 22% (50.1 against 40.8 ms) and the serial frame rate falls from about 24.5 to 19.9. In F24 the same model scored +0.071 mAP@50 on DeepFish val (one seed), so on this measure the extra accuracy costs about 9 ms per frame.
- **The crowded frame costs the same** (50.2 against 50.1 ms; 3 or more detections), as in F8: NMS is not the limit.
- **p95 is tight** (50.6 to 50.9 ms), so the figure is steady at this load.

**Limits.** Serial pipeline with JPEG decode, not video: the F9 run (video decode, about 34 fps for YOLOv8n) was **not** repeated for YOLOv8s, so no sustained-load, thermal or video frame rate is claimed for it. One engine per model, one frame pair, one power mode (25 W), clocks not locked, warm board, ambient unrecorded.
**Not done at the time of F28 (run afterwards, see F30):** the 12-frame engine-against-PyTorch parity (`scripts/engine_parity.py`, F7) had not been run on these two engines, and the YOLOv8s ONNX failed the strict raw-tensor gate by 0.0006 px (see `docs/onnx_manifest.json`; detections matched exactly), so their agreement with PyTorch is **unverified on the board**; there is no on-device mAP, no INT8 and no 1280 px engine.

### F29. A 30-minute run of YOLOv8s at 960 px held about 24.4 fps with no throttling; the junction temperature plateaued at about 58 C (one run, 25 W)
`edge/sustained.py` on the board (same script and method as F9), YOLOv8s FP16 engine `20261005-yolov8s-960-kakadu-ozfish-s0-544x960-fp16.engine`, looping `sample_clip.mp4`, serial pipeline with software H.264 decode, **nvpmodel 25W**, clocks not locked, governor `schedutil`, fan on.
Started 2026-10-07T16:50:46Z, ended 17:20:46Z, 179 ten-second windows. **Ambient temperature not measured** (F9 used 26 C assumed by the user; nobody recorded one for this run). Logs: `runs/device-20261007/sustained_yolov8s_960_20261007.jsonl` and `.out`; the comparison is the F9 log in `runs/device-20261004/sustained_960.jsonl`, recomputed by the same method.

| | YOLOv8s, first 2 min | YOLOv8s, last 2 min | YOLOv8n (F9), first / last 2 min |
|---|---|---|---|
| Frames per second (mean of windows) | 24.39 | 24.37 | 34.17 / 33.99 |
| Total p50 / p95 per frame | 40.6 / 42.6 ms | 40.4 / 42.5 ms | 28.6 / 30.2 and 28.8 / 29.9 ms |
| Copy in + GPU forward + copy out (p50) | 23.5 ms | 23.5 ms | 14.2 / 14.2 ms |
| Video decode (p50) | 9.8 ms | 9.8 ms | 8.3 / 8.4 ms |
| Junction temperature (mean) | 52.1 C | 57.8 C | 51.9 / 56.5 C |

- **Throughput did not decay.** Per-window fps: min 23.02, median 24.36, max 25.27; the first and last two minutes differ by 0.1%, inside the window-to-window spread. No throttling is visible at this load and ambient.
- **Temperature climbed then flattened.** Junction 49.8 C at the start, 58.2 C maximum and at the end; slope over the last 15 min is +0.06 C/min (F9: +0.03). The fan rose from 1,511 to 2,271 RPM (maximum 2,360) to hold it.
  The end temperature is about 1.3 C above F9's, but ambient differs by an unknown amount between the two runs, so this is not attributed to the model.
- **Input power 6.47 to 7.13 W, mean 6.92 W** (F9: 6.51 to 7.09, mean 6.82); the channel mapping is assumed, as in F9.
- **The GPU clock reading is again unusable:** `gpu_mhz` read 306 in all 179 windows against a 918 maximum while the GPU was working, as in F9. No claim about GPU clocks is made; throughput flat while the temperature rose 8 C is the evidence offered.
- **Cost of the larger model on this path:** about 12 ms more per frame at the median (40.5 against 28.7 ms) and 28% fewer frames per second (24.4 against 34.1). The GPU stage accounts for 9.3 ms of that (23.5 against 14.2 ms); video decode was also 1.5 ms slower (9.8 against 8.3 ms) and mean detections per frame were higher (1.76 against 1.48), which this run does not explain.
- **Compared with F28:** the serial JPEG pipeline gave 50.1 ms (19.9 fps) for the same engine; with video decode instead of JPEG decode (about 9.8 against 17.5 ms) the same model reaches about 24.4 fps.

**Limits.** One run, one clip (the same one as F9), one engine, 25 W only, ambient not measured, software H.264 decode (the board's NVDEC is not used), light duty cycle, serial pipeline, no engine-against-PyTorch parity check yet for this engine (F28), and run-to-run spread unknown. A heavier load, MAXN_SUPER or a warmer room may behave differently.

### F30. On the Orin, the YOLOv8s engine agrees with PyTorch on all 12 frames; the new YOLOv8n engine matches every detection but fails the IoU gate on one tiny box (12 frames, FP16)
`edge/engine_dump.py` on the board (`trtexec`, FP16 engines of F28) fed the **same 12 frames** the PyTorch reference used (4 OzFish and 8 DeepFish frames, empty ones included; the preprocessed inputs were rebuilt on the board from a lossless uint8 copy, checked bit-exact against the
workstation tensors before upload), and `scripts/engine_parity.py` compared the raw outputs and the detections after NMS (conf 0.25, IoU 0.7) with PyTorch, the same method as F7. Results: `runs/device-20261007/engine_parity_fp16_*.json`; the gate is set in `src/fish/parity.py`
(no confident miss or extra, worst matched IoU at least 0.9, worst confidence drift at most 0.05).

| | New YOLOv8n engine | **YOLOv8s engine** |
|---|---|---|
| Detections, PyTorch / engine (12 frames) | identical counts on every frame | identical counts on every frame |
| Confident misses / extras / borderline | 0 / 0 / 0 | 0 / 0 / 0 |
| Worst matched box IoU | **0.708 (gate fails)** | 0.990 |
| Worst confidence drift | 0.045 | 0.007 |
| Largest raw-tensor difference (box pixels) | 6.98 | 21.70 (an empty frame) |
| Gate verdict | **FAIL** (IoU 0.708 below 0.9) | PASS |

- **YOLOv8s agrees with PyTorch on the board.** Every detection matches (including 63 on the most crowded frame), the worst IoU is 0.990 and confidence drifts by at most 0.007. Its ONNX export had narrowly missed the strict raw-tensor gate (F28 / `docs/onnx_manifest.json`, by 0.0006 px), but the engine passes the detection-level gate that matters here.
- **The new YOLOv8n engine fails the gate on one box.** All detection counts match and there are no misses or extras. The single failure is on the crowded frame `A000007_L-avi-74577` (74 detections), where one box 9.4x8.7 px, confidence 0.325 in PyTorch and 0.281 in the engine, matches with IoU 0.708. The other 73 matched pairs on that frame have a median IoU 0.997 and a 5th percentile of 0.992; on the other 11 frames the worst IoU is 0.965 or better.
  A shift of about one pixel is enough to take a box that small from 1.0 to about 0.7, so this is consistent with FP16 jitter on a tiny, low-confidence box in a crowd (three engine boxes overlap it), but that explanation was not verified against the weights. **The gate was not relaxed**: by the rule fixed in advance this engine did not pass.
- **Raw differences are larger than in F7** (7 and 22 px against 3 and 15 px for the older engines), which F7 already noted occur mostly on frames with few or no detections (the 21.7 px case is a frame with none), and detection-level agreement is what is judged.

**Limits.** 12 frames, one engine per model, FP16 only; agreement with PyTorch on a small sample, **not** an mAP measurement on the device. The uint8 rebuild of the inputs was verified exact on the workstation but the board-side tensors were not checksummed.

### F31. MAXN_SUPER did not speed up the network on the Orin Nano: the end-to-end gain (4 to 5 ms) came from faster JPEG decode, and the video frame rate did not change (10-minute runs)
Same engines, runner and method as F28 and F29 (YOLOv8n `20261005-yolov8n-960-kakadu-ozfish-s0` and YOLOv8s `20261005-yolov8s-960-kakadu-ozfish-s0`, FP16, 544x960), with the board set to **nvpmodel MAXN_SUPER (mode id 2)** by `maxn_super.sh` on the board, `jetson_clocks` inactive, governor `schedutil`, fan on, and **restored to 25 W afterwards**
(the status file records `RESTORED mode=1` at 18:38:41Z, and the board was confirmed at 25 W again at check time). Run 2026-10-07 18:17 to 18:38 UTC; board junction temperature 49.0 to 50.4 C at the start of each timing case. Ambient not measured. Median of 200 runs per frame, serial, JPEG decode; then a 10-minute loop of `sample_clip.mp4` (software H.264 decode) per model.
Logs: `runs/device-20261007/e2e_maxn_super_*`, `sustained_maxn_super_{n,s}.jsonl` and `maxn_super_status.txt`.

| Median per frame (DeepFish frame) | YOLOv8n, 25 W (F28) | **YOLOv8n, MAXN_SUPER** | YOLOv8s, 25 W (F28) | **YOLOv8s, MAXN_SUPER** |
|---|---|---|---|---|
| JPEG decode (CPU) | 17.43 ms | **13.25 ms** | 17.51 ms | **13.20 ms** |
| Letterbox + normalise (CPU) | 8.04 ms | 8.00 ms | 8.07 ms | 8.03 ms |
| Copy in + GPU forward + copy out | 14.33 ms | **14.33 ms** | 23.59 ms | **23.54 ms** |
| Threshold + NMS (CPU) | 0.93 ms | 0.93 ms | 0.93 ms | 0.92 ms |
| **Total, serial** | 40.79 ms | **36.54 ms (p95 39.45)** | 50.13 ms | **45.73 ms (p95 46.37)** |
| Frames per second, serial | 24.5 | 27.4 | 19.9 | 21.9 |

The crowded OzFish frame gave the same picture (YOLOv8n 35.97 ms, YOLOv8s 46.08 ms at MAXN_SUPER).

| 10-minute video loop (59 windows) | YOLOv8n, MAXN_SUPER | YOLOv8s, MAXN_SUPER | 30-minute runs at 25 W (F9 / F29) |
|---|---|---|---|
| Frames per second, first / last 2 min | 34.6 / 34.5 | 24.4 / 24.4 | 34.2 / 34.0 and 24.4 / 24.4 |
| Total p50 per frame | 28.4 to 28.5 ms | 40.5 to 40.4 ms | 28.6 and 40.5 ms |
| Copy + GPU p50 / video decode p50 | 14.1 / 8.2 ms | 23.4 / 9.9 ms | 14.2 / 8.3 and 23.5 / 9.8 ms |
| Junction temperature | 51.8 to 56.7 C | 56.3 to 57.4 C (started warm after the first run) | plateau 56.5 and 57.8 C |
| Fan RPM / input power | 1,561 to 2,174 / 6.5 to 7.3 W | 2,174 to 2,187 / 6.7 to 7.1 W (mean 6.94) | similar |

- **The GPU stage did not get faster** (14.3 ms for YOLOv8n and 23.5 ms for YOLOv8s at both modes), so MAXN_SUPER does nothing for the network at this load. The earlier unmeasured assumption that the mode swing would be large is not supported for this pipeline.
- **The serial end-to-end gain is all CPU:** JPEG decode fell by about 4.2 ms (17.5 to 13.2 ms), which accounts for the total dropping from 40.8 to 36.5 ms (YOLOv8n) and from 50.1 to 45.7 ms (YOLOv8s), about 10% and 9%. The CPU's top speed in the log rose from 1,344 to 1,728 MHz; the board's reported GPU maximum rose from 918 to 1,020 MHz, but the log's live GPU clock read 306 MHz in every window again (see F9), so **no claim about GPU clocks is made**.
- **Video frame rate did not change** (34.5 and 24.4 fps against 34.1 and 24.4 at 25 W), because the video path's decode (about 8 to 10 ms) is not the part that sped up.
- **No throttling visible; temperature and power are about the same** as at 25 W in these 10-minute runs (about 57 C, about 7 W), so on this evidence the faster mode costs little heat at this duty cycle. Whether that holds for a longer run or a warmer room is untested.

**Limits.** 10-minute loops (F9 and F29 were 30 minutes), one run per case, ambient not measured, `jetson_clocks` inactive (clocks not locked), warm board for the second sustained run, one clip, serial pipeline, software decode, no NVDEC. Only mode id 2 was tried; the 15 W and 7 W modes were not measured.

---

## 6. Corrections — claims that were wrong and what replaced them

Kept deliberately. A correction deleted is a mistake someone repeats.

| Earlier claim | Reality | Found by |
|---|---|---|
| OzFish boxes span 507 species; collapse to one class | The ~45k boxes are fish/no-fish only. 507 species is the ~80k *crops*. | Reading the OzFish README, line 57 |
| "Pawsey URLs confirmed dead" | Unsupported. Every path returns 200 — so does a nonsense path (JS portal). curl proves neither. | Testing a nonsense path |
| HuggingFace "mirror" has the OzFish data | It is a 3-file *model* repo (one 74.7 MB `.pt`). | Querying the HF API |
| DeepFish has no usable boxes | A box-labelled darknet export exists (6,517 images). | Trying the YOLO-Fish Drive IDs |
| Power mode changes FPS 2–3x | Inherited from a memo; 4GB modes are 7W/10W. Must be measured on the board. | Review |
| INT8 is required on a 4GB Orin | False; YOLOv8n is ~6 MB in FP16. INT8 is a research question, not a necessity. | Review |
| A single frame showed heavy colour dependence (confidence 0.71 → 0.03 in greyscale) | Full-set result is a 7–9% dip for greyscale. One frame sat just above threshold. | Running the full experiment |
| Pixel buckets "under 16 px" | Audit figures are at *model input*; first bucketing used native pixels (3x off at 640). | Frame selection returned zero matches |
| "Best epoch" from a mAP@50 scan | `best.pt` is selected by fitness; e.g. 960 seed 0 best is epoch 89, not 18. Metrics were right; the epoch attribution was not. | Cross-checking against fitness |
| Two seeds looked tight (spread 0.008) | Third seed landed 0.055 above the first. Two seeds hid the true spread. | Running seed 2 |
| The board has no cooling fan (owner-reported; then assumed throughout `CLAUDE.md` and `docs/hardware.md`) | A fan is fitted and spinning: tachometer 1,411 RPM, PWM 68/255. The dev kit ships with one. | Reading the fan tach over sysfs |
| `fan_interface_present` implies a fan | It only shows the kernel driver exists. The tachometer is the evidence. | Checking the tach after the probe contradicted the owner |
| The 960 px engine costs 7.1 ms on the Orin | That engine was built for a 960x960 square. At the evaluated shape (544x960) the measured GPU time is 4.37 ms. | Reading the engine's input binding in the build log |
| The board log dated 14 May 2026 was a May measurement | The model was trained in September, so the board's clock was stale when it was written. NTP is now synchronised and new logs carry the right date. Trust hashes, not timestamps. | The date predating the model |
| The model fits comfortably: ~7 ms of GPU against ~14 ms per frame at 60 fps (on-board report) | The 60 fps budget is 16.7 ms, not 14. More importantly the GPU kernel is only a small part of a frame: the serial end-to-end pipeline is 40.8 ms (24.5 fps) at 960, so 60 fps is not reachable serially, and "leaving the CPU free" was an untested hypothesis that the CPU is in fact the bottleneck. | Building the end-to-end runner |

---

## 7. Bugs found and fixed

| Bug | Effect | Caught by |
|---|---|---|
| `configs/data/*.yaml` written with an absolute drive path | Every teammate's config pointed at one machine | `test_no_absolute_paths` |
| `.gitignore` `data/` matched `configs/data/` | Dataset configs silently excluded from the repo | Reviewing staged files |
| `runs/_kaggle` and `runs/_supervisor` committed | Clone shipped old failures; reports showed phantom failures | Reading a failed Kaggle report |
| `data/processed/` on the GPU server was emptied while a run was training (`deepfish`, `ozfish_eval` and the converted Kakadu set gone, loose unzipped PNGs left in their place) | The YOLOv8s run crashed at epoch 50 with `Image Not Found`, and the queued YOLOv8m and YOLO11s runs failed at once. Someone was working on the server at the time; who and why is not known. Recovered with `scripts/get_data.py` from `data/raw` and `data/interim` (no download): 5,178 train and 1,339 val images, 352 OzFish test images, and the DeepFish split identical to the workstation copy (same site groups and statistics). Kakadu was converted again from its zip. | A training log ending in `FileNotFoundError`, then `ls data/processed` |
| `scripts/archive_paths.py` named its archive folder by the second | Two archive commands run in the same second shared one folder and the second manifest overwrote the first; the files were safe but the undo record of the first move was lost. Found by reading the printed manifest paths, fixed (a counter suffix, a test), and the lost manifest rebuilt from the archived files (`MANIFEST_loose_copies.json`, 398 items). | Output of the second and third archive commands |
| `scripts/export_onnx.py` crashed on any checkpoint not named like `yolov8n-960-s2` | Its name pattern `-(\d+)-s(\d+)` did not match `yolov8s-960-kakadu-ozfish-s0`, so the first export attempt of the new models died with an AttributeError (nothing was written). Fixed with a pattern that allows extra name parts. | The traceback on the first export |
| The first Nano timing run of 2026-10-07 produced nothing | `trt_runner.py` was started without the board's `pylibs` folder on `PYTHONPATH`, so `cuda-python` was not found (`No module named 'cuda'`), and the redirect `>` overwrote the board's copies of the earlier 960 px e2e logs with the traceback. No result was reported from it; the repo copies of the original logs (`runs/device-20261004/`) are intact, and the rerun writes date-stamped files (`e2e_20261007_*`). | The `Traceback` in every output log |
| `bootstrap.py` chose the `cu124` wheel on a CUDA 13 driver | RTX 5060 (compute capability 12.0) needs cu128; cu124 would fail only at the first CUDA op. The driver header said `CUDA UMD Version`, which the parser did not read, so it fell back to cu124. | Reading the dry-run output on the new server |
| `import cv2` fails on the remote server (`libxcb.so.1`) | The headless container has no display libraries, so `opencv-python` cannot load and `check_env.py` reported ultralytics missing. Fixed there by swapping in `opencv-python-headless` (no sudo needed). Not yet automated in `bootstrap.py`. | `check_env.py --require-gpu` |
| `bootstrap.py` refused to install outside a venv | Kaggle run died on `import ultralytics` after downloading data | Reading the supervisor log |
| Inherited-run filter matched on name only | A real smoke result was dropped as "inherited" | Empty metrics on a finished session |
| `train.py` indexed optional config keys directly | Missing `patience` raised a KeyError traceback | Testing the failure path |
| Supervisor echo loop raised `UnicodeEncodeError` on cp1252 | Would kill an overnight run over cosmetics | First supervised run |
| `cv2.VideoWriter` `mp4v` output | Saved fine, would not play in browsers | User report |
| Training results lived only in git-ignored scratch | Reported numbers had no committed provenance | Asked where results were saved |
| A personal file swept in by `git add -A` | Private document pushed to a team repo (removed from HEAD; still in history) | Noticing the filename |
| `probe_board.py` returned a command's error text as if it were a value | `libnvinfer` reported "no packages found" and the fallback to `libnvinfer10` never ran | First run on the real board |
| Parity results held numpy `float32` values | A passing check printed PASS, then crashed writing its result, leaving no record | The first real engine run |
| `trtexec --warmUp=0` intermittently exports no output | Looked like per-frame failures (1 of 12, then 10 of 12, succeeded); the same frame worked on re-run | Re-running a "failing" frame by hand |
| Parity-set label counted OzFish frames by an `A0` filename prefix | Reported 1 OzFish frame when 4 were used (OzFish names also start with E and G) | Listing the actual frames |
| Aug "looks barely better" (said in conversation: +0.04 mAP@50, +0.012 mAP@50-95 against the Kaggle baseline mean) | Against the same-machine control it is +0.087 and +0.033: the server control scores 0.031 below the Kaggle run at the same seed, so the Kaggle baseline was the wrong comparator. Still one seed and a bundle. | The control run finishing (F14) |
| F12 compared the P2 head with the Kaggle baseline and called the batch difference "5 against 23" | The same-machine control used batch 10, and P2 used batch 5 (the automatic batch for `yolov8n-p2` differs from `yolov8n`). Against the control, P2 is +0.051 mAP@50 and -0.013 mAP@50-95, with 16-32 px recall 3.7% against 2.6%: mixed, and confounded by batch size, so "no small-fish gain" is weaker than F12 stated. | Reading the control's `args.yaml` (F14) |
| F12, F14 and section 8 gave the Kaggle runs' Ultralytics version as 8.4.130 | That is the workstation's version. The Kaggle runs' `env.txt` says **8.4.159** (torch 2.10.0+cu128, Tesla T4). The server runs use 8.4.172. The difference in library versions between Kaggle and the server is therefore 13 releases, not 42, which does not change any conclusion (the same-machine control is what the comparisons rest on). | Reading `runs/20260922-yolov8n-960-s0/env.txt` while writing the protocol rules |
| `docs/hardware.md` said the board's firewall drops everything but SSH and documented the app as localhost-only | The code default is `--host 0.0.0.0`. The timeout was real at the time; `ufw` now lists allow rules for 7860 and 7861 (not added by this work), and direct access works. | A user could not reach the app on the board's address; `ufw status` |
| `fish.parity.nms()` modified the array it was given | Ultralytics' NMS converts boxes xywh to xyxy in place and `torch.from_numpy` shares memory, so the caller's raw output was silently overwritten and a second decoder converted it twice (boxes like `[167, 52, 628, 248]`). Fixed with a copy and a regression test. | Cross-checking the edge decoder against it |
| `cuda-python` pinned at 12.6.2 | That release was yanked by its maintainers. Replaced by 12.6.2.post1. | pip's yanked-version warning |
| `pkill -f "python3 app.py"` run over SSH | It matched the SSH session's own command line and killed it (exit 255). The bracket trick fails too when the pattern also appears elsewhere in the command. Kill by PID found from the port owner, never by name: NVIDIA's system services also run as `python3`. | Two failed launches |

---

## 8. Not done / open

- **Device measurement is only partly done.** GPU kernel time is measured for both engines (F6) and parity holds on 12 frames (F7). End-to-end latency is measured for a serial pipeline (F8). Sustained 30-min throughput and thermals are measured once at 960 / 25 W (F9); still open: other power modes, a repeat, a warmer ambient, an explained GPU-clock reading, **hardware video decode (NVDEC)**, memory under a real pipeline, any **overlap or pinned-memory optimisation** of the CPU stages that dominate, INT8, and **mAP on the device**. Engines exist only for seed 2, and the board boots from an SD card with about 8 GB free.
- **No false-alarm rate on unseen footage.** OzFish has zero empty frames; DeepFish val (31% empty) is the only place it can be measured, and it has not been reported yet.
- **No annotation-quality audit.** OzFish boxes came from crowd annotation; an unlabelled fish scores as a false positive.
- **No hue-augmentation retraining** (F4 hypothesis).
- **FD-129 screens on `ai-gpu`** (started 2026-10-04): one seed (0) each of `yolov8n-960-p2`, `yolov8n-960-aug` and a
  same-machine control, all done (F12, F14); the aug gain and the YOLOv8s gain need 3 seeds before they count. About 65 s per epoch for the P2 run on
  the RTX 5060 at batch 5 (the Kaggle T4 runs used batch 23 and about 115 s per epoch), so the control rerun exists to
  keep the comparison fair across machine, Ultralytics version (8.4.172 here, 8.4.159 on Kaggle) and batch size. One
  seed cannot decide anything: the 960 baseline seed spread is 0.034 mAP@50, so only a difference clearly larger than
  that justifies 3 seeds. Logs are on the server (`~/workspace/train_*.log`) until copied into `runs/`.
- **OzFish full training set: the route is real, and it is smaller than assumed (2026-10-04).** The official AIMS
  catalogue record (DOI 10.25845/5e28f062c5097, licence Creative Commons Attribution Australia, read from the public
  catalogue index) says the boxes are "~45k bounding box annotations ... of fish/no fish across **1800 frames**", about
  25 boxes per frame. So it adds crowded scenes, not many images: against our 6,517 DeepFish images it is 1,800 frames
  (the sealed test set's 352 frames are probably part of them, which must be checked before any training on it, by
  video and not only by file name, because frames from one video are near-duplicates). The Pawsey store is a real
  file portal (seen in a browser by the user): `frames.zip` 106 GB (out of scope), `labelled-zip.zip` 2.91 GB (likely
  the labelled frames, unverified), `crops-zip.zip` 3.18 GB (species crops, not needed). Direct `curl` to the zips
  returns 404, so the download link has to come from the browser. Attribution to use:
  Australian Institute of Marine Science, University of Western Australia and Curtin University (2019), OzFish Dataset,
  https://doi.org/10.25845/5e28f062c5097. An earlier chat estimate of "about 7x the images" was wrong; it was never in
  these docs.
- **Other open fish-detection datasets found (2026-10-04, not downloaded, figures from a curated third-party list and
  the Zenodo page, to be verified at the source).** No third-party mirror of OzFish exists on HuggingFace, Kaggle or
  Zenodo (searched). Candidates with direct downloads and open licences: **Kakadu** (Zenodo 7250921; 44,112 optical
  1920x1080 frames from remote underwater video in northern Australia, 82,904 boxes, 23 species to be collapsed to one
  class, COCO format, CC BY 4.0, training zip 5.9 GB); **NOAA Puget Sound Nearshore Fish** (LILA; 77,739 images,
  67,990 boxes, CDLA-permissive 1.0, so many empty frames: a source of hard negatives and of a false-alarm test);
  **PomerFish** (Zenodo, CC BY 4.0), **OBSEA** (Zenodo, CC BY 4.0), **Brackish** (Kaggle, CC BY-SA 4.0, share-alike).
  Excluded by licence: TORSI (CC BY-NC-SA). Whatever is used needs the same site/video-level split and leakage check as
  DeepFish, because all of it is video frames.
- **Kakadu on the server (2026-10-04):** the full archive reached `ai-gpu` by someone copying it there; `prepare_kakadu.py` read all 44,112 members with a CRC check (so the copy is intact), keeping 7,318 frames and 11,695 boxes (max 15 per group, 960 px wide). The workstation copy of the zip was deleted afterwards, as asked. The server's own download (about 45 KB/s) and the workstation upload (about 60 KB/s) were abandoned. Next: the `yolov8n_960_kakadu` comparison, in the order given in `docs/OVERNIGHT.md`.
- (superseded, kept for the record) **Kakadu was downloading to `ai-gpu`** (about 0.4 MB/s from that host, roughly 4 hours, started 2026-10-04; the PC
  copy exists and is verified). Uploading from the PC measured 0.09 MB/s, so the server fetches it itself. After it is
  verified there: run `prepare_kakadu.py`, then the `yolov8n_960_kakadu` screen. The PC copy is to be deleted once the
  server copy's md5 matches (user's instruction).
- **Label audit is one reviewer's sample** (F11: 48 missed, 24 found, 36 false alarms). A second reviewer, and a larger sample outside site 7398, would firm it up; it is not a blocker for the small-fish experiments.
- **Error analysis covers one seed per size and DeepFish validation only.** It has not been run on the OzFish set (sealed; would need a decision about analysing without tuning).
- **No ONNX / TensorRT export or parity check.**
- **No trivial baselines** (predict-nothing, zero-shot COCO, frame differencing).
- **Licence position undecided.** Ultralytics and the COCO-pretrained weights are AGPL-3.0; weights are published
  publicly alongside the code, which is the compliant posture, but a closed device would not be.
- **`report_sections.md` is stale** — still the LCFCN counting experiment.
- **One personal file remains in git history** (commit `b56ca8d`). Purging needs a history rewrite and force-push
  on a shared repo; owner's decision.

---

## 9. Where things live

| What | Where |
|---|---|
| Run metrics, configs, epoch curves | `runs/<run>/` (tracked) |
| Board facts | `docs/hardware.md`, raw probe output `docs/board_probe_20261004.json` |
| Engine provenance, build and benchmark logs, parity results | `runs/device-20261004/` (`engine_registry.json` links each engine to its ONNX, checkpoint, TensorRT and L4T) |
| ONNX export manifest | `docs/onnx_manifest.json` |
| End-to-end per-stage timing logs | `runs/device-20261004/e2e_*.log` |
| 30-minute sustained run (F9) | `runs/device-20261004/sustained_960.jsonl` |
| Label audit (F11) | `runs/audit-labels-20260923-yolov8n-960-s2/` (`summary.json`, `verdicts.json`, sheets) |
| Error analysis (F10) | `runs/analysis-20261004-*/analysis.json` and the worst-frame images beside it |
| Weights | GitHub release `v0.1.0-models` (not in git) |
| Checksums | `docs/model_manifest.json` |
| Evaluation contract | `docs/eval_protocol.md` |
| Research narrative, methodology, next steps | `docs/RESEARCH.md` |
| Report draft and reference tracker | `docs/report/` |
| Plain-language guide to the concepts and mechanisms (for the owner) | `docs/GUIDE.md` |
| Dataset statistics | `docs/data_audit.md` |
| Setup and training | `README.md`, `docs/TRAINING.md` |
| Tickets | `docs/tickets/BACKLOG.md` |
| Training board / comparison bench / field unit | claude.ai artifacts (private until shared) |
