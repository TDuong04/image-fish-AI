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
| Data | **In hand.** DeepFish (train/val) and an OzFish test set (sealed). |
| Training | **Done for the planned grid.** 6 converged runs: 3 seeds x {640, 960} px. |
| Evaluation | In-domain, cross-dataset (sealed OzFish), size-stratified recall, and a colour-perturbation study. |
| Weights | Published as GitHub release `v0.1.0-models`, fetched and checksum-verified by `scripts/fetch_models.py`. |
| Demos | Three shareable pages and a runnable Gradio app. |
| Automation | Crash-recovering supervisor and a Kaggle session runner. |
| **Jetson Orin Nano** | **In hand and probed:** 8 GB Super dev kit, L4T 36.4.7, TensorRT 10.3.0, active fan, SD-card boot with ~8 GB free (`docs/hardware.md`). **Two FP16 engines built on the board at the shapes the accuracy was evaluated at; both match PyTorch on real frames, and GPU kernel time is measured (2.56 ms at 384x640, 4.37 ms at 544x960, a 20 s burst).** **End to end, a serial pipeline takes about 33 ms per frame at 640 and 41 ms at 960 (about 30 and 24 fps), and a web app runs on the board** (`edge/app.py`). **A 30-minute sustained run held about 34 fps at 960 with no throttling (F9, one run, 25 W mode, ambient 26 C assumed).** Not yet measured: hardware video decode, INT8, on-device mAP, other power modes. The deployment half of the thesis is only partly answered. |
| Report | `report_sections.md` still describes the older LCFCN *counting* experiment, not this work. Needs rewriting. |

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
`runs/20261004-yolov8n-960-p2-s0`. Compared with the Kaggle baseline `20260922-yolov8n-960-s0` (T4, Ultralytics 8.4.130,
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

**Limits.** Grouping validated by eye on 3 pairs and by distance statistics, not exhaustively. 23 species merged to
one class without checking whether any species labels cover non-fish objects.

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
- **FD-129 screens on `ai-gpu`** (started 2026-10-04): one seed (0) each of `yolov8n-960-p2` (done, F12), then
  `yolov8n-960-aug` (running) and a control rerun of `yolov8n-960` on the same machine (queued). About 65 s per epoch for the P2 run on
  the RTX 5060 at batch 5 (the Kaggle T4 runs used batch 23 and about 115 s per epoch), so the control rerun exists to
  keep the comparison fair across machine, Ultralytics version (8.4.172 here, 8.4.130 on Kaggle) and batch size. One
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
| Dataset statistics | `docs/data_audit.md` |
| Setup and training | `README.md`, `docs/TRAINING.md` |
| Tickets | `docs/tickets/BACKLOG.md` |
| Training board / comparison bench / field unit | claude.ai artifacts (private until shared) |
