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
| **Jetson Orin Nano** | **In hand and probed:** 8 GB Super dev kit, L4T 36.4.7, TensorRT 10.3.0, active fan, SD-card boot with ~8 GB free (`docs/hardware.md`). **Two FP16 engines built on the board at the shapes the accuracy was evaluated at; both match PyTorch on real frames, and GPU kernel time is measured (2.56 ms at 384x640, 4.37 ms at 544x960, a 20 s burst).** **End to end, a serial pipeline takes about 33 ms per frame at 640 and 41 ms at 960 (about 30 and 24 fps), and a web app runs on the board** (`edge/app.py`). Not yet measured: sustained throughput, hardware video decode, INT8, on-device mAP. The deployment half of the thesis is only partly answered. |
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
  timing. Reached through an SSH tunnel; setup is in `docs/hardware.md`.

### Safety net
- 161 pytest checks: frame grouping, label parsing, no absolute paths, no hardcoded `cuda`,
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
49 to 50 C because the GPU is idle most of each frame, so this is **not** a sustained-load test. Both engines are
from seed 2.

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
| `fish.parity.nms()` modified the array it was given | Ultralytics' NMS converts boxes xywh to xyxy in place and `torch.from_numpy` shares memory, so the caller's raw output was silently overwritten and a second decoder converted it twice (boxes like `[167, 52, 628, 248]`). Fixed with a copy and a regression test. | Cross-checking the edge decoder against it |
| `cuda-python` pinned at 12.6.2 | That release was yanked by its maintainers. Replaced by 12.6.2.post1. | pip's yanked-version warning |
| `pkill -f "python3 app.py"` run over SSH | It matched the SSH session's own command line and killed it (exit 255). The bracket trick fails too when the pattern also appears elsewhere in the command. Kill by PID found from the port owner, never by name: NVIDIA's system services also run as `python3`. | Two failed launches |

---

## 8. Not done / open

- **Device measurement is only partly done.** GPU kernel time is measured for both engines (F6) and parity holds on 12 frames (F7). End-to-end latency is measured for a serial pipeline (F8). Still unmeasured: **sustained throughput and thermals** over 30+ minutes, **hardware video decode (NVDEC)**, memory under a real pipeline, any **overlap or pinned-memory optimisation** of the CPU stages that dominate, INT8, and **mAP on the device**. Engines exist only for seed 2, and the board boots from an SD card with about 8 GB free.
- **No false-alarm rate on unseen footage.** OzFish has zero empty frames; DeepFish val (31% empty) is the only place it can be measured, and it has not been reported yet.
- **No annotation-quality audit.** OzFish boxes came from crowd annotation; an unlabelled fish scores as a false positive.
- **No hue-augmentation retraining** (F4 hypothesis).
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
| Weights | GitHub release `v0.1.0-models` (not in git) |
| Checksums | `docs/model_manifest.json` |
| Evaluation contract | `docs/eval_protocol.md` |
| Dataset statistics | `docs/data_audit.md` |
| Setup and training | `README.md`, `docs/TRAINING.md` |
| Tickets | `docs/tickets/BACKLOG.md` |
| Training board / comparison bench / field unit | claude.ai artifacts (private until shared) |
