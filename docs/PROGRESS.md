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
| **Jetson Orin Nano** | **Not acquired. No device measurement exists.** The whole deployment half of the thesis is unanswered. |
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

### Safety net
- 104 pytest checks: frame grouping, label parsing, no absolute paths, no hardcoded `cuda`,
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

---

## 8. Not done / open

- **No Jetson measurements.** FPS, latency, power mode, sustained throughput, memory — all unmeasured.
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
| Weights | GitHub release `v0.1.0-models` (not in git) |
| Checksums | `docs/model_manifest.json` |
| Evaluation contract | `docs/eval_protocol.md` |
| Dataset statistics | `docs/data_audit.md` |
| Setup and training | `README.md`, `docs/TRAINING.md` |
| Tickets | `docs/tickets/BACKLOG.md` |
| Training board / comparison bench / field unit | claude.ai artifacts (private until shared) |
