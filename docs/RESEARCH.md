# Research and methodology

The narrative of the work: what question was asked, how it was answered, in what order, and what the answers are.
It is a map, not the record. **Every number here lives, with its run folder and caveats, in `docs/PROGRESS.md`
(findings `F#`)**; the measurement rules live in `docs/eval_protocol.md`. If this file and those disagree, those win.
Written 2026-10-06 from the state at commit `5727068` and its successors.

## 1. Question

Can a small single-class fish detector (bounding boxes, underwater and BRUVS footage) be made accurate enough to be
useful and still run on a Jetson Orin Nano 8 GB Super, with the performance claims **measured on that board** and not
extrapolated from a desktop?

Two sub-questions came out of the first results (YOLOv8n at 640/960 px on DeepFish: mAP@50 0.42 / 0.50, see F1):

1. Why are the scores low, and what raises them?
2. What does the accuracy cost on the device?

## 2. Data

| Source | Use | Notes |
|---|---|---|
| DeepFish (darknet export, 6,517 images) | train (14 sites) and validation (6 sites), split by **site** | the benchmark every run is scored on |
| Kakadu (Zenodo 7250921, CC BY 4.0) | extra **training** data only (7,318 frames kept, 11,695 boxes) | cannot be split without leaking (F13) |
| OzFish (AIMS, CC BY 3.0 AU) | 622 train / 157 held-out frames from the labelled set; the 352-frame test set is **sealed** | the test frames overlap the labelled set, so training on OzFish changes what the sealed test measures (F21 to F24) |
| CFC (sonar) | not used | different modality |

Details and counts: `docs/data_audit.md`.

## 3. Methodology

**Model and training.** Ultralytics YOLOv8n (also s and m, YOLO11s), COCO-pretrained, one class, 960 px (letterbox
544x960), early stopping with patience 30 and a 150-epoch cap, checkpoint chosen by Ultralytics' fitness
(0.1·mAP@50 + 0.9·mAP@50-95). Batch auto-scaled to the GPU. Configs under `configs/train/`, data under `configs/data/`.

**Splits by site, never by image.** Frames come from continuous video; a per-image split puts near-duplicates on both
sides. Splits are over whole capture sites, checked by assertion. For Kakadu, which has no site labels, frames were
grouped by thumbnail distance (48x27 mean-removed grayscale, L1, single linkage, threshold 4.0, 2,296 groups). A leak
check was required before any split was trusted. The cost of getting this wrong was measured: on a random per-image
split the same pipeline scores 0.98 mAP@50 against about 0.47 on the site split (F20), which is why our numbers sit far
below published DeepFish figures. Published splits were not verified, so no like-for-like comparison is claimed.

**Same-machine controls.** Machine, batch size and library version moved the same-seed score by about 0.031 mAP@50
(F12, F14). Every comparison therefore uses a control trained on the same machine at the same batch, not a number from
another machine.

**Seeds.** Seed spread was measured, not assumed (control range 0.041 mAP@50). One seed is a screen: it nominates a
change for three seeds and decides nothing. A claim needs at least three seeds, reported as mean and range, with a
difference larger than the range.

**Error analysis** (`scripts/error_analysis.py`) at IoU 0.5 and confidence 0.25: fish found and missed, false alarms
by cause and on empty frames, and recall by size bucket (at model-input pixels). The best-F1 threshold is reported
because it moves between seeds (F24), so an operating threshold is chosen on validation per model.

**Label audit** (F11): one reviewer checked a sample of misses and false alarms. Most "misses" were real visible
fish, so the labels are mostly sound.

**Sealed test.** The OzFish test set is not used for tuning or model choice. Cross-dataset checks use held-out OzFish
deployments instead (`scripts/eval_ozfish_val.py`, which refuses the sealed set). The sealed set is run once, on the
final model, when the owner decides.

**Device measurement** (earlier phase, F5 to F9). Engines are built on the Orin from ONNX (TensorRT engines are not
portable), at the shape the accuracy was evaluated at. Latency is end to end (decode, letterbox, inference, NMS), warm-up
discarded, median and p95, every figure tagged with its power mode and clock state. A kernel-time benchmark is not
called a frame rate.

**Documentation and reproducibility.** Seeds, configs, git SHA and environment are logged per run
(`runs/<date>-<slug>/`); `tests/test_docs.py` requires every run folder and script to be named in `docs/PROGRESS.md`;
corrections are kept in a table (§6 there), not overwritten.

## 4. Order of work and results

| Step | What was asked | Answer (finding) |
|---|---|---|
| Baseline grid | 640 vs 960 px, 3 seeds each | 960 beats 640 by +0.087 mAP@50 in-domain (F1), but only +0.013 on OzFish (F2); gain is in medium fish (F3); colour loss tolerated, wrong colour not (F4) |
| Device | does it run on the Orin | FP16 engines match PyTorch (F7); end to end about 41 ms/frame at 960; the GPU kernel is under 11% of a frame (F8); 30 min sustained about 34 fps, no throttling (F9, one run) |
| Why so low | where do errors come from | missing fish, not wrong boxes or the threshold (F10); labels mostly sound (F11); empty frames explain under 0.03 mAP@50 (F19); the split protocol explains the gap to published scores (F20) |
| Architecture | stride-4 P2 head for small fish | did not help (F12, one seed) |
| Augmentation, model size | stronger augmentation; s, m, YOLO11s | augmentation and size help in-domain on one seed (F14 to F16); later shown to add nothing once data is added (F23) |
| More data | Kakadu | the biggest lever: 0.698 vs 0.473 mAP@50, 3 seeds each (F17, F18); holds on unseen OzFish deployments (F21) |
| OzFish data | add the labelled OzFish frames | +0.052 mAP@50 over Kakadu-only on three seeds (F22 to F24); held-out OzFish 0.689 vs 0.357 |
| Model size on combined data | YOLOv8s | best accuracy so far: 0.821 mAP@50, 0.562 mAP@50-95, one seed, batch 5 (F24); Orin latency unmeasured |
| Device cost of the bigger model | time YOLOv8s on the Orin | 50.1 ms per frame against 40.8 ms for YOLOv8n (19.9 against 24.5 fps serial, 25 W, JPEG decode); only the GPU stage grows (F28); a 30-minute video run held 24.4 fps against 34 fps for YOLOv8n, with no throttling and a plateau near 58 C (F29) |
| Higher training resolution | YOLOv8s at 1280 px, data re-cut at 1280 | no gain in small-fish recall (24.5% against 26.4%) and DeepFish mAP@50 0.785 against 0.821; held-out OzFish 0.736 against 0.724; one seed, batch 3 (F27) |
| Small fish at inference | tile the frame and merge | finds +8% fish and lifts 16-32 px recall from 26% to 37-39%, but false alarms rise 3.7-5x and AP50 falls 0.817 to 0.78-0.79, at 5-10x the compute (F25); a small-box merge (four rules fixed in advance) also did not beat the whole frame (F26): dropped |

## 5. What the evidence supports

- **Data diversity is the dominant lever**, larger than resolution, architecture or augmentation.
- **The low scores are mostly a protocol effect** (site split vs random split) plus narrow data, not a broken model.
- **The best model by accuracy is YOLOv8s on DeepFish + Kakadu + OzFish** (one seed); **the best confirmed and
  deployable one is YOLOv8n on the same data** (three seeds, timed on the Orin).
- Small fish (16 to 32 px) are still mostly missed: 26% recall at best.

## 6. What it does not show

- The seed spread of YOLOv8s, and any effect of its batch size (5 against 10).
- YOLOv8s on the Orin is timed end to end (F28: 50.1 ms serial) and for 30 minutes with video decode (F29: 24.4 fps, no throttling); one run at 25 W, ambient not measured; no on-device mAP, no INT8, no other power modes, no NVDEC decode, and engine-against-PyTorch parity not yet run for the new engines.
- Generalisation to a new reef: held-out OzFish deployments may share reefs with the training ones, and DeepFish val
  resembles Kakadu (F13).
- The sealed OzFish test result (not run).
- Comparison with published numbers (different splits; not verified).
- Frames from one video are not independent samples, so per-frame mAP overstates confidence.

## 7. Suggested next steps

In order of value for the report:

1. **YOLOv8s on the Orin was timed (F28):** 50.1 ms per frame against 40.8 ms for YOLOv8n at 25 W. A 30-minute video run held 24.4 fps (F29). Still to do: engine-against-PyTorch parity on the board.
2. **Two more YOLOv8s seeds** at the same batch (about 3.5 h each) to turn the one-seed lead into a confirmed one.
3. **Run the sealed OzFish test once** on the chosen final model, after steps 1 and 2. The owner decides the timing.
4. **Small fish** (tiling and a small-box merge were tried and did not help, F25 and F26): training at 1280 px did not help either, F27): training on tiles is untried; the likelier route is more data with small, hard fish.
5. **Decide the licence question** (will any copy of software or weights leave the institution?). If yes, the AGPL
   encumbrance on Ultralytics weights means YOLOX-nano or NanoDet-Plus.
6. **Write the report** from this file, `docs/PROGRESS.md` and `docs/eval_protocol.md`; `report_sections.md` is the
   older counting experiment and should be replaced.
