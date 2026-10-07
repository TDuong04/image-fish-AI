# Detecting fish in underwater video on a Jetson Orin Nano: data, evaluation and deployment of a single-class YOLO detector

*Working title. DRAFT, started 2026-10-06. Sections marked **[PENDING]** cannot be written until the experiments they describe
are finished. Every number carries a finding id (F#) that points to the evidence, run folder and limits in
`docs/PROGRESS.md`; if this text and that file disagree, that file wins. Citations are author-year and tracked in
`references.md` / `references.bib`. Reading order for the structure: Introduction, Related work, Datasets, Method,
Results, Discussion, Limitations, Conclusion, as in comparable fish-detection papers (for example Zhang et al., 2024).*

## Abstract

**[PENDING: written last, when the final numbers exist. Provisional statement of what it will have to say, so nothing
is forgotten.]**

A single-class fish detector for underwater video is trained from COCO-pretrained YOLOv8 weights on three openly
licensed datasets and evaluated with a site-level split that avoids frame leakage. [Headline result with seeds and
spread.] [Cross-dataset result on the sealed test set, run once.] [Measured end-to-end frame rate and sustained thermal
behaviour on a Jetson Orin Nano, with the power mode stated.] The analysis shows that [what limits recall], that
[what raised it], and that [what did not help].

## 1. Introduction

Monitoring fish with underwater cameras produces far more footage than people can review. Baited remote underwater
video stations (BRUVS) and fixed cameras record for hours, and most frames contain no fish at all. Automatic detection
would turn that footage into counts and presence records, and running the detector on a small computer at the camera
would let a field unit filter its own footage. This project asks a practical version of that question: **how accurate
can a fish detector be made, and what does it cost to run on a Jetson Orin Nano, when both answers are measured on the
real device and on data the model has not seen?**

Three things make the question harder than a benchmark number suggests.

1. **Scores depend heavily on how the data are split.** Underwater datasets are made of video frames, and neighbouring
   frames are nearly identical. A random per-image split lets the model see almost the same picture in training and in
   testing. We measured the effect on our own pipeline: the same model scores 0.98 mAP@50 on a random per-image split of
   DeepFish and about 0.47 when whole capture sites are held out (F20). Results from the literature should not be
   compared with ours unless the protocol is known to match (Kapoor and Narayanan, 2023, discuss leakage of this kind
   across machine-learning-based science).
2. **Fish are small.** At the resolutions that suit a small device, many fish occupy a few tens of pixels, and detection
   quality collapses below about 32 pixels (F3, F10).
3. **A desktop speed is not a device speed.** On the Jetson the network itself is a small part of the time per frame
   (F8), so GPU-only benchmarks mislead.

**Contributions.** (i) A leakage-controlled evaluation protocol and a record of how much it changes the numbers.
(ii) An error analysis that locates the failures (which fish are missed, which boxes are false alarms) instead of
reporting a single score. (iii) A controlled study of what raises accuracy: input resolution, additional training
data, model size, and test-time tiling, with seed variance measured rather than assumed. (iv) End-to-end and
sustained measurements of the detector on a Jetson Orin Nano, with the power mode and clocks stated.

Throughout, we separate **observations** from **claims**: one seed, one split or one dataset supports "observed", and
the report says so.

## 2. Related work

### 2.1 Fish detection in underwater images

Deep-learning fish detection has mostly used the YOLO family because it combines a single-pass design (Redmon et al.,
2016) with real-time speed. YOLO-Fish adapted YOLOv3 and YOLOv4 variants to realistic underwater scenes using DeepFish
imagery and reported average precision of roughly three quarters (Muksit et al., 2022) **[figure from a search
snippet; confirm in the paper before publishing]**. Later work modifies YOLOv8-class detectors for fish: Zhang et al.
(2024) add a space-to-depth convolution, an attention module and an extra small-target detection layer to YOLOv8n
for fish in natural waters, and AquaYOLO (Vijayalakshmi and Sasithradevi, 2025) targets aquaculture monitoring and
reports evaluation on DeepFish and OzFish among other data. These papers differ in datasets, splits, image sizes
and metric definitions, so their numbers are not directly comparable with each other or with ours; we therefore
quote none of them as a baseline in the results. **[Read AquaYOLO and YOLO-Fish fully and state their splits here.]**

### 2.2 BRUVS and stereo video

BRUVS are a standard tool in fish ecology. Deep-learning models for BRUVS include generalised models for
semi-automated length measurement in stereo-BRUVS (Marrable et al., 2023). Our footage includes baited, crowded
scenes (OzFish) as well as sparse reef scenes (DeepFish), and the two behave differently (F2).
**[Read and expand: what BRUVS studies need from a detector, and the annotation practice behind OzFish.]**

### 2.3 Small objects and tiling

Small objects are a known weakness of single-stage detectors. A common remedy is slicing the image into overlapping
tiles at test time and merging the results (Akyon et al., 2022). We test this on our best model and report that it
did not help here (Section 5.6), including a small-box merge rule fixed in advance.

### 2.4 Evaluation practice

Average precision with an IoU criterion of 0.5 follows the PASCAL VOC convention (Everingham et al., 2010) and
the COCO-style average over IoU 0.50 to 0.95 (Lin et al., 2014). Duplicate boxes are removed by non-maximum
suppression (Neubeck and Van Gool, 2006).

## 3. Datasets

All annotation is converted to a single class, `fish`. Frames are 1920x1080 unless stated.

### 3.1 DeepFish (training and validation)

DeepFish (Saleh et al., 2020) provides underwater video frames from tropical Australian habitats. We use the
YOLO-format box export released with YOLO-Fish: 6,517 images and 15,463 boxes. We split by **capture site** (14
training sites, 5,178 images and 14,078 boxes, 29.9% empty frames; 6 validation sites, 1,339 images and 1,385
boxes, 34.7% empty frames). The split is committed and the code asserts that no site appears on both sides (`docs/data_audit.md`).
This validation split is the common yardstick for all models in this report.

### 3.2 Kakadu (extra training data only)

The Kakadu dataset (Jansen et al., 2022; CC BY 4.0) has 44,112 frames and 82,904 boxes of 23 freshwater taxa from
remote underwater video in northern Australia. It carries no site or video labels, so we grouped frames by picture
similarity (small mean-removed grayscale thumbnails, L1 distance, single linkage at a threshold of 4.0, giving 2,296
groups), kept at most 15 frames per group, and obtained 7,318 frames with 11,695 boxes. A held-out split inside Kakadu
still leaked (the closest train-validation pairs were the same scene), so Kakadu is used **for training only**
(F13). Every Kakadu frame contains a fish, so it cannot measure false alarms.

### 3.3 OzFish (held-out evaluation, then partly training)

OzFish (Australian Institute of Marine Science et al., 2019; CC BY 3.0 AU) holds crowded BRUVS frames with
crowd-sourced boxes, unlabelled by species for the detection task. Our sealed test set has 352 frames, 8,329 boxes
and no empty frames (297 clips, 23.7 boxes per frame against 2.4 in DeepFish). The sealed set was used for the
baseline cross-dataset result (F2) and is **not touched again** until the final model is chosen. We later obtained the
labelled OzFish frames. Because the test frames come from the same labelled pool, we excluded every deployment that
appears in the test set (leaving 794 of 1,758 frames, 779 after a near-duplicate filter), and held out whole deployments (157 frames) as a decision-only validation set, with 622 frames for
training. Training on OzFish changes what the sealed test measures: it then tests new deployments of the same
collections, not an unseen source.

### 3.4 Data not used

The CFC dataset is sonar imagery, a different modality, and is not used for optical training. YOLO-Fish weights
are a reference, not a baseline we reproduce.

## 4. Method

### 4.1 Detector

We use YOLOv8 (Ultralytics; Jocher et al., 2023) in nano, small and medium sizes, and YOLO11s for one comparison. The
network is a convolutional backbone, a multi-scale neck and an anchor-free head that predicts boxes at three strides
(8, 16, 32); overlapping predictions are removed by non-maximum suppression (Neubeck and Van Gool, 2006). We use the
single `fish` class. A variant with an extra stride-4 head was tried for small fish (F12).

### 4.2 Training

Models start from COCO-pretrained weights (Lin et al., 2014) and are fine-tuned for at most 150 epochs with
early stopping (patience 30) and Ultralytics' default augmentation (mosaic with the last 10 epochs off, small
hue and scale jitter). Checkpoints are chosen by Ultralytics' fitness, 0.1 x mAP@50 + 0.9 x mAP@50-95, on the
DeepFish validation split. Input size is 640, 960 or 1280 px on the long side, letterboxed (a 1080p frame is
evaluated at 544x960 for 960). The first baseline runs (YOLOv8n, 640 and 960, three seeds each) used a Kaggle T4; all
later runs used one 8 GB RTX 5060 server, with batch sizes chosen automatically (YOLOv8n 10, YOLOv8s 5, 1280 px 3).
Seeds, configs, library versions and the git commit are stored with every run. Ultralytics versions differ across the
phases (8.4.130 workstation, 8.4.159 Kaggle, 8.4.172 server).

### 4.3 Evaluation protocol

- **Splits** are by capture site, never by image, and the sealed OzFish test set is used once (Section 3). Leakage
  is checked by a distance test on thumbnails as well as by site labels, which we found necessary but not sufficient (F13).
- **Metrics.** mAP@50 and mAP@50-95 (Everingham et al., 2010; Lin et al., 2014) from Ultralytics for the headline
  tables; a separate error analysis at IoU 0.5 and confidence 0.25 reports fish found and missed, false alarms by cause
  (duplicate, badly placed, background), false alarms on empty frames, and recall by box size (under 16, 16-32, 32-96 and
  over 96 px, measured as seen at 640 px input). Empty frames matter because BRUVS footage is mostly empty and mAP hides
  false alarms.
- **Controls and seeds.** Every comparison uses a same-machine, same-batch control. The seed spread of the baseline on
  the same machine was 0.041 mAP@50, so differences below that are not results, and any claim needs three seeds
  reported as a mean and range. One seed is called a screen. The machine, batch size and library version alone moved the same-seed
  score by about 0.03 mAP@50, which is why cross-machine comparisons are not used.
- **Frames are not independent.** Frames from one video are near duplicates, so per-frame scores look more certain
  than they are; we say so with each table.
- **Rules fixed in advance.** The tiling merge rules (Section 5.6) were fixed before running and all were reported.

### 4.4 Test-time tiling

For the tiling study (F25, F26) the checkpoint is run on overlapping tiles (20% overlap, 2x2 and 3x3) as well as the
whole frame, tile boxes touching an inner tile edge are dropped (a cut fish gives a partial box), and boxes are merged
by NMS. A small-box variant keeps a tile box only if the fish is at most 32 px at 640-px scale, optionally with a 0.4
confidence floor on tile boxes. One average-precision routine scores every variant, including the untiled control
(it gives 0.8167 against Ultralytics' 0.8209 for the same checkpoint). The idea follows sliced inference (Akyon et al.,
2022); the merge rules are ours.

### 4.5 Deployment on the Jetson Orin Nano

Target: Jetson Orin Nano 8 GB Super developer kit, L4T 36.4.7, CUDA 12.6, TensorRT 10.3.0, active fan, SD-card boot.
Models are exported to ONNX on a desktop and TensorRT engines are built **on the board**, since engines are specific to
the GPU, TensorRT version and driver. FP16 engines are built at the shape the accuracy was evaluated at (384x640 and
544x960), so latency and accuracy describe the same workload. Every frame-rate figure states the `nvpmodel` mode.
Latency is end to end (decode, letterbox, inference, NMS), warm-up discarded, median and 95th percentile. A GPU
kernel benchmark is reported as such and never called a frame rate. Agreement between the engine and PyTorch is
checked on real frames (F7).

## 5. Results

### 5.1 Resolution and the baseline (F1, F2)

On the DeepFish validation split, YOLOv8n trained at 960 px reaches 0.502 mAP@50 (mean of three seeds; sd 0.018)
against 0.416 at 640 px (sd 0.030): +0.087, and every 960 run beats every 640 run (F1). The first Kaggle-trained
baselines are in-domain figures from a short, single-source dataset and sit well below published numbers for the
reasons in Section 4.3; we do not compare against them.

### 5.2 The in-domain advantage largely vanishes on a different dataset (F2)

On the sealed OzFish set, the two models score 0.204 (640) and 0.217 (960) mAP@50, a loss of 51% and 57% from in-domain,
and the 960 advantage shrinks from +0.087 to +0.013. The test is crowded baited footage (23.7 boxes per image) and the
training data sparse reef footage (2.4), so the loss mixes a change of scene, density and camera. These are the
baseline models trained before any OzFish data were used.

### 5.3 Where the detector fails (F3, F10, F11, F19)

Recall depends strongly on fish size (YOLOv8n, 960 px): about 3% for 16-32 px fish, 49% for 32-96 px and 91% above
96 px (one seed, F10; the three-seed means in F3 agree to within a few points). **Missing fish, not wrongly placed
boxes, limits the detector:** at 960 px, 523 of 688 missed fish had no prediction at all, 161 were found below the 0.25
threshold, and only 4 were badly placed (F10). Most false alarms fall on background (321 of 365). A manual audit of a
sample found that most "misses" were visible fish rather than label errors (F11, one reviewer, a sample). Empty frames
explain less than 0.03 of the mAP@50 gap to published figures (F19), and the split protocol explains the rest of that gap
(F20, Section 1). A colour perturbation study showed the model tolerates missing colour (greyscale -7% to -9%) but not
wrong colour (hue shifts up to -49%; F4), which is relevant because water colour shifts with depth.

### 5.4 More diverse data is the largest lever (F17, F18, F21, F23, F24)

Same machine, same batch, three seeds each, DeepFish validation, YOLOv8n at 960 px:

| Training data | mAP@50 (mean, range) | mAP@50-95 | Held-out OzFish mAP@50 |
|---|---|---|---|
| DeepFish only (control) | 0.473 (0.456 to 0.497) | 0.301 | 0.256 |
| + Kakadu | 0.698 (0.684 to 0.709) | 0.471 | 0.357 |
| + Kakadu + OzFish | 0.750 (0.740 to 0.764) | 0.514 | 0.689 |

Adding Kakadu raised the score by 0.225, every seed far above every control seed (F18), and the gain held on unseen OzFish
deployments (about +0.10 mAP@50; F21). Adding the OzFish training frames raised it by a further 0.052; the weakest
combined seed (0.740) is above the best Kakadu-only seed (0.709) (F24). Stronger augmentation added nothing measurable
on top of OzFish data (F23, one seed per arm), so the gain is attributed to the data. Held-out OzFish deployments may share
reefs with the training deployments, and DeepFish validation resembles Kakadu (F13), so these gains may be flattered.
False alarms at the default threshold varied a lot between seeds (155, 477, 175) and the best-F1 threshold moved from 0.2
to 0.4, so an operating threshold has to be chosen per model on validation (F24).

### 5.5 Model size (F14 to F16, F24)

Larger models on DeepFish alone gave one-seed gains that were later shown to depend on data. On the combined data
YOLOv8s scored 0.821 mAP@50 and 0.562 mAP@50-95 (one seed, batch 5), against 0.750 and 0.514 for the YOLOv8n mean: it found
1,004 of 1,385 fish with 122 false alarms (YOLOv8n arm: 910 and 269) and scored 0.724 on held-out OzFish (0.689). It
costs about three times the compute per image and its latency on the Orin has not been measured; the YOLOv8s result is
a single seed with a different batch size from the YOLOv8n arm and is a screen (F24).

### 5.6 Test-time tiling does not help (F25, F26)

| Variant (YOLOv8s, DeepFish val) | AP50 | AP50-95 | Fish found (of 1,385) | False alarms |
|---|---|---|---|---|
| Whole frame (control) | 0.817 | 0.556 | 1,004 | 123 |
| 2x2 tiles + whole frame | 0.791 | 0.544 | 1,085 | 459 |
| 3x3 tiles + whole frame | 0.784 | 0.549 | 1,080 | 611 |
| 2x2, small tile boxes, conf floor 0.4 | 0.812 | 0.554 | 1,019 | 181 |

Tiling finds 8% more fish and raises recall for 16-32 px fish from 26% to 37 to 39%, but false alarms rise 3.7 to 5
times and AP falls; restricting tile boxes to small fish recovers most of the loss but no variant beats the whole frame.
It also costs 5 to 10 network passes per frame. One checkpoint, one split, fixed tilings.

### 5.7 Higher input resolution (F27)

A YOLOv8s run at 1280 px, with the Kakadu and OzFish frames re-cut at 1280 px so that they are not up-scaled from 960, tests whether more pixels per fish raise small-fish recall. They did not: recall on 16-32 px fish was 24.5% against 26.4% for the 960 px model (fish under 16 px: 0 of 10 in both), the number of fish found was the same (1,009 against 1,004), DeepFish validation mAP@50 was 0.785 against 0.821 (mAP@50-95 0.555 against 0.562), and false alarms at the default threshold were higher (344 against 122; the best-F1 confidence moved from 0.25 to 0.5). On held-out OzFish the 1280 px model scored slightly higher (0.736 against 0.724 mAP@50), similar in size to the seed spread of the YOLOv8n arm. One seed each and different batch sizes (3 against 5), so this is a screen; training took 1.7 times longer per epoch.

### 5.8 Device measurements on the Jetson Orin Nano (F6 to F9, F28 to F31)

All figures are for YOLOv8n FP16 engines built on the board (25 W mode, clocks not locked, fan on, `trtexec` and the
project's own runner).

| Measure | 640 (384x640) | 960 (544x960) |
|---|---|---|
| GPU kernel time, median (a 20 s burst, not a frame rate) | 2.56 ms | 4.37 ms |
| End to end, serial pipeline, median | 32.8 ms (30.5 fps) | 40.8 ms (24.5 fps) |
| Share of the frame spent in the GPU kernel | 8% | 11% |

The GPU kernel is a small part of each frame; JPEG decode, letterboxing and host-device copies dominate (F8). In a 30
minute run at 960 px with video decode, the frame rate held at about 34 fps (first and last two minutes within 0.5%),
the junction temperature rose from about 50 to 56.5 C and flattened, input power was about 6.5 to 7.1 W, and no
throttling was visible (F9; one run, ambient assumed 26 C, GPU clock reading unusable). The engines match PyTorch on 12
real frames (F7). **The larger model costs more on the device (F28):** a YOLOv8s engine trained on the combined data takes 50.1 ms per frame end to end (serial, JPEG decode, 25 W, 200 runs) against 40.8 ms for the YOLOv8n engine, because only the GPU stage grows (23.6 against 14.3 ms); the serial frame rate falls from about 24.5 to 19.9 fps. In a 30 minute run with video decode the YOLOv8s engine held 24.4 fps (against 34 fps for YOLOv8n), the junction temperature plateaued near 58 C and no throttling was visible (F29; one run, 25 W, ambient not measured). Switching the board to MAXN_SUPER left the GPU stage unchanged (14.3 and 23.5 ms) and sped up only CPU JPEG decode (about 4 ms), so the serial totals fell by 9 to 10% but the 10-minute video frame rates did not change (F31). A control rerun of the original baseline engine gave 41.0 ms, matching F8. **Not measured:** the 15 W and 7 W power modes, INT8, mAP computed on the device, NVDEC hardware decode, and agreement with PyTorch beyond 12 frames. On those 12 frames the YOLOv8s engine agrees with PyTorch on every detection (worst IoU 0.990), while the new YOLOv8n engine matches every detection but fails the preset IoU gate on one 9-pixel box in a crowded frame (IoU 0.708, F30).

## 6. Discussion

**What limits accuracy.** Our evidence points to data diversity and fish size rather than the detector design.
Adding two openly licensed datasets raised mAP@50 on a site-held-out split from 0.47 to 0.75 (F24), more than
resolution (+0.09), augmentation (none after data), or test-time tiling (negative). The remaining misses are mostly
small and faint fish, so neither test-time tiling (more fish found for more false alarms) nor training at a higher resolution (no gain, F27) fixed them, so more data that contains small, hard fish is the more likely route.

**Why our scores sit below published figures.** The same pipeline scores 0.98 mAP@50 on a random per-image split
(F20). Published results may use random per-image splits or other protocols, which we could not verify, so the
comparison is not like for like. Precision at our operating point (about 0.89 to 0.92) is in the same range as the figures
reported in the literature, while recall (about 0.72) is lower **[compare only after reading the cited papers]**.

**Deployment.** Because the network is a small part of each frame on the Orin, the larger model is affordable: YOLOv8s adds about 9 ms (22%) per frame (F28) for a gain of about 0.07 mAP@50 (F24, one seed).

## 7. Limitations

- Most comparisons are one seed (screens); the three-seed results are the DeepFish-only control, the Kakadu arm and the
  OzFish arm of YOLOv8n.
- Per-frame metrics on near-duplicate video frames overstate certainty.
- DeepFish validation resembles Kakadu, and the held-out OzFish deployments may share reefs with training ones, so
  cross-source generalisation is only partly tested; the sealed OzFish test has not been run on any model trained with
  OzFish data.
- OzFish labels come from crowd annotation, so an unlabelled fish scores as a false alarm; the label audit is one
  reviewer's sample.
- Device results cover one clip and one 30-minute run each for YOLOv8n and YOLOv8s (plus one frame pair for timing), at one power mode (25 W); thermal behaviour at other power modes,
  ambient temperatures or loads is unknown. The Nano was offline for the later experiments.
- The AGPL-3.0 licence of Ultralytics code and its pretrained weights applies to any copy that leaves the institution;
  this is an open decision (PROGRESS section 8).
- Multiple Ultralytics versions and machines were used across the phases; comparisons are made within a machine.

## 8. Conclusion **[PENDING]**

**[Written after: 1280 px result, YOLOv8s seeds, the final model's sealed OzFish test (once), YOLOv8s on the Orin.]**

## Reproducibility statement

Every table above is regenerable from a committed config and run folder (`runs/`), and the sealed-set rule, splits and
seeds are in `docs/eval_protocol.md` and `docs/PROGRESS.md`. Data licences: DeepFish (MIT), Kakadu (CC BY 4.0), OzFish
(CC BY 3.0 AU; converted labels are a derivative work and need attribution).

## References

See `references.bib` (machine-readable) and `references.md` (how each was verified and what it is used for). Entries
cited above: Akyon et al. (2022); Australian Institute of Marine Science et al. (2019); Everingham et al. (2010); Jansen et
al. (2022); Jocher et al. (2023); Kapoor and Narayanan (2023); Lin et al. (2014); Marrable et al. (2023); Muksit et al. (2022);
Neubeck and Van Gool (2006); Redmon et al. (2016); Saleh et al. (2020); Vijayalakshmi and Sasithradevi (2025); Zhang et al. (2024).
