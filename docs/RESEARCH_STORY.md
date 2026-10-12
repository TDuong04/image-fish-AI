# The research story: what we did, in what order, and why

A long, self-contained account of this project from the first day to the draft paper, written so that
**(a)** the project owner can follow the reasoning from the beginning, and **(b)** a second AI assistant, given this
file, can answer questions about how the methods work and why each step was taken.

Written 2026-10-12 from the repository at commit `1fe6086` and its predecessors. It is a narrative and a teaching
document, not the record. **Every number in it comes from a finding (`F#`) in `docs/PROGRESS.md`**, where the run
folder, the number of seeds and the limits are stated. If this file and those files disagree, they win:

| Authority (highest first) | What it holds |
|---|---|
| `docs/PROGRESS.md` | every result (F1 to F41), every correction (section 6), every bug (section 7), open items (section 8) |
| `docs/eval_protocol.md` | the measurement rules, written before the results they govern |
| `runs/<name>/` | the raw records behind each number |
| `docs/report/paper.md` | the draft paper |
| this file | the story that connects them |

---

## 0. How to use this file

### 0.1 If you are the owner

Read sections 1 to 3 once (the question, the vocabulary, the principles). Then read section 4 (the story) in order;
each phase says *what we wanted to know, what we did, what we saw, and why that led to the next step*. Section 5
explains the mechanisms behind the methods. Sections 6 to 9 are for looking things up.

### 0.2 If you are giving this file to another AI assistant

Paste the block below first, then the file (or the file plus `docs/PROGRESS.md`, `docs/eval_protocol.md` and
`docs/GUIDE.md` if the assistant can take them). Then ask questions from section 10.

```
You are helping me understand a research project. The attached document is the project's narrative.
Rules for you:
1. Separate three kinds of statement and label them when you answer:
   (A) what this project measured (cite the finding number, e.g. F18);
   (B) general machine-learning or engineering knowledge that the project relies on but did not test;
   (C) your own speculation. Do not present (B) or (C) as if the project had shown it.
2. If I ask for a number, give it with its finding number, its number of seeds, and its stated limit.
   If the document does not contain it, say so; do not estimate it.
3. The document is deliberately honest about weak evidence ("one seed", "a screen", "not shown").
   Keep that calibration. Do not upgrade "observed" to "proved".
4. When I ask "why does this work?", explain the mechanism in plain language first, then say what the
   project's own evidence for it is (often: none, it is background).
5. If you think the project made a mistake, say so and say what evidence would show it.
```

### 0.3 Evidence labels used throughout

| Label | Meaning |
|---|---|
| **Established** | measured with 3 or more seeds (or a cluster bootstrap), a stated spread, and a difference larger than the spread |
| **Screen** | one seed; it can nominate an idea, it cannot support a claim |
| **Diagnostic** | deliberately constructed (for example a leaky split) to explain something; not a result |
| **Observed** | seen once or in a small sample; no more than that |
| **Background** | general knowledge the project relies on; not tested here |

---

## 1. The project in one page

**Question.** Can a small single-class fish detector (draws a box around every fish in underwater or baited-camera
video, no species) be made accurate enough to be useful, and still run on a **Jetson Orin Nano 8 GB Super**, with the
performance claims **measured on that board** and on data the model has not seen, instead of extrapolated from a
desktop and a convenient test split?

**Why that question.** Fish-detection papers usually report one number (a score on a held-out split). Two things
make such numbers weak for deployment: the split may let near-identical video frames sit on both sides (inflating
the score), and the number is rarely tied to a real small device. The project's purpose is to produce numbers a
reader can trust and a pipeline that runs on the real hardware.

**Two deliverables, both required** (`CLAUDE.md`): a reproducible *report* with honest baselines and cross-dataset
evaluation, and a *device* pipeline with measured, reproducible frame rates.

**Team context** (as of 2026-10-11, `docs/report/fishiot_sections.md`): this detector is one half of a team system in
which a reinforcement-learning agent operating fish infrastructure takes the fish count as one of its observations.
The detection work is also the owner's thesis and a planned publication (venue, template and page limit not yet
decided). No footage from a real deployment site exists yet.

**Where it stands (2026-10-12).** Five headline findings, each detailed later:

1. **The split protocol changes the score more than any model change.** On the same six sites, a model that saw
   neighbouring frames scored 0.989 AP50; models that had not scored 0.491 (F38, diagnostic).
2. **More diverse training data was the largest lever.** DeepFish only 0.473, plus Kakadu 0.698, plus OzFish 0.750
   mAP@50 for YOLOv8n (three seeds each, F18, F24); YOLOv8s on the same data 0.812 (F32). Resolution, stronger
   augmentation, a stride-4 head, test-time tiling and training at 1280 px did not help (F12, F23, F25 to F27).
3. **Generalisation to a truly unseen source is poor.** On a Mediterranean set every detector scored AP50 0.05 to 0.23
   (F39); the "sealed" OzFish test (0.70) measures new deployments of data also used in training (F33, F37).
   A little labelled footage from the target camera recovers most of it for that camera (0.18 to 0.65 AP50 with 25
   frames, F41) but only slightly for other cameras.
4. **On the Orin Nano** (25 W, TensorRT FP16, engine built on the board) the shipped YOLOv8s takes about 50 ms per frame
   end to end (serial, JPEG), holds about 24 fps on video over 30 minutes with no visible throttling (junction
   temperature plateau about 58 to 59 C), and agrees with PyTorch on 12 test frames (F28, F29, F34).
5. **The limits are stated, not hidden:** no deployment footage, one device run per model, single-seed screens for
   several comparisons, checkpoint selection on a reused validation split.

---

## 2. Vocabulary (short; the long version is `docs/GUIDE.md`)

| Term | Plain meaning | Why it matters here |
|---|---|---|
| **Object detection** | find objects and draw a box with a confidence score around each | our task; one class, `fish` |
| **YOLO / YOLOv8** | a family of fast one-pass detectors; n, s, m are small, medium, larger sizes | n is about 3 M parameters; s costs roughly three times the compute of n (F24) and 9 ms more per frame on the Orin (F28) |
| **COCO-pretrained** | the model starts from weights learned on a large general image set, then is fine-tuned | speeds learning; also carries the AGPL licence (section 5.16) |
| **Confidence threshold** | boxes below a score are discarded; we report at 0.25 unless stated | a deployment setting, not a model property (F32, F39) |
| **IoU** | overlap of a predicted and a true box divided by their union | a box counts as correct above an IoU cut (0.5 for "AP50") |
| **Precision / recall** | of the boxes drawn, the share that are real / of the real fish, the share found | recall limits us more than precision (F10) |
| **AP, mAP@50, mAP@50-95** | one number summarising the precision-recall curve across thresholds; @50 uses IoU 0.5, @50-95 averages IoU 0.5 to 0.95 | the scoreboard; "AP50" in later findings is our own 101-point routine, close to but not identical to Ultralytics' mAP (F35) |
| **Found / false alarms** | counts at a fixed threshold (0.25): true fish matched vs boxes with no fish | mAP can hide false alarms; we report both |
| **Seed** | the random number that fixes initial weights and data order; a different seed gives a different model | seed-to-seed spread can exceed the effect being tested (section 5.6) |
| **Control** | the same experiment without the change being tested, on the same machine, batch size and library versions | without it, a "gain" can be a machine effect (F14) |
| **Site / clip / frame** | a camera location / one video / one image from a video | frames from one clip are near-duplicates, so they are not independent samples |
| **Leakage** | information from the test set reaching training (here: neighbouring frames of the same clip) | inflates the score (F20, F38) |
| **Sealed set** | a test set used only once, at the end, under rules written in advance | stops us tuning to it (section 5.14) |
| **Cluster bootstrap** | re-sampling whole clips (or sites, or dates), not frames, to get an honest interval | frames are not independent (section 5.5) |
| **Domain shift** | the test footage differs from training (camera, water, species) | the reason scores fall on new sources (F39) |
| **ONNX / TensorRT engine** | a portable model file / a GPU-specific optimised program built from it | engines are built on the Orin, not copied (section 5.9) |
| **FP16** | half-precision numbers: faster and smaller than FP32, tiny numeric differences | checked by a parity test (F7, F30, F34) |
| **End-to-end latency** | time from reading a frame to final boxes, including decode and NMS | the only latency we call a frame rate (section 5.10) |
| **NMS** | non-maximum suppression: removes duplicate overlapping boxes | CPU step; cheap here (F8) |
| **MaxN** | the largest count of fish in one frame of a video | the standard ecology statistic for baited cameras (F36) |
| **BRUVS** | baited remote underwater video stations | the footage type OzFish comes from |

---

## 3. The principles behind the method (and why each exists)

These are written in `CLAUDE.md` and `docs/eval_protocol.md`. Most were added **after** the project got something
wrong; the cause is given.

| # | Principle | Why | Where it came from |
|---|---|---|---|
| P1 | **Write the evaluation protocol before results exist** (`eval_protocol.md`, committed 27 Aug) | every unfixed choice (threshold, split, seed, checkpoint) is a knob that can be turned until the answer looks good | design; later items 9 to 12 were each committed *before* their run |
| P2 | **Split by capture site/clip, never by image** | adjacent video frames are near-duplicates; a per-image split lets the model "see" the test | the F20 and F38 diagnostics later measured the cost |
| P3 | **Measure on the device, at the shape the accuracy was evaluated at** | a desktop number does not transfer (F5), and an engine built for a 960x960 square runs 1.76x the pixels of the geometry behind the reported accuracy (F6) | the first board engine was built square (corrections table) |
| P4 | **End-to-end latency, not kernel time** | the GPU kernel was only 8 to 11% of a frame (F8) | building the end-to-end runner |
| P5 | **At least three seeds for a comparison that drives a decision; one seed is a screen** | the seed spread (0.04 mAP@50) exceeds many effects being tested | two seeds hid the true spread (corrections) |
| P6 | **A same-machine, same-batch control for every comparison** | the same seed scored 0.031 lower on a different machine (F14) | the Kaggle-vs-server discrepancy |
| P7 | **A big gain triggers a leak check before it is reported** (about 0.05 mAP@50) | large gains are the usual signature of leakage | the first Kakadu run jumped far above everything else (F13) |
| P8 | **The sealed set is run once, on the final candidate, under rules fixed in advance** | repeated looks turn a test set into a validation set (section 5.14) | design, tightened at item 8 to 11 |
| P9 | **Choose the threshold and checkpoint on validation only** | a threshold tuned on the test inflates it; and the best threshold moves between seeds (F32) | observed threshold variation |
| P10 | **Report what a number does not show, in the same entry** | prevents quiet over-claiming | project habit; enforced in review |
| P11 | **Corrections are recorded, not overwritten** (PROGRESS section 6) | a deleted mistake is repeated | 16 recorded corrections |
| P12 | **Docs updated in the same change as the work** (`tests/test_docs.py` fails otherwise) | results that live in a chat are lost | early loss of Kaggle outputs (PROGRESS section 7) |
| P13 | **Do not compare with published numbers across different splits, sizes or metrics without saying so** | published scores often come from different protocols | the 0.92 vs 0.47 gap (F19, F20) |
| P14 | **Code runs on any machine** (no absolute paths, device resolved at run time, Python entrypoints) | the project spans a Windows laptop, a Linux GPU server, Kaggle and an ARM64 Jetson | `CLAUDE.md` portability rules |

---

## 4. The story, in order

Dates are commit dates. The "why next" line of each phase is the logic that links it to the one after.

### Phase 0 — Before the detector (up to 27 Aug 2026)

*Starting point.* An earlier experiment used **LCFCN**, a method that learns to count fish from single clicks per fish
(point labels) on DeepFish (`report_sections.md`: validation MAE 0.3375 after 10 epochs on a Kaggle T4). It is a
different task (counting, no boxes), not converged, and not the baseline for this work. It is kept only as history;
`report_sections.md` is stale.

*What was decided.* On 27 Aug the project charter, backlog (`docs/tickets/BACKLOG.md`, tickets `FD-###`) and a portable
training scaffold were committed, with the evaluation protocol **before any detection result existed** (P1). The goal
became **bounding-box detection**. The repository does not record a written reason for dropping counting; the likely ones
(boxes give both a count and a location, and box labels exist for DeepFish and OzFish) are an inference, so ask the owner if it matters.

*Why next.* A protocol with no data is nothing. Data came first.

### Phase 1 — Data in hand, and the finding that shaped everything (27 Aug to 22 Sep)

*What we wanted.* Boxes of fish, single class, from footage resembling baited or reef cameras, with licences allowing
use and release.

*What we found.*

- **DeepFish** (6,517 images, 15,463 boxes) and the **OzFish test set** (352 images, 8,329 boxes) came from the
  YOLO-Fish authors' public Google Drive archives. An earlier belief that DeepFish had no usable boxes was wrong
  (corrections table). The full OzFish training set was not available at that time.
- **OzFish boxes are fish/no-fish only.** The "507 species" figure belongs to the ~80,000 crops, not the boxes. An
  earlier plan to "collapse species" was therefore unnecessary (corrections).
- **Box size is the central difficulty** (`docs/data_audit.md`). At the common 640 px input, 14% of DeepFish and 35%
  of OzFish boxes are under 16 px, roughly the smallest object a stride-8 detection head can represent. Raising the
  input resolution helps, and costs the most on a small device. The **accuracy-per-watt trade-off** between those two
  is the project's tension.
- **Splitting is the hard part.** DeepFish clips run to 462 near-identical frames. The splits were made over whole
  capture sites (14 train sites, 6 validation sites) with an assertion that no site appears on both sides.

*Why next.* With data, a baseline: how good is a plain small detector, and does resolution matter?

### Phase 2 — The baseline grid (22 to 24 Sep): F1 to F4

*Design.* YOLOv8n, COCO-pretrained, one class, trained on DeepFish at 640 and at 960 px, **three seeds each**, on a
Kaggle T4 (an ephemeral hosted GPU), early stopping with patience 30, checkpoint chosen by Ultralytics' fitness
(`0.1 x mAP@50 + 0.9 x mAP@50-95`). A supervisor script restarted crashed runs only when a retry could help.

*Results (Established, 3 seeds).*

| | 640 | 960 |
|---|---|---|
| mAP@50 on DeepFish validation | 0.4156 (sd 0.030) | 0.5021 (sd 0.018) |
| mAP@50 on the OzFish test set | 0.2038 | 0.2165 |

- **F1:** 960 beats 640 by +0.087 mAP@50 in-domain; every 960 run beats every 640 run.
- **F2:** on the OzFish test the gap shrinks to +0.013. Most of the resolution advantage does not survive a change
  of scene.
- **F3:** the gain is in 32 to 96 px fish; fish under 32 px are almost never found at either resolution.
- **F4:** greyscale costs 7 to 9% of the score, but wrong hues cost up to 49%. Absent colour is tolerated, wrong
  colour is not. (A single-frame observation had suggested heavy colour dependence; the full-set run corrected it.)
- **F5:** desktop timing (about 13 ms for either resolution) says little about the Orin.

*Why next.* Two puzzles: the scores were far lower than published DeepFish numbers (about 0.92), and the board had
not been touched. We pursued both, device first because the board arrived.

### Phase 3 — Onto the Orin Nano (2 to 4 Oct): F6 to F9

*Design.* The board was probed read-only (`edge/probe_board.py`) rather than assumed: Orin Nano 8 GB Super, L4T 36.4.7,
CUDA 12.6, TensorRT 10.3.0, SD-card boot with about 8 GB free. Models were exported to ONNX **at the shapes the
accuracy was measured at** (384x640 and 544x960, not squares), checked against PyTorch, and built into **FP16 engines
on the board**.

*Results.*

- **F6:** GPU kernel time 2.56 ms (640) and 4.37 ms (960): 960 costs 1.71x for 2.12x the pixels. A kernel number,
  not a frame rate.
- **F7:** both engines match PyTorch on 12 real frames (no confident misses or extras).
- **F8:** end to end, a serial pipeline takes 32.8 ms (640) and 40.8 ms (960): the GPU kernel is only 8 to 11% of a
  frame. JPEG decode (17 ms), letterbox (5 to 8 ms) and copies dominate; NMS is under 1 ms.
- **F9:** a 30-minute loop at 960 held about 34 fps with video decode, no visible throttling, junction temperature
  plateau about 56 C. One run, 25 W mode, ambient assumed 26 C.

*Corrections made on the way.* The board was assumed fanless; the tachometer showed a 1,411 RPM fan. A "7.1 ms" 960
engine was actually a 960x960 square (the right shape gives 4.37 ms). A report claiming the model "fits comfortably in
a 14 ms budget" ignored that the GPU is a small part of a frame.

*Why next.* A working device pipeline exists. The model is still weak, so improving it is the main task.

### Phase 4 — Why are the scores low? (4 to 5 Oct): F10, F11, F19, F20

This phase is the intellectual core. Three hypotheses for the gap to published numbers: the detector is bad; the
labels are bad; the evaluation differs.

- **F10 (error analysis, one seed per size).** At 960 px and confidence 0.25, 688 of 1,385 validation fish were
  missed; **523 of those had no prediction at all**, 161 were found below the threshold and only 4 were badly placed.
  So the failure is *missing fish* (mostly 16 to 96 px), not sloppy boxes and not a threshold that could be tuned away.
- **F11 (label audit, one reviewer, a sample).** Of 48 sampled misses, 32 were plainly visible fish, 12 ambiguous, 4
  invisible. Of 36 false alarms, only 2 might be unlabelled fish; 16 were mangrove leaves. The labels are mostly
  sound; the model misses fish a person can see; leaves are a hard-negative problem.
- **F19 (empty frames).** Published DeepFish detection sets seem to contain only the 4,505 images with fish; our
  validation keeps empty frames. Re-scoring on fish-only frames adds only 0.006 to 0.026 mAP@50. Empty frames do not
  explain the gap.
- **F20 (the split diagnostic).** We built a deliberately leaky random per-image split of the same images, trained the
  same pipeline, and scored **0.979 mAP@50** at 640 and 0.977 at 960 (against about 0.47 for the site split). The
  published 0.92 is plausibly a random-split number. **This is a Diagnostic, not a result**: the model is not
  better, the test is easier.
- **F38 (later, the tight version).** F20 could be criticised: maybe the site-split validation sites are just harder.
  F38 holds the sites fixed. On the same six sites, the random-split model scored **0.989 AP50** and the site-split
  controls **0.491**; gap +0.498 (interval 0.225 to 0.752). Seeing neighbouring frames of the same sites, not site
  difficulty, accounts for the difference.

*Why next.* If the model is not broken and the labels are sound, the remaining levers are the model, the training
recipe, and the data.

### Phase 5 — The improvement screens on a GPU server (4 to 5 Oct): F12 to F16

*Context.* Training moved from Kaggle to a shared Linux server (`ai-gpu`, RTX 5060 8 GB). Every change was a
**one-seed screen** compared with a same-machine control.

*The lesson learned first (F14).* The control (the baseline recipe re-run on the server) scored 0.4562 mAP@50, the same
seed on Kaggle 0.4876: **a 0.031 difference caused by machine, batch size and library version alone.** The Kaggle
baseline was therefore the wrong comparator, and a first claim that stronger augmentation "looks barely better"
(+0.04) became +0.087 against the right control. From then on, P6.

*Screens (one seed each, all screens).*

| Idea | Result | Verdict |
|---|---|---|
| **P2 head** (stride-4 output for small fish), F12 | 16 to 32 px recall 3.7% vs 2.6%; mAP@50-95 lower | no clear gain; confounded by batch size |
| **Stronger augmentation**, F14 | +0.087 mAP@50 vs control | later shown to add nothing once data are added (F23) |
| **YOLOv8s**, F14 | +0.128 mAP@50 vs control | real, later confirmed on three seeds (F32) |
| **YOLOv8m**, F15 | best strict score, far fewer false alarms, no more fish found than s | costs 3x compute |
| **YOLO11s**, F16 | gain is fewer false alarms, not more fish found | no recall gain |

*Why next.* Nothing here moved the 16 to 32 px fish. Bigger models help; but the control for the model-size effect
suggested the data might matter more than architecture. Data was the next lever.

### Phase 6 — The data lever (4 to 6 Oct): F13, F17, F18, F21 to F24

**Kakadu (F13).** A large open set (44,112 frames, 82,904 boxes, 23 species merged to one class, CC BY 4.0) with
**no video or site field**. A split by picture-similarity groups still leaked (the nearest validation/train pairs
were the same scene), so Kakadu is **training data only**, capped at 15 frames per similarity group (7,318 frames
kept), and never a judge. A leak check against the DeepFish validation set found no near-duplicates, but Kakadu
looks more like the validation sites than DeepFish's own training sites do, so part of the gain is "data that
resembles the validation sites".

**The first Kakadu run (F17)** jumped from 0.456 to 0.709 mAP@50, so P7 applied and the leak check ran *before* anything was reported.

**Three seeds each (F18, Established).** Control (DeepFish only) 0.4733 (range 0.041); with Kakadu 0.6978 (range
0.024): +0.2245 mAP@50. The weakest Kakadu seed is 4.6 times the control range above the best control seed. About
200 more fish found of 1,385, 350 fewer false alarms, 16 to 32 px recall from about 3% to about 20%.

**Does it generalise beyond DeepFish? (F21).** From the 779 trainable OzFish frames, whole deployments (stereo pairs
together) were held out: 622 training and 157 held-out frames from 79 deployments. The existing models, which had
seen no OzFish, scored 0.256 (DeepFish only) and 0.357 (plus Kakadu) mAP@50 on them: the Kakadu gain carries over.

**Adding OzFish training data (F22 to F24).** OzFish training frames overlap the sealed test frames' collections;
the audit (`audit_ozfish_overlap.py`) kept only deployments absent from the sealed test, and dropped frames within a
thumbnail distance of a test frame. A first combined recipe (Kakadu + OzFish + strong augmentation) scored 0.755 but
bundled two changes; a **split test (F23)** separated them: OzFish data alone gives 0.764, augmentation alone gives
nothing (0.698 vs 0.698). Three seeds of the OzFish recipe (F24): 0.7499 (range 0.024), +0.052 over Kakadu-only.

*Status of the OzFish gain, a correction.* F24 said the OzFish effect "holds on three seeds". The later cluster
bootstrap (F35) shows the paired gain is +0.037 with a 95% interval of -0.001 to +0.091, **not separated from zero**
once clips are the unit. It is recorded as "positive on every seed, probably real, its size uncertain" (corrections).

*Why next.* Data diversity is the biggest lever. Small fish are still mostly missed (about 20 to 30% recall at 16 to 32 px).
Before concluding, we tried to fix small fish directly.

### Phase 7 — Routes that did not work (6 Oct): F25 to F27

Each was a hypothesis with a mechanism, tested and dropped.

- **Tiled inference (F25).** Cut the frame into overlapping tiles, run the detector on each (the fish appear
  larger), merge. It does find +76 to 81 fish (+8%) and lifts 16 to 32 px recall from 26% to 37 to 39%, but false
  alarms rise 3.7 to 5x and AP50 falls from 0.817 to 0.78 to 0.79, at 5 to 10x the compute.
- **A small-box merge (F26).** Only keep tile boxes for small fish. Four rules were **fixed in advance and all
  reported**; none beat the whole frame (best: 0.812 vs 0.817).
- **Training at 1280 px (F27).** Give small fish more pixels in training. Recall at 16 to 32 px did not rise
  (24.5% vs 26.4%), DeepFish mAP@50 fell 0.036 (within seed spread); held-out OzFish rose 0.012.

*Why next.* The small-fish problem is not solved by input size or tiling alone; more diverse data is the better
bet. Time to firm up the best model and measure it on the device.

### Phase 8 — The bigger model on the board (7 to 11 Oct): F28 to F31, F34

- **F28.** YOLOv8s costs about 9 ms more per frame than YOLOv8n: 50.1 vs 40.8 ms serial at 960 px, 25 W. Only the GPU
  stage grows (23.6 vs 14.3 ms).
- **F29.** A 30-minute video loop: 24.4 fps, no throttling, plateau about 58 C (YOLOv8n: 34.1 fps).
- **F30.** Engine vs PyTorch on 12 frames: YOLOv8s passes; the new YOLOv8n engine fails the IoU gate on one 9 px box
  (the gate was not relaxed).
- **F31.** MAXN_SUPER did not speed up the network (GPU stage unchanged); the 4 to 5 ms serial gain was faster CPU
  JPEG decode; the video frame rate did not change.
- **F34.** The owner chose **seed 2** to ship; its engine, built on the board, behaves the same (49.9 ms serial, 24.3
  fps over 30 minutes, parity PASS).

*Why next.* Before the sealed test, the YOLOv8s result needed its seeds.

### Phase 9 — Seeds and the sealed test (8 Oct): F32, F33

- **F32 (Established).** Three seeds of YOLOv8s on the combined data: mean **0.812 mAP@50** (range 0.041), against 0.750
  for YOLOv8n. Every YOLOv8s seed is above every YOLOv8n seed. False alarms at 0.25 range from 80 to 601 across
  seeds: **an operating threshold must be chosen per model on validation**.
- **F33 (sealed OzFish, run once).** The plan (three seeds, once each, nothing chosen from the result) was committed
  *before* the run. Mean **0.700 mAP@50** and 0.380 mAP@50-95, against 0.217 for the DeepFish-only baseline. It
  measures new deployments of footage also used in training, **not an unseen source**.
- A protocol error was found while preparing this step: item 8 said the sealed set had not been run on any model;
  it had (F2). It was corrected in the open (PROGRESS section 7).

*Why next.* An independent review argued that the numbers needed honest uncertainty, not only seeds.

### Phase 10 — Statistical hardening (11 Oct): F35 to F38

- **F35 (cluster bootstrap on clips).** The 1,339 validation frames come from only **20 clips of 6 sites**.
  Re-sampling whole clips: the Kakadu gain (+0.234, interval 0.116 to 0.339) and the YOLOv8s lead (+0.058, 0.012 to
  0.104) stay clearly positive; the OzFish gain does not (see above). Absolute scores are uncertain by about +-0.1.
- **F36 (counting).** The detector counts 79 to 84% of the labelled fish per frame (YOLOv8s), but the clip maximum
  (MaxN) is **over-estimated** (119 to 150% of the true maximum), because one frame with false alarms sets it. Both
  facts are true; which matters depends on how the count is used (the team's RL agent consumes it).
- **F37 (sealed intervals).** A second pass over the same three checkpoints, rules committed before (item 11), only to
  add uncertainty: mean AP50 **0.697, 95% interval 0.669 to 0.724**, resampling 249 deployments. The sealed set has
  therefore been scored twice, and the paper must say so.
- **F38.** The fixed-sites leakage control described in Phase 4.

*Why next.* The strongest remaining question for a deployment is domain shift: what happens on footage from a source
nobody trained on?

### Phase 11 — Unseen source, second family, adaptation (11 Oct): F39 to F41

- **F39 (rules fixed beforehand, item 10).** A Mediterranean set (DEEPECOMAR; 1,516 frames, 17,157 fish boxes, mostly
  small) from a project and region that contributed nothing to training. Every detector did poorly: **AP50 0.05 to
  0.23**. Order for our arms: DeepFish only 0.045, plus Kakadu 0.109, plus OzFish 0.154, YOLOv8s 0.166 (the same
  ordering as on DeepFish). The published **YOLO-Fish YOLOv4** weights scored highest (0.227); we do not claim to beat
  them (different size, 11x larger, different training data). A validation-chosen threshold did not carry over
  (the same recipe landed at precision 0.54 to 0.86 and recall 0.11 to 0.16 depending on the seed).
- **F40 (second detector family, a Screen).** A torchvision Faster R-CNN shows the same data effect (+0.105 AP50 on
  validation, +0.036 on the unseen source), so the finding is not specific to YOLO.
- **F41 (adaptation, item 12, a Screen).** Fine-tune on early frames of one fixed camera, test on later frames of that
  camera and on other cameras: 25 frames take AP50 from 0.18 to 0.65; 100 frames 0.72; about 530 frames 0.76. A
  COCO-only control matches ours on the adapted camera (0.759 vs 0.756), so pretraining did not matter there; on other
  cameras the gain is small and uncertain, and the COCO-only control is worse than zero-shot (0.367 with 11,286 false
  alarms). A practical recipe, not a new-site result: label a few dozen frames of the deployment camera and fine-tune.

### Phase 12 — Writing and literature (6 to 12 Oct)

- **Documents.** `docs/GUIDE.md` (concepts), `docs/RESEARCH.md` (earlier narrative), the draft paper
  `docs/report/paper.md` (abstract to conclusion, five figures built from committed records by `scripts/make_figures.py`
  and `scripts/make_examples.py`), thesis structure, a team-report draft, and a licence decision (**AGPL-3.0**, because
  Ultralytics and the COCO-pretrained weights are AGPL; code, configs and run records prepared for release, not yet public).
- **Literature.** References carry a verification level (read in full / abstract / snippet / not read). Papers read in
  full include work on fish detection with Mask R-CNN (Ditria 2020), BRUVS length measurement (Marrable 2023), SAHI tiling, leakage taxonomy
  (Kapoor and Narayanan 2023), and three embedded-fish-detection papers. The **YOLO-Fish paper itself (Muksit 2022, closed access) has
  not been read**; its AP figures stay flagged as unverified, and the paper makes no claim that depends on them.
- **Honest comparisons.** The Discussion compares with Marrable 2022 (precision 0.898, recall 0.699) with an explicit
  statement that data and splits differ.

---

## 5. Mechanisms: what is actually happening, and how to think about it

Each topic: *what happens*, *why it matters here*, *what the project's evidence is*. Statements marked **Background**
are general knowledge the project relies on and did not test.

### 5.1 How a detector like YOLOv8 turns an image into boxes (Background; F10 for the failures)

The image is resized and padded ("letterboxed") to the model's input shape (for 1080p video at `imgsz` 960: 544x960).
A convolutional network (backbone) extracts features at several scales; a "neck" mixes them; the "head" predicts, on
grids at strides 8, 16 and 32, a box and a confidence for each grid cell. A stride-8 grid cell covers 8 input pixels, so an object much
smaller than about two cells has very little representation, which is why fish under about 16 px at the input scale are
almost never found (F3, F18). The head emits thousands of candidate boxes; **NMS** keeps the best-scoring box among
overlapping ones. A confidence threshold removes weak boxes. Both are post-processing on the CPU.
*Project evidence:* the failures are missed small fish, not placement (F10); a finer P2 head did not fix them (F12).

### 5.2 Why mAP can mislead (Background; our practice P9, F32)

AP is the area under a precision-recall curve built by sweeping the threshold. It measures *ranking quality*, not how
a model behaves at the one threshold you deploy at. Two models with the same mAP can have very different numbers of
false alarms at 0.25 (F32: 80 to 601 across seeds of one recipe). So we report found fish and false alarms at a
stated threshold alongside AP, and we choose a deployment threshold per model on validation. AP at IoU 0.5 ("AP50") is
lenient about box placement; AP50-95 averages stricter cut-offs and is more sensitive to box tightness.

### 5.3 Leakage and why splitting by site matters (Background; tested: F20, F38)

A video produces hundreds of nearly identical frames. If some go to training and some to testing, the test frames
look like training frames; the model can score highly by recognising scenes rather than generalising. The project
tested this directly: same pipeline, only the split changed, 0.98 vs about 0.47 (F20), and on identical sites 0.989 vs 0.491 (F38).
A site split removes both adjacent-frame leakage and shared background/camera/water; the F38 control cannot separate those two, and says so.
**Kakadu** shows the practical difficulty: with no site labels, similarity grouping only partly works (F13), so it is used for training, not for judging.

### 5.4 Why the "sealed" set has a specific, limited meaning (F33, F37)

A sealed test is a set you do not tune on and look at once. It tells you how the final model does on that set. The
OzFish sealed frames come from collections that also supplied training frames (different deployments, same footage
families), so it measures "new deployments of familiar footage", not "a new source". That is why the Mediterranean set (F39) exists.

### 5.5 Cluster bootstrap: honest uncertainty from correlated frames (Background; used F35, F37, F38, F39, F41)

If you bootstrap individual frames from 20 clips, you treat 1,339 near-duplicate frames as 1,339 independent
samples and get intervals that are far too narrow. A cluster bootstrap repeatedly draws *whole clips* (or sites, or
dates, or deployments) with replacement, recomputes the metric, and takes the 2.5th and 97.5th percentiles. The same
resample is applied to every model so differences are *paired*. It captures sampling variability of *which clips
were in the set*; it does not capture seed variability (reported separately) and is rough with few clusters (F39 has 18 groups, one holding 886 of 1,516 images).

### 5.6 Seeds, controls and what counts as a result (Background; measured: F14, F18, F32)

Training involves randomness (initial layers, data order, augmentation). The same recipe with different seeds gives
different scores; measured here, mAP@50 varies by about 0.04 for YOLOv8n and YOLOv8s. An effect smaller than that
cannot be distinguished from luck with one seed, so: one seed is a *screen*; a decision needs three seeds and a difference larger than their range; and a control must match
in machine, batch, library and seed. The discovery that machine alone moved the score by 0.031 (F14) is why these rules are strict.

### 5.7 Why more diverse training data helped most (Hypotheses; tested: F18, F21, F24, F39, F40)

The *effect* is established: adding Kakadu then OzFish raised scores on DeepFish validation, on held-out OzFish deployments and (more weakly, with wide
intervals) on an unseen source; a second architecture shows the same direction (F40). The *mechanism* is not isolated.
Plausible contributors, none separately tested: more variety of water, light, species and camera; more small and
crowded examples (Kakadu's median box at 640 input about 50 px; OzFish crowded scenes); more fish-containing frames
per training step. Caveats that limit the claim: Kakadu resembles the validation sites (F13); the OzFish part is
not separated from zero under clip resampling (F35); on the Mediterranean set all detectors are poor (F39).

### 5.8 Domain shift and why a few labelled frames help (Background; F39, F41)

A detector learns the appearance statistics of its training footage: colours, textures, water clarity, camera
optics, species, size distribution, labelling style. On a new source those shift. Fine-tuning on a few frames of the
target teaches the model the target's statistics quickly (F41: 25 frames, 0.18 to 0.65), which is why a COCO-only
model catches up on the adapted camera (0.759 vs 0.756). For *other* cameras nothing is learned about them, and the
adapted models only get more false alarms. Practical implication: label a handful of frames from the deployment
camera; choose the threshold on that camera.

### 5.9 From PyTorch weights to a TensorRT engine on the Orin (Background; F6, F7, F30, F34)

The trained PyTorch checkpoint is exported to **ONNX**, a framework-neutral graph (here with a fixed input shape and
without NMS). NVIDIA's `trtexec` compiles it for the specific GPU, TensorRT version and driver into an **engine**.
Engines are not portable (a different GPU or TensorRT version will not load them), so they are built on the Orin from
the ONNX file. **FP16** halves numeric precision, roughly halves memory traffic and uses tensor cores; it changes
outputs slightly, so a **parity test** compares the engine's detections with PyTorch's on identical preprocessed
inputs (F7, F30, F34). *What this is not:* an accuracy measurement on the device (no on-device mAP exists yet) and
not an INT8 study.

### 5.10 Why the GPU kernel is a small part of a frame (F6, F8)

`trtexec` times the network on a static input already in GPU memory (2.56 ms or 4.37 ms). A real frame also needs:
read and decode the image or video (CPU, 8 to 17 ms), letterbox and normalise (CPU, 5 to 8 ms), copy to the GPU
and back (pageable memory, several ms), and NMS (under 1 ms). On the Orin the Arm CPU and memory traffic dominate.
This is why the project insists that a kernel time is not a frame rate (P4), why MAXN_SUPER sped up decode but not the network (F31), and why the next
optimisation targets are pinned memory, overlap of stages, and hardware video decode (NVDEC), none yet tried. The Orin Nano has no hardware video encoder, so writing annotated video would be CPU-bound; the app deliberately writes none.

### 5.11 Unified memory, power modes and heat (Background; F9, F29, F31, F34)

The Jetson shares one memory pool between CPU and GPU, so "8 GB" is not dedicated GPU memory. Power modes
(`nvpmodel`: 7 W, 15 W, 25 W, MAXN_SUPER) cap clocks and power; a thermal limit lowers clocks when the chip gets hot ("throttling"). A short
test on a cold board proves nothing, hence 30-minute runs, with the fan, junction temperature and frame rate logged
in 10-second windows. Observed: flat frame rate, temperature plateau about 56 to 59 C at about 7 W input, in 25 W mode, one run each. The GPU-clock
reading in the log was unusable (306 MHz throughout) and no clock claim is made. Not covered: 15 W and 7 W modes, long runs in warm rooms, ambient temperature.

### 5.12 Thresholds and calibration (F10, F17, F32, F39)

A model's confidence is not a probability. The score distribution differs between seeds, recipes and
datasets: the best-F1 threshold on validation was 0.4 to 0.6 for early runs and 0.2 to 0.25 for Kakadu runs, ranged 0.2 to 0.5 across YOLOv8s seeds, and
did not carry to the Mediterranean source (F39). So a threshold is a *deployment setting* to be chosen per model on data from the deployment.

### 5.13 Counting versus detecting (F33, F36)

The team's end use is a fish *count*. Per-frame sums come out about a fifth low (79 to 84% for YOLOv8s on sparse validation footage) and 61%
recall on crowded OzFish suggests undercounting there; but a *maximum* over a clip (MaxN) is inflated because a single frame with several false
alarms sets it. Different statistics have different error signs. A smoothed statistic (for example a median over frames) is safer, and its error profile
must be measured on the deployment footage. This was not tested on a deployment.

### 5.14 Test-set discipline and why we disclose scoring the sealed set twice (Background; eval_protocol items 8, 9, 11)

Reusing a test set to choose among options leaks information from it into the choices; the literature (Dwork et al. 2015) shows adaptive reuse can invalidate estimates,
and Recht et al. (2019) and Roelofs et al. (2019) found less damage in practice than feared. The project's rule: the sealed set is scored once, on
a pre-declared candidate; the second pass (F37) re-measures the *same fixed models* only to attach uncertainty, with the rules committed first, nothing chosen from it, and disclosure in the paper.
That is not adaptive reuse, but it is a second look and is stated as such.

### 5.15 Why we kept negative results (Principle)

F12, F16, F25 to F27 and F23 (augmentation) are results of "no gain". Reporting them stops others repeating them and
shows the data finding is not a product of not trying other levers. Several are one-seed screens, so "did not help in this test"
is the claim, not "can never help".

### 5.16 Licensing, briefly (Background; `CLAUDE.md`, `NOTICE.md`, `docs/report/LICENCE_OPTIONS.md`)

Ultralytics YOLOv8 is AGPL-3.0, and the licence reaches the weights initialised from the COCO-pretrained `yolov8n.pt`. Conveying a copy of software or weights outside
the institution triggers the obligations. The decision (2026-10-11) is AGPL-3.0 for everything; the datasets keep their own licences (OzFish CC BY 3.0 AU needs attribution; the converted labels are derivative works; DeepFish and CFC are MIT; Kakadu CC BY 4.0). A permissive alternative exists in principle (YOLOX-nano, NanoDet-Plus, or the BSD
Faster R-CNN code of F40), but its pretrained-weight licence is unverified.

---

## 6. Decision log

| Decision | Alternatives considered | Reason | Evidence |
|---|---|---|---|
| Detection (boxes) instead of LCFCN counting | continue point-supervised counting | reason not written down in the repo; probably that boxes give location and count and box labels exist (inferred) | charter |
| Split by site/clip, assert no overlap | per-image split | near-duplicate frames inflate scores | F20, F38 |
| 960 px as the main resolution | 640 | +0.087 mAP@50, small fish; GPU cost modest (F6, F8) | F1, F6 |
| Engines built at the evaluated shape, on the board | square engine; build on PC | accuracy and latency must describe the same workload; engines are not portable | F6, correction |
| Same-machine controls | compare with the Kaggle baseline | machine effect 0.031 | F14 |
| Kakadu as train-only, never a judge | split Kakadu | any split leaks | F13 |
| Sealed set: once, three seeds, rules first | choose best seed from it | avoid tuning on the test | F33 |
| Ship YOLOv8s seed 2 | seed 0, 1 | best on validation (0.829); chosen *before* looking at the sealed result for it; the honest sealed figure is the three-seed mean | PROGRESS section 8 |
| YOLOv8s over YOLOv8n | n (faster) | +0.062 mAP@50 for 9 ms per frame | F28, F32 |
| Do not use tiling, 1280 px, P2 head | adopt | no gain | F12, F25 to F27 |
| Unseen-source test on a Mediterranean set | rely on the sealed OzFish set | sealed set shares collections with training | F33, F39 |
| Report YOLO-Fish YOLOv4 scoring higher than ours | omit external comparison | honesty; also states size/data differences | F39 |
| AGPL-3.0 | permissive path | simplest and compatible with Ultralytics | NOTICE.md |
| Cut the YOLO-Fish AP figure from reliance | wait for the paper | no claim depends on it | paper.md flag |

---

## 7. Mistakes and corrections (the useful ones)

The full table is `docs/PROGRESS.md` section 6 (16 rows) and bugs in section 7. The patterns:

1. **Inherited numbers were wrong** (fan, power-mode swing, INT8 "required", 60 fps "fits"). Rule: a hardware number is
   measured on the board or not stated.
2. **The wrong comparator.** The Kaggle baseline was compared with a server run; machine alone moved the score by 0.031.
3. **Too few seeds.** Two seeds looked tight (spread 0.008), the third landed 0.055 above.
4. **Bundled changes.** The first combined recipe changed data and augmentation together; the split test (F23) separated them.
5. **One example generalised.** One greyscale frame suggested heavy colour dependence; the full set showed 7 to 9%.
6. **Over-claiming from intervals.** "OzFish effect holds on three seeds" was narrowed once clips were resampled (F35).
7. **Process slips.** The protocol said the sealed set had never been run (F2 had run it); a queued analysis step failed because a script was not copied to the server; someone emptied `data/processed` on the shared server while a run was training (recovered from raw data).
8. **Tooling.** `cuda-python` pinned to a yanked release; a shell `pkill -f` killed its own SSH session; the board's decimal-comma locale corrupted a logging helper.

---

## 8. What is established, what is suggestive, what is not shown

### 8.1 Headline numbers (each with its finding)

| Claim | Number | Evidence level | Finding |
|---|---|---|---|
| Resolution, in-domain | 960 beats 640 by +0.087 mAP@50 (3 seeds) | Established | F1 |
| Resolution gain on a new scene | only +0.013 (sealed OzFish, DeepFish-only models) | Established (3 seeds) | F2 |
| Why scores were low | random split 0.98 vs site split about 0.47 | Diagnostic | F20 |
| Same sites, split only | 0.989 vs 0.491 AP50; gap 0.498 (0.225 to 0.752) | Diagnostic with interval | F38 |
| Kakadu data | +0.2245 mAP@50 (3 seeds each); clip interval +0.234 (0.116 to 0.339) | Established | F18, F35 |
| OzFish data | +0.052 mAP@50; clip interval -0.001 to +0.091 | Suggestive | F24, F35 |
| YOLOv8s over YOLOv8n | 0.812 vs 0.750; clip interval +0.058 (0.012 to 0.104) | Established | F32, F35 |
| Sealed OzFish (YOLOv8s) | 0.700 mAP@50 (once); 0.697 AP50 (0.669 to 0.724) | Established for "new deployments of familiar footage" | F33, F37 |
| Unseen Mediterranean source | AP50 0.05 to 0.23, best (published YOLOv4) 0.227, ours 0.166 | Established as "poor" (single run, 18 groups) | F39 |
| Same data effect, other family | Faster R-CNN +0.105 / +0.036 AP50 | Screen | F40 |
| Adapt to one camera | 0.18 to 0.65 (25 frames), 0.72 (100), 0.76 (530); other cameras small, uncertain | Screen | F41 |
| Orin Nano, YOLOv8s, 25 W | about 50 ms serial; 24.3 fps over 30 min video; plateau 59.5 C max | Observed (one run each) | F28, F29, F34 |
| MAXN_SUPER | no change in network time; video fps unchanged | Observed (10-minute runs) | F31 |
| Counting | per-frame 79 to 84% counted; MaxN 119 to 150% | Observed (one split) | F36 |

### 8.2 Not shown (the honest list)

- No footage from a real deployment; no false-alarm measurement on real deployment footage.
- Device: one run per model, 25 W mainly (one 10-minute MAXN_SUPER run), no 15 W/7 W, no ambient temperature, no INT8, no on-device mAP, no hardware decode, 12-frame parity only.
- Single-seed screens: P2 head, augmentation alone, YOLOv8m, YOLO11s, 1280 px, tiling, Faster R-CNN, adaptation.
- Validation reuse: checkpoints were selected on the DeepFish validation split that also reports the headline.
- The OzFish held-out deployments may share reefs with training deployments, so those scores are optimistic.
- Annotation quality is one reviewer's sample (F11); crowd-sourced OzFish boxes mean unlabelled fish count as false alarms.
- The Mediterranean false alarms were not audited; whether many are unlabelled fish is unknown.
- Comparisons with published numbers: splits unverified; the YOLO-Fish paper is unread.

---

## 9. Open problems and what would change the conclusions

| Open problem | What would settle it | Effect on conclusions |
|---|---|---|
| Deployment-site performance | footage and labels from the intended site | tests the adaptation recipe (F41) and sets the real threshold |
| Selection and reporting share a split | a second validation source | removes the optimism in 0.81 |
| Device generality | repeated runs across 15/7 W, ambient logged, long runs, INT8, NVDEC | firms the 24 fps claim |
| Mediterranean false alarms | audit of "no label nearby" boxes | may raise the zero-shot score |
| Why diversity helps | ablate source by source with 3 seeds | turns a hypothesis into a mechanism |
| YOLO-Fish reported AP | the paper (closed access; an email to the authors has been drafted) | only affects related-work wording |
| Counting as the field metric | clip-level counts on deployment footage | decides which statistic the RL agent should read |

---

## 10. Questions to put to the other AI (starter list)

Beginner, to build the picture:

1. In plain language, what does a YOLO detector do between receiving a 1080p frame and printing boxes? Where can a small fish get lost?
2. What is the difference between mAP@50 and the number of fish found at a threshold of 0.25? When can they disagree?
3. Why does a random per-image split give 0.98 and a site split 0.47 on the same pipeline? What would you expect a "clip-level split" to give?
4. Why is a TensorRT engine built on the Orin instead of copied from my laptop? What does FP16 change?

Intermediate, to challenge the method:

5. Section 5.5: why does resampling clips give a wider interval than resampling frames, and when is a cluster bootstrap itself unreliable (F39 has one group with 886 of 1,516 images)?
6. F18 and F35 both show the Kakadu gain. Why is the OzFish gain "positive on every seed" yet not significant under clip resampling?
7. F14 says the machine changed the score by 0.031. What differs between a Kaggle T4 and an RTX 5060 run that could do that?
8. F8: the GPU is 8 to 11% of a frame. What specifically would you do to raise the frame rate on an Orin Nano, in order of expected benefit, and what would each need measuring first?
9. F31: why would MAXN_SUPER speed up JPEG decode but not the network?

Advanced, to find weaknesses:

10. Where could this project still be fooling itself? (Suggested angles: validation reuse; Kakadu resembling the validation sites; crowd-sourced labels; the sealed set sharing collections.)
11. F41 shows the COCO-only control equals our model on the adapted camera. What does that imply about the value of the 6,500-image training set, and what experiment would test it with three seeds?
12. F36: MaxN is inflated while per-frame totals are low. What statistic would you give a controller that needs a robust fish count, and how would you validate it?
13. Given the evidence, what is the minimum deployment experiment that would convince a sceptical reviewer?

### 10.1 Things an assistant often gets wrong about this project

- Calling F20's 0.98 a result. It is a diagnostic of leakage.
- Treating the sealed 0.70 as "performance on a new site". It is new deployments of familiar footage; the unseen-source score is 0.17 to 0.23.
- Quoting the 50 ms or 24 fps as general: it is one board, 25 W, FP16, serial pipeline or software-decoded video, one clip.
- Assuming the OzFish gain is proved or that augmentation helps (F23: no).
- Assuming published YOLO-Fish scores are comparable to ours (different splits, sizes; the paper is unread).
- Inferring pronouns or identity of the owner; refer to "the owner".

---

## 11. File map for the other AI (what to attach alongside this file)

| File | What it adds |
|---|---|
| `docs/PROGRESS.md` | every number with run folders and limits (long) |
| `docs/eval_protocol.md` | the rules, including items 8 to 12 fixed before runs |
| `docs/GUIDE.md` | the concept guide from the basics (written at F27) |
| `docs/RESEARCH.md` | the earlier, shorter narrative |
| `docs/data_audit.md` | dataset statistics, box sizes, overlaps |
| `docs/hardware.md` | the verified board facts |
| `docs/report/paper.md` | the draft paper |
| `docs/report/figures/` | the five figures (leakage, examples, device, unseen source, adaptation) |
| `CLAUDE.md` | project rules and constraints |

Code that matters for the methods: `scripts/prepare_dataset.py` (site split), `scripts/bootstrap_ci.py` (cluster
bootstrap and AP routine), `scripts/error_analysis.py`, `scripts/export_onnx.py`, `edge/build_engines.py`,
`edge/trt_runner.py`, `edge/sustained.py`, `scripts/engine_parity.py`, `scripts/eval_unseen.py`,
`scripts/prepare_adaptation.py`, `scripts/eval_adaptation.py`.

## 12. Where to go next (for the paper, not the story)

The remaining paper work is a consistency pass (numbers and cross-references), cutting to the venue's page limit once it is chosen, author and
affiliation details, and the release decisions (repository address, weights). The tasks that need the owner are listed in `docs/PROGRESS.md` section 8
and the session notes; none require new experiments.
