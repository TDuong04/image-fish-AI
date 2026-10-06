# A guide to this project, from the basics

Written for the project owner, to explain what everything is, why it works, and how each piece of this project fits
together. It assumes no machine learning background. It explains mechanisms and vocabulary; it does not repeat every
result. For the results and their evidence, see `docs/PROGRESS.md` (findings F1 to F27) and `docs/RESEARCH.md`
(the short story). Where this guide quotes a number it names the finding to look up.

Contents

1. What the project is for
2. What "object detection" means
3. How YOLO works
4. How a model learns (training)
5. How we measure a detector
6. Data, and why splitting it is the hardest part
7. How to run a fair experiment
8. What we did and found, in order
9. Why the detector misses fish
10. The Jetson Orin Nano: getting the model onto a small computer
11. Licences
12. A map of the repository
13. Glossary

---

## 1. What the project is for

We want a program that looks at underwater video, such as the footage from baited remote underwater video stations
(BRUVS), and draws a box around every fish. That is useful for counting fish and monitoring ecosystems without a
person watching hours of footage.

The program must run on a **Jetson Orin Nano**, a small computer (about the size of a credit card plus a heatsink)
that could sit in a field unit. That constraint matters: a model that is accurate but needs a big server is no use
there. So the project has two halves:

- **Accuracy:** how good is the detector, measured honestly?
- **Speed on the device:** how many frames per second does it process on the real Orin Nano?

The report we are building has to back both claims with measurements, not guesses.

---

## 2. What "object detection" means

There are several related computer-vision tasks:

| Task | Question it answers | Output |
|---|---|---|
| Classification | "What is in this picture?" | A label for the whole image |
| **Detection** (ours) | "What is where?" | A list of **boxes**, each with a label and a confidence |
| Segmentation | "Which pixels belong to each object?" | A mask per object |
| Counting | "How many?" | A number (an earlier experiment of ours used this, LCFCN) |

A detector's output for one image is a list like: *box at (x1, y1, x2, y2), class "fish", confidence 0.83*. The
**confidence** is a number from 0 to 1: how sure the model is that the box contains a fish. We usually keep only
boxes above a **confidence threshold** (we use 0.25 as the default operating point).

We have **one class**, `fish`. The model does not say which species. That keeps the task simple and is all that
the datasets' boxes support.

Detection is harder than classification because the model has to find the objects as well as recognise them, and a
photo can have none, one or hundreds.

---

## 3. How YOLO works

**YOLO** stands for "You Only Look Once". Older detectors looked at an image many times (propose regions, then
classify each). YOLO looks at the image once and predicts all boxes in one pass, which is why it is fast enough for a
small device.

We use **YOLOv8** (and tried YOLO11) from the Ultralytics library. Sizes: **n** (nano, 3 million parameters), **s**
(small, about 11 million), **m** (medium, about 26 million). A **parameter** is one learned number inside the
network; more parameters usually means more accuracy and more compute.

### The three parts

1. **Backbone.** A stack of convolutional layers. A **convolution** slides a small filter over the image to detect
   a pattern (an edge, a texture, a fin shape). Early layers find edges and colours; deeper layers combine them into
   parts and whole objects. The image gets smaller and the descriptions richer at each stage.
2. **Neck.** Combines the backbone's outputs from several depths so the model can see both fine detail (good for
   small fish) and the big picture (good for large fish). Each depth is called a **scale**, defined by its
   **stride**: stride 8 means one output cell per 8x8 pixels, stride 16 per 16x16, stride 32 per 32x32.
3. **Head.** For every cell at every scale, predicts: is there an object centred here, where exactly are its box
   edges, and how confident. YOLOv8 is **anchor-free**: it predicts the box directly, not an offset from preset box
   shapes.

### Why small fish are hard

A fish that is 10 pixels across fits inside a single stride-8 cell, so the model has very little information about
it. By the time the image has passed through the deeper layers it is nearly gone. That is why recall on tiny fish
is the weak point (section 9). A "P2" head adds a stride-4 output to see finer detail; we tried it and it did not
help here (F12).

### Predictions are many; NMS keeps the best

The head makes thousands of candidate boxes, and several overlap the same fish. **Non-maximum suppression (NMS)**
fixes that: take the highest-confidence box, delete any other box that overlaps it too much (above an IoU
threshold, see section 5), repeat. Without NMS you would see five boxes on one fish.

### Image size

The image is resized to a fixed size before it goes in (`imgsz`, in pixels along the long side). Video frames here
are 1920x1080. At `imgsz 960` the frame is shrunk to 960x544 (the model needs sides divisible by 32). At 640 it is
shrunk to 640x384. Lower size is faster but small fish shrink further. We trained at 640, 960 and (now) 1280. The
shape "544x960" is called the **letterbox** shape: the picture keeps its proportions and any spare space is padded.

---

## 4. How a model learns (training)

### The loop

1. Show the model a batch of images with the correct boxes (the **labels** or **ground truth**).
2. Compare its predictions to the truth with a **loss function**: a single number for "how wrong". For YOLO it has
   three parts: box position (`box_loss`), whether something is there and how confident (`cls_loss`), and a finer
   box-edge term (`dfl_loss`).
3. Compute how each parameter should change to reduce the loss (**backpropagation**) and nudge them
   (**gradient descent**, step size set by the **learning rate**).
4. Repeat. One full pass through all training images is an **epoch**. We train for up to 150.

### Vocabulary you will see in our logs and configs

| Term | Meaning |
|---|---|
| **Batch size** | Images processed per step. Bigger batch = smoother learning and faster per image, but more GPU memory. Our server GPU has 8 GB, so YOLOv8n gets about 10, YOLOv8s 5 and the 1280 px run 3. |
| **Pretrained / transfer learning** | We start from `yolov8n.pt`, a model already trained on COCO (80 everyday object classes, 330k images). It already knows edges, textures and animal shapes. We only teach it "fish". This needs far less data than training from nothing. |
| **Fine-tuning** | What we do: continue training the pretrained model on fish data. |
| **Seed** | The number that fixes the random choices (initial layer for the new class, image order, augmentation). Same seed and setup should give nearly the same result. Different seeds give different results; that spread is real and we measure it (section 7). |
| **Overfitting** | The model memorises the training images instead of learning "fish". Training loss keeps falling while the score on unseen images stops improving or falls. |
| **Validation set** | Images the model never trains on, used to choose settings and to know when to stop. |
| **Early stopping, patience** | If the validation score hasn't improved for `patience` epochs (we use 30), stop. This is why runs end at different epochs (88, 119, 71...). |
| **Checkpoint (`best.pt`)** | The saved weights from the epoch with the best **fitness**: `0.1 x mAP@50 + 0.9 x mAP@50-95`. Not necessarily the last epoch. |
| **Augmentation** | Randomly altering training images (flip, colour shift, scale, **mosaic**: stitching four images into one) so the model sees more variety and overfits less. We found stronger augmentation added nothing once the data was good (F23). |
| **AMP** | Automatic mixed precision: uses 16-bit numbers where safe, for speed and memory. |
| **Workers** | Background processes loading images. On Windows they must be guarded by `if __name__ == "__main__":` or the program launches itself repeatedly; that is why our Windows scripts use 0 workers. |

### Why 960 px beat 640 px (F1)

More pixels per fish means more information. At 960 the score was higher by 0.087 mAP@50 on the same data, but only
by 0.013 on an unrelated dataset (F2), so part of the gain was fitting the training scene better.

---

## 5. How we measure a detector

This is the part that most often confuses people. Take it in order.

### 5.1 IoU: when does a box count as right?

**Intersection over Union (IoU)** measures how much two boxes overlap: area of overlap divided by area of the union.
It is 0 for no overlap and 1 for identical boxes. We call a predicted box a hit if its IoU with a true box is at
least a chosen value; the usual one is **0.5** (half overlap).

### 5.2 True positives, false positives, false negatives

At a given confidence threshold, going through predictions from most to least confident:

- **True positive (TP):** a predicted box matches a real fish (IoU >= 0.5) not already matched. The fish was **found**.
- **False positive (FP) = false alarm:** a predicted box that matches no unmatched real fish. Causes: a duplicate on a
  fish already found, a badly placed box, or a box on background (rocks, shadows, bubbles).
- **False negative (FN) = missed fish:** a real fish no prediction matched.

### 5.3 Precision and recall

- **Precision = TP / (TP + FP)**: of the boxes the model drew, what fraction are real fish. High precision = few false alarms.
- **Recall = TP / (TP + FN)**: of the real fish, what fraction did it find. High recall = few misses.

They trade off. Lower the confidence threshold: the model draws more boxes, finds more fish (recall up) but also
more junk (precision down). Raise it: the reverse. For our best model at threshold 0.25, precision is about 0.89 and
recall about 0.72; at 0.10, recall is 0.78 and precision 0.75 (F24 and the error analysis).

**F1 score** combines them: the harmonic mean `2PR / (P + R)`. The threshold with the highest F1 is the "best-F1
threshold"; it differs from model to model and seed to seed.

### 5.4 AP and mAP: one number that ignores the threshold

Choosing a threshold is arbitrary, so people sweep it from very low to very high, plot precision against recall
(the **precision-recall curve**), and take the **area under it**: the **Average Precision (AP)**. A model that stays
precise as recall grows has a high AP.

- **mAP@50** = AP using IoU 0.5 as the "right" criterion. "mAP" is "mean AP"; with one class it is just AP.
- **mAP@50-95** = the average of AP at IoU 0.50, 0.55, ..., 0.95. Much stricter: it rewards boxes drawn tightly.
  It is always lower than mAP@50 (ours: 0.82 and 0.56).

Rule of thumb for what they tell you: mAP@50 says "does it find the fish?", mAP@50-95 says "and are the boxes tight?"

### 5.5 Our extra measurements

mAP hides things, so we add:

- **Fish found / false alarms** at threshold 0.25 (counts, not percentages).
- **Recall by size**: fish bucketed by size as seen at 640 px input: under 16 px, 16-32, 32-96, over 96. This is how we
  learned small fish are the problem.
- **False alarms on empty frames**: BRUVS footage is mostly empty water, so false alarms there matter more than
  mAP suggests.
- **Cross-dataset score**: test on a different dataset than the one trained on.

---

## 6. Data, and why splitting it is the hardest part

### 6.1 The datasets

| Dataset | What it is | How we use it |
|---|---|---|
| **DeepFish** | About 6,500 images from 20 habitats in tropical Australia, one box per fish | The main benchmark. Split by capture site into train and validation. |
| **Kakadu** | 44,112 frames from remote underwater video in northern Australia | Extra training data only (we keep 7,318 frames). |
| **OzFish** | BRUVS frames with crowd-sourced boxes (AIMS, UWA, Curtin) | Part for training, part held out, and a **sealed test set** we have not touched. |
| **CFC** | Sonar images | Not used. Sonar is a different kind of image from optical video. |
| **YOLO-Fish** | A published detector (weights only) | A reference point. |

### 6.2 The three kinds of data split

- **Training set:** the model learns from it.
- **Validation set:** used during development to choose settings and the checkpoint. Using it to decide things
  slightly "leaks" it into the model's design.
- **Test set:** touched once, at the very end, for the final honest number. Ours is the **sealed OzFish test set**.
  Looking at it repeatedly and tuning would destroy its value, so scripts refuse to use it.

### 6.3 Why you can't split video frames randomly (the key lesson)

A video of 462 frames shows the same scene, the same rocks, nearly the same fish in each frame. If you randomly
send half the frames to training and half to validation, the model sees almost-identical pictures in both. It scores
high by remembering, not by generalising. This is **data leakage**.

We proved it (F20): the same model and pipeline score **0.98 mAP@50 on a random per-image split** but about 0.47 on
a split by capture site. That gap is why our numbers are far below the 0.9+ that some papers quote: they very likely
use easier splits (we could not verify). Our rule: **split whole sites, never single images**, and the code checks it.

For Kakadu there were no site labels, so we grouped frames by visual similarity (shrink each to a tiny grayscale
thumbnail, measure distances, link close ones into scenes). Even then a split inside Kakadu still leaked (F13), so
Kakadu is training data only.

### 6.4 Why more data helped, and what kind

Adding Kakadu raised DeepFish validation mAP@50 from 0.47 to 0.70, and OzFish added 0.05 more. The detector is not
mainly limited by its design; it is limited by how many kinds of scene (water colour, fish species, camera, lighting)
it has seen. This is the most important result so far.

### 6.5 Honest caveats about our data

- DeepFish validation resembles Kakadu (similar tropical scenes), so the Kakadu gain may be flattered.
- The held-out OzFish deployments might share reefs with the training ones.
- Frames from one video are not independent samples, so per-frame scores look more certain than they are.

---

## 7. How to run a fair experiment

Machine learning results are noisy. Without rules it is easy to fool yourself. Ours:

- **Control.** Every change is compared against the same recipe without the change, run on the same machine, same
  batch size and library version. Moving machine or batch size alone shifted scores by about 0.03 mAP@50, which can
  be bigger than the effect you are trying to measure.
- **Seeds.** Retrain with different seeds. The spread between seeds (about 0.04 mAP@50 for our baseline) is the noise
  floor. A difference smaller than that is not a result. One seed is a **screen** (a hint that justifies more runs);
  **three seeds** supports a claim, reported as mean and range.
- **Pre-registered rules.** For tiling (F26) we fixed the four merge rules before looking, and reported all four,
  because choosing the best of many after the fact (on the same data you test on) overstates the result.
- **Sealed test.** The final number is computed once, on data nothing was tuned against.
- **Logging.** Every run folder holds its config, metrics, environment and the git commit, so any number can be
  regenerated. `tests/test_docs.py` fails if a run is not documented.
- **Corrections table.** When we find we were wrong, we add a row (PROGRESS section 6) instead of editing history.

---

## 8. What we did and found, in order

1. **Baseline.** YOLOv8n at 640 and 960 px on DeepFish, 3 seeds each. 960 won in-domain (F1) but the advantage
   mostly vanished on a different dataset (F2). It is concentrated in medium fish (F3). The model tolerates losing
   colour but not wrong colour (F4).
2. **Put it on the Jetson.** Exported to ONNX, built TensorRT engines on the board, checked they match the original
   (F7), measured speed (F5 to F9): about 41 ms per frame end to end at 960 px, and 30 minutes sustained at about 34
   fps without throttling.
3. **Why is it weak?** Error analysis: misses, not wrong boxes, limit the model (F10); the labels are mostly sound
   (F11); empty frames explain very little (F19); the split protocol explains the gap to published scores (F20).
4. **Try model changes.** A stride-4 head (F12), stronger augmentation, larger models YOLOv8s/m and YOLO11s (F14 to
   F16): small gains on one seed each, later shown to be mostly a data effect.
5. **Get more data.** Kakadu raised the score the most (F17, F18, three seeds), and it held on unseen OzFish
   deployments (F21). OzFish training frames added 0.05 more (F22 to F24).
6. **Bigger model on the best data.** YOLOv8s scored best: 0.82 mAP@50 (F24, one seed).
7. **Try tiling at test time.** Cutting the frame into pieces found more small fish but many more false alarms, so the
   score fell (F25, F26). Dropped.
8. **Now:** train at 1280 px (F27 when finished) to see whether more pixels per fish raises recall.

---

## 9. Why the detector misses fish

From the error analysis of our best model on DeepFish validation (1,385 real fish):

- **Found 1,004; missed 381.** Of the misses, 271 are fish the model **does** see but scores below the 0.25
  threshold, and 105 it does not see at all.
- **Size matters most:** recall is 0% under 16 px, 26% for 16-32 px, 77% for 32-96 px and 99% above 96 px.
- **Why small and faint fish are missed:** few pixels to learn from; dim, dark, blurred or camouflaged fish look
  like background; and many tiny fish in the labels are ones even people disagree on (the label audit, F11).
- **What would raise recall:** more diverse data (the biggest lever so far), higher training resolution (running now),
  a larger model (it helped: 94 more fish found, at higher cost), cleaner labels, and, as a trade-off only, lowering
  the confidence threshold.

False alarms are not the problem: about 89% of boxes are real fish, comparable to published figures, although
published papers use different splits so the comparison is loose.

---

## 10. The Jetson Orin Nano: getting the model onto a small computer

### 10.1 The hardware

- A small ARM computer with an NVIDIA Ampere GPU (1,024 CUDA cores) and 8 GB of memory **shared** between the CPU and
  GPU ("unified memory"), so the operating system and your program draw on the same 8 GB.
- It runs **JetPack** (NVIDIA's Linux, here L4T 36.4.7, CUDA 12.6, TensorRT 10.3).
- **Power modes** (`nvpmodel`): 7 W, 15 W, 25 W or MAXN_SUPER. More power allows higher clock speeds and more
  speed, but more heat. **Every speed figure must say which mode it was measured in.**
- No hardware video encoder and no DLA (a special accelerator some Orin modules have), so everything runs on the one
  GPU and writing annotated video is slow CPU work.
- It has an active fan, but a short test on a cold board proves nothing: **thermal throttling** (the chip slowing
  itself to avoid overheating) shows up after minutes, so we ran 30 minutes.

### 10.2 From PyTorch weights to a fast engine

1. **PyTorch** (`best.pt`): the training format. Slow and heavy on the board; not used there.
2. **ONNX:** a neutral file format describing the network. We export it on the desktop.
3. **TensorRT engine:** NVIDIA's inference compiler turns the ONNX into an optimised program for one specific GPU. It
   fuses layers and picks the fastest kernels. **An engine only works on the GPU, TensorRT and driver it was built
   for**, so it must be built **on the Orin itself**; you cannot ship one built on the desktop.
4. **Precision:** **FP32** (32-bit numbers, slow), **FP16** (half the size, faster, almost no accuracy loss; what we
   deploy) and **INT8** (8-bit integers, fastest but needs calibration and can lose accuracy; a research question for
   us, not a necessity, since the model is only about 6 MB at FP16).

### 10.3 What "FPS" means, and what it does not

A fair speed measurement is **end to end**: read and decode a frame, resize and pad it (letterbox), run the network,
run NMS, convert the boxes back. We found the GPU network itself is under 11% of the time per frame (F8); the rest is
CPU work. So a "GPU kernel time" benchmark is not a frame rate and we do not report it as one. We also discard the
first few "warm-up" runs and report the **median** and the **95th percentile** (the slow tail), not just the mean.

We also **evaluate and time at the same shape** the accuracy was measured at (544x960), otherwise latency describes a
different workload from the accuracy.

---

## 11. Licences

- Ultralytics YOLO is **AGPL-3.0**, a strong copyleft licence. The COCO-pretrained starting weights are covered too,
  so our trained weights are encumbered. The trigger is **conveying** a copy, not selling: if a copy of the software or
  weights leaves the institution (for example a unit given to a partner agency), the AGPL obligations apply.
- If that is a problem, alternatives are YOLOX-nano or NanoDet-Plus (permissive licences).
- Datasets: OzFish is CC BY 3.0 AU (credit required, and our converted labels count as a derivative); Kakadu is CC BY 4.0;
  DeepFish is MIT.

This is an open decision for you (PROGRESS section 8).

---

## 12. A map of the repository

| Where | What |
|---|---|
| `docs/PROGRESS.md` | The master record: what was built, every finding F1 to F27, every correction |
| `docs/RESEARCH.md` | The short research narrative and next steps |
| `docs/GUIDE.md` | This file |
| `docs/eval_protocol.md` | The measurement rules, fixed before results |
| `docs/data_audit.md` | What is in each dataset |
| `docs/TRAINING.md` | How to set up and train |
| `docs/OVERNIGHT.md` | How the unattended server runs work |
| `docs/tickets/BACKLOG.md` | Tickets FD-001 to FD-141 and their status |
| `src/fish/` | Shared code: paths, device selection, dataset grouping |
| `scripts/` | Entry points: `train.py`, `error_analysis.py`, `eval_tiled.py`, `prepare_*.py`, `queue_runner.py` |
| `configs/` | `train/` (model and recipe), `data/` (dataset lists), `env/` (per-machine settings), `plan_*.yaml` (job queues) |
| `edge/` | Code for the Jetson (TensorRT only; no PyTorch) |
| `runs/` | One folder per experiment with config, metrics and environment |
| `models/` | Weights (git-ignored) |
| `splits/` | Committed record of which images belong to which group and split |
| `tests/` | Automated checks, including that docs name every run |

**Three machines:** your Windows workstation (RTX 4060, 8 GB; evaluation and small jobs), the remote server `ai-gpu`
(RTX 5060, 8 GB; the long training runs, reached via Cloudflare SSH) and the Jetson (inference only, currently offline).
Paths never contain a drive letter: everything is computed from the repo root or `$FISH_DATA_ROOT`, and per-machine
settings live in `configs/env/<profile>.yaml` selected by `$FISH_ENV`.

**The queue runner** (`scripts/queue_runner.py`) runs a YAML list of steps one after another on the server, skipping
steps whose results already exist, so a long programme can run overnight and survive a restart.

---

## 13. Glossary

| Term | Plain meaning |
|---|---|
| Anchor-free | The model predicts boxes directly, not as adjustments to preset shapes |
| AP / mAP | Area under the precision-recall curve; "mean" over classes (we have one) or IoU thresholds |
| Backbone / neck / head | Feature extractor / scale mixer / box predictor |
| Batch | Images processed per training step |
| BRUVS | Baited Remote Underwater Video Stations |
| Checkpoint | Saved model weights |
| COCO | A large everyday-objects dataset; our pretrained starting point |
| Confidence | The model's 0-1 certainty for a box |
| Control | The same recipe without the change under test |
| Convolution | A sliding pattern filter, the basic building block of image networks |
| Epoch | One pass through the training set |
| FP16 / INT8 | 16-bit / 8-bit number formats for faster inference |
| Ground truth | The human-made correct boxes |
| IoU | Overlap measure between two boxes |
| Leakage | Test or validation information sneaking into training, which inflates scores |
| Letterbox | Resize keeping proportions and padding the rest |
| Loss | A number for how wrong the model is, which training reduces |
| NMS | Removes overlapping duplicate boxes |
| ONNX | A neutral model file format |
| Overfitting | Memorising training data instead of learning the task |
| Pretrained | Starting from a model already trained on other data |
| Precision / recall | Share of boxes that are real / share of real fish found |
| Seed | Fixes the random choices of a run |
| Stride | Pixels per output cell at a given scale |
| Tiling | Running the model on overlapping crops of a frame |
| TensorRT | NVIDIA's compiler for fast inference on its GPUs |
| Unified memory | CPU and GPU share one memory pool |
| Validation / test set | Held-out data used to tune / to give the final honest score |
