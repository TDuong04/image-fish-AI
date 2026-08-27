# Dataset audit

**FD-014.** Regenerate with `python scripts/audit_dataset.py <dir>`.

Two datasets are on disk and prepared. Both came from the YOLO-Fish authors'
Google Drive links, which were listed in `fish-research/README.md` and had never
been tried. Neither required the Pawsey portal.

---

## The finding that drives the whole project

**Fish are small relative to the frame, and the standard 640 px input throws
away a large fraction of them.**

Box size, measured as √area at native 1920×1080, scaled to candidate input sizes:

| Input | DeepFish median | <32 px | <16 px | OzFish median | <32 px | <16 px |
|---|---|---|---|---|---|---|
| 416 | 18.9 px | 77.5% | 38.3% | 14.5 px | 80.6% | 54.2% |
| 512 | 23.3 px | 68.3% | 25.2% | 17.9 px | 73.8% | 45.1% |
| **640** | **29.1 px** | **56.3%** | **14.4%** | **22.4 px** | **65.3%** | **34.8%** |
| 800 | 36.4 px | 40.9% | 5.9% | 28.0 px | 55.9% | 26.5% |
| 960 | 43.7 px | 29.1% | 2.3% | 33.5 px | 48.0% | 20.4% |
| 1280 | 58.3 px | 14.4% | 0.2% | 44.7 px | 34.8% | 12.4% |

32 px is COCO's small-object boundary. **16 px is roughly the floor for a
stride-8 P3 head** — two feature cells across the object.

At the conventional 640, **14% of DeepFish boxes and 35% of OzFish boxes fall
below that floor**. A model trained at 640 is not merely inaccurate on those
fish; it is close to structurally unable to represent them.

**This is the project's central tension.** Recovering small fish means raising
input resolution. Raising input resolution is the most expensive thing you can
ask of a Jetson Orin Nano. The accuracy-per-watt frontier between those two is
the actual research contribution — not a parameter to be tuned quietly.

Practical consequences:
- `imgsz` is a **first-class experimental variable** (FD-022), not a default.
- A **P2 head** (stride 4) is worth testing: it targets exactly this failure
  mode at lower input cost than upscaling.
- **Tiling** is a live option for inference and should be measured against plain
  upscaling rather than assumed better or worse.
- Any published FPS number must state its input size, or it means nothing.

---

## DeepFish — training set

`data/processed/deepfish`, from `deepfish_darknet.zip` (1.09 GB).

| | |
|---|---|
| Images | 6517 (5178 train / 1339 val) |
| Boxes | 15463, single class `Fish` |
| Native resolution | 1920×1080, uniform |
| Boxes per image | mean 2.37, median 1, p95 9, max 16 |
| **Empty frames** | **2012 (30.9%)** |
| Structure | 20 sites × 68 clips |
| Frames per clip | median 73, max 462 — **consecutive video frames** |

**Explicit negatives.** `Nagative_samples/` (sic) supplies 2012 no-fish frames.
This is what makes false-positives-per-empty-frame measurable at all (FD-024),
and it is the reason the DeepFish val split — not OzFish — is where that metric
is reported.

**Leakage risk was severe and is now handled.** Clips run to 462 consecutive
frames of the same scene. A random per-image split would have scattered ~73
near-identical frames of one clip across train and val. Splits are made over
whole **sites**, and `prepare_dataset.py` asserts no site appears on both sides.

Val draws from 6 of 20 sites: `7398, 7623, 9862, 9870, 9892, 9898`.

---

## OzFish — held-out cross-dataset test set

`data/processed/ozfish_eval`, from `ozfish_test.zip` (60 MB). Never trained on.

| | |
|---|---|
| Images | 352, all val |
| Boxes | 8329, single class |
| Native resolution | 1920×1080, uniform |
| Boxes per image | mean **23.66**, median 16, max **232** |
| **Empty frames** | **0 (0.0%)** |
| Source clips | 297 distinct, median 1 frame each |

Filenames (`A000010_L-avi-43493`) preserve the source video, so grouping is
intact — the acceptance criterion FD-011 was written to protect.

**Two properties that constrain what this set can measure:**

1. **No empty frames.** It cannot produce a false-alarm rate. The cross-dataset
   number is silent on false positives, and the report must say so.
2. **232 boxes in the densest frame.** Ultralytics' default `max_det=300` is not
   exceeded but is close enough to check rather than assume.

---

## Domain shift between the two

| | DeepFish (train) | OzFish (test) |
|---|---|---|
| Boxes per image | 2.37 | **23.66** |
| Empty frames | 30.9% | **0%** |
| Median box @640 | 29.1 px | **22.4 px** |
| Scene | reef habitat, sparse | BRUVS bait-station, schooling |

These are **very different distributions**, not two samples of one. Training on
DeepFish and testing on OzFish measures a large domain shift: denser scenes,
smaller fish, no negatives.

That makes it an honest generalisation stress test and a legitimate FD-026
result — but it is not a like-for-like estimate of field performance, and
reporting it as one would overstate what was measured.

---

## Open gaps

- **Annotation quality is unmeasured** (FD-105). OzFish boxes came from
  SageMaker Ground Truth crowd consensus. An unlabelled fish scores as a false
  positive, so every FP figure is an upper bound until this is quantified.
- **No OzFish training split.** Only the authors' 352-image test set was
  recoverable. Training on OzFish still depends on FD-113.
- **No target-domain set** (FD-016) — no real deployment footage yet.
