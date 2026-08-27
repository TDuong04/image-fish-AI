# Evaluation protocol

**FD-005. Fixed before results exist. Changing anything here after seeing a number
requires saying so in the report.**

The point of writing this first is that every degree of freedom left open — which
threshold, which split, which seed, which checkpoint — is a knob that can be turned
until the answer looks good. Fixing them in advance is what separates a measurement
from a search.

---

## 1. Task and class definition

Single-class detection: **`fish`**, class id `0`.

Both datasets in use are already class-agnostic — OzFish's box annotations carry no
species labels, and DeepFish's darknet export uses a single `Fish` class. There is
no species collapse being performed and therefore no species-related limitation to
declare.

---

## 2. Datasets and splits

| Dataset | Role | Images | Boxes | Empty | Groups |
|---|---|---|---|---|---|
| `deepfish` train | training | 5178 | 14078 | 29.9% | 14 sites |
| `deepfish` val | model selection | 1339 | 1385 | 34.7% | 6 sites |
| `ozfish_eval` | **held-out cross-dataset test** | 352 | 8329 | 0% | 297 clips |

**Splitting rule.** Frames come from continuous video — DeepFish clips run to 462
consecutive frames. Splits are made over whole **capture sites**, never per image.
A per-image split would place adjacent, near-identical frames on both sides of the
boundary and inflate mAP by an unknown margin.

`scripts/prepare_dataset.py` asserts that no group appears in two splits and fails
rather than warning. Split membership is committed under `splits/` and is not
regenerated per run.

**Validation diversity.** The val side must draw from **at least 4 groups**. An
earlier split hit the 20% image target using only 2 sites; that measures two sites,
not the dataset.

**`ozfish_eval` is sealed.** It is not used for model selection, early stopping,
threshold tuning, or quantisation calibration. It is read once per reported model.

---

## 3. Primary metrics

- **mAP@50** and **mAP@50-95**, COCO-style, single class.
- **Precision, recall, F1** at the operating threshold defined in §4.
- **False positives per empty frame**, reported separately from mAP.

mAP is computed at `conf=0.001`, `iou=0.7` for NMS. That low confidence is an
integration bound for the PR curve, **not** an operating point, and must never be
reported as one.

**Why FP-per-empty-frame is first-class:** DeepFish is ~30% empty frames and BRUVS
footage in the field is mostly empty water. A detector that fires on kelp, bubbles
and particulates produces a useless count, and mAP does not surface that.

**Caveat carried into the report:** `ozfish_eval` contains **zero** empty frames, so
it cannot measure this metric at all. FP-per-empty-frame is reported on the DeepFish
val split only, and the cross-dataset number is silent on false alarms.

---

## 4. Operating threshold

Selected on the **DeepFish val split**, never on `ozfish_eval`.

Criterion: the confidence maximising F1 on val. Recorded per model in its run
directory and quoted in every table that reports P/R/F1.

**Thresholds are re-selected per precision.** Quantisation shifts the score
distribution, so reusing an FP32 threshold on an INT8 engine compares two different
operating points and silently attributes the difference to quantisation.

---

## 5. Statistical rules

- **≥3 seeds** for any comparison that drives a decision. Report **mean ± range**.
- **Never report the best seed.** `scripts/train.py` prints the spread specifically
  to make single-seed claims uncomfortable.
- A difference smaller than the seed spread is **not a result**. State the observed
  spread whenever claiming one configuration beats another.
- Frames from one clip are **not independent samples**. Per-frame metrics over
  adjacent frames overstate confidence; prefer clip-level aggregates where the claim
  allows it, and say so where it doesn't.

---

## 6. Model selection

- Train to convergence with early stopping on **val mAP@50-95**, `patience=50`.
- A fixed small epoch count is not convergence. Report epochs trained and whether
  early stopping fired.
- Select the checkpoint by best val mAP, not last epoch.

---

## 7. Latency (deferred to E4, defined here)

Every latency number is **end-to-end**: decode → preprocess → inference → NMS →
postprocess. GPU-kernel-only timing is not a deployment result.

- Discard warm-up iterations; report **median and p95** over ≥500 frames.
- Every FPS figure cites its `nvpmodel` mode and `jetson_clocks` state.
- Report **sustained** throughput after ≥30 min of load, not just burst.
- Desktop FPS is never compared against board FPS in the same table without saying
  so in the same sentence.

---

## 8. Reporting rules

- Report accuracy at every stage of the export chain (PyTorch → ONNX → TensorRT
  FP16 → INT8) as a **parity check**. TensorRT FP32 is not a deployment
  configuration and is not a results-table row.
- Any number from a session that no longer exists is marked unreproduced.
- Published numbers computed on different splits or image sizes are labelled as such
  in the same sentence, never placed in a bare comparison column.

---

## Known limitations, declared up front

1. `ozfish_eval` has no empty frames → no cross-dataset false-alarm measurement.
2. `ozfish_eval` is 352 images → wide confidence intervals on the cross-dataset number.
3. Train and test distributions differ sharply — DeepFish averages 2.4 boxes/image,
   OzFish 23.7. The cross-dataset result measures a large domain shift, which is
   informative but is not a like-for-like generalisation estimate.
4. DeepFish val draws from 6 sites; site-to-site variation is not separable from
   model variation at that count.
5. Annotation quality is unmeasured (FD-105). An unlabelled fish scores as a false
   positive, so FP rates are an upper bound until label noise is quantified.
