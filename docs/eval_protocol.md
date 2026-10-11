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

## 9. Rules added during the improvement programme (2026-10-04)

These were applied from 2026-10-04 and are why some earlier numbers are not comparable with later ones.

1. **Compare only against a same-machine, same-batch control.** The same seed on the Kaggle T4 (batch 23, Ultralytics 8.4.159, torch 2.10.0) and
   on the GPU server (batch 10, 8.4.172) differs by 0.031 mAP@50 (PROGRESS F14). A comparison across machines, batch sizes or
   library versions is not evidence. Record the machine, batch size and library version with every run (`env.txt`, `args.yaml`).
2. **Seed spread is measured, not assumed.** Control (DeepFish only, server, batch 10): mAP@50 0.4562, 0.4970, 0.4669 (range 0.041,
   against 0.034 on Kaggle); mAP@50-95 0.2957, 0.2966, 0.3106 (range 0.015). A difference below the spread is not a result.
3. **One-seed runs are screens.** They can nominate an idea, not support a claim. An entry from one seed says "one seed" in its title.
4. **A gain larger than about 0.05 mAP@50 triggers a leak check before it is reported.** Method: thumbnail nearest-neighbour distance
   between the added training data and the validation split, plus a look at the closest pairs. It can rule out near-duplicates; it cannot
   rule out shared scenes or sites, and it cannot show that a gain generalises (PROGRESS F13, F17).
5. **Data added to training must be checked against both judges** (the DeepFish val split and the sealed OzFish set) for overlap, by
   content and not only by file name.
6. **The operating threshold and the checkpoint are chosen on the DeepFish val split only.** The best-F1 threshold moved from 0.4-0.6
   (earlier runs) to 0.2-0.25 (Kakadu runs); a deployment threshold is a setting, not a property of the model.
7. **Error analysis is part of a result**: `scripts/error_analysis.py` at IoU 0.5 and conf 0.25 (found and missed, false alarms by
   cause, recall by box size), so a gain in mAP is attributed to finding more fish or to fewer false alarms (F16 is the example).
8. **The sealed OzFish set is run once, at the end, on the final candidate, and not used to choose between candidates.** (Correction
   2026-10-08: an earlier version of this line said it had not been run on any model. It had been run on the six DeepFish-only baseline models
   of PROGRESS F2, before any OzFish data were used for training.)
9. **Sealed-set run of 2026-10-08, fixed before it was run.** The final candidate is the YOLOv8s recipe (DeepFish + Kakadu + OzFish train data, 960 px,
   default augmentation). Its **three seeds** (`20261005-yolov8s-960-kakadu-ozfish-s0`, `20261007-yolov8s-960-kakadu-ozfish-s1`, `20261007-yolov8s-960-kakadu-ozfish-s2`)
   are each evaluated **once** with `scripts/evaluate.py --data ozfish_eval` (imgsz 960, conf 0.001, NMS IoU 0.7, the script's defaults) and reported as a mean and
   range. No seed is chosen from the result, no threshold is tuned on it, nothing is re-run, and the YOLOv8n arm is not evaluated on it. The test frames come from
   deployments excluded from the OzFish training frames, but from the same collections, so this measures new deployments of the same footage and not an unseen source.

10. **Unseen-source test (DEEPECOMAR), fixed before it was run, 2026-10-11.** The test set is the Mediterranean fish dataset of Catalan and Alvarez-Ellacuria
    (Zenodo 7534425, CC BY 4.0: 1,548 images, 17,965 boxes, 20 species and the categories `stain`, `small` and `fish poly`). No image of it, nor any image from
    that project or region, was used for training, validation, threshold choice or model selection. Rules, all fixed here:
    - **One class.** Every box of a fish class, `small` and `fish poly` is a fish; `stain` boxes (lens stains) are dropped and are not fish. A second score excludes the `small` category.
    - **Models, each evaluated once:** the YOLOv8s recipe (3 seeds, F32), the YOLOv8n arms (DeepFish only, plus Kakadu, plus Kakadu and OzFish; 3 seeds each), and the published YOLO-Fish
      detectors (the `merge` darknet weights: yolo-fish-1, yolo-fish-2, yolov3, yolov4) as the external comparison; a detector from a second, permissively licensed family only if it is trained before the run.
    - **Scoring:** one AP routine for every model (101-point AP at IoU 0.50 and 0.50 to 0.95), ours at 960 px and darknet models at their configured input size; found fish and false alarms at confidence 0.25;
      no threshold is tuned on this set. Frames are scored as given; if the dataset has no clip or site identifiers the intervals are frame-level and are labelled as such.
    - **No selection from the result:** nothing is chosen, tuned or re-run because of it. A failure to read a model (for example a darknet file that does not load) may be repaired and retried, a poor score may not.
    - **YOLO-Fish caveat stated in advance:** those weights were trained on DeepFish and OzFish frames, so DEEPECOMAR is also new to them, but their training data and ours overlap in source, which is why the comparison is on a source neither saw and not on our validation split.

    Addendum, written before any model was scored on it, after inspecting the files only: the stain class is `taca` (Catalan for stain; checked by drawing its boxes, which are blurred lens blobs, not fish);
    the export holds 1,590 image files of which 1,516 are unique source frames (74 frames appear twice under different Roboflow hashes; the first copy is kept; train and valid folders share no source frame);
    after conversion the set has 1,516 images, 17,157 fish boxes (16,488 without the `small` category), 794 stain boxes dropped and 30 images without a fish.

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
