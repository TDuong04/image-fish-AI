# `scripts/`: entry points

Every script is a standalone Python entry point (`python scripts/<name>.py --help`). Descriptions below are the first line of each script's own docstring. A normal order: set up, get data, train, evaluate. The experiments are described in `docs/PROGRESS.md` (F1 to F41) and the rules they follow in `docs/eval_protocol.md`.

## Set up and check the machine

| Script | What it does |
|---|---|
| `bootstrap.py` | One cross-platform installer. Picks the right torch wheel for this machine. |
| `check_env.py` | Report whether this machine can train, and exit non-zero if it cannot. |

## Get and prepare data

| Script | What it does |
|---|---|
| `get_data.py` | Download, extract and prepare every dataset. One command, safe to re-run. |
| `fetch_url.py` | Download one large file over several connections, resumably, and verify its checksum. |
| `prepare_dataset.py` | FD-011 + FD-013: build an Ultralytics dataset with a leakage-free split. |
| `prepare_kakadu.py` | FD-135: turn the Kakadu archive into a single-class YOLO dataset with a leakage-free split. |
| `prepare_ozfish_train.py` | FD-113: turn the OzFish labelled frames into a single-class YOLO training set that excludes the sealed test. |
| `prepare_deepecomar.py` | Convert the DEEPECOMAR Mediterranean fish dataset into a single-class evaluation set (the unseen-source test). |
| `prepare_adaptation.py` | Time-ordered splits of the converted DEEPECOMAR set for the adaptation experiment (docs/eval_protocol.md item 12). |
| `prepare_random_split.py` | Build a DELIBERATELY LEAKY DeepFish split, only to place our numbers next to published ones. |
| `audit_dataset.py` | FD-014: characterise a YOLO-format dataset before training on it. |
| `audit_ozfish_overlap.py` | How much of the OzFish labelled set can be used for training without touching the sealed test set? |

## Train

| Script | What it does |
|---|---|
| `train.py` | Train a fish detector. One entrypoint, every machine. |
| `train_tv.py` | Train a detector from a second family (torchvision Faster R-CNN, MobileNetV3-Large FPN) on the same fish data. |
| `supervise.py` | Run training unattended: survive crashes, resume, and report why it stopped. |
| `queue_runner.py` | Run a plan of steps one after another, unattended, and record what happened. |
| `kaggle_runner.py` | Drive Kaggle training sessions from here: push, poll, fetch, decide. |
| `post_run.py` | After a training run: run the error analysis and the held-out OzFish score on its best checkpoint. |

## Evaluate and analyse

| Script | What it does |
|---|---|
| `evaluate.py` | FD-026: evaluate trained checkpoints on a dataset they were not trained on. |
| `error_analysis.py` | Step 0 of the improvement plan: find out HOW a checkpoint fails before changing anything. |
| `eval_ozfish_val.py` | Score checkpoints on the held-out OzFish deployments (decision-only validation, NOT the sealed test). |
| `eval_positive_only.py` | Re-score checkpoints on the validation images that contain at least one fish. |
| `eval_tiled.py` | Does tiled inference find more small fish than one pass over the whole frame? |
| `eval_unseen.py` | Score detectors of several kinds on a source none of them saw (the unseen-source test), with one AP routine. |
| `eval_adaptation.py` | Score the adaptation experiment (docs/eval_protocol.md item 12): zero-shot checkpoints and the fine-tuned ones on the two test lists. |
| `count_accuracy.py` | How well do the detections count fish? Per-frame counts and the clip-level maximum (MaxN) against the labels. |
| `experiment_colour.py` | Does the detector depend on colour? Measure it over the full validation split. |
| `label_audit.py` | FD-105: look at the labels behind the errors, instead of assuming they are right. |
| `leakage_control.py` | Isolate leakage from site difficulty: the same sites, scored by a model that saw neighbouring frames and by models that did not. |
| `threshold_transfer.py` | Does a threshold chosen on validation transfer to an unseen source? (declared before the unseen run was read, docs/eval_protocol.md item 10, second addendum) |
| `sealed_intervals.py` | Interval-only second pass on the sealed OzFish test set (docs/eval_protocol.md item 11, declared before it was run). |
| `bootstrap_ci.py` | Confidence intervals that respect the structure of the data: resample whole clips (or sites), not frames. |

## Export and check on the device

| Script | What it does |
|---|---|
| `export_onnx.py` | Export trained checkpoints to ONNX at the shapes they were EVALUATED at, and prove parity. |
| `engine_parity.py` | Compare a TensorRT engine's outputs against the PyTorch reference (FD-041). |

## Figures and demos

| Script | What it does |
|---|---|
| `make_figures.py` | Draw the paper figures from committed run records (no new measurements). |
| `make_examples.py` | Draw example detections (a figure for the paper): two frames from each of three sources, picked by a fixed random rule, not by looks. |
| `build_demo.py` | Build the data behind the comparison demo: detections, matches, timings. |
| `build_device_demo.py` | Build the frame-by-frame data behind the device demo. |

## Models and housekeeping

| Script | What it does |
|---|---|
| `fetch_models.py` | Download the trained checkpoints from the GitHub release, verified. |
| `archive_paths.py` | Archive files we no longer use, safely: move, never delete, and keep a manifest that can undo it. |
| `assemble_parts.py` | Join uploaded parts back into a tar and unpack it, after verifying every checksum. |
| `upload_parts.py` | Upload a folder of file parts to a remote host over ssh/scp, resumably, and verify them there. |
