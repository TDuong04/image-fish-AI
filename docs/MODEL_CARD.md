# Model card: single-class fish detectors (YOLOv8s and YOLOv8n)

Every number here points to a finding in `docs/PROGRESS.md` (F#), which holds the run folder, the number of seeds and the limits. Licence: **AGPL-3.0** (see `LICENSE` and `NOTICE.md`).

## Model details

- **What:** single-class (`fish`) object detectors, YOLOv8s (11.1 M parameters) and YOLOv8n (3.0 M), fine-tuned from the COCO-pretrained Ultralytics weights.
- **Input:** a 1920x1080 frame is letterboxed to 544x960 (`imgsz` 960). **Output:** boxes with confidence; the operating threshold must be chosen per model on validation footage (F32).
- **Recipe:** DeepFish + Kakadu + OzFish training data, default Ultralytics augmentation, early stopping (patience 30), checkpoint by fitness. Three seeds per size.
- **Deployment:** FP16 TensorRT engines built on a Jetson Orin Nano (engines are not portable and are not part of the release); ONNX files are in `models/onnx/` locally.

## Intended use

Research on automated fish detection in underwater video (for example baited remote underwater video), offline analysis of recorded footage, and reproduction of the results in the accompanying report.
**Not validated for:** a specific deployment site (none exists yet), species identification, crowded schools, fish under about 20 pixels, murky or very dark water, or any decision where a missed fish matters without human review.

## Training and evaluation data

DeepFish, Kakadu and OzFish (training); DeepFish validation sites (decisions); a sealed OzFish test set (evaluated once, F33); the Mediterranean DEEPECOMAR set as an unseen source (evaluated once, see the findings list). Splits are by capture site; the protocol is `docs/eval_protocol.md`. Attributions are in `NOTICE.md`.

## Performance (all at 960 px)

| | YOLOv8s (3 seeds) | YOLOv8n (3 seeds) | Finding |
|---|---|---|---|
| DeepFish validation mAP@50 / mAP@50-95 | 0.812 / 0.567 (range 0.041 / 0.017) | 0.750 / 0.514 | F32, F24 |
| Held-out OzFish deployments mAP@50 | 0.729 | 0.689 | F32, F24 |
| Sealed OzFish test mAP@50 / mAP@50-95 (new deployments of the same footage, not an unseen source; 95% interval over deployments 0.669 to 0.724 for AP50) | 0.700 / 0.380 | not evaluated | F33, F37 |
| Unseen Mediterranean source, AP50 (95% interval over groups) | 0.166 (0.136 to 0.442) | 0.154 (0.123 to 0.413) | F39 |
| Recall on 16-32 px fish / fish under 16 px (DeepFish val) | about 29% / 0% | about 15% / 0% | F32 |
| Jetson Orin Nano, 25 W, video, sustained 30 min (YOLOv8s: seed 2 engine, F34, same as seed 0, F29) | about 24 fps, 59 C maximum, no throttling | about 34 fps, about 57 C, no throttling | F34, F29, F9 |

Uncertainty: with whole clips resampled the YOLOv8s lead over YOLOv8n is positive (+0.058 AP50, interval 0.012 to 0.104) and absolute scores carry roughly +-0.1 (F35).

## Known limitations

- **Generalisation to an unseen source is weak:** on a Mediterranean set that no training or selection data came from, YOLOv8s scores AP50 0.17 (the DeepFish-only YOLOv8n 0.05; the published YOLO-Fish YOLOv4 0.23), finding about 13 to 15% of the fish at the default threshold (F39). Expect large drops on footage unlike the training data.
- Small and faint fish are missed (about 28% of fish overall on DeepFish validation); false alarms at a fixed confidence differ widely between seeds (80 to 601 at 0.25), so calibrate the threshold on your own footage (F32).
- As a counter, per-frame totals are about a fifth low, and the clip-level maximum count is over-estimated because false alarms inflate the largest frame (F36).
- Training data are tropical Australian marine and freshwater footage (Mediterranean footage was used only for testing); behaviour on pond or aquaculture footage is unmeasured.
- Single run per device configuration, one ambient temperature that was not recorded (F28 to F31).

## Weights (checksums)

| File | SHA-256 | MB |
|---|---|---|
| `20261005-yolov8s-960-kakadu-ozfish-s0.pt` | `a87e329b696b4e49bcdcab6b6ee62178264a690dbe87979c456a9dd151f23c97` | 22.5 |
| `20261007-yolov8s-960-kakadu-ozfish-s1.pt` | `169a1b68a454c8c6a59b2fc1f2baced547cd439f1b269da344cfc8c4d900b400` | 22.5 |
| `20261007-yolov8s-960-kakadu-ozfish-s2.pt` | `5a63722e0274f8fed7985bf86d786d381f88a92f5b54dcb99188bb54d82eaac3` | 22.5 |
| `20261005-yolov8n-960-kakadu-ozfish-s0.pt` | `f70a8bc99e3f604f3c48bc58c717d150cb48e02c945b88f3ccbe40d2be3f5ff1` | 6.3 |
| `20261005-yolov8n-960-kakadu-ozfish-s1.pt` | `0347ee2328cb8f10eb57d63973045cedc94bdd464a9492548566ced7c4de5479` | 6.3 |
| `20261005-yolov8n-960-kakadu-ozfish-s2.pt` | `765bd348afbd4e53a3c35329e2d2f15de97b75dcf7aadb145bf9c068e954ae67` | 6.3 |

The weights are not in git (`.gitignore` excludes `*.pt`); they are published as release assets with this card (not yet released). Corresponding source: this repository at the release tag.

## Use and citation

```python
from ultralytics import YOLO
model = YOLO("20261007-yolov8s-960-kakadu-ozfish-s2.pt")
results = model.predict("frame.jpg", imgsz=960, conf=0.25)
```

Cite the accompanying report (reference to follow) and the datasets in `NOTICE.md`.
