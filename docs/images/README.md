# Images used in the README

| File | What it is | How it was made |
|---|---|---|
| `demo.gif`, `demo.mp4` | Detections on a 10-second DeepFish clip (`7623_F2`, MIT licence) | The shipped YOLOv8s checkpoint (`20261007-yolov8s-960-kakadu-ozfish-s2`) at 960 px, confidence 0.25, drawn frame by frame; the count shown is boxes above the threshold, not labels |
| `demo_still.jpg` | One frame of the same video (frame 60), chosen by eye to show two clear detections | Same run |
| `examples.jpg` | Example detections on three sources | Downsized copy of `docs/report/figures/fig5_examples.png`, drawn by `scripts/make_examples.py` with a fixed random rule |

These illustrate behaviour. They are not measurements; the measured numbers are in `docs/PROGRESS.md`.
