# `demo/`: the workstation demo

`app.py` is a browser demo (Gradio): upload a photo or a clip and see the detections, with the number of fish found, timings and confidence. It runs the real checkpoints from `models/` (fetch them with `python scripts/fetch_models.py`), so nothing is precomputed. Timings are for **your** machine, not a Jetson; the board has its own app in `edge/`.

```bash
python demo/app.py                # http://127.0.0.1:7860
python demo/app.py --model 640    # start on the 640 px model
python demo/app.py --share        # temporary public link; the app has no login
```

| File | What it is |
|---|---|
| `app.py` | The demo app |
| `sample_clip.mp4` | A short clip from DeepFish (MIT) to try the video tab. Large; not in git |
| `demo_data.json`, `device_data.json` | Precomputed data for two earlier comparison pages (built by `scripts/build_demo.py` and `scripts/build_device_demo.py`) |

The pictures and video in the repository README are in `docs/images/` and were made with the shipped YOLOv8s checkpoint on this sample clip.
