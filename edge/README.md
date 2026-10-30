# `edge/`: running the detector on the Jetson Orin Nano

Code that runs **on the board**. It must not import training-only packages: no `ultralytics`, no `torch`. It needs TensorRT, NumPy, OpenCV and (for the web app) Gradio.

| File | What it does |
|---|---|
| `probe_board.py` | Reports the facts that decide everything downstream: module, L4T and TensorRT versions, power modes, memory, storage, fan. Output for this board: `docs/board_probe_20261004.json`; summary in `docs/hardware.md`. |
| `build_engines.py` | Builds static-shape FP16 TensorRT engines from ONNX **on the Jetson**. Engines are specific to the GPU, TensorRT version and driver, so they cannot be built on a desktop. Shapes match the shapes accuracy was evaluated at (384x640 and 544x960). |
| `trt_runner.py` | Runs an engine on images with a per-stage timing breakdown (decode, letterbox, GPU, NMS). |
| `app.py` | The web app served from the board (port 7860): photo and video tabs, counts, MaxN, a per-frame CSV and the busiest frame. It deliberately writes no annotated video: the Orin Nano has no hardware video encoder, so encoding would cost more than detection. |
| `sustained.py` | Long run (for example 30 minutes) that logs frame rate against board telemetry (junction temperature, power). |
| `engine_dump.py` | Runs an engine on saved input tensors and writes its raw outputs, for the parity test against PyTorch (`scripts/engine_parity.py`). |

## Typical order

```bash
python3 edge/probe_board.py                        # what is this board?
python3 edge/build_engines.py --help               # ONNX in, engines out (see docs/hardware.md)
python3 edge/trt_runner.py --help                  # timed run on images
python3 edge/sustained.py --help                   # 30-minute run
python3 edge/app.py --engines-dir engines          # the web app
```

## What was measured

Every figure names its power mode. Measured results (YOLOv8n 960 px: 40.8 ms end to end on images; YOLOv8s: 50 ms and 24.3 fps over 30 minutes of video at 25 W) are in `docs/PROGRESS.md`, findings F6 to F9, F28 to F31 and F34, with the raw logs under `runs/device-*/`. Not measured: INT8, mAP on the device, the 15 W and 7 W modes, hardware (NVDEC) decode, and any live camera.

The app has no login. On a network you do not trust, bind it to `127.0.0.1` and use an SSH tunnel.
