# Fish Detection for Jetson Orin Nano

Finds fish in underwater video with a box around each one, using a model small enough
for a Jetson Orin Nano. **Status:** trained and evaluated on a desktop GPU, and running on
a Jetson Orin Nano (8 GB) as TensorRT FP16 engines. End to end, a simple serial pipeline
takes about 33 ms per frame at 640 px and 41 ms at 960 px. Sustained load, the hardware
video decoder and INT8 are not measured yet. Details: `docs/PROGRESS.md`.

## 1. Set up (once)

You need **Python 3.10 or 3.11** (`python --version`).

```bash
git clone https://git.grifufu.io.vn/IoT-Group/image-fish-AI.git
cd image-fish-AI

py -3.11 -m venv .venv            # Windows
.venv\Scripts\activate

python3.11 -m venv .venv          # Mac / Linux
source .venv/bin/activate

python scripts/bootstrap.py       # installs everything (~5 min)
python scripts/check_env.py       # last line should say READY
```

Run the `activate` line again whenever you open a new terminal.
Don't `pip install torch` yourself: on Windows it silently gives a CPU-only build.

## 2. Try the demo

Upload a photo or a clip and see the detections.

1. Download the trained models. They are not stored in git; this fetches them from the
   GitHub release and checks each one against a checksum:

```bash
python scripts/fetch_models.py
```

2. Run:

```bash
python demo/app.py
```

It opens in your browser. Add `--share` to get a temporary public link (~72 hours) to
send someone.

The timings it shows are for **your** machine, not a Jetson.

## 2b. Run it on the Jetson

`demo/app.py` needs PyTorch, which the Jetson does not have. The Jetson runs a separate
app, `edge/app.py`, that uses the TensorRT engines directly. Engines only work on the
board they were built on, so they are built there (`edge/build_engines.py`); setup is in
`docs/hardware.md`.

```bash
# on the Jetson
python3 edge/app.py --engines-dir engines        # serves 127.0.0.1:7860 only

# on your computer
ssh -N -L 7861:127.0.0.1:7860 <user>@<jetson-address>
# then open http://127.0.0.1:7861
```

It is reached through an SSH tunnel on purpose: the app has no login, so it is not
exposed to the network. Timings shown there are measured on the board, with its power
mode and temperature displayed next to them.

## 3. Train

```bash
python scripts/get_data.py                        # ~1.2 GB, safe to re-run
python scripts/train.py --config smoke            # 2-minute test, accuracy will be bad
python scripts/train.py --config yolov8n_640      # real run, hours, 3 seeds
```

Results go to `runs/<date>-<name>/`. **No `metrics.json` means the run did not finish.**

## If something breaks

| It says | Do this |
|---|---|
| `Python 3.x is not supported` | Wrong Python. Recreate the venv with `py -3.11`. |
| `Refusing to install into the system Python` | You skipped `activate`. |
| `no GPU available ... CPU-only build` | Run the `pip install` command it prints. |
| `No model files found in models/` | `python scripts/fetch_models.py` |
| `Dataset ... does not exist` | `python scripts/get_data.py` |
| Download stopped halfway | Run `python scripts/get_data.py` again. |

## More

| File | Read it when |
|---|---|
| `docs/PROGRESS.md` | **Start here** — what was built, every result, every correction, what is open |
| `docs/TRAINING.md` | You want the long version of training |
| `docs/eval_protocol.md` | **Before quoting any number** |
| `docs/data_audit.md` | You want to know what is in the data |
| `docs/tickets/BACKLOG.md` | You want to know what is done and next |
| `CLAUDE.md` | You are changing how the project works |

Tests: `python -m pytest`
