# Fish Detection for Jetson Orin Nano

Finds fish in underwater video with a box around each one, using a model small enough
for a Jetson Orin Nano. **Status:** trained and evaluated on a desktop GPU. Nothing has
run on the Jetson yet; the board is not in hand.

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

1. Get the trained `.pt` model files from a teammate (they are not stored in git).
2. Put them in the `models/` folder.
3. Run:

```bash
python demo/app.py
```

It opens in your browser. Add `--share` to get a temporary public link (~72 hours) to
send someone.

The timings it shows are for **your** machine, not a Jetson.

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
| `No model files found in models/` | Step 2 above: get the `.pt` files. |
| `Dataset ... does not exist` | `python scripts/get_data.py` |
| Download stopped halfway | Run `python scripts/get_data.py` again. |

## More

| File | Read it when |
|---|---|
| `docs/TRAINING.md` | You want the long version of training |
| `docs/eval_protocol.md` | **Before quoting any number** |
| `docs/data_audit.md` | You want to know what is in the data |
| `docs/tickets/BACKLOG.md` | You want to know what is done and next |
| `CLAUDE.md` | You are changing how the project works |

Tests: `python -m pytest`
