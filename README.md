# Fish Detection for Jetson Orin Nano

Finds fish in underwater video with a box around each one, using a model small enough
for a Jetson Orin Nano. **Status:** trained and evaluated on a desktop GPU, and running on
a Jetson Orin Nano (8 GB) as TensorRT FP16 engines. End to end, a simple serial pipeline
takes about 33 ms per frame at 640 px and 41 ms at 960 px, and held about 34 fps at 960 for
30 minutes on a video clip (one run). The hardware video decoder and INT8 are not measured yet. Adding a second training set (Kakadu)
raised mean validation mAP@50 from 0.47 to 0.70 (3 seeds each, same machine); that validation set resembles the added data, and the
sealed OzFish test (run once, three YOLOv8s seeds, new deployments of the same footage) gave 0.70 mAP@50. Adding OzFish training data raised it to 0.75 (3 seeds), and a larger model (YOLOv8s, 3 seeds, about 50 ms per frame on the
Jetson) to 0.81. Details: `docs/PROGRESS.md`.

## 1. Set up (once)

You need **Python 3.10 or 3.11** (`python --version`).

```bash
git clone <repository-url>
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
python3 edge/app.py --engines-dir engines        # serves 0.0.0.0:7860

# on a computer on the same network: open http://<jetson-address>:7860
# private option: run with --host 127.0.0.1, then on your computer
ssh -N -L 7861:127.0.0.1:7860 <user>@<jetson-address>
# and open http://127.0.0.1:7861
```

The app has no login, so anyone on the network can use it; use the tunnel option on a
network you do not trust. Timings shown there are measured on the board, with its power
mode and temperature displayed next to them.

## 3. Train

```bash
python scripts/get_data.py                        # ~1.2 GB, safe to re-run
python scripts/train.py --config smoke            # 2-minute test, accuracy will be bad
python scripts/train.py --config yolov8n_640      # real run, hours, 3 seeds
```

Results go to `runs/<date>-<name>/`. **No `metrics.json` means the run did not finish.**

Optional extra data (Kakadu, about 7,300 frames, CC BY 4.0) is described in `docs/TRAINING.md` section 5b. It raised the validation score a lot,
but the validation split resembles it, so treat that as unproven until the sealed OzFish test has been run.

## If something breaks

| It says | Do this |
|---|---|
| `Python 3.x is not supported` | Wrong Python. Recreate the venv with `py -3.11`. |
| `Refusing to install into the system Python` | You skipped `activate`. |
| `no GPU available ... CPU-only build` | Run the `pip install` command it prints. |
| `No model files found in models/` | `python scripts/fetch_models.py` |
| `Dataset ... does not exist` | `python scripts/get_data.py` |
| Download stopped halfway | Run `python scripts/get_data.py` again. |

## Licence

AGPL-3.0 (`LICENSE`): the code, configurations, documentation and trained weights are free software under the GNU Affero General Public License, version 3, because they build on Ultralytics YOLOv8 (AGPL-3.0).
Third-party software, dataset credits and the changes made to each dataset are in `NOTICE.md`; the models are described in `docs/MODEL_CARD.md`. If you serve a modified version over a network, offer its source to your users
(the apps print a source link from `$FISH_SOURCE_URL`). Datasets are not included and keep their own licences (CC BY, MIT). Before making the repository public, read `docs/PUBLISH_CHECKLIST.md`.

## More

| File | Read it when |
|---|---|
| `docs/PROGRESS.md` | **Start here** — what was built, every result, every correction, what is open |
| `docs/TRAINING.md` | You want the long version of training |
| `docs/report/` | The report draft (`paper.md`), its references and their verification status |
| `docs/GUIDE.md` | You want to understand what everything is and how it works, from the basics |
| `docs/RESEARCH.md` | The research story, methodology and suggested next steps |
| `docs/eval_protocol.md` | **Before quoting any number** |
| `docs/data_audit.md` | You want to know what is in the data |
| `docs/tickets/BACKLOG.md` | You want to know what is done and next |
| `CLAUDE.md` | You are changing how the project works |

Tests: `python -m pytest`
