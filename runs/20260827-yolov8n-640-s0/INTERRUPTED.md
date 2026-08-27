# This run was stopped manually. It is NOT a result.

Stopped at **epoch 4 of 150**, by request, to free the GPU.

No `metrics.json` was written, because the run never reached the end. That is the
intended behaviour: an unfinished run must not leave behind an artifact that looks
like a finished one.

Last values from `results.csv` (epoch 4):

| metric | value |
|---|---|
| mAP@50 | 0.2162 |
| mAP@50-95 | 0.08399 |
| precision | 0.42948 |
| recall | 0.24549 |

**Do not cite these numbers anywhere.** The model was still in the steep early part
of the curve (mAP@50 went 0.038 -> 0.066 -> 0.216 over three epochs) and had not
begun to converge. `patience` is 30 and never fired.

Per `CLAUDE.md`: "A model trained for 10 epochs is not a converged result. Label it
as what it is." Four epochs is further from convergence still.

To run it properly:

    python scripts/train.py --config yolov8n_640

Expect roughly 63 s/epoch on an RTX 4060 after the first epoch, so about 2.5 hours
per seed, and the config asks for three seeds.
