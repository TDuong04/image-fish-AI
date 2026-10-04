# Target hardware: Jetson Orin Nano

The record of what the deployment board actually is. **Nothing in the export or benchmark
chain should be written to depend on a fact that is not recorded here as confirmed.**

Last updated: 2026-10-04. Status: **verified by `edge/probe_board.py` on the real board.**
Raw output is committed verbatim at `docs/board_probe_20261004.json`.

## Verified on the board

| Item | Value |
|---|---|
| Module | **NVIDIA Jetson Orin Nano Engineering Reference Developer Kit Super** (device-tree string) |
| **Super mode** | **Available.** Power modes: `0` 15W, `1` 25W, `2` MAXN_SUPER, `3` 7W. Currently in `25W` (id 1). |
| L4T | **36.4.7** (R36, revision 4.7) |
| CUDA | 12.6 |
| TensorRT | **10.3.0** (Python module 10.3.0, libnvinfer 10.3.0.30, `trtexec` at `/usr/src/tensorrt/bin/trtexec`) |
| Python / arch | 3.10.12, aarch64 |
| CPU | 6-core Arm Cortex-A78AE (owner-reported) |
| Memory | 7.44 GB visible, **unified** with the GPU; 4.77 GB available at idle; 3.72 GB swap |
| **Storage** | **SD card** (`/dev/mmcblk0p1`), 32 GB, **about 8 GB free** (74% used) |
| **Cooling** | **Active fan, spinning.** Tachometer 1,411–1,412 RPM at PWM duty 68/255 at idle |
| Idle temperature | 47–48 degrees C across all thermal zones |

The `nvidia-jetpack` metapackage is **not installed**, so the JetPack release name is not
recorded by the probe. L4T 36.4.7 is what identifies the software stack.

## Corrections to what was assumed or reported earlier

| Earlier statement | Reality | How it was found |
|---|---|---|
| "No cooling fan" (owner-reported) | A fan **is** fitted and spinning (about 1,411 RPM). The dev kit ships with one. | Reading the fan tachometer over sysfs. An interface existing proves nothing; a non-zero tach does. |
| "Unclear on Super" | **Super is present.** MAXN_SUPER and 25W are both listed. | The probe, reading `/etc/nvpmodel.conf` |
| Board assumed passively cooled throughout CLAUDE.md and the first version of this file | Wrong, for the fan reason above. Thermal risk is lower than assumed, not zero. | Same |

The owner-reported "no cooling fan" may have meant "no extra cooler beyond the stock one". Either
way, the tachometer is the evidence, and it says a fan is running.

## Documented by NVIDIA, not verified on this unit

| Item | Value | Why it matters |
|---|---|---|
| GPU | 1024 CUDA cores, Ampere, 32 tensor cores | Everything infers here |
| DLA | None | No second inference engine to offload to |
| Hardware video encoder (NVENC) | None | Annotated video output falls to CPU x264 |
| Hardware video decode (NVDEC) | Present | Decode path can be accelerated |

## Still unknown

| Item | Why it matters |
|---|---|
| PSU rating | Limits which power modes can be sustained |
| Ambient temperature | Throttling depends on it; log it with every sustained run |
| SD card speed class | Decides how badly storage limits video decode |
| Whether `jetson_clocks` is applied | Unlocked clocks make short benchmarks variable |

## What this implies for the plan

- **Storage is now the practical constraint.** About 8 GB free on an SD card has to hold engines,
  swap, any video being processed and benchmark logs. Video decode from an SD card will likely be
  storage-limited, so note that before blaming the model. An NVMe drive (or at least an external
  SSD for the footage) is the fix.
- **Memory is not the constraint.** YOLOv8n is about 6 MB in FP16. **INT8 is a research question
  about accuracy cost, not a deployment necessity.**
- **TensorRT 10.3 changes how INT8 would be done.** The implicit calibrator
  (`IInt8EntropyCalibrator2`) is deprecated there; explicit quantisation (Q/DQ nodes) is the
  supported route. Do not write against the deprecated path.
- **Thermal behaviour is still unmeasured.** The fan lowers throttling risk, but a short benchmark
  on a cold board proves nothing about sustained load, and 25 W and MAXN_SUPER draw the most. The
  30-minute sustained test remains essential, with temperature and clocks logged.
- **Power-mode swings are not assumed.** Their size is an output of the benchmark, not an input.
- **No hardware encoder.** Detections and counts are the unit's output; any annotated video is
  rendered off-board.

## Corroboration

An on-board benchmark report received on 2026-10-04 states 25 W mode, L4T 36.4.7 and TensorRT
10.3.0. The probe independently shows the same three facts. **Its timing figures are not recorded
as results**, because the `trtexec` log and engine hashes behind them have not been committed.

## Re-running the probe

```bash
python3 edge/probe_board.py          # human-readable, with a plain-language reading
python3 edge/probe_board.py --json   # machine-readable
```

Standard library only, no sudo. Over SSH without copying anything onto the board:
`ssh user@board "python3 - --json" < edge/probe_board.py`.

## Freeze rule

Do not upgrade JetPack/L4T for the duration of the study. A change invalidates every TensorRT
engine and every latency number collected with it. If an upgrade becomes unavoidable, every
measurement is re-taken and the old ones are marked superseded in `docs/PROGRESS.md`.
