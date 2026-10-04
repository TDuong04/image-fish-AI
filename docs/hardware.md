# Target hardware: Jetson Orin Nano

The record of what the deployment board actually is. **Nothing in the export or benchmark
chain should be written to depend on a fact that is not recorded here as confirmed.**

Last updated: 2026-10-04. Status: **partly known. Probe output pending.**

## Confirmed (reported by the project owner, not yet measured by a script)

| Item | Value |
|---|---|
| Module | **Orin Nano, 8 GB** |
| CPU | 6-core Arm Cortex-A78AE (Armv8.2-A) |
| Memory | 8 GB, 128-bit LPDDR5, **unified** (CPU and GPU share it) |
| Cooling | **No cooling fan** (passive) |

## Documented by NVIDIA, not verified on this unit

| Item | Value | Why it matters |
|---|---|---|
| GPU | 1024 CUDA cores, Ampere, 32 tensor cores | Everything infers here |
| DLA | None | No second inference engine to offload to |
| Hardware video encoder (NVENC) | None | Annotated video output falls to CPU x264 |
| Hardware video decode (NVDEC) | Present | Decode path can be accelerated |

## Unknown. These decide the plan.

| Item | Why it matters | How to find out |
|---|---|---|
| **Super mode available?** | Changes the power modes and the throughput ceiling | Run `edge/probe_board.py`: a `MAXN_SUPER` mode in the list means yes |
| JetPack / L4T version | Pins TensorRT, which pins the usable ONNX opset | Same probe (`/etc/nv_tegra_release`) |
| TensorRT version | Decides whether INT8 uses the deprecated calibrator or explicit Q/DQ | Same probe |
| Boot storage (SD vs NVMe) | SD card will bottleneck video decode | Same probe |
| Heatsink present? | "No fan" can mean a passive heatsink or a bare module | Visual check |
| PSU rating | Limits which power modes are usable | Check the adapter label |
| Ambient temperature | Throttling depends on it; must be logged with every sustained run | Measure |

**About Super.** The specs above (6-core A78AE, 8 GB 128-bit LPDDR5, 1024-core GPU) are
**identical with and without Super**. Super is a software and firmware change delivered by
JetPack 6.2 or later, not different silicon, so the hardware description alone cannot answer it.
The authoritative answer is the list of power modes the board offers.

## What this implies for the plan

- **Memory is not the constraint.** YOLOv8n is about 6 MB of weights in FP16. With 8 GB unified and
  roughly 1.5 to 2 GB used by the OS before any process starts, there is ample room for the engine,
  buffers and decode pipeline. **INT8 is therefore a research question about accuracy cost, not a
  deployment necessity.**
- **Cooling is the main risk to measurement integrity.** A passively cooled module will throttle
  under sustained load, so a burst benchmark overstates what the unit can do. The 30-minute
  sustained-load test is essential, and the higher power modes may throttle quickly. Record ambient
  temperature, clocks and junction temperature alongside every sustained number.
- **Power modes are not assumed.** Published information says the original 8 GB module offers 7 W
  and 15 W, and that Super adds 25 W and MAXN_SUPER. That is *published*, not measured here, and the
  magnitude of any mode-to-mode swing is an output of the benchmark, not an input to it.
- **No hardware encoder.** Detections and counts are the unit's output. Any annotated video is
  rendered off-board.

## How to fill in the unknowns

On the board:

```bash
python3 edge/probe_board.py          # human-readable, with a plain-language reading
python3 edge/probe_board.py --json   # machine-readable
```

It needs no sudo and only the standard library. Paste the output back and this file gets updated
from it. One limit to know about: the probe reads power-mode names from `/etc/nvpmodel.conf`. That
parsing was tested against simulated files, not a real board's, so if it reports "could not read"
the fallback is `sudo nvpmodel -q --verbose`. Do not guess.

## Freeze rule

Once JetPack is recorded, **do not upgrade it for the duration of the study.** A JetPack change
invalidates every TensorRT engine and every latency number collected with it. If an upgrade becomes
unavoidable, every measurement is re-taken and the old ones are marked superseded in
`docs/PROGRESS.md`.
