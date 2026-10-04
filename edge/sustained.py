#!/usr/bin/env python3
"""Run the on-device pipeline for a long time and log throughput against board telemetry.

    python3 edge/sustained.py --engine engines/X.engine --video clip.mp4 --minutes 30 --log run.jsonl

A short benchmark on a cold board says nothing about what a unit does after an hour in
the field: temperature climbs, and clocks may be capped to stay inside the thermal
budget, which shows up as throughput falling. This runs the same decode -> letterbox ->
GPU -> NMS loop as edge/trt_runner.py, repeating one clip, and every WINDOW seconds
writes one line holding that window's throughput and latency percentiles together with
temperatures, fan speed, GPU/CPU clocks and input power. Reading throughput next to
temperature and clocks is what separates "slower because it throttled" from "slower for
some other reason".

Telemetry is read from sysfs and needs no sudo. Anything that cannot be read is recorded
as null, never guessed, so a gap in the log is visible as a gap.

It does not change the power mode, lock clocks or touch anything else: the board is
measured in whatever state it is in, and that state is recorded at the start.

Standard library, numpy, OpenCV and the project's own runner; Python 3.8-safe.
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402

import trt_runner as R  # noqa: E402


def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def _int(path):
    v = _read(path)
    try:
        return int(v)
    except (TypeError, ValueError):
        return None


def _first(globs):
    for g in globs:
        hits = sorted(glob.glob(g))
        if hits:
            return hits[0]
    return None


GPU_FREQ_GLOBS = ["/sys/devices/platform/*/17000000.gpu/devfreq/*/cur_freq",
                  "/sys/devices/platform/17000000.gpu/devfreq/*/cur_freq"]
GPU_MAX_GLOBS = [g.replace("cur_freq", "max_freq") for g in GPU_FREQ_GLOBS]


def telemetry():
    """One snapshot of the board. Every field is None if it cannot be read."""
    temps = {}
    for z in sorted(glob.glob("/sys/class/thermal/thermal_zone*")):
        name, val = _read(z + "/type"), _int(z + "/temp")
        if name and val is not None:
            temps[name] = round(val / 1000.0, 1)

    rpm = None
    for f in glob.glob("/sys/class/hwmon/hwmon*/rpm"):
        rpm = _int(f)
        if rpm is not None:
            break

    gpu = _first(GPU_FREQ_GLOBS)
    gmax = _first(GPU_MAX_GLOBS)

    # Input power from the INA3221 monitor: channel 1 is the module's main supply on
    # the dev kit. Best effort and labelled as such; None if the layout differs.
    power_mw = None
    for h in glob.glob("/sys/class/hwmon/hwmon*"):
        if _read(h + "/name") == "ina3221":
            mv, ma = _int(h + "/in1_input"), _int(h + "/curr1_input")
            if mv is not None and ma is not None:
                power_mw = round(mv * ma / 1000.0)
            break

    return {
        "temps_c": temps,
        "tj_c": temps.get("tj-thermal"),
        "fan_rpm": rpm,
        "gpu_mhz": round(_int(gpu) / 1e6) if gpu and _int(gpu) else None,
        "gpu_max_mhz": round(_int(gmax) / 1e6) if gmax and _int(gmax) else None,
        "cpu0_mhz": round((_int("/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq") or 0) / 1e3) or None,
        "power_mw_vdd_in": power_mw,
    }


def environment(args):
    nv = None
    try:
        nv = subprocess.run(["nvpmodel", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            universal_newlines=True, timeout=5).stdout.strip().replace("\n", " ")
    except Exception:
        pass
    return {
        "engine": os.path.basename(args.engine), "video": os.path.basename(args.video),
        "minutes": args.minutes, "window_s": args.window,
        "power_mode": nv,
        "cpu_governor": _read("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"),
        "gpu_governor": _read((_first(GPU_FREQ_GLOBS) or "").replace("cur_freq", "governor")),
        "clocks_locked": "not applied by this script",
        "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "note": "ambient temperature is not measured by the board and must be recorded by hand",
    }


def pct(vals, p):
    v = sorted(vals)
    return v[min(len(v) - 1, int(len(v) * p))]


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--minutes", type=float, default=30.0)
    ap.add_argument("--window", type=float, default=10.0, help="seconds per log line")
    ap.add_argument("--log", required=True, help="output .jsonl")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        sys.exit("cannot open %s" % args.video)
    det = R.TRTDetector(args.engine)
    for _ in range(30):                              # warm-up, discarded and not logged
        ok, fr = cap.read()
        if not ok:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, fr = cap.read()
        det.detect(fr)

    env = environment(args)
    t_end = time.time() + args.minutes * 60.0
    t0 = time.time()
    with open(args.log, "w") as log:
        log.write(json.dumps({"type": "env", **env, "telemetry_start": telemetry()}) + "\n")
        log.flush()
        win_start, totals, decs, gpus, dets_n, frames = time.time(), [], [], [], 0, 0
        while time.time() < t_end:
            t = time.perf_counter()
            ok, fr = cap.read()
            if not ok:                               # loop the clip
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                ok, fr = cap.read()
            dec = (time.perf_counter() - t) * 1e3
            dets, tm = det.detect(fr)
            totals.append(dec + tm["total_ms"])
            decs.append(dec)
            gpus.append(tm["gpu_ms"])
            dets_n += len(dets)
            frames += 1

            now = time.time()
            if now - win_start >= args.window:
                span = now - win_start
                row = {"type": "window", "t_s": round(now - t0, 1), "frames": frames,
                       "fps": round(frames / span, 2),
                       "total_ms_p50": round(pct(totals, 0.5), 2),
                       "total_ms_p95": round(pct(totals, 0.95), 2),
                       "decode_ms_p50": round(pct(decs, 0.5), 2),
                       "copy_gpu_ms_p50": round(pct(gpus, 0.5), 2),
                       "mean_dets": round(dets_n / float(frames), 2), **telemetry()}
                log.write(json.dumps(row) + "\n")
                log.flush()
                win_start, totals, decs, gpus, dets_n, frames = now, [], [], [], 0, 0
        log.write(json.dumps({"type": "end", "ended_utc": time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "telemetry_end": telemetry()}) + "\n")
    det.close()
    print("DONE")
    return 0


if __name__ == "__main__":
    sys.exit(main())
