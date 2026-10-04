#!/usr/bin/env python3
"""Run a TensorRT engine on saved input tensors and write its raw outputs to disk.

    python3 edge/engine_dump.py --engine engines/X.engine --parity-dir parity/X --out-dir parity/X/trt

Run ON the Jetson. For each frame_NN.bin (a preprocessed float32 NCHW tensor written by
scripts/export_onnx.py) it feeds the engine through `trtexec` and saves the raw output as
trt_NN.npy. scripts/engine_parity.py then compares those against the PyTorch reference
on the workstation.

Why trtexec and not a Python runner: it needs no extra dependencies (pycuda is not
installed here), and it keeps the thing under test independent of any runner code written
by us. If the engine and the PyTorch reference disagree, there is exactly one suspect.

Standard library plus numpy; Python 3.8-safe, because edge/ must not import training
dependencies and JetPack ships an old interpreter.
"""

import argparse
import glob
import json
import os
import subprocess
import sys

import numpy as np

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def run_one(engine, input_name, bin_path, out_json, attempts=3):
    """One inference through trtexec, retried if it exports nothing.

    With --warmUp=0 trtexec intermittently times zero queries and writes no output, which
    looked like a per-frame failure but was not (the same frame succeeds on re-run). The
    default warm-up runs queries reliably, and every query sees the same input, so the
    exported output is the same whichever one is captured. Retries stay as a safeguard so
    a flaky run cannot silently thin out the evidence.
    """
    cmd = [TRTEXEC, "--loadEngine=" + engine,
           "--loadInputs=%s:%s" % (input_name, bin_path),
           "--exportOutput=" + out_json,
           "--iterations=1", "--duration=0"]
    log = ""
    for _ in range(attempts):
        if os.path.exists(out_json):
            os.remove(out_json)
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              universal_newlines=True, timeout=300)
        log = proc.stdout
        if proc.returncode == 0 and os.path.exists(out_json):
            return 0, log
    return 1, log


def parse_output(path):
    """trtexec exports a list of {name, dimensions, values}."""
    with open(path) as f:
        data = json.load(f)
    entry = data[0] if isinstance(data, list) else data
    dims = [int(x) for x in str(entry["dimensions"]).lower().split("x")]
    arr = np.asarray(entry["values"], dtype=np.float32)
    if arr.size != int(np.prod(dims)):
        raise ValueError("value count %d does not match dimensions %s" % (arr.size, dims))
    return arr.reshape(dims), entry.get("name")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--parity-dir", required=True, help="dir holding frame_NN.bin + frames.json")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    if not os.path.exists(TRTEXEC):
        sys.exit("trtexec not found at %s -- this must run on the Jetson." % TRTEXEC)
    meta_path = os.path.join(args.parity_dir, "frames.json")
    with open(meta_path) as f:
        meta = json.load(f)
    input_name = meta["input_name"]

    os.makedirs(args.out_dir, exist_ok=True)
    bins = sorted(glob.glob(os.path.join(args.parity_dir, "frame_*.bin")))
    if not bins:
        sys.exit("no frame_*.bin in %s" % args.parity_dir)

    print("engine : %s" % args.engine)
    print("frames : %d   input '%s'" % (len(bins), input_name))
    failed = 0
    for b in bins:
        idx = os.path.basename(b)[len("frame_"):-len(".bin")]
        tmp = os.path.join(args.out_dir, "trt_%s.json" % idx)
        code, log = run_one(args.engine, input_name, b, tmp)
        if code != 0 or not os.path.exists(tmp):
            failed += 1
            print("  frame %s FAILED (exit %s):" % (idx, code))
            print("\n".join(log.splitlines()[-6:]))
            continue
        arr, name = parse_output(tmp)
        np.save(os.path.join(args.out_dir, "trt_%s.npy" % idx), arr)
        os.remove(tmp)                      # the JSON is large; keep only the .npy
        print("  frame %s ok  %s  shape %s" % (idx, name, tuple(arr.shape)))
    print("done: %d ok, %d failed" % (len(bins) - failed, failed))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
