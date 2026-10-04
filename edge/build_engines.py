#!/usr/bin/env python3
"""Build static-shape TensorRT engines from ONNX models, on the Jetson itself.

    python3 edge/build_engines.py --onnx-dir onnx --engine-dir engines
    python3 edge/build_engines.py --onnx-dir onnx --engine-dir engines --dry-run

Engines are compiled for one specific GPU, TensorRT version and driver, so they cannot
be built on the workstation and copied over. This runs `trtexec` once per ONNX file, one
at a time: two concurrent builds can exhaust the unified memory of an 8 GB module, and
TensorRT picks kernels by timing them, so other GPU work during a build can skew which
ones it chooses. Leave the GPU alone while this runs.

Flags are deliberately minimal and identical for every model (`--fp16`), so engines stay
comparable. `--skipInference` keeps the build separate from any benchmark.

Each build writes its trtexec log, and the run appends to build_status.txt, so a build
that is interrupted or detached still leaves a record of what finished.

Standard library only; Python 3.8-safe, because edge/ must not import training
dependencies and JetPack ships an old interpreter.
"""

import argparse
import glob
import os
import subprocess
import sys
import time

TRTEXEC = "/usr/src/tensorrt/bin/trtexec"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--onnx-dir", required=True)
    ap.add_argument("--engine-dir", required=True)
    ap.add_argument("--log-dir", default="logs")
    ap.add_argument("--precision", choices=("fp16",), default="fp16",
                    help="only FP16 is built here; INT8 needs a calibration set and is separate")
    ap.add_argument("--dry-run", action="store_true", help="print the commands, build nothing")
    args = ap.parse_args()

    onnx_files = sorted(glob.glob(os.path.join(args.onnx_dir, "*.onnx")))
    if not onnx_files:
        sys.exit("no .onnx files in %s" % args.onnx_dir)
    if not args.dry_run and not os.path.exists(TRTEXEC):
        sys.exit("trtexec not found at %s -- this must run on the Jetson." % TRTEXEC)

    for d in (args.engine_dir, args.log_dir):
        if not args.dry_run:
            os.makedirs(d, exist_ok=True)
    status = os.path.join(args.log_dir, "build_status.txt")

    failed = 0
    for path in onnx_files:
        tag = os.path.splitext(os.path.basename(path))[0]
        engine = os.path.join(args.engine_dir, "%s-%s.engine" % (tag, args.precision))
        cmd = [TRTEXEC, "--onnx=" + path, "--" + args.precision,
               "--saveEngine=" + engine, "--skipInference"]
        print(" ".join(cmd))
        if args.dry_run:
            continue
        start = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        with open(status, "a") as f:
            f.write("BUILD_START %s %s\n" % (tag, start))
        log_path = os.path.join(args.log_dir, "trtexec_build_%s.log" % tag)
        with open(log_path, "w") as log:
            code = subprocess.call(cmd, stdout=log, stderr=subprocess.STDOUT)
        with open(status, "a") as f:
            f.write("BUILD_END %s exit=%d %s\n" % (tag, code, time.strftime(
                "%Y-%m-%dT%H:%M:%SZ", time.gmtime())))
        failed += 1 if code else 0
        print("  %s -> exit %d  (log: %s)" % (tag, code, log_path))
    if not args.dry_run:
        with open(status, "a") as f:
            f.write("ALL_BUILDS_DONE failed=%d\n" % failed)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
