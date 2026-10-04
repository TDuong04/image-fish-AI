#!/usr/bin/env python3
"""Run a TensorRT fish-detector engine on images, with a per-stage timing breakdown.

    python3 edge/trt_runner.py --engine engines/X.engine --image frame.jpg
    python3 edge/trt_runner.py --engine engines/X.engine --image frame.jpg --bench 200

This is the on-device inference path: decode (by the caller), letterbox, host-to-device,
GPU forward, device-to-host, threshold + NMS, map back to the original frame. It is
what `trtexec` is NOT: trtexec times the GPU on a static input, so it cannot say what a
frame actually costs once the CPU work around it is included. The breakdown here can.

Dependencies are the ones the Jetson already has (TensorRT, numpy, OpenCV) plus
`cuda-python` for device memory. There is deliberately no torch and no ultralytics: the
edge side must not depend on training code, and neither fits comfortably on this board.

Preprocessing reproduces Ultralytics' LetterBox exactly (scripts/export_onnx.py proves
the engine matches PyTorch on that preprocessing; if this letterbox differed, that proof
would not transfer). tests/test_edge_runner.py checks the two against each other.

Python 3.8-safe: JetPack ships an old interpreter.
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np

CONF = 0.25
NMS_IOU = 0.7
PAD_VALUE = 114


def letterbox(img_bgr, hw):
    """Resize to fit (H, W) keeping aspect ratio, pad with grey. Matches Ultralytics.

    Returns the padded BGR image, the scale, and the (left, top) padding, which are
    needed to map detections back onto the original frame.
    """
    h0, w0 = img_bgr.shape[:2]
    H, W = hw
    r = min(H / h0, W / w0)
    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    dw, dh = (W - new_w) / 2.0, (H - new_h) / 2.0
    if (w0, h0) != (new_w, new_h):
        img_bgr = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    out = cv2.copyMakeBorder(img_bgr, top, bottom, left, right, cv2.BORDER_CONSTANT,
                             value=(PAD_VALUE, PAD_VALUE, PAD_VALUE))
    assert out.shape[:2] == (H, W), "letterbox produced %s, wanted %s" % (out.shape[:2], hw)
    return out, r, (left, top)


def to_tensor(letterboxed_bgr):
    """BGR HxWx3 uint8 -> float32 1x3xHxW in [0, 1], RGB order."""
    x = letterboxed_bgr[:, :, ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(x, dtype=np.float32)[None] / 255.0


def decode(raw, r, pad, orig_hw, conf=CONF, iou=NMS_IOU):
    """(1, 5, N) raw output -> (k, 5) [x1, y1, x2, y2, conf] in ORIGINAL-frame pixels.

    Single class, so a plain NMS is exactly what Ultralytics does.
    """
    p = raw[0].T                                    # (N, 5): cx, cy, w, h, score
    keep = p[:, 4] >= conf
    p = p[keep]
    if len(p) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    cx, cy, w, h, s = p[:, 0], p[:, 1], p[:, 2], p[:, 3], p[:, 4]
    boxes_xywh = np.stack([cx - w / 2, cy - h / 2, w, h], axis=1)
    idx = cv2.dnn.NMSBoxes(boxes_xywh.tolist(), s.tolist(), conf, iou)
    idx = np.array(idx).reshape(-1).astype(int)
    if len(idx) == 0:
        return np.zeros((0, 5), dtype=np.float32)
    x1 = (boxes_xywh[idx, 0] - pad[0]) / r
    y1 = (boxes_xywh[idx, 1] - pad[1]) / r
    x2 = (boxes_xywh[idx, 0] + boxes_xywh[idx, 2] - pad[0]) / r
    y2 = (boxes_xywh[idx, 1] + boxes_xywh[idx, 3] - pad[1]) / r
    H0, W0 = orig_hw
    out = np.stack([np.clip(x1, 0, W0), np.clip(y1, 0, H0),
                    np.clip(x2, 0, W0), np.clip(y2, 0, H0), s[idx]], axis=1)
    return out.astype(np.float32)


class TRTDetector(object):
    """One engine, one execution context, device buffers allocated once and reused."""

    def __init__(self, engine_path):
        import tensorrt as trt
        from cuda import cudart
        self._trt, self._cudart = trt, cudart

        with open(engine_path, "rb") as f:
            blob = f.read()
        logger = trt.Logger(trt.Logger.WARNING)
        self.engine = trt.Runtime(logger).deserialize_cuda_engine(blob)
        if self.engine is None:
            raise RuntimeError("could not deserialise %s -- an engine only loads on the GPU, "
                               "TensorRT version and driver it was built with" % engine_path)
        self.ctx = self.engine.create_execution_context()

        self.in_name = self.out_name = None
        for i in range(self.engine.num_io_tensors):
            name = self.engine.get_tensor_name(i)
            if self.engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
                self.in_name = name
            else:
                self.out_name = name
        self.in_shape = tuple(self.engine.get_tensor_shape(self.in_name))
        self.out_shape = tuple(self.engine.get_tensor_shape(self.out_name))
        self.hw = (self.in_shape[2], self.in_shape[3])
        self.in_dtype = trt.nptype(self.engine.get_tensor_dtype(self.in_name))
        self.out_dtype = trt.nptype(self.engine.get_tensor_dtype(self.out_name))

        self.h_out = np.empty(self.out_shape, dtype=self.out_dtype)
        self.d_in = self._check(cudart.cudaMalloc(int(np.prod(self.in_shape)) *
                                                  np.dtype(self.in_dtype).itemsize))[0]
        self.d_out = self._check(cudart.cudaMalloc(self.h_out.nbytes))[0]
        self.stream = self._check(cudart.cudaStreamCreate())[0]
        self.ctx.set_tensor_address(self.in_name, int(self.d_in))
        self.ctx.set_tensor_address(self.out_name, int(self.d_out))

    def _check(self, res):
        if int(res[0]) != 0:
            raise RuntimeError("CUDA call failed with error %d" % int(res[0]))
        return res[1:]

    def infer_raw(self, x):
        """float32 NCHW tensor -> raw (1, 5, N) output. One sync at the end."""
        cr = self._cudart
        x = np.ascontiguousarray(x, dtype=self.in_dtype)
        kind = cr.cudaMemcpyKind
        self._check(cr.cudaMemcpyAsync(self.d_in, x.ctypes.data, x.nbytes,
                                       kind.cudaMemcpyHostToDevice, self.stream))
        if not self.ctx.execute_async_v3(int(self.stream)):
            raise RuntimeError("execute_async_v3 failed")
        self._check(cr.cudaMemcpyAsync(self.h_out.ctypes.data, self.d_out, self.h_out.nbytes,
                                       kind.cudaMemcpyDeviceToHost, self.stream))
        self._check(cr.cudaStreamSynchronize(self.stream))
        return self.h_out

    def detect(self, img_bgr, conf=CONF, iou=NMS_IOU):
        """Full path on one decoded frame. Returns (detections, timings_ms)."""
        t0 = time.perf_counter()
        lb, r, pad = letterbox(img_bgr, self.hw)
        x = to_tensor(lb)
        t1 = time.perf_counter()
        raw = self.infer_raw(x)                       # H2D + GPU + D2H, one sync
        t2 = time.perf_counter()
        dets = decode(raw, r, pad, img_bgr.shape[:2], conf, iou)
        t3 = time.perf_counter()
        return dets, {"preprocess_ms": (t1 - t0) * 1e3, "gpu_ms": (t2 - t1) * 1e3,
                      "postprocess_ms": (t3 - t2) * 1e3, "total_ms": (t3 - t0) * 1e3}

    def close(self):
        cr = self._cudart
        cr.cudaFree(self.d_in)
        cr.cudaFree(self.d_out)
        cr.cudaStreamDestroy(self.stream)


def summarise(samples):
    """median / p95 / mean per stage from a list of timing dicts."""
    out = {}
    for k in samples[0]:
        v = sorted(s[k] for s in samples)
        out[k] = {"median": v[len(v) // 2], "p95": v[min(len(v) - 1, int(len(v) * 0.95))],
                  "mean": sum(v) / len(v)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--bench", type=int, default=0,
                    help="repeat N times (after warm-up) and report median / p95 per stage")
    ap.add_argument("--conf", type=float, default=CONF)
    args = ap.parse_args()

    img = cv2.imread(args.image)
    if img is None:
        sys.exit("could not read %s" % args.image)
    det = TRTDetector(args.engine)
    print("engine %s   input %sx%s   output %s" % (os.path.basename(args.engine),
          det.hw[0], det.hw[1], det.out_shape))

    dets, t = det.detect(img, args.conf)
    print("%d detection(s) on a %dx%d frame" % (len(dets), img.shape[1], img.shape[0]))
    for d in dets[:10]:
        print("  [%.0f, %.0f, %.0f, %.0f]  conf %.3f" % tuple(d))

    if args.bench:
        for _ in range(20):                         # warm-up, discarded
            det.detect(img, args.conf)
        samples = [det.detect(img, args.conf)[1] for _ in range(args.bench)]
        # JPEG decode is a real per-frame cost on the Arm cores (OpenCV has no hardware
        # JPEG path here), so time it too rather than leave it out of the total.
        for smp in samples:
            t = time.perf_counter()
            cv2.imread(args.image)
            smp["jpeg_decode_ms"] = (time.perf_counter() - t) * 1e3
            smp["frame_total_ms"] = smp["total_ms"] + smp["jpeg_decode_ms"]
        s = summarise(samples)
        print("\nper-stage over %d runs, each a separate wall-clock figure:" % args.bench)
        for k in ("jpeg_decode_ms", "preprocess_ms", "gpu_ms", "postprocess_ms", "frame_total_ms"):
            print("  %-15s median %7.2f   p95 %7.2f   mean %7.2f" %
                  (k, s[k]["median"], s[k]["p95"], s[k]["mean"]))
        print("  (gpu_ms = host-to-device + GPU forward + device-to-host, one sync. It is larger\n"
              "   than trtexec's kernel-only figure because the copies are included.)")
    det.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
