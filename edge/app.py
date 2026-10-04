#!/usr/bin/env python3
"""The fish detector as a web app, running ON the Jetson with the TensorRT engines.

    python3 edge/app.py --engines-dir engines                    # http://<board>:7860
    python3 edge/app.py --engines-dir engines --host 127.0.0.1   # this board only

demo/app.py is the workstation version: it needs PyTorch and Ultralytics, which this board
neither has nor should have. This one talks to the engines directly through
edge/trt_runner.py, so what you see here is the real deployment path, and every timing is
measured on this device in its current power mode.

What it deliberately does NOT do: write annotated video. The Orin Nano has no hardware
video encoder, so encoding would fall to the Arm cores and cost more than the detection.
The video tab returns counts, MaxN, a per-frame CSV, and the single frame with the most
fish -- the outputs a survey actually uses -- rather than a re-encoded clip.

Python 3.8-safe syntax; needs gradio, plus the board's own TensorRT, numpy and OpenCV.
"""

import argparse
import csv
import glob
import os
import re
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

import trt_runner as R  # noqa: E402

COLORS = {640: (85, 160, 232), 960: (221, 168, 109)}      # BGR
_detectors = {}
PHOTO_REPEATS = 7


# ---------------------------------------------------------------- board status
def _read(path):
    try:
        with open(path) as f:
            return f.read().strip()
    except Exception:
        return None


def board_status():
    """Power mode, temperature and fan, read live. Shown so a timing is never read
    without the conditions it was taken in."""
    mode = None
    try:
        out = subprocess.run(["nvpmodel", "-q"], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             universal_newlines=True, timeout=5).stdout.split()
        mode = " ".join(out[-2:]) if out else None
    except Exception:
        pass
    temp = None
    for z in glob.glob("/sys/class/thermal/thermal_zone*"):
        if _read(z + "/type") == "tj-thermal":
            v = _read(z + "/temp")
            temp = float(v) / 1000.0 if v and v.lstrip("-").isdigit() else None
    rpm = None
    for f in glob.glob("/sys/class/hwmon/hwmon*/rpm"):
        v = _read(f)
        if v and v.isdigit():
            rpm = int(v)
            break
    parts = []
    parts.append("power mode **%s**" % (mode or "unknown"))
    parts.append("junction **%.1f C**" % temp if temp is not None else "temperature unknown")
    parts.append("fan **%s RPM**" % rpm if rpm is not None else "fan unknown")
    return "Board now: " + "  ·  ".join(parts) + "  ·  clocks governor-managed (not locked)"


# -------------------------------------------------------------------- engines
def find_engines(engines_dir):
    """Map input width (640 / 960) to its engine file."""
    found = {}
    for p in sorted(glob.glob(os.path.join(engines_dir, "*.engine"))):
        m = re.search(r"-(\d+)x(\d+)-fp16", os.path.basename(p))
        if m:
            found[int(m.group(2))] = p
    return found


def get_detector(size, engines):
    if size not in engines:
        raise gr.Error("No %dpx engine in the engines directory. Build one with "
                       "edge/build_engines.py." % size)
    if size not in _detectors:
        _detectors[size] = R.TRTDetector(engines[size])
    return _detectors[size]


def draw(frame_bgr, dets, size, show_conf):
    out = frame_bgr.copy()
    col = COLORS.get(size, (0, 255, 0))
    thick = max(2, int(round(out.shape[1] / 700)))
    for x1, y1, x2, y2, c in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), col, thick)
        if show_conf:
            cv2.putText(out, "%.2f" % c, (int(x1) + 3, max(14, int(y1) - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
    return out


def _median(v):
    v = sorted(v)
    return v[len(v) // 2]


# ------------------------------------------------------------------- photo tab
def run_photo(image, size_choice, conf, show_conf, compare, engines):
    if image is None:
        raise gr.Error("Add a photo first.")
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    sizes = [640, 960] if compare else [int(size_choice)]
    panels, rows = [], []
    for sz in sizes:
        det = get_detector(sz, engines)
        det.detect(frame, conf)                      # warm this shape up; first call is slow
        # One run is too noisy to quote (a single call here swung from 14 to 18 ms), so
        # report the median of several. Detections are identical on every repeat.
        runs = [det.detect(frame, conf) for _ in range(PHOTO_REPEATS)]
        dets = runs[0][0]
        t = {k: _median([r[1][k] for r in runs]) for k in runs[0][1]}
        panels.append((cv2.cvtColor(draw(frame, dets, sz, show_conf), cv2.COLOR_BGR2RGB),
                       "%dpx -- %d fish" % (sz, len(dets))))
        rows.append("| %d px | **%d** | %.1f | %.1f | %.1f | **%.1f** |" % (
            sz, len(dets), t["preprocess_ms"], t["gpu_ms"], t["postprocess_ms"], t["total_ms"]))
    table = ("| Input | Fish | Letterbox ms | Copy+GPU ms | NMS ms | Total ms |\n"
             "|---|---|---|---|---|---|\n" + "\n".join(rows))
    note = ("\n\nTimings are the median of %d warm runs on this board. JPEG decode is **not** included "
            "(the browser already decoded the upload); a 1080p JPEG costs about 17 ms more on "
            "these Arm cores. A single frame proves nothing about which resolution is better."
            % PHOTO_REPEATS)
    return panels, table + note, board_status()


# ------------------------------------------------------------------- video tab
def run_video(video, size_choice, conf, stride, max_seconds, engines,
              progress=gr.Progress()):
    if video is None:
        raise gr.Error("Add a clip first.")
    sz = int(size_choice)
    det = get_detector(sz, engines)
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise gr.Error("Could not open that video.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    limit = int(fps * max_seconds) if max_seconds else total
    det.detect(np.zeros((1080, 1920, 3), np.uint8), conf)        # warm-up

    counts, stage, best, best_n, i = [], [], None, -1, 0
    wall0 = time.perf_counter()
    while i < limit:
        t0 = time.perf_counter()
        ok, frame = cap.read()
        dec = (time.perf_counter() - t0) * 1e3
        if not ok:
            break
        if i % stride == 0:
            dets, t = det.detect(frame, conf)
            t["decode_ms"] = dec
            stage.append(t)
            counts.append((i, i / fps, len(dets)))
            if len(dets) > best_n:
                best_n, best = len(dets), draw(frame, dets, sz, False)
            if total:
                progress(min(i / float(min(limit, total)), 1.0), desc="frame %d" % i)
        i += 1
    cap.release()
    wall = time.perf_counter() - wall0
    if not counts:
        raise gr.Error("No frames were processed.")

    csv_path = os.path.join(tempfile.mkdtemp(), "detections_per_frame.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["frame", "seconds", "fish_count"])
        w.writerows([(a, "%.2f" % b, c) for a, b, c in counts])

    n = len(counts)
    med = lambda k: _median([s[k] for s in stage])          # noqa: E731
    per_frame = med("decode_ms") + med("total_ms")
    summary = (
        "| Metric | Value |\n|---|---|\n"
        "| Frames processed | %d of %d read (every %d) |\n"
        "| **MaxN** (most fish in one frame) | **%d** |\n"
        "| Mean per frame | %.1f |\n"
        "| Frames with no fish | %d |\n"
        "| Decode (software, CPU) median | %.1f ms |\n"
        "| Letterbox median | %.1f ms |\n"
        "| Copy + GPU median | %.1f ms |\n"
        "| NMS median | %.1f ms |\n"
        "| **Per processed frame, serial** | **%.1f ms  (%.1f fps)** |\n"
        "| Wall time for this clip | %.1f s |\n\n"
        "**MaxN** is the standard count from baited video: the most fish visible in any single "
        "frame, since summing frames would count one fish many times. Decode here is software "
        "H.264 on the Arm cores; the board's hardware decoder (NVDEC) is not used by this "
        "path, so real video throughput could be better. Stages run one after another with no "
        "overlap. No annotated video is written: the Orin Nano has no hardware encoder."
        % (n, i, stride, max(c for _, _, c in counts), sum(c for _, _, c in counts) / float(n),
           sum(1 for _, _, c in counts if c == 0), med("decode_ms"), med("preprocess_ms"),
           med("gpu_ms"), med("postprocess_ms"), per_frame, 1000.0 / per_frame, wall))
    peak = cv2.cvtColor(best, cv2.COLOR_BGR2RGB) if best is not None else None
    return summary, peak, csv_path, board_status()


# --------------------------------------------------------------------- the UI
def build_ui(engines):
    sizes = sorted(engines)
    default = str(sizes[-1]) if sizes else "960"
    with gr.Blocks(title="Fish Detector -- Jetson Orin Nano") as ui:
        gr.Markdown(
            "# Fish Detector on the Jetson Orin Nano\n"
            "Real TensorRT FP16 engines running on this board. Nothing is precomputed, and "
            "every timing is measured here, in the conditions shown below.\n\n"
            "*Engines loaded: %s. Trained on DeepFish, single class.*"
            % (", ".join("%dpx" % s for s in sizes) or "none found"))
        status = gr.Markdown(board_status())

        with gr.Tab("Photo"):
            with gr.Row():
                with gr.Column(scale=1):
                    img = gr.Image(type="numpy", label="Photo", height=300)
                    size = gr.Radio([str(s) for s in sizes], value=default,
                                    label="Input resolution")
                    cmp_ = gr.Checkbox(len(sizes) > 1, label="Compare both resolutions")
                    conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05,
                                     label="Confidence threshold")
                    lab = gr.Checkbox(False, label="Show confidence on each box")
                    go = gr.Button("Detect", variant="primary")
                with gr.Column(scale=2):
                    gallery = gr.Gallery(label="Detections", columns=2, height=420)
                    table = gr.Markdown()
            go.click(lambda a, b, c, d, e: run_photo(a, b, c, d, e, engines),
                     [img, size, conf, lab, cmp_], [gallery, table, status], api_name="photo")

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid = gr.Video(label="Clip")
                    vsize = gr.Radio([str(s) for s in sizes], value=default,
                                     label="Input resolution")
                    vconf = gr.Slider(0.05, 0.9, value=0.25, step=0.05,
                                      label="Confidence threshold")
                    vstride = gr.Slider(1, 10, value=2, step=1, label="Process every Nth frame")
                    vsecs = gr.Slider(2, 60, value=15, step=1, label="Seconds to process")
                    vgo = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    vsum = gr.Markdown()
                    vpeak = gr.Image(label="Frame with the most fish")
                    vfile = gr.File(label="Per-frame counts (CSV)")
            vgo.click(lambda a, b, c, d, e: run_video(a, b, c, d, e, engines),
                      [vid, vsize, vconf, vstride, vsecs], [vsum, vpeak, vfile, status],
                      api_name="video")

        gr.Markdown(
            "---\n**What these numbers are.** Stage timings are wall-clock on this board for one "
            "frame at a time, with no overlap between stages, so they describe a simple serial "
            "pipeline, not the best this hardware could do. They are short bursts: sustained "
            "behaviour under heat has not been measured.")
    return ui


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--engines-dir", default="engines")
    ap.add_argument("--host", default="0.0.0.0",
                    help="0.0.0.0 serves your local network; 127.0.0.1 serves this board only")
    ap.add_argument("--port", type=int, default=7860)
    args = ap.parse_args()

    engines = find_engines(args.engines_dir)
    if not engines:
        sys.exit("No *-<H>x<W>-fp16.engine files in %s. Run edge/build_engines.py first."
                 % args.engines_dir)
    print("engines:", {k: os.path.basename(v) for k, v in sorted(engines.items())})
    ui = build_ui(engines)
    ui.queue(max_size=4).launch(server_name=args.host, server_port=args.port)
    return 0


if __name__ == "__main__":
    sys.exit(main())
