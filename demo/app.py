#!/usr/bin/env python
"""Runnable demo: drop in a photo or a clip and watch the detector work.

    python demo/app.py                 # http://127.0.0.1:7860
    python demo/app.py --share         # also a public link, ~72h, for teammates
    python demo/app.py --model 960     # start on the high-resolution model

Runs the real trained checkpoints from models/. Nothing is precomputed and
nothing is faked: the boxes come from the same weights that produced the
reported mAP, and the timings are measured on whatever GPU this machine has.

The one thing it cannot do is tell you what this costs on a Jetson Orin Nano.
No such board exists in this project yet, so the timing panel reports the
machine it is actually running on and says so.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import cv2  # noqa: E402
import gradio as gr  # noqa: E402
import numpy as np  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402

COLORS = {640: (232, 160, 85), 960: (109, 168, 221)}   # BGR-ish, matches the web demo
_models: dict[int, object] = {}
_device = None


def available_checkpoints() -> dict[int, Path]:
    """Prefer seed 0 so the app matches the published comparison page."""
    found: dict[int, Path] = {}
    for mp in sorted(paths.MODELS_ROOT.glob("*.pt")):
        for sz in (640, 960):
            if f"-{sz}-" in mp.name and (sz not in found or "s0" in mp.name):
                found[sz] = mp
    return found


def get_model(imgsz: int):
    global _device
    if _device is None:
        _device = resolve_device("auto")
    if imgsz not in _models:
        ck = available_checkpoints()
        if imgsz not in ck:
            raise gr.Error(
                f"No {imgsz}px checkpoint in {paths.rel(paths.MODELS_ROOT)}/. "
                f"Run scripts/build_demo.py's prerequisites, or copy a best.pt there."
            )
        from ultralytics import YOLO
        _models[imgsz] = YOLO(str(ck[imgsz]))
    return _models[imgsz]


def draw(frame: np.ndarray, boxes, confs, imgsz: int, show_conf: bool) -> np.ndarray:
    out = frame.copy()
    col = COLORS[imgsz]
    thick = max(2, int(round(out.shape[1] / 700)))
    for b, c in zip(boxes, confs):
        x1, y1, x2, y2 = (int(v) for v in b)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, thick)
        if show_conf:
            label = f"{c:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(out, (x1, max(0, y1 - th - 6)), (x1 + tw + 6, y1), col, -1)
            cv2.putText(out, label, (x1 + 3, max(10, y1 - 4)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (20, 20, 20), 1, cv2.LINE_AA)
    return out


def run_image(image, imgsz_choice, conf, show_conf, compare):
    if image is None:
        raise gr.Error("Add a photo first.")
    sizes = [640, 960] if compare else [int(imgsz_choice)]
    frame = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    panels, lines = [], []
    for sz in sizes:
        model = get_model(sz)
        t0 = time.perf_counter()
        r = model.predict(frame, imgsz=sz, conf=conf, device=_device.ultralytics,
                          verbose=False)[0]
        ms = (time.perf_counter() - t0) * 1000
        boxes = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        drawn = draw(frame, boxes, confs, sz, show_conf)
        panels.append((cv2.cvtColor(drawn, cv2.COLOR_BGR2RGB), f"{sz}px — {len(boxes)} fish"))
        lines.append(f"| {sz} px | **{len(boxes)}** | {ms:.0f} ms | "
                     f"{np.mean(confs):.2f} |" if len(confs) else
                     f"| {sz} px | **0** | {ms:.0f} ms | — |")

    table = ("| Input | Fish found | Time | Mean conf |\n|---|---|---|---|\n"
             + "\n".join(lines))
    note = (f"\n\nRunning on **{_device}**. Timings are this machine, not a Jetson — "
            f"no such board exists in this project yet.")
    if compare and len(sizes) == 2:
        a, b = [int(l.split("**")[1]) for l in lines]
        delta = b - a
        note = (f"\n\n**960 found {delta:+d} compared with 640.** A single frame proves nothing "
                f"on its own — across the full validation set the gain is concentrated in fish "
                f"roughly 32–96 px wide." + note)
    return panels, table + note


def run_video(video, imgsz_choice, conf, stride, max_seconds, progress=gr.Progress()):
    if video is None:
        raise gr.Error("Add a clip first.")
    sz = int(imgsz_choice)
    model = get_model(sz)

    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        raise gr.Error("Could not open that video.")
    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    limit = int(fps_in * max_seconds) if max_seconds else total

    out_path = str(Path(video).with_name(Path(video).stem + f"_detected_{sz}.mp4"))
    writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"mp4v"),
                             fps_in / max(1, stride), (w, h))

    counts, times, i, kept = [], [], 0, 0
    while True:
        ok, frame = cap.read()
        if not ok or i >= limit:
            break
        if i % stride == 0:
            t0 = time.perf_counter()
            r = model.predict(frame, imgsz=sz, conf=conf, device=_device.ultralytics,
                              verbose=False)[0]
            times.append((time.perf_counter() - t0) * 1000)
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            counts.append(len(boxes))
            writer.write(draw(frame, boxes, confs, sz, False))
            kept += 1
            if total:
                progress(min(i / min(limit, total), 1.0), desc=f"frame {i}")
        i += 1
    cap.release()
    writer.release()

    if not counts:
        raise gr.Error("No frames were processed.")

    times.sort()
    med = times[len(times) // 2]
    p95 = times[int(len(times) * 0.95)]
    # MaxN is the count ecologists actually use from baited video: the peak number
    # visible in any single frame, which avoids double-counting the same fish.
    summary = (
        f"| Metric | Value |\n|---|---|\n"
        f"| Frames processed | {kept} of {i} (every {stride}) |\n"
        f"| **MaxN** (peak in one frame) | **{max(counts)}** |\n"
        f"| Mean per frame | {np.mean(counts):.1f} |\n"
        f"| Frames with no fish | {sum(1 for c in counts if c == 0)} |\n"
        f"| Median latency | {med:.0f} ms ({1000/med:.0f} fps) |\n"
        f"| p95 latency | {p95:.0f} ms |\n\n"
        f"Running on **{_device}** at {sz}px. **MaxN** is the standard count from baited "
        f"underwater video — the most fish visible in any single frame — because it cannot "
        f"double-count one fish swimming past twice.\n\n"
        f"These timings are this machine. A Jetson Orin Nano would be substantially slower, "
        f"and this project has not measured one."
    )
    return out_path, summary


def build_ui(default_model: int) -> gr.Blocks:
    ck = available_checkpoints()
    found = ", ".join(f"{k}px" for k in sorted(ck)) or "none found"

    with gr.Blocks(title="Fish Detector") as ui:
        gr.Markdown(
            "# Fish Detector\n"
            "Drop in a photo or a clip and watch the model work. These are the real trained "
            "checkpoints behind the reported results — nothing precomputed, nothing faked.\n\n"
            f"*Checkpoints loaded: {found}. Trained on DeepFish, single class, site-level splits. "
            f"The first detection takes a few seconds while the weights load; every one after "
            f"that is fast.*"
        )

        with gr.Tab("Photo"):
            with gr.Row():
                with gr.Column(scale=1):
                    img_in = gr.Image(type="numpy", label="Photo", height=300)
                    img_size = gr.Radio(["640", "960"], value=str(default_model),
                                        label="Input resolution")
                    img_cmp = gr.Checkbox(True, label="Compare 640 against 960")
                    img_conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05,
                                         label="Confidence threshold")
                    img_lab = gr.Checkbox(False, label="Show confidence on each box")
                    img_go = gr.Button("Detect", variant="primary")
                with gr.Column(scale=2):
                    img_out = gr.Gallery(label="Detections", columns=2, height=420)
                    img_txt = gr.Markdown()
            img_go.click(run_image, [img_in, img_size, img_conf, img_lab, img_cmp],
                         [img_out, img_txt])

        with gr.Tab("Video"):
            with gr.Row():
                with gr.Column(scale=1):
                    vid_in = gr.Video(label="Clip")
                    vid_size = gr.Radio(["640", "960"], value=str(default_model),
                                        label="Input resolution")
                    vid_conf = gr.Slider(0.05, 0.9, value=0.25, step=0.05,
                                         label="Confidence threshold")
                    vid_stride = gr.Slider(1, 10, value=2, step=1,
                                           label="Process every Nth frame")
                    vid_secs = gr.Slider(2, 60, value=15, step=1,
                                         label="Seconds to process")
                    vid_go = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    vid_out = gr.Video(label="Annotated clip")
                    vid_txt = gr.Markdown()
            vid_go.click(run_video, [vid_in, vid_size, vid_conf, vid_stride, vid_secs],
                         [vid_out, vid_txt])

        gr.Markdown(
            "---\n"
            "**What the numbers mean.** In-domain the 960px model scores mAP@50 0.503 against "
            "0.416 for 640 (3 seeds each). On held-out OzFish footage the models have never "
            "seen, both drop by more than half and the gap narrows to 0.013 — so most of the "
            "in-domain advantage does not survive a change of scene.\n\n"
            "**No Jetson numbers anywhere.** The board has not been acquired, so any FPS figure "
            "for it would be copied from a datasheet rather than measured."
        )
    return ui


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--share", action="store_true",
                    help="also expose a public link (~72h) so teammates can try it")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--model", type=int, default=960, choices=(640, 960))
    args = ap.parse_args()

    if not available_checkpoints():
        raise SystemExit(
            f"No model files found in {paths.rel(paths.MODELS_ROOT)}/.\n"
            f"The trained models are not stored in git. Get the .pt files from a teammate\n"
            f"and put them in that folder (names like '20260922-yolov8n-960-s0.pt' -- the\n"
            f"name must contain '-640-' or '-960-')."
        )
    # Gradio 6 moved `theme` from the Blocks constructor to launch().
    build_ui(args.model).launch(server_port=args.port, share=args.share,
                                theme=gr.themes.Soft(),
                                inbrowser=not args.share)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
