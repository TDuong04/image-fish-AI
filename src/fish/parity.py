"""Compare two detectors' outputs on the same input. Shared by every parity check.

Three stages of the deployment chain each need proving against the same PyTorch
reference: PyTorch -> ONNX, ONNX -> TensorRT FP16, and later TensorRT INT8. They must
all use one definition of "the same detections", or a pass at one stage and a pass at
the next mean different things.

The one subtlety is the confidence threshold. A detection whose score sits a hair above
the cutoff in FP32 can land a hair below it in FP16, so it appears to vanish. That is
numerical noise, not a defect, and counting it as one would fail a correct engine. Such
**borderline** detections are reported separately and excluded from the pass/fail count.
A confident detection that goes missing, or a confident one that appears from nowhere,
is a real disagreement.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

CONF = 0.25
NMS_IOU = 0.7
# Scores within this margin of CONF may flip across the threshold under FP16 rounding.
BORDERLINE = 0.03


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def box_iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    # float(): with float32 detections this was a numpy scalar, which json cannot
    # serialise -- so a passing check wrote no result file at all.
    return float(inter / ua) if ua > 0 else 0.0


def nms(raw: np.ndarray, conf: float = CONF, iou: float = NMS_IOU) -> np.ndarray:
    """(1, 5, N) raw network output -> (k, 6) [x1, y1, x2, y2, conf, cls]."""
    import torch
    try:
        from ultralytics.utils.nms import non_max_suppression   # recent releases
    except ImportError:
        from ultralytics.utils.ops import non_max_suppression   # older releases
    out = non_max_suppression(torch.from_numpy(np.ascontiguousarray(raw, dtype=np.float32)),
                              conf_thres=conf, iou_thres=iou, max_det=300, nc=1)[0]
    return out.cpu().numpy()


def compare_detections(ref: np.ndarray, got: np.ndarray, conf: float = CONF,
                       borderline: float = BORDERLINE, match_iou: float = 0.5) -> dict:
    """Match each reference detection to its best unused candidate.

    `unmatched_ref` / `extra` count only CONFIDENT disagreements. Detections within
    `borderline` of the threshold are tallied under `*_borderline` and not held against
    the engine.
    """
    used, ious, dconf = set(), [], []
    unmatched_ref = unmatched_ref_borderline = 0
    for r in ref:
        best, bi = 0.0, None
        for i, g in enumerate(got):
            if i in used:
                continue
            v = box_iou(r[:4], g[:4])
            if v > best:
                best, bi = v, i
        if bi is not None and best >= match_iou:
            used.add(bi)
            ious.append(best)
            dconf.append(abs(float(r[4]) - float(got[bi][4])))
        elif float(r[4]) < conf + borderline:
            unmatched_ref_borderline += 1
        else:
            unmatched_ref += 1

    extra = extra_borderline = 0
    for i, g in enumerate(got):
        if i in used:
            continue
        if float(g[4]) < conf + borderline:
            extra_borderline += 1
        else:
            extra += 1

    return {"ref": int(len(ref)), "got": int(len(got)), "matched": len(ious),
            "min_iou": min(ious) if ious else None,
            "max_dconf": max(dconf) if dconf else None,
            "unmatched_ref": unmatched_ref, "unmatched_ref_borderline": unmatched_ref_borderline,
            "extra": extra, "extra_borderline": extra_borderline}


def verdict(rows: list[dict], max_dconf: float = 0.05, min_iou: float = 0.9) -> dict:
    """Aggregate per-frame comparisons into one pass/fail with the reasons spelled out."""
    ious = [r["min_iou"] for r in rows if r["min_iou"] is not None]
    dcs = [r["max_dconf"] for r in rows if r["max_dconf"] is not None]
    summary = {
        "frames": len(rows),
        "unmatched_ref": sum(r["unmatched_ref"] for r in rows),
        "extra": sum(r["extra"] for r in rows),
        "borderline": sum(r["unmatched_ref_borderline"] + r["extra_borderline"] for r in rows),
        "worst_iou": min(ious) if ious else None,
        "worst_dconf": max(dcs) if dcs else None,
    }
    reasons = []
    if summary["unmatched_ref"]:
        reasons.append(f"{summary['unmatched_ref']} confident reference detection(s) missing")
    if summary["extra"]:
        reasons.append(f"{summary['extra']} confident detection(s) with no reference")
    if summary["worst_iou"] is not None and summary["worst_iou"] < min_iou:
        reasons.append(f"worst matched IoU {summary['worst_iou']:.3f} < {min_iou}")
    if summary["worst_dconf"] is not None and summary["worst_dconf"] > max_dconf:
        reasons.append(f"worst confidence drift {summary['worst_dconf']:.3f} > {max_dconf}")
    summary["pass"] = not reasons
    summary["reasons"] = reasons
    return summary
