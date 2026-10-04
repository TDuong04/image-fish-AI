"""The on-device preprocessing and decoding must match what the accuracy was measured on.

The engine was proven to match PyTorch *on Ultralytics' preprocessing*. If the edge
letterbox differed by even a pixel of padding, that proof would not transfer to what the
Jetson actually runs. Likewise the decoder must select what Ultralytics' NMS selects.
Both are pinned here on synthetic data, so no dataset or board is needed.

edge/trt_runner.py imports TensorRT lazily, so it imports cleanly on a workstation.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "edge"))
sys.path.insert(0, str(REPO / "src"))

import trt_runner as R  # noqa: E402

ultra = pytest.importorskip("ultralytics")
from ultralytics.data.augment import LetterBox  # noqa: E402

from fish.parity import compare_detections, nms as ref_nms  # noqa: E402

SHAPES = [(544, 960), (384, 640)]
# Awkward sizes on purpose: odd dimensions, non-16:9, tiny, huge, square.
SOURCES = [(1080, 1920), (720, 1280), (480, 640), (1081, 1921), (100, 300), (2160, 3840), (900, 900)]


@pytest.mark.parametrize("src", SOURCES)
@pytest.mark.parametrize("hw", SHAPES)
def test_letterbox_is_bit_identical_to_ultralytics(src, hw):
    img = np.random.default_rng(0).integers(0, 255, (*src, 3), dtype=np.uint8)
    mine, _, _ = R.letterbox(img, hw)
    ref = LetterBox(hw, auto=False, stride=32)(image=img)
    assert mine.shape == ref.shape and np.array_equal(mine, ref)


def test_letterbox_padding_and_scale_map_back_exactly():
    """A box drawn on the original frame must survive a round trip through the letterbox."""
    img = np.zeros((1080, 1920, 3), np.uint8)
    _, r, (left, top) = R.letterbox(img, (544, 960))
    x, y = 640.0, 300.0                          # a point in the original frame
    lx, ly = x * r + left, y * r + top           # where it lands in the network input
    assert abs((lx - left) / r - x) < 1e-9 and abs((ly - top) / r - y) < 1e-9


def test_tensor_is_rgb_chw_float01():
    img = np.zeros((4, 4, 3), np.uint8)
    img[:, :, 0] = 255                           # pure BLUE in BGR
    t = R.to_tensor(img)
    assert t.shape == (1, 3, 4, 4) and t.dtype == np.float32
    assert t[0, 2].min() == 1.0 and t[0, 0].max() == 0.0     # blue lands in the LAST channel


def _raw(n=800, seed=0):
    """A synthetic (1, 5, N) output: cx, cy, w, h, score, with a few real clusters."""
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(40, 920, n), rng.uniform(40, 500, n)
    w, h = rng.uniform(10, 120, n), rng.uniform(10, 120, n)
    score = rng.uniform(0, 0.2, n)
    for k, (bx, by) in enumerate([(200, 150), (600, 300), (800, 100)]):
        sl = slice(k * 5, k * 5 + 5)             # five overlapping candidates per fish
        cx[sl], cy[sl] = bx + rng.normal(0, 2, 5), by + rng.normal(0, 2, 5)
        w[sl], h[sl] = 90 + rng.normal(0, 2, 5), 60 + rng.normal(0, 2, 5)
        score[sl] = rng.uniform(0.5, 0.95, 5)
    return np.stack([cx, cy, w, h, score])[None].astype(np.float32)


def test_decode_selects_the_same_detections_as_ultralytics_nms():
    raw = _raw()
    ref = ref_nms(raw)                           # nms() copies, so raw is untouched
    got = R.decode(raw, 1.0, (0, 0), (544, 960))
    c = compare_detections(ref, got)
    assert c["ref"] == c["got"] == 3 and c["matched"] == 3
    assert c["unmatched_ref"] == 0 and c["extra"] == 0 and c["min_iou"] > 0.99


def test_decode_clips_to_the_frame_and_maps_back_through_the_letterbox():
    raw = np.zeros((1, 5, 3), np.float32)
    raw[0, :, 0] = [470, 300, 100, 80, 0.9]      # in the network input, left padding 0
    out = R.decode(raw, r=0.5, pad=(0, 22), orig_hw=(1080, 1920))
    x1, y1, x2, y2, _ = out[0]
    assert (x1, x2) == (840.0, 1040.0)            # (420, 520) / 0.5
    assert (y1, y2) == (476.0, 636.0)             # ((260 - 22), (340 - 22)) / 0.5
    far = np.zeros((1, 5, 1), np.float32)
    far[0, :, 0] = [955, 300, 100, 80, 0.9]       # sticks out of the right edge
    assert R.decode(far, 1.0, (0, 0), (544, 960))[0][2] == 960.0     # clipped, not overflowing


def test_decode_returns_nothing_when_nothing_is_confident():
    raw = np.zeros((1, 5, 50), np.float32)
    raw[0, 4] = 0.1
    assert R.decode(raw, 1.0, (0, 0), (544, 960)).shape == (0, 5)


def test_summarise_reports_median_p95_mean():
    s = R.summarise([{"a": float(i)} for i in range(1, 101)])["a"]
    assert s["median"] == 51.0 and s["p95"] == 96.0 and abs(s["mean"] - 50.5) < 1e-9
