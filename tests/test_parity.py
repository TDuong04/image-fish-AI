"""The parity comparison decides whether an engine is trusted, so it must be right.

A comparison that wrongly passes would let a broken engine's latency be reported as a
result; one that wrongly fails would reject a correct FP16 engine over rounding noise.
Both directions are pinned here on synthetic detections.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish.parity import BORDERLINE, CONF, box_iou, compare_detections, nms, verdict  # noqa: E402


def det(x1, y1, x2, y2, conf):
    return np.array([x1, y1, x2, y2, conf, 0.0], dtype=np.float32)


def test_identical_detections_pass():
    ref = np.stack([det(0, 0, 10, 10, 0.9), det(50, 50, 80, 90, 0.6)])
    r = compare_detections(ref, ref.copy())
    assert r["matched"] == 2 and r["unmatched_ref"] == 0 and r["extra"] == 0
    assert verdict([r])["pass"]


def test_a_confident_detection_going_missing_fails():
    """The failure that matters: the engine silently drops a clear fish."""
    ref = np.stack([det(0, 0, 10, 10, 0.9), det(50, 50, 80, 90, 0.8)])
    got = ref[:1]
    r = compare_detections(ref, got)
    assert r["unmatched_ref"] == 1
    v = verdict([r])
    assert not v["pass"] and any("missing" in x for x in v["reasons"])


def test_a_confident_phantom_detection_fails():
    ref = np.stack([det(0, 0, 10, 10, 0.9)])
    got = np.stack([det(0, 0, 10, 10, 0.9), det(200, 200, 240, 260, 0.7)])
    r = compare_detections(ref, got)
    assert r["extra"] == 1 and not verdict([r])["pass"]


def test_borderline_flip_is_noise_not_a_defect():
    """FP16 can push a 0.26 score under the 0.25 cutoff. That must not fail an engine."""
    just_above = CONF + BORDERLINE / 3
    ref = np.stack([det(0, 0, 10, 10, 0.9), det(50, 50, 80, 90, just_above)])
    got = ref[:1]                                    # the borderline one dropped out
    r = compare_detections(ref, got)
    assert r["unmatched_ref"] == 0 and r["unmatched_ref_borderline"] == 1
    assert verdict([r])["pass"]


def test_borderline_phantom_is_also_tolerated():
    ref = np.stack([det(0, 0, 10, 10, 0.9)])
    got = np.stack([det(0, 0, 10, 10, 0.9), det(200, 200, 240, 260, CONF + 0.01)])
    r = compare_detections(ref, got)
    assert r["extra"] == 0 and r["extra_borderline"] == 1 and verdict([r])["pass"]


def test_a_displaced_box_fails_on_iou():
    ref = np.stack([det(0, 0, 100, 100, 0.9)])
    got = np.stack([det(20, 20, 120, 120, 0.9)])     # IoU 0.47: a match, but misplaced
    r = compare_detections(ref, got, match_iou=0.3)
    v = verdict([r], min_iou=0.9)
    assert r["matched"] == 1 and not v["pass"]
    assert any("IoU" in x for x in v["reasons"])


def test_large_confidence_drift_fails():
    ref = np.stack([det(0, 0, 10, 10, 0.9)])
    got = np.stack([det(0, 0, 10, 10, 0.6)])
    assert not verdict([compare_detections(ref, got)], max_dconf=0.05)["pass"]


def test_empty_frames_pass_trivially():
    """No detections on either side is agreement, e.g. the empty-water frames."""
    empty = np.zeros((0, 6), dtype=np.float32)
    r = compare_detections(empty, empty)
    assert r["ref"] == 0 and r["got"] == 0 and verdict([r])["pass"]


def test_each_candidate_matches_at_most_one_reference():
    """Two reference fish must not both claim one engine box."""
    ref = np.stack([det(0, 0, 10, 10, 0.9), det(1, 1, 11, 11, 0.9)])
    got = np.stack([det(0, 0, 10, 10, 0.9)])
    r = compare_detections(ref, got)
    assert r["matched"] == 1 and r["unmatched_ref"] == 1


def test_box_iou_basics():
    assert box_iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert box_iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert abs(box_iou([0, 0, 10, 10], [5, 0, 15, 10]) - (50 / 150)) < 1e-9


def test_results_are_json_serialisable_with_float32_detections():
    """Detections arrive as float32. A passing check must still be able to save itself:
    this once printed PASS and then crashed writing the result, leaving no record."""
    ref = np.stack([det(0, 0, 10, 10, 0.9)]).astype(np.float32)
    got = np.stack([det(0, 0, 11, 10, 0.88)]).astype(np.float32)
    r = compare_detections(ref, got)
    json.dumps({"row": r, "verdict": verdict([r])})        # raises TypeError if not


def test_nms_does_not_modify_its_input():
    """Ultralytics' NMS converts boxes xywh -> xyxy in place, and torch.from_numpy shares
    memory. Without a copy the caller's raw output was silently overwritten, so a later
    reader (or a second decoder) saw already-converted boxes and converted them again."""
    raw = np.zeros((1, 5, 6), dtype=np.float32)
    raw[0, :, 0] = [100, 80, 40, 30, 0.9]          # cx, cy, w, h, score
    before = raw.copy()
    nms(raw)
    assert np.array_equal(raw, before), "nms() overwrote the array it was given"
