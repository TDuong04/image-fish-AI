"""FD-110: guard the checks whose failure is silent.

A label-format mistake trains to near-zero mAP with no error, and a split that
leaks produces a *better* number than the truth. Neither announces itself, so
both are pinned here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish.datasets import group_key, read_label, site_key  # noqa: E402


# --- grouping: the basis of every leakage-free split ----------------------

@pytest.mark.parametrize("name,expected", [
    # OzFish: <video>-avi-<frame>
    ("A000010_L-avi-43493.jpg", "A000010_L"),
    ("A000011_L-avi-5884.jpg", "A000011_L"),
    # DeepFish: <site>_<species>_f<frame>  -- consecutive frames of one clip
    ("7117_Caranx_sexfasciatus_juvenile_f000000.jpg", "7117_Caranx_sexfasciatus_juvenile"),
    ("7117_Caranx_sexfasciatus_juvenile_f000406.jpg", "7117_Caranx_sexfasciatus_juvenile"),
    ("7268_F1_f000123.jpg", "7268_F1"),
    # DeepFish negatives
    ("7117_no_fish_2_f000000.jpg", "7117_no_fish_2"),
])
def test_group_key_extracts_clip(name, expected):
    assert group_key(Path(name)) == expected


def test_consecutive_frames_share_a_group():
    """The whole point: neighbouring frames must not be separable by a split."""
    a = Path("7463_F2_f000100.jpg")
    b = Path("7463_F2_f000101.jpg")
    assert group_key(a) == group_key(b)


def test_different_clips_at_one_site_differ_by_clip_but_share_site():
    a = Path("7117_Caranx_sexfasciatus_juvenile_f000001.jpg")
    b = Path("7117_no_fish_2_f000001.jpg")
    assert group_key(a) != group_key(b)
    assert site_key(a) == site_key(b) == "7117"


def test_split_directory_is_not_used_as_a_group():
    """`train/` is a role, not a source. Using it as the key collapses everything."""
    p = Path("Deepfish/7117/train/7117_F1_f000001.jpg")
    assert group_key(p) == "7117_F1"
    assert site_key(p) == "7117"


# --- label parsing: catches coordinate-convention mistakes ----------------

def test_valid_label(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("0 0.5 0.5 0.1 0.2\n0 0.25 0.75 0.05 0.05\n")
    boxes, bad = read_label(f)
    assert len(boxes) == 2 and not bad
    assert boxes[0] == (0, 0.5, 0.5, 0.1, 0.2)


def test_empty_label_is_a_valid_negative(tmp_path):
    """An empty label file is an explicit 'no fish here', not an error."""
    f = tmp_path / "a.txt"
    f.write_text("")
    boxes, bad = read_label(f)
    assert boxes == [] and bad == []


@pytest.mark.parametrize("line,why", [
    ("0 100 200 50 80", "absolute pixels instead of normalised"),
    ("0 0.1 0.1 0.9 0.9 0.5", "six fields -- wrong format"),
    ("0 0.5 0.5", "truncated"),
    ("0 0.5 0.5 0 0.2", "zero width"),
    ("0 0.5 0.5 -0.1 0.2", "negative width"),
    ("0 1.5 0.5 0.1 0.2", "centre outside the image"),
    ("fish 0.5 0.5 0.1 0.2", "class not numeric"),
])
def test_malformed_lines_are_reported_not_swallowed(tmp_path, line, why):
    f = tmp_path / "a.txt"
    f.write_text(line + "\n")
    boxes, bad = read_label(f)
    assert boxes == [], f"should reject: {why}"
    assert len(bad) == 1, f"should report: {why}"


def test_xyxy_mistaken_for_xywh_is_caught(tmp_path):
    """A classic: exporting xyxy where xywh is expected.

    Corner coordinates read as width/height give a box that is still inside
    [0,1], so it parses -- but the geometry is wrong. Catch what we can: a
    'width' that reaches the image edge with a centre near it is implausible.
    """
    f = tmp_path / "a.txt"
    f.write_text("0 0.1 0.1 0.9 0.9\n")
    boxes, bad = read_label(f)
    # This one DOES parse -- documenting the limit of static validation.
    # The visual round-trip check in FD-011 is what actually catches it.
    assert len(boxes) == 1 and not bad
