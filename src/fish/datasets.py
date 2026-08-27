"""YOLO-format dataset primitives: reading labels, and grouping frames by source.

The grouping key is the load-bearing part. Frames sampled from video are highly
correlated with their neighbours, so a random per-image split puts near-identical
frames in both train and test and inflates mAP by an unknown margin (FD-013).
Splitting by source video is the fix, which means the source video has to be
recoverable from the filename.
"""

from __future__ import annotations

import re
from pathlib import Path

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}

# Frame-numbering suffixes seen in the two datasets actually in use:
#   OzFish   A000010_L-avi-43493        -> clip "A000010_L"
#   DeepFish 7117_Caranx_..._f000123    -> clip "7117_Caranx_..."
#            7117_no_fish_2_f000000     -> clip "7117_no_fish_2"
# DeepFish frames are CONSECUTIVE frames of one continuous clip, so splitting
# anywhere below the clip level leaks near-identical frames across the boundary.
_PATTERNS = (
    re.compile(r"^(?P<g>.+?)-(?:avi|mp4|mov|mkv)-\d+$", re.I),   # A000010_L-avi-43493
    re.compile(r"^(?P<g>.+?)_f\d{3,}$", re.I),                   # 7117_Species_f000123
    re.compile(r"^(?P<g>.+?)_\d{4,}$"),                          # clip_000123
    re.compile(r"^(?P<g>.+?)-\d{4,}$"),                          # clip-000123
)

# Directory names that describe a split or a role rather than a source.
_NON_SOURCE_DIRS = {"images", "labels", "train", "val", "valid", "test", "."}

_SITE = re.compile(r"^(?P<s>\d{3,})_")


def group_key(image_path: Path) -> str:
    """Source-clip identifier for a frame -- the unit a split must not straddle.

    Falls back to the nearest meaningful parent directory, then the stem. A
    dataset whose keys are mostly unique stems has no usable grouping; callers
    should treat that as "grouping unavailable" rather than trusting a split
    built on it.
    """
    stem = image_path.stem
    for pat in _PATTERNS:
        m = pat.match(stem)
        if m:
            return m.group("g")
    for parent in image_path.parents:
        if parent.name and parent.name not in _NON_SOURCE_DIRS:
            return parent.name
    return stem


def site_key(image_path: Path) -> str:
    """Coarser grouping: the capture site / habitat a clip belongs to.

    DeepFish encodes it as a numeric prefix (7117, 9852, ...) on both the clip
    name and the containing directory. Splitting at this level is stricter than
    by clip: it prevents two clips filmed at the same site, in the same water
    and lighting, from landing on opposite sides of the split.

    Returns the clip key when no site can be identified, so callers always get
    a usable -- if less strict -- grouping.
    """
    clip = group_key(image_path)
    m = _SITE.match(clip)
    if m:
        return m.group("s")
    for parent in image_path.parents:
        if _SITE.match(parent.name + "_") or parent.name.isdigit():
            return parent.name
    return clip


def label_for(image_path: Path) -> Path:
    """The .txt label beside an image, or under a sibling labels/ directory."""
    direct = image_path.with_suffix(".txt")
    if direct.exists():
        return direct
    if image_path.parent.name == "images":
        alt = image_path.parent.parent / "labels" / f"{image_path.stem}.txt"
        if alt.exists():
            return alt
    return direct


def iter_pairs(root: Path) -> list[tuple[Path, Path]]:
    """All (image, label) pairs under root. Images without labels are skipped.

    An image with no label file is ambiguous -- it could be a genuine negative
    or a conversion bug -- so it is excluded here and counted by the caller.
    An image with an empty label file is an explicit negative and is kept.
    """
    root = Path(root)
    out = []
    for img in sorted(root.rglob("*")):
        if img.suffix.lower() not in IMAGE_EXTS:
            continue
        lbl = label_for(img)
        if lbl.exists():
            out.append((img, lbl))
    return out


def read_label(path: Path) -> tuple[list[tuple[int, float, float, float, float]], list[str]]:
    """Parse a YOLO label file.

    Returns (boxes, malformed_lines). A box is (class, cx, cy, w, h) with the
    geometry normalised to [0, 1]. Lines that do not parse, or whose geometry is
    outside [0, 1] or has non-positive extent, are reported rather than silently
    dropped -- a coordinate-convention mistake (xyxy vs xywh, absolute vs
    normalised) otherwise trains to near-zero mAP with no error message.
    """
    boxes: list[tuple[int, float, float, float, float]] = []
    bad: list[str] = []
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return boxes, [f"unreadable: {exc}"]

    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) != 5:
            bad.append(line)
            continue
        try:
            cls = int(float(parts[0]))
            cx, cy, w, h = (float(v) for v in parts[1:])
        except ValueError:
            bad.append(line)
            continue
        if not (0.0 < w <= 1.0 and 0.0 < h <= 1.0):
            bad.append(line)
            continue
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            bad.append(line)
            continue
        boxes.append((cls, cx, cy, w, h))
    return boxes, bad
