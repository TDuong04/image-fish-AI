"""Kakadu preparation: Deflate64 reading, label conversion, scene grouping and splitting."""
import importlib.util
import sys
import zipfile
import zlib
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fish import zip64  # noqa: E402

SPEC = importlib.util.spec_from_file_location("prepare_kakadu", ROOT / "scripts" / "prepare_kakadu.py")
pk = importlib.util.module_from_spec(SPEC)
sys.modules["prepare_kakadu"] = pk
SPEC.loader.exec_module(pk)


def test_reader_handles_stored_and_deflated_members(tmp_path):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("stored.txt", b"hello" * 100, compress_type=zipfile.ZIP_STORED)
        z.writestr("deflated.txt", b"world" * 1000, compress_type=zipfile.ZIP_DEFLATED)
    with zip64.Zip64Reader(path) as r:
        assert r.read("stored.txt") == b"hello" * 100
        assert r.read("deflated.txt") == b"world" * 1000
        assert "stored.txt" in r


def test_reader_rejects_a_corrupt_member(tmp_path):
    path = tmp_path / "a.zip"
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x.bin", b"A" * 500, compress_type=zipfile.ZIP_STORED)
    data = bytearray(path.read_bytes())
    data[data.index(b"A" * 500) + 10] ^= 0xFF          # flip a byte inside the payload
    path.write_bytes(bytes(data))
    with zip64.Zip64Reader(path) as r, pytest.raises(ValueError, match="corrupt"):
        r.read("x.bin")


def test_deflate64_round_trip():
    inflate64 = pytest.importorskip("inflate64")
    payload = bytes(range(256)) * 400
    comp = inflate64.Deflater()
    blob = comp.deflate(payload) + comp.flush()
    assert zip64.decompress(9, blob) == payload


def test_unknown_method_is_refused():
    with pytest.raises(NotImplementedError):
        zip64.decompress(99, b"")


def test_coco_to_yolo_normalises_and_clips():
    cx, cy, w, h = pk.coco_to_yolo([100, 200, 300, 100], 1000, 1000)
    assert (round(cx, 3), round(cy, 3), round(w, 3), round(h, 3)) == (0.25, 0.25, 0.3, 0.1)
    # a box hanging off the right edge is clipped to the image
    cx, _, w, _ = pk.coco_to_yolo([900, 0, 300, 100], 1000, 1000)
    assert round(w, 3) == 0.1 and round(cx, 3) == 0.95
    assert pk.coco_to_yolo([10, 10, 0.5, 40], 100, 100) is None      # under a pixel wide
    assert pk.coco_to_yolo([-50, -50, 40, 40], 100, 100) is None     # entirely outside


def test_grouping_joins_near_duplicates_and_keeps_distinct_scenes_apart():
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(0)
    scene_a, scene_b = rng.normal(0, 30, 200), rng.normal(0, 30, 200)
    rows = [scene_a + rng.normal(0, 0.3, 200) for _ in range(5)] + [scene_b + rng.normal(0, 0.3, 200) for _ in range(4)]
    thumbs = np.array(rows, dtype=np.float32)
    thumbs -= thumbs.mean(axis=1, keepdims=True)
    group = pk.group_frames(thumbs, 2.0, "cpu")
    assert len(set(group[:5])) == 1 and len(set(group[5:])) == 1 and group[0] != group[5]


def test_split_puts_whole_groups_on_one_side():
    group = np.repeat(np.arange(20), 10)
    is_val = pk.split_groups(group, 0.2, seed=1)
    assert set(group[is_val]).isdisjoint(set(group[~is_val]))
    assert 0.15 <= is_val.mean() <= 0.35
    assert (pk.split_groups(group, 0.2, seed=1) == is_val).all()     # deterministic


def test_default_is_train_only():
    text = (ROOT / "scripts" / "prepare_kakadu.py").read_text(encoding="utf-8")
    assert '"--val-frac", type=float, default=0.0' in text
