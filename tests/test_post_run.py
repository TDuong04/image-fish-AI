"""post_run picks the newest finished run, not an unfinished one."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from post_run import find_run  # noqa: E402


def make(root: Path, name: str, finished: bool):
    d = root / name
    (d / "weights").mkdir(parents=True)
    (d / "weights" / "best.pt").write_text("x")
    if finished:
        (d / "metrics.json").write_text("{}")


def test_newest_finished_run_is_chosen(tmp_path):
    make(tmp_path, "20261005-yolov8s-960-x-s1", True)
    make(tmp_path, "20261008-yolov8s-960-x-s1", True)
    make(tmp_path, "20261009-yolov8s-960-x-s1", False)      # still training
    make(tmp_path, "20261008-yolov8s-960-x-s2", True)       # another seed
    assert find_run(tmp_path, "yolov8s-960-x-s1").name == "20261008-yolov8s-960-x-s1"


def test_no_finished_run_returns_none(tmp_path):
    make(tmp_path, "20261008-yolov8s-960-x-s1", False)
    assert find_run(tmp_path, "yolov8s-960-x-s1") is None
