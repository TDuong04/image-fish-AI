"""archive_paths: dry run by default, never deletes, refuses protected paths, restores exactly."""
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run(tmp_path, *args):
    env = {**os.environ, "FISH_DATA_ROOT": str(tmp_path / "data")}
    return subprocess.run([sys.executable, str(ROOT / "scripts" / "archive_paths.py"), *map(str, args)],
                          capture_output=True, text=True, env=env, cwd=str(tmp_path))


def make(tmp_path):
    d = tmp_path / "data"
    (d / "interim" / "junk").mkdir(parents=True)
    (d / "interim" / "junk" / "a.txt").write_text("hello")
    (d / "processed" / "deepfish" / "images").mkdir(parents=True)
    (d / "raw").mkdir()
    (d / "raw" / "kakadu_training_data.zip").write_text("zip")
    return d


def test_dry_run_moves_nothing(tmp_path):
    d = make(tmp_path)
    r = run(tmp_path, d / "interim" / "junk", "--reason", "test")
    assert r.returncode == 0 and "dry run" in r.stdout
    assert (d / "interim" / "junk" / "a.txt").exists() and not (d / "archive").exists()


def test_move_writes_manifest_and_restore_undoes_it(tmp_path):
    d = make(tmp_path)
    r = run(tmp_path, d / "interim" / "junk", "--reason", "test", "--yes")
    assert r.returncode == 0, r.stderr
    assert not (d / "interim" / "junk").exists()
    manifests = list((d / "archive").glob("*/MANIFEST.json"))
    assert len(manifests) == 1
    m = json.loads(manifests[0].read_text())
    assert m["reason"] == "test" and m["items"][0]["bytes"] == 5
    assert (Path(m["items"][0]["archived"]) / "a.txt").read_text() == "hello"      # content intact, nothing deleted
    r = run(tmp_path, "--restore", manifests[0], "--yes")
    assert r.returncode == 0, r.stderr
    assert (d / "interim" / "junk" / "a.txt").read_text() == "hello"


def test_protected_paths_are_refused(tmp_path):
    d = make(tmp_path)
    for target in (d / "processed" / "deepfish", d / "raw" / "kakadu_training_data.zip", d / "processed"):
        r = run(tmp_path, target, "--reason", "test", "--yes")
        assert r.returncode != 0 and "refusing" in (r.stdout + r.stderr), target
        assert target.exists()


def test_outside_data_root_is_refused_unless_allowed(tmp_path):
    make(tmp_path)
    outside = tmp_path / "elsewhere"
    outside.mkdir()
    (outside / "f.txt").write_text("x")
    r = run(tmp_path, outside, "--reason", "test", "--yes")
    assert r.returncode != 0 and "outside DATA_ROOT" in (r.stdout + r.stderr)
    assert (outside / "f.txt").exists()


def test_restore_refuses_to_overwrite(tmp_path):
    d = make(tmp_path)
    run(tmp_path, d / "interim" / "junk", "--reason", "test", "--yes")
    (d / "interim" / "junk").mkdir()
    manifest = next((d / "archive").glob("*/MANIFEST.json"))
    r = run(tmp_path, "--restore", manifest, "--yes")
    assert r.returncode != 0 and "already exists" in (r.stdout + r.stderr)


def test_two_archives_never_share_a_manifest(tmp_path):
    d = make(tmp_path)
    (d / "interim" / "other").mkdir()
    (d / "interim" / "other" / "b.txt").write_text("b")
    run(tmp_path, d / "interim" / "junk", "--reason", "first", "--yes")
    run(tmp_path, d / "interim" / "other", "--reason", "second", "--yes")
    manifests = sorted((d / "archive").glob("*/MANIFEST.json"))
    assert len(manifests) == 2
    assert {json.loads(m.read_text())["reason"] for m in manifests} == {"first", "second"}
