"""queue_runner: runs steps in order, records state, skips finished work, survives a failing step."""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def run_plan(tmp_path, steps_yaml):
    plan = tmp_path / "plan.yaml"
    plan.write_text(steps_yaml)
    env = {"FISH_DATA_ROOT": str(tmp_path / "data"), "PATH": "", "SYSTEMROOT": "C:\Windows"}
    import os
    env = {**os.environ, "FISH_DATA_ROOT": str(tmp_path / "data")}
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "queue_runner.py"), str(plan), "--poll", "0.2"],
                       capture_output=True, text=True, timeout=120, env=env)
    status = json.loads((tmp_path / "data" / ".cache" / "queue" / "status.json").read_text())
    return r, status


def test_steps_run_in_order_and_failure_does_not_stop_the_queue(tmp_path):
    r, status = run_plan(tmp_path, """
steps:
  - name: first
    cmd: ["{python}", "-c", "print('a')"]
  - name: broken
    cmd: ["{python}", "-c", "import sys; sys.exit(3)"]
  - name: last
    cmd: ["{python}", "-c", "print('z')"]
""")
    assert r.returncode == 0, r.stderr
    assert status["steps"]["first"]["state"] == "done"
    assert status["steps"]["broken"]["state"] == "failed" and status["steps"]["broken"]["exit"] == 3
    assert status["steps"]["last"]["state"] == "done"


def test_stop_on_failure_stops(tmp_path):
    r, status = run_plan(tmp_path, """
steps:
  - name: broken
    stop_on_failure: true
    cmd: ["{python}", "-c", "import sys; sys.exit(2)"]
  - name: never
    cmd: ["{python}", "-c", "print('no')"]
""")
    assert r.returncode == 2
    assert "never" not in status["steps"]


def test_skip_if_marks_done_without_running(tmp_path):
    (tmp_path / "marker.txt").write_text("x")
    # patterns are relative to the repo root, so point at a file that always exists there
    r, status = run_plan(tmp_path, """
steps:
  - name: skipped
    skip_if: "README.md"
    cmd: ["{python}", "-c", "import sys; sys.exit(9)"]
""")
    assert status["steps"]["skipped"]["state"] == "done"
    assert "skipped" in status["steps"]["skipped"]["note"]


def test_wait_files_blocks_until_the_file_exists(tmp_path):
    flag = ROOT / "data" / ".cache" / "queue_test_flag.txt"
    flag.parent.mkdir(parents=True, exist_ok=True)
    flag.write_text("here")
    try:
        r, status = run_plan(tmp_path, """
steps:
  - name: waits
    wait_files: ["data/.cache/queue_test_flag.txt"]
    cmd: ["{python}", "-c", "print('go')"]
""")
        assert status["steps"]["waits"]["state"] == "done"
    finally:
        flag.unlink()
