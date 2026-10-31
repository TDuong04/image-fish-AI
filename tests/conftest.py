"""Shared pytest setup.

pytest.ini points --basetemp at data/.cache/pytest (the default temp location is not always
writable on Windows). pytest creates that folder without creating its parents, and a fresh
clone has no data/ directory, so every test using tmp_path errored on a clean checkout (found
by the first CI run). Create the parent here, before any fixture asks for a temp path.
"""

from pathlib import Path

(Path(__file__).resolve().parents[1] / "data" / ".cache").mkdir(parents=True, exist_ok=True)
