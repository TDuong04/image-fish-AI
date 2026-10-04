"""Enforce the part of the documentation rule that a machine can check.

CLAUDE.md requires findings and code changes to be recorded in docs/PROGRESS.md in the
same change. A rule that nothing checks decays, so two mechanical halves of it live here:

  * every committed training run that produced metrics is named in PROGRESS.md, so no
    reported number is without a documented source;
  * every script and demo entrypoint is named in PROGRESS.md, so new code cannot land
    undocumented.

These cannot check that a *description is accurate* -- only that one exists. That is
still the half that silently rots.
"""

from __future__ import annotations

from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
PROGRESS = REPO / "docs" / "PROGRESS.md"


@pytest.fixture(scope="module")
def progress_text() -> str:
    assert PROGRESS.exists(), "docs/PROGRESS.md is missing -- CLAUDE.md requires it"
    return PROGRESS.read_text(encoding="utf-8")


def _real_runs() -> list[str]:
    """Run directories that finished and reported metrics. Smoke runs are pipeline
    checks, not results, and are deliberately excluded."""
    return sorted(
        d.name for d in (REPO / "runs").glob("*")
        if d.is_dir() and (d / "metrics.json").exists()
        and "smoke" not in d.name and not d.name.startswith("_")
    )


@pytest.mark.parametrize("run", _real_runs())
def test_every_result_run_is_documented(run, progress_text):
    assert run in progress_text, (
        f"runs/{run}/ has metrics but docs/PROGRESS.md never mentions it.\n"
        f"A number with no documented source is not a result. Add it to PROGRESS.md "
        f"section 4 (see CLAUDE.md, 'Documentation is part of the work')."
    )


def _entrypoints() -> list[str]:
    files = [p.name for p in (REPO / "scripts").glob("*.py")]
    files += ["demo/app.py"] if (REPO / "demo" / "app.py").exists() else []
    return sorted(files)


@pytest.mark.parametrize("entry", _entrypoints())
def test_every_entrypoint_is_documented(entry, progress_text):
    name = entry.split("/")[-1]
    assert name in progress_text, (
        f"{entry} exists but docs/PROGRESS.md section 2 never mentions {name}.\n"
        f"New code must be recorded in the same change (CLAUDE.md, "
        f"'Documentation is part of the work')."
    )


def test_progress_has_the_required_sections(progress_text):
    for heading in ("## 2. What was built", "## 5. Findings",
                    "## 6. Corrections", "## 7. Bugs found", "## 8. Not done"):
        assert heading in progress_text, f"PROGRESS.md lost its section: {heading!r}"


def test_progress_is_dated(progress_text):
    assert "Last updated:" in progress_text, "PROGRESS.md must say when it was last updated"
