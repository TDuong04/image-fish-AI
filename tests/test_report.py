"""The report draft must stay traceable: every finding it cites exists, every reference is tracked."""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "docs" / "report"
PENDING = set()   # findings the draft refers to as pending; remove each one once it is written up in PROGRESS.md


def test_every_cited_finding_exists_in_progress():
    paper = (REPORT / "paper.md").read_text(encoding="utf-8")
    progress = (ROOT / "docs" / "PROGRESS.md").read_text(encoding="utf-8")
    cited = {int(n) for n in re.findall(r"\bF(\d+)\b", paper)}
    defined = {int(n) for n in re.findall(r"^### F(\d+)\.", progress, flags=re.M)}
    assert cited, "paper cites no findings"
    cited -= PENDING
    assert cited <= defined, f"findings cited but not in PROGRESS.md: {sorted(cited - defined)}"


def test_every_bib_key_is_tracked_and_cited():
    bib = (REPORT / "references.bib").read_text(encoding="utf-8")
    tracker = (REPORT / "references.md").read_text(encoding="utf-8")
    keys = re.findall(r"^@\w+\{([^,]+),", bib, flags=re.M)
    assert keys
    missing = [k for k in keys if f"`{k}`" not in tracker]
    assert not missing, f"references without a tracker row: {missing}"
