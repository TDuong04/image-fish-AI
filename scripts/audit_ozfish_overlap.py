#!/usr/bin/env python
"""How much of the OzFish labelled set can be used for training without touching the sealed test set?

    python scripts/audit_ozfish_overlap.py
    python scripts/audit_ozfish_overlap.py --labelled <folder of labelled frames> --json out.json

The sealed test set (data/processed/ozfish_eval, 352 images) was cut from the same 1,758 labelled frames, so
training on all of them would put the test frames into training. Names carry the video and frame number
(`A000007_L-avi-74577.jpg` in the test set, `A000007_L.avi.74577.png` in the labelled set); the same name
without the `_L`/`_R` suffix is one deployment filmed by a stereo pair, whose two views show the same scene at
the same moment. This counts identical frames, shared videos and shared deployments and reports how many frames
remain once the test deployments are excluded. It only reads file names; nothing is moved or deleted.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish import paths  # noqa: E402

KEY = re.compile(r"^(?P<vid>.+?)[-.](?P<ext>avi|mp4|mov|mkv)[-.](?P<frame>\d+)(?:\.png|\.jpg)", re.I)


def parse(name: str):
    m = KEY.match(name)
    if not m:
        return None
    video = m["vid"].lower()
    return video, re.sub(r"_(l|r)$", "", video), int(m["frame"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelled", type=Path,
                    default=paths.DATA_ROOT / "processed" / "ozfish" / "assets" / "projects" / "FDFML" / "labelled" / "frames")
    ap.add_argument("--test", type=Path, default=paths.DATA_ROOT / "processed" / "ozfish_eval" / "images" / "val")
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()
    for d in (args.labelled, args.test):
        if not d.exists():
            sys.exit(f"{d} does not exist.")

    lab_names = [p.name for p in sorted(args.labelled.rglob("*.png"))]
    ev_names = [p.name for p in sorted(args.test.iterdir()) if p.suffix.lower() in {".jpg", ".png"}]
    lab = [parse(n) for n in lab_names]
    ev = [parse(n) for n in ev_names]
    if None in lab or None in ev:
        sys.exit("some file names do not look like <video>.<avi|mp4>.<frame>; cannot compare safely.")

    ev_vid, ev_dep = {v for v, _, _ in ev}, {d for _, d, _ in ev}
    lab_vid = collections.Counter(v for v, _, _ in lab)
    res = {
        "test_frames": len(ev), "test_videos": len(ev_vid), "test_deployments": len(ev_dep),
        "labelled_frames": len(lab), "labelled_videos": len(lab_vid), "labelled_deployments": len({d for _, d, _ in lab}),
        "identical_frames": len({(v, f) for v, _, f in ev} & {(v, f) for v, _, f in lab}),
        "labelled_frames_in_test_videos": sum(c for v, c in lab_vid.items() if v in ev_vid),
        "frames_left_excluding_test_videos": sum(1 for v, _, _ in lab if v not in ev_vid),
        "frames_left_excluding_test_deployments": sum(1 for _, d, _ in lab if d not in ev_dep),
        "deployments_left": len({d for _, d, _ in lab if d not in ev_dep}),
    }
    for k, v in res.items():
        print(f"{k:42s} {v}")
    if args.json:
        args.json.write_text(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
