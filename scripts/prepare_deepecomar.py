#!/usr/bin/env python
"""Convert the DEEPECOMAR Mediterranean fish dataset into a single-class evaluation set (the unseen-source test).

    python scripts/prepare_deepecomar.py            # writes data/processed/deepecomar/

Source: Catalan and Alvarez-Ellacuria, Zenodo record 7534425 (CC BY 4.0), `DEEPECOMAR_DATASET_YOLO.zip` placed in data/raw/. It holds Roboflow-exported
YOLO labels with 25 classes (20 species plus `Afish poly`, `labrid unid-`, `small` and `taca`, Catalan for a lens stain). The rules were fixed before any model
was scored on it (docs/eval_protocol.md item 10):

  * every box of a fish class, including `small` and `fish poly`, is the single class `fish`; `taca` (stain) boxes are dropped, because a lens stain is not a fish;
  * the export contains augmented duplicates (a few frames appear twice under different Roboflow hashes), so one copy per source frame is kept;
  * the train and valid folders of the export share no source frame, and both are used here as one evaluation set: nothing in this project was trained on them.

Also writes `groups.json` (a coarse clip id per image taken from the file name) so that intervals can resample groups instead of frames; the ids come from file-name
prefixes and are only an approximation of the real video structure. A second label set without the `small` category is written to labels_nosmall/ for the pre-declared second score.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish import paths  # noqa: E402

NAMES = ["Afish poly", "Chromis chromis", "Coris julis", "Dentex dentex", "Diplodus annularis", "Diplodus puntazzo", "Diplodus sargus", "Diplodus sp-",
         "Diplodus vulgaris", "Epinephelus marginatus", "Lithognathus mormyrus", "Mugilidae prob Chelon", "Mullus sp-", "Oblada melanura", "Pomatous salator",
         "Sarpa salpa", "Seriola dumerilii", "Serranus cabrilla", "Serranus scriba", "Sparus aurata", "Sphyraena sp-", "Symphodus sp-", "labrid unid-", "small", "taca"]
STAIN, SMALL = NAMES.index("taca"), NAMES.index("small")


def base_name(member: str) -> str:
    return re.sub(r"_jpg\.rf\.[0-9a-f]+\.jpg$", "", member.split("/")[-1])


def group_of(base: str) -> str:
    if "_frame_" in base:
        return base.split("_frame_")[0]
    m = re.match(r"(vlcsnap-\d{4}-\d{2}-\d{2}-\d{2}h)", base)
    return m.group(1) if m else base.split("_")[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=paths.DATA_ROOT / "raw" / "deepecomar_yolo.zip")
    ap.add_argument("--out", type=Path, default=paths.DATA_ROOT / "processed" / "deepecomar")
    args = ap.parse_args()
    if not args.zip.exists():
        sys.exit(f"missing {args.zip}. Download Zenodo 7534425 (see docs/eval_protocol.md item 10).")

    z = zipfile.ZipFile(args.zip)
    members = sorted(n for n in z.namelist() if "/images/" in n and n.endswith(".jpg"))
    chosen: dict[str, str] = {}
    for m in members:
        chosen.setdefault(base_name(m), m)                      # first copy per source frame
    img_dir, lab_dir, lab2_dir = (args.out / "images" / "val", args.out / "labels" / "val", args.out / "labels_nosmall" / "val")
    for d in (img_dir, lab_dir, lab2_dir):
        d.mkdir(parents=True, exist_ok=True)
    groups, sources, n_boxes, n_boxes2, n_stain, n_empty = {}, {}, 0, 0, 0, 0
    for k, (base, m) in enumerate(sorted(chosen.items())):
        stem = f"dem_{k:05d}"
        (img_dir / f"{stem}.jpg").write_bytes(z.read(m))
        lab_member = m.replace("/images/", "/labels/").rsplit(".", 1)[0]
        lab_member = re.sub(r"\.jpg$", "", m).replace("/images/", "/labels/") + ".txt"
        rows, rows2 = [], []
        if lab_member in z.namelist():
            for line in z.read(lab_member).decode().splitlines():
                p = line.split()
                if len(p) < 5:
                    continue
                c = int(p[0])
                if c == STAIN:
                    n_stain += 1
                    continue
                row = "0 " + " ".join(p[1:5])
                rows.append(row)
                if c != SMALL:
                    rows2.append(row)
        n_boxes += len(rows)
        n_boxes2 += len(rows2)
        n_empty += not rows
        (lab_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        (lab2_dir / f"{stem}.txt").write_text("\n".join(rows2) + ("\n" if rows2 else ""))
        groups[stem] = group_of(base)
        sources[stem] = base
    (args.out / "groups.json").write_text(json.dumps(groups, indent=1))
    (args.out / "sources.json").write_text(json.dumps(sources, indent=1))     # stem -> original frame name (for time-ordered splits)
    summary = {"source": "Zenodo 7534425 (CC BY 4.0)", "images": len(chosen), "boxes_fish": n_boxes, "boxes_fish_without_small": n_boxes2,
               "stain_boxes_dropped": n_stain, "empty_images": n_empty, "groups": len(set(groups.values()))}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
