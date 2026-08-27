#!/usr/bin/env python
"""FD-014: characterise a YOLO-format dataset before training on it.

    python scripts/audit_dataset.py data/interim/ozfish_test/test
    python scripts/audit_dataset.py <dir> --json docs/audit_ozfish.json

Answers the questions that decide the training recipe, and that are expensive to
discover after a training run rather than before it:

  * How small are the objects at each candidate input resolution? A YOLO P3 head
    has stride 8, so a box under ~16 px at the training size is close to
    structurally invisible. This is what decides imgsz, whether a P2 head is
    needed, and whether tiling is required.
  * How many frames contain no objects? Empty frames are what a false-positive
    rate is measured on; a set with none cannot support that metric.
  * How many objects per image? Dense scenes interact with the NMS and
    max_det settings, and with how much a single mistake costs.
  * How many distinct source videos, and how concentrated are frames within
    them? This is the raw material for a leakage-free split (FD-013).
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fish.datasets import IMAGE_EXTS, group_key, iter_pairs, read_label  # noqa: E402

# COCO's small/medium boundaries, in pixels of sqrt(area).
COCO_SMALL = 32
# Below this, a stride-8 feature map has under ~2 cells on the object.
EFFECTIVELY_INVISIBLE = 16

CANDIDATE_SIZES = (416, 512, 640, 800, 960, 1280)


def audit(root: Path, sample_dims: int = 80) -> dict:
    from PIL import Image

    pairs = list(iter_pairs(root))
    if not pairs:
        raise SystemExit(
            f"No image/label pairs found under {root}.\n"
            f"Expected images ({', '.join(sorted(IMAGE_EXTS))}) beside .txt labels."
        )

    dims = Counter()
    for img, _ in pairs[:sample_dims]:
        with Image.open(img) as im:
            dims[im.size] += 1
    native_w, native_h = dims.most_common(1)[0][0]

    sides: list[float] = []      # sqrt(area) in native pixels
    per_image: list[int] = []
    classes: Counter = Counter()
    malformed: list[str] = []
    empty = 0
    groups: Counter = Counter()

    for img, lbl in pairs:
        boxes, bad = read_label(lbl)
        malformed.extend(f"{lbl.name}: {b}" for b in bad)
        per_image.append(len(boxes))
        if not boxes:
            empty += 1
        groups[group_key(img)] += 1
        for cls, _cx, _cy, w, h in boxes:
            classes[cls] += 1
            sides.append(((w * native_w) * (h * native_h)) ** 0.5)

    sides.sort()

    def pct(p: float) -> float:
        return sides[min(int(len(sides) * p), len(sides) - 1)] if sides else 0.0

    by_size = {}
    for imgsz in CANDIDATE_SIZES:
        scale = imgsz / max(native_w, native_h)
        scaled = [s * scale for s in sides]
        by_size[imgsz] = {
            "median_px": round(st.median(scaled), 1) if scaled else 0,
            "pct_under_32px": round(100 * sum(s < COCO_SMALL for s in scaled) / len(scaled), 1)
            if scaled else 0,
            "pct_under_16px": round(
                100 * sum(s < EFFECTIVELY_INVISIBLE for s in scaled) / len(scaled), 1
            ) if scaled else 0,
        }

    frames_per_group = sorted(groups.values())
    return {
        "root": str(root),
        "images": len(pairs),
        "labels_malformed": len(malformed),
        "malformed_examples": malformed[:10],
        "native_resolution": [native_w, native_h],
        "resolutions_seen": {f"{w}x{h}": n for (w, h), n in dims.items()},
        "boxes_total": sum(per_image),
        "classes": dict(classes),
        "boxes_per_image": {
            "mean": round(sum(per_image) / len(per_image), 2),
            "median": st.median(per_image),
            "p95": sorted(per_image)[int(len(per_image) * 0.95)],
            "max": max(per_image),
        },
        "empty_images": empty,
        "empty_pct": round(100 * empty / len(pairs), 1),
        "box_side_px_native": {
            "p5": round(pct(0.05), 1), "p25": round(pct(0.25), 1),
            "median": round(pct(0.50), 1), "p75": round(pct(0.75), 1),
            "p95": round(pct(0.95), 1), "max": round(sides[-1], 1) if sides else 0,
        },
        "by_input_size": by_size,
        "source_groups": len(groups),
        "frames_per_group": {
            "median": st.median(frames_per_group),
            "max": max(frames_per_group),
            "singletons": sum(1 for v in frames_per_group if v == 1),
        },
    }


def report(a: dict) -> None:
    print(f"\n=== {a['root']} ===")
    print(f"  images {a['images']}  boxes {a['boxes_total']}  classes {a['classes']}")
    if a["labels_malformed"]:
        print(f"  !! {a['labels_malformed']} malformed label lines")
        for m in a["malformed_examples"]:
            print(f"       {m}")
    print(f"  native {a['native_resolution'][0]}x{a['native_resolution'][1]}"
          f"   variants: {len(a['resolutions_seen'])}")

    b = a["boxes_per_image"]
    print(f"\n  boxes/image  mean {b['mean']}  median {b['median']}  p95 {b['p95']}  max {b['max']}")
    print(f"  empty frames {a['empty_images']} ({a['empty_pct']}%)")
    if a["empty_pct"] == 0:
        print("    !! No empty frames: this set cannot measure false-positives-per-empty-frame")
        print("       (FD-024). Negatives must come from elsewhere.")
    if b["max"] > 100:
        print(f"    !! Dense scenes present (max {b['max']} boxes). Ultralytics' default")
        print("       max_det=300 and NMS settings need checking before evaluation.")

    s = a["box_side_px_native"]
    print(f"\n  box side (sqrt area) at native resolution:")
    print(f"    p5 {s['p5']}  p25 {s['p25']}  median {s['median']}  "
          f"p75 {s['p75']}  p95 {s['p95']}")

    print(f"\n  scaled to candidate input sizes:")
    print(f"    {'size':>6} {'median':>8} {'<32px':>8} {'<16px':>8}")
    for size, v in a["by_input_size"].items():
        flag = ""
        if v["pct_under_16px"] > 25:
            flag = "  <-- a quarter of boxes effectively invisible to a P3 head"
        elif v["pct_under_32px"] > 60:
            flag = "  <-- dominated by COCO-small objects"
        print(f"    {size:>6} {v['median_px']:>7}px {v['pct_under_32px']:>7}% "
              f"{v['pct_under_16px']:>7}%{flag}")

    g = a["frames_per_group"]
    print(f"\n  source groups {a['source_groups']}  "
          f"(median {g['median']} frames each, max {g['max']}, {g['singletons']} singletons)")
    if a["source_groups"] < a["images"] * 0.1:
        print("    !! Few groups relative to frames: a random split WILL leak")
        print("       near-duplicate frames across train/test (FD-013).")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("root", type=Path, help="directory containing images and .txt labels")
    ap.add_argument("--json", type=Path, default=None, help="also write the audit as JSON")
    args = ap.parse_args()

    a = audit(args.root)
    report(a)
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(a, indent=2), encoding="utf-8")
        print(f"\n  written: {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
