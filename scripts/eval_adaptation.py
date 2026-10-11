#!/usr/bin/env python
"""Score the adaptation experiment (docs/eval_protocol.md item 12): zero-shot checkpoints and the fine-tuned ones on the two test lists.

    python scripts/eval_adaptation.py             # after scripts/prepare_adaptation.py and the runs of configs/plan_adapt.yaml

Test lists (built by scripts/prepare_adaptation.py, never used in training or epoch selection):
  test_vivo   the latest 30% of the fixed camera's frames (same camera, later months)
  test_other  every frame of the other 17 groups (other cameras and dates)
Models: the three zero-shot YOLOv8s seeds of F32 (no Mediterranean data) and the four fine-tuned runs (25, 100, all early frames; COCO-only control on all early frames).
One AP routine for everything (101-point AP, scripts/bootstrap_ci.py), a cluster bootstrap over dates (test_vivo) or groups (test_other), fish found and false alarms at 0.25.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402
import bootstrap_ci as bc  # noqa: E402
import error_analysis as ea  # noqa: E402

ZERO_SHOT = ["20261005-yolov8s-960-kakadu-ozfish-s0.pt", "20261007-yolov8s-960-kakadu-ozfish-s1.pt", "20261007-yolov8s-960-kakadu-ozfish-s2.pt"]
RUNS = {"fine-tuned on 25 frames": "yolov8s_adapt_25", "fine-tuned on 100 frames": "yolov8s_adapt_100", "fine-tuned on all early frames": "yolov8s_adapt_all",
        "COCO-only control, all early frames": "yolov8s_adapt_coco_all"}


def find_best(slug: str) -> Path:
    cands = sorted(paths.RUNS_ROOT.glob(f"*-{slug}-s0/weights/best.pt"))
    if not cands:
        sys.exit(f"no finished run for {slug}")
    return cands[-1]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = paths.DATA_ROOT / "processed" / "deepecomar"
    spl = root / "splits_adapt"
    sources = json.loads((root / "sources.json").read_text())
    groups = json.loads((root / "groups.json").read_text())
    import cv2

    def load(name):
        stems = [Path(l.strip()).stem for l in (spl / f"{name}.txt").read_text().splitlines() if l.strip()]
        imgs = [root / "images" / "val" / f"{s}.jpg" for s in stems]
        gts = []
        for p in imgs:
            h, w = cv2.imread(str(p)).shape[:2]
            gts.append(ea.read_labels(root / "labels" / "val" / (p.stem + ".txt"), w, h))
        if name == "test_vivo":
            keys = [re.match(r"vivo_(\d{8})_", sources[s]).group(1) for s in stems]
        else:
            keys = [groups[s] for s in stems]
        return stems, imgs, gts, keys

    device = resolve_device("auto")
    cache = paths.DATA_ROOT / ".cache" / "adapt"
    models = {Path(m).stem: paths.MODELS_ROOT / m for m in ZERO_SHOT}
    labels = {Path(m).stem: "zero-shot (no Mediterranean data)" for m in ZERO_SHOT}
    for label, slug in RUNS.items():
        p = find_best(slug)
        models[p.parent.parent.name] = p
        labels[p.parent.parent.name] = label
    report = {"date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "boot": args.boot, "protocol": "docs/eval_protocol.md item 12", "sets": {}}
    for tname in ("test_vivo", "test_other"):
        stems, imgs, gts, keys = load(tname)
        n_gt = np.array([len(g) for g in gts], dtype=float)
        uniq = sorted(set(keys))
        cid = np.array([uniq.index(k) for k in keys])
        report["sets"][tname] = {"images": len(imgs), "boxes": int(n_gt.sum()), "clusters": len(uniq), "models": {}, "arms": {}}
        scorers, found = {}, {}
        for name, mp in models.items():
            boxes, confs = bc.predict_all(mp, imgs, device, 960, 0.005, cache / f"{name}-{tname}.npz")
            scorers[name] = bc.Scorer(gts, boxes, confs)
            tp = fp = 0
            for gt, b, c in zip(gts, boxes, confs):
                a, f, _ = ea.match_at(gt, b, c, 0.25)
                tp, fp = tp + a, fp + len(f)
            found[name] = (tp, fp)
        rng = np.random.default_rng(args.seed)
        i50, iall = [0], list(range(len(bc.IOUS)))
        zs = [Path(m).stem for m in ZERO_SHOT]
        arms = {"zero-shot (3 seeds, mean)": zs}
        for label, slug in RUNS.items():
            arms[label] = [n for n in models if labels[n] == label]
        full = np.ones(len(imgs))
        point = {a: (float(np.mean([scorers[n].ap(full, n_gt, i50) for n in ns])), float(np.mean([scorers[n].ap(full, n_gt, iall) for n in ns]))) for a, ns in arms.items()}
        boots = {a: ([], []) for a in arms}
        for _ in range(args.boot):
            draw = rng.integers(0, len(uniq), len(uniq))
            w = np.bincount(draw, minlength=len(uniq)).astype(float)[cid]
            for a, ns in arms.items():
                boots[a][0].append(float(np.mean([scorers[n].ap(w, n_gt, i50) for n in ns])))
                boots[a][1].append(float(np.mean([scorers[n].ap(w, n_gt, iall) for n in ns])))

        def ci(v):
            return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

        for a, ns in arms.items():
            report["sets"][tname]["arms"][a] = {"AP50": round(point[a][0], 4), "AP50_ci95": ci(boots[a][0]), "AP50_95": round(point[a][1], 4), "AP50_95_ci95": ci(boots[a][1]),
                                                "found_at_0.25_mean": round(float(np.mean([found[n][0] for n in ns])), 1), "false_alarms_at_0.25_mean": round(float(np.mean([found[n][1] for n in ns])), 1)}
        report["sets"][tname]["models"] = {n: {"label": labels[n], "found_at_0.25": found[n][0], "false_alarms_at_0.25": found[n][1]} for n in models}
    out = args.out or (paths.RUNS_ROOT / f"adaptation-{datetime.now().strftime('%Y%m%d')}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "adaptation.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for t, d in report["sets"].items():
        print(f"== {t}: {d['images']} images, {d['boxes']} boxes, {d['clusters']} clusters")
        for a, v in d["arms"].items():
            print(f"  {a:40s} AP50 {v['AP50']:.4f} {v['AP50_ci95']}  AP50-95 {v['AP50_95']:.4f}  found {v['found_at_0.25_mean']} FA {v['false_alarms_at_0.25_mean']}")
    print(f"wrote {out / 'adaptation.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
