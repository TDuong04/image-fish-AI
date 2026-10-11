#!/usr/bin/env python
"""Isolate leakage from site difficulty: the same sites, scored by a model that saw neighbouring frames and by models that did not.

    python scripts/leakage_control.py

The headline contrast of F20 (0.98 mAP@50 on a random per-image split against about 0.47 on a split by site) compares two different test sets, so a reviewer can say it mixes leakage with
the difficulty of the validation sites. This control uses the SAME sites for both: the random-split model (trained on a random 80% of the fish-containing DeepFish frames, so it saw
neighbouring frames of every site) is scored on its validation frames that belong to the six validation sites of the site split, and the site-split control models (YOLOv8n, DeepFish only, three seeds,
trained without those sites) are scored on the fish-containing frames of those same six sites. The remaining gap is what seeing neighbouring frames adds on identical footage. The random-split
validation list is rebuilt by the same seeded shuffle as scripts/prepare_random_split.py and checked against its recorded size (901 images, 3,058 boxes).
Same AP routine and cluster bootstrap over clips as scripts/bootstrap_ci.py (a clip is resampled for both sets together). Diagnostic only, not a result.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from fish import paths  # noqa: E402
from fish.datasets import iter_pairs, read_label  # noqa: E402
from fish.device import resolve_device  # noqa: E402
import bootstrap_ci as bc  # noqa: E402
import error_analysis as ea  # noqa: E402

RANDOM_MODEL = "20261005-yolov8n-960-randomsplit-s0.pt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import cv2
    # 1. the random-split validation list, rebuilt exactly as scripts/prepare_random_split.py builds it
    src = paths.DATA_ROOT / "interim" / "deepfish_darknet"
    pairs = [(i, l) for i, l in iter_pairs(src) if read_label(l)[0]]
    names = sorted(p[0].name for p in pairs)
    random.Random(0).shuffle(names)
    val_names = set(names[: int(len(names) * 0.2)])
    val_pairs = [(i, l) for i, l in pairs if i.name in val_names]
    n_boxes = sum(len(read_label(l)[0]) for _, l in val_pairs)
    print(f"random-split validation: {len(val_pairs)} images, {n_boxes} boxes (recorded in F20: 901 and 3,058)")
    if (len(val_pairs), n_boxes) != (901, 3058):
        sys.exit("the rebuilt random split does not match the recorded one; stop (the source data differ from the original run)")

    # 2. the six validation sites of the site split, and the control images (fish-containing frames of those sites)
    site_val_dir = paths.DATA_ROOT / "processed" / "deepfish" / "images" / "val"
    site_val = sorted(p for p in site_val_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    sites = {p.stem.split("_")[0] for p in site_val}
    lab_dir = paths.DATA_ROOT / "processed" / "deepfish" / "labels" / "val"
    ctrl_gts_all = []
    for p in site_val:
        h, w = cv2.imread(str(p)).shape[:2]
        ctrl_gts_all.append(ea.read_labels(lab_dir / (p.stem + ".txt"), w, h))
    fish_idx = [i for i, g in enumerate(ctrl_gts_all) if len(g)]
    print(f"validation sites: {sorted(sites)}; control images with fish: {len(fish_idx)} of {len(site_val)}")

    # 3. random-split model on its validation frames
    rs_images = [i for i, _ in val_pairs]
    rs_gts = []
    for i, l in val_pairs:
        h, w = cv2.imread(str(i)).shape[:2]
        rs_gts.append(ea.read_labels(l, w, h))
    device = resolve_device("auto")
    cache = paths.DATA_ROOT / ".cache" / "leakage" / f"{RANDOM_MODEL}-randomval.npz"
    rb, rc = bc.predict_all(paths.MODELS_ROOT / RANDOM_MODEL, rs_images, device, 960, 0.005, cache)
    rs_site = np.array([i.stem.split("_")[0] in sites for i in rs_images])
    rs_clip = [bc.clip_of(i.stem) for i in rs_images]
    print(f"random-split val frames from the six validation sites: {int(rs_site.sum())}; from the other sites: {int((~rs_site).sum())}")

    # 4. site-split controls on the fish frames of the six sites, from the cached predictions of bootstrap_ci.py
    cfg = yaml.safe_load((paths.CONFIG_DIR / "stats_arms.yaml").read_text(encoding="utf-8"))
    ctrl_models = cfg["arms"]["deepfish_only_n"]
    ctrl_clip = [bc.clip_of(site_val[i].stem) for i in fish_idx]
    ctrl_gts = [ctrl_gts_all[i] for i in fish_idx]
    ctrl_scorers = []
    for m in ctrl_models:
        z = np.load(paths.DATA_ROOT / ".cache" / "bootstrap" / f"{m}-deepfish-val-960-0.005-{len(site_val)}.npz", allow_pickle=True)
        ctrl_scorers.append(bc.Scorer(ctrl_gts, [z["boxes"][i] for i in fish_idx], [z["conf"][i] for i in fish_idx]))

    def subset_scorer(mask):
        idx = [k for k in range(len(rs_images)) if mask[k]]
        return bc.Scorer([rs_gts[k] for k in idx], [rb[k] for k in idx], [rc[k] for k in idx]), [rs_clip[k] for k in idx], np.array([len(rs_gts[k]) for k in idx], dtype=float)

    same_sc, same_clip, same_n = subset_scorer(rs_site)
    other_sc, other_clip, other_n = subset_scorer(~rs_site)
    ctrl_n = np.array([len(g) for g in ctrl_gts], dtype=float)

    clips = sorted(set(same_clip) | set(ctrl_clip) | set(other_clip))
    cidx = {c: k for k, c in enumerate(clips)}
    same_cid, ctrl_cid, other_cid = (np.array([cidx[c] for c in x]) for x in (same_clip, ctrl_clip, other_clip))
    i50, iall = [0], list(range(len(bc.IOUS)))

    def score(w_clip):
        a = same_sc.ap(w_clip[same_cid], same_n, i50), same_sc.ap(w_clip[same_cid], same_n, iall)
        o = other_sc.ap(w_clip[other_cid], other_n, i50), other_sc.ap(w_clip[other_cid], other_n, iall)
        c = (float(np.mean([s.ap(w_clip[ctrl_cid], ctrl_n, i50) for s in ctrl_scorers])),
             float(np.mean([s.ap(w_clip[ctrl_cid], ctrl_n, iall) for s in ctrl_scorers])))
        return a, o, c

    ones = np.ones(len(clips))
    pa, po, pc = score(ones)
    rng = np.random.default_rng(args.seed)
    boots = []
    for _ in range(args.boot):
        draw = rng.integers(0, len(clips), len(clips))
        boots.append(score(np.bincount(draw, minlength=len(clips)).astype(float)))

    def ci(v):
        return [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]

    rep = {"random_split_model": RANDOM_MODEL, "control_models": ctrl_models, "validation_sites": sorted(sites), "boot": args.boot, "clips": len(clips),
           "images": {"random_model_same_sites": int(rs_site.sum()), "random_model_other_sites": int((~rs_site).sum()), "control_fish_frames_same_sites": len(fish_idx)},
           "date_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
           "note": "diagnostic only; 101-point AP of bootstrap_ci.py; interval resamples clips for both sets together"}
    for metric, k in (("AP50", 0), ("AP50_95", 1)):
        rep[metric] = {
            "random_split_model_on_the_same_sites": {"point": round(pa[k], 4), "ci95": ci([b[0][k] for b in boots])},
            "site_split_controls_on_the_same_sites": {"point": round(pc[k], 4), "ci95": ci([b[2][k] for b in boots])},
            "gap_same_sites": {"point": round(pa[k] - pc[k], 4), "ci95": ci([b[0][k] - b[2][k] for b in boots])},
            "random_split_model_on_the_other_sites": {"point": round(po[k], 4), "ci95": ci([b[1][k] for b in boots])},
        }
    out = args.out or (paths.RUNS_ROOT / f"stats-{datetime.now().strftime('%Y%m%d')}-leakage-control")
    out.mkdir(parents=True, exist_ok=True)
    (out / "leakage.json").write_text(json.dumps(rep, indent=2), encoding="utf-8")
    for metric in ("AP50", "AP50_95"):
        r = rep[metric]
        print(f"{metric}: random-split model on the same sites {r['random_split_model_on_the_same_sites']['point']} {r['random_split_model_on_the_same_sites']['ci95']} | "
              f"site-split controls {r['site_split_controls_on_the_same_sites']['point']} {r['site_split_controls_on_the_same_sites']['ci95']} | "
              f"gap {r['gap_same_sites']['point']} {r['gap_same_sites']['ci95']} | other sites {r['random_split_model_on_the_other_sites']['point']}")
    print(f"wrote {out / 'leakage.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
