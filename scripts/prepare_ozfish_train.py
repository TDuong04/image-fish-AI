#!/usr/bin/env python
"""FD-113: turn the OzFish labelled frames into a single-class YOLO training set that excludes the sealed test.

    python scripts/prepare_ozfish_train.py                 # writes data/processed/ozfish_train
    python scripts/prepare_ozfish_train.py --report-only   # leak statistics, writes nothing

The sealed test set (data/processed/ozfish_eval, 352 images) is cut from the same 1,758 labelled frames. BRUVS films
with stereo pairs, so the `_L` and `_R` videos of one deployment show the same scene at the same moment. This keeps
only frames whose *deployment* is absent from the test set (794 of 1,758), converts the SageMaker Ground Truth boxes to
YOLO labels (one class, `fish`), resizes frames to --max-side (default 960, the training size), then measures how close
each kept frame is to the sealed test frames and to the DeepFish validation frames (thumbnail distance, as in the
Kakadu check) and writes the figures to prepare.json. Frames closer than --drop-below to a test frame are dropped.

With --val-frac, whole deployments are held out as a decision-only validation set (images/val). It is not perfectly leak-free: one reef can appear
in several deployments.

Training on these frames changes what the sealed OzFish test measures (new deployments of the same collections, not an
unseen kind of footage); see docs/data_audit.md. Nothing here reads or evaluates the sealed images' labels.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402

from audit_ozfish_overlap import parse  # noqa: E402
from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402
from prepare_kakadu import coco_to_yolo  # noqa: E402

THUMB_W, THUMB_H = 48, 27
OZ_ROOT = paths.DATA_ROOT / "processed" / "ozfish" / "assets" / "projects" / "FDFML" / "labelled"


def frame_key(name: str):
    """(video, frame) for a manifest source-ref or a file name; the export suffix `-N-1` is ignored."""
    p = parse(name)
    return (p[0], p[2]) if p else None


def thumb(path: Path) -> np.ndarray:
    import cv2
    g = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_REDUCED_GRAYSCALE_8)
    if g is None:
        raise SystemExit(f"cannot decode {path}")
    t = cv2.resize(g, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    return t - t.mean()


def nearest(a: np.ndarray, b: np.ndarray, device: str, block: int = 500):
    """Nearest-neighbour mean-abs distance from each row of a to the rows of b."""
    import torch
    tb = torch.from_numpy(b).to(device)
    out, arg = [], []
    for s in range(0, len(a), block):
        d = torch.cdist(torch.from_numpy(a[s:s + block]).to(device), tb, p=1) / a.shape[1]
        m, i = d.min(dim=1)
        out.append(m.cpu().numpy())
        arg.append(i.cpu().numpy())
    return np.concatenate(out), np.concatenate(arg)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labelled", type=Path, default=OZ_ROOT)
    ap.add_argument("--test", type=Path, default=paths.DATA_ROOT / "processed" / "ozfish_eval" / "images" / "val")
    ap.add_argument("--deepfish-val", type=Path, default=paths.DATA_ROOT / "processed" / "deepfish" / "images" / "val")
    ap.add_argument("--out", type=Path, default=paths.DATA_ROOT / "processed" / "ozfish_train")
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--drop-below", type=float, default=3.0,
                    help="drop a kept frame whose thumbnail distance to any sealed-test frame is below this")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="hold out about this fraction of the kept FRAMES as a validation set, by whole deployments (stereo pairs stay together)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    for d in (args.labelled, args.test, args.deepfish_val):
        if not d.exists():
            sys.exit(f"{d} does not exist.")
    frames = {frame_key(p.name): p for p in sorted((args.labelled / "frames").rglob("*.png"))}
    if None in frames:
        sys.exit("some frame names could not be parsed; refusing to guess.")
    test_files = sorted(p for p in args.test.iterdir() if p.suffix.lower() in {".jpg", ".png"})
    test_dep = {parse(p.name)[1] for p in test_files}

    records = {}
    for m in sorted((args.labelled / "manifests").iterdir()):
        for line in m.read_text().splitlines():
            r = json.loads(line)
            body = next(v for k, v in r.items() if k != "source-ref" and not k.endswith("-metadata"))
            w, h = body["image_size"][0]["width"], body["image_size"][0]["height"]
            records[frame_key(r["source-ref"])] = (w, h, body["annotations"])
    missing = [k for k in frames if k not in records]
    print(f"frames {len(frames)} | manifest records {len(records)} | frames without a record {len(missing)}")

    kept = [(k, p) for k, p in frames.items() if k in records and parse(p.name)[1] not in test_dep]
    print(f"kept after excluding the {len(test_dep)} test deployments: {len(kept)}")

    import cv2
    device_info = resolve_device("auto")
    dev = "cuda" if device_info.kind == "cuda" else "cpu"
    kt = np.stack([thumb(p) for _, p in kept])
    tt = np.stack([thumb(p) for p in test_files])
    dv = np.stack([thumb(p) for p in sorted(args.deepfish_val.glob("*.jpg"))])
    d_test, i_test = nearest(kt, tt, dev)
    d_df, _ = nearest(kt, dv, dev)
    dep = np.array([parse(p.name)[1] for _, p in kept])
    base = []                                   # distance to the nearest frame of a DIFFERENT deployment inside the kept set
    for s in range(0, len(kt), 400):
        import torch
        d = torch.cdist(torch.from_numpy(kt[s:s + 400]).to(dev), torch.from_numpy(kt).to(dev), p=1) / kt.shape[1]
        same = torch.from_numpy(dep[s:s + 400][:, None] == dep[None, :]).to(dev)
        d[same] = 1e9
        base.append(d.min(dim=1).values.cpu().numpy())
    base = np.concatenate(base)

    def stats(x):
        return {"min": round(float(x.min()), 2), "p1": round(float(np.percentile(x, 1)), 2),
                "median": round(float(np.median(x)), 2), "below_3": int((x < 3).sum()),
                "below_4": int((x < 4).sum()), "below_5": int((x < 5).sum())}

    report = {"kept_before_drop": len(kept), "test_deployments_excluded": len(test_dep),
              "to_sealed_test": stats(d_test), "to_deepfish_val": stats(d_df), "to_other_deployments_in_kept_set": stats(base)}
    print(json.dumps(report, indent=2))
    if args.report_only:
        return 0

    keep_mask = d_test >= args.drop_below
    # Hold out whole deployments (a deployment's frames, including its stereo partner, never straddle the split).
    import random
    dep_frames = {}
    for (key, path), ok in zip(kept, keep_mask):
        if ok:
            dep_frames.setdefault(parse(path.name)[1], []).append(key)
    order = sorted(dep_frames)
    random.Random(args.seed).shuffle(order)
    val_deps, got, target = set(), 0, args.val_frac * sum(len(v) for v in dep_frames.values())
    for d in order:
        if got >= target:
            break
        val_deps.add(d)
        got += len(dep_frames[d])
    if args.val_frac <= 0:
        val_deps = set()
    if args.out.exists() and (args.out / 'prepare.json').exists():
        import shutil
        shutil.rmtree(args.out)            # our own earlier output of this script, rebuilt so train and val cannot mix
    dropped = [kept[i][1].name for i in np.where(~keep_mask)[0]]
    n_img = {'train': 0, 'val': 0}
    n_box = {'train': 0, 'val': 0}
    for (key, path), ok in zip(kept, keep_mask):
        if not ok:
            continue
        w, h, anns = records[key]
        rows = []
        for a in anns:
            yolo = coco_to_yolo([a["left"], a["top"], a["width"], a["height"]], w, h)
            if yolo:
                rows.append("0 " + " ".join(f"{v:.6f}" for v in yolo))
        stem = f"ozf_{key[0].replace('.', '_')}_{key[1]:06d}"
        img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_COLOR)
        scale = args.max_side / max(img.shape[:2])
        if scale < 1:
            img = cv2.resize(img, (round(img.shape[1] * scale), round(img.shape[0] * scale)), interpolation=cv2.INTER_AREA)
        split = "val" if parse(path.name)[1] in val_deps else "train"
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "images" / split / f"{stem}.jpg").write_bytes(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes())
        (args.out / "labels" / split / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
        n_img[split] += 1
        n_box[split] += len(rows)

    (args.out / "prepare.json").write_text(json.dumps({
        "name": "ozfish_train", "source": "OzFish labelled frames, doi:10.25845/5e28f062c5097, CC BY 3.0 AU",
        "grouping": "deployment (stereo _L/_R pairs merged); frames of every sealed-test deployment excluded",
        "max_side": args.max_side, "drop_below": args.drop_below, "dropped_for_closeness": dropped,
        "images": n_img, "boxes": n_box, "val_frac": args.val_frac, "seed": args.seed,
        "val_deployments": sorted(val_deps), "leak_report": report,
        "note": "train only; training on it changes what the sealed OzFish test measures"}, indent=2))
    print(f"wrote {paths.rel(args.out)}: images {n_img}, boxes {n_box}; val deployments {len(val_deps)}; dropped {len(dropped)} for closeness to a test frame")
    return 0


if __name__ == "__main__":
    sys.exit(main())
