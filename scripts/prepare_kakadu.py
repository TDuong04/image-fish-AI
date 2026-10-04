#!/usr/bin/env python
"""FD-135: turn the Kakadu archive into a single-class YOLO dataset with a leakage-free split.

    python scripts/get_data.py                      # not needed for this one
    python scripts/fetch_url.py <zenodo url> data/raw/kakadu_training_data.zip --md5 ...
    python scripts/prepare_kakadu.py                # writes data/processed/kakadu/

The archive (Jansen et al. 2022, CC BY 4.0, doi 10.5281/zenodo.7250921) holds 44,112 frames of
remote underwater video and a COCO file with 23 species. This script:

  1. merges the 23 species into one class, `fish`;
  2. groups frames into scenes. File names are bare numbers and the COCO file carries no video or
     site field, and consecutive ids are NOT neighbouring frames, so nothing says which frames
     came from the same static camera. Frames are grouped by picture similarity instead (L1
     distance between small mean-removed grayscale thumbnails, single linkage). A threshold of 4.0
     gives about 2,300 groups, the largest 4.4% of frames; at 6.0 scenes start merging into one
     giant cluster, so 4.0 is deliberately below that;
  3. keeps at most --max-per-group frames per group (default 15): frames of one static scene are
     near-duplicates, so thousands of them cost training time and add little;
  4. by default writes TRAIN ONLY. A held-out split inside Kakadu is possible (--val-frac) but it
     leaks: at threshold 4.0 the closest train/val pairs were checked by eye and are the same scene
     (same plants and branches, small camera or lighting change that broke the similarity chain),
     and looser thresholds merge unrelated scenes before they join those fragments. A distance
     check is necessary but not sufficient. So do not make decisions on a Kakadu val split; judge
     models on the DeepFish val split and the sealed OzFish set, which share no scenes with it.

The grouping is saved to splits/kakadu_groups.json so every machine gets the same split without
recomputing it. Every Kakadu image contains a fish, so this dataset has no empty frames and cannot
measure false alarms.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402

from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402
from fish.zip64 import Zip64Reader  # noqa: E402

ANN_NAME = "KakaduFishAI_boundingbox.json"
THUMB_W, THUMB_H = 48, 27
ALGORITHM = "L1 of mean-removed 48x27 grayscale thumbnails (cv2 IMREAD_REDUCED_GRAYSCALE_8 then INTER_AREA); single linkage"
GROUPS_FILE = paths.REPO_ROOT / "splits" / "kakadu_groups.json"


def coco_to_yolo(bbox, width: int, height: int):
    """COCO [x, y, w, h] in pixels -> normalised (cx, cy, w, h), clipped; None if degenerate."""
    x, y, w, h = (float(v) for v in bbox)
    x0, y0 = max(x, 0.0), max(y, 0.0)
    x1, y1 = min(x + w, float(width)), min(y + h, float(height))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)


def group_frames(thumbs: np.ndarray, threshold: float, torch_device, block: int = 1000) -> np.ndarray:
    """Group row indices whose thumbnails are within `threshold` (mean abs diff) of each other.

    Single linkage via union-find over every pair under the threshold. Returns group ids 0..G-1.
    """
    import torch

    n = len(thumbs)
    t = torch.from_numpy(thumbs).to(torch_device)
    parent = np.arange(n)

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    for s in range(0, n, block):
        d = torch.cdist(t[s:s + block], t, p=1) / thumbs.shape[1]
        ii, jj = torch.nonzero(d < threshold, as_tuple=True)
        for a, b in zip((ii + s).cpu().tolist(), jj.cpu().tolist()):
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[ra] = rb
    roots = np.array([find(i) for i in range(n)])
    _, group = np.unique(roots, return_inverse=True)
    return group


def split_groups(group: np.ndarray, val_frac: float, seed: int) -> np.ndarray:
    """Boolean mask, True = validation. Whole groups go to one side, picked in seeded random order."""
    rng = np.random.default_rng(seed)
    ids = np.unique(group)
    rng.shuffle(ids)
    sizes = {g: int((group == g).sum()) for g in ids}
    target, got, val = val_frac * len(group), 0, set()
    for g in ids:
        if got >= target:
            break
        val.add(int(g))
        got += sizes[g]
    return np.array([int(g) in val for g in group])


def min_cross_distance(thumbs: np.ndarray, is_val: np.ndarray, torch_device, block: int = 1000):
    """Smallest and 1st-percentile nearest-neighbour distance from each val frame to the train set."""
    import torch

    train = torch.from_numpy(thumbs[~is_val]).to(torch_device)
    val = thumbs[is_val]
    nn = []
    for s in range(0, len(val), block):
        d = torch.cdist(torch.from_numpy(val[s:s + block]).to(torch_device), train, p=1) / thumbs.shape[1]
        nn.append(d.min(dim=1).values.cpu().numpy())
    nn = np.concatenate(nn)
    return float(nn.min()), float(np.percentile(nn, 1)), nn


def thumbnails(reader: Zip64Reader, images: list[dict], cache: Path) -> np.ndarray:
    import cv2

    if cache.exists():
        arr = np.load(cache)
        if arr.shape[0] == len(images):
            return arr
    out = np.empty((len(images), THUMB_W * THUMB_H), np.float32)
    for k, im in enumerate(images):
        raw = np.frombuffer(reader.read(im["file_name"]), np.uint8)
        g = cv2.imdecode(raw, cv2.IMREAD_REDUCED_GRAYSCALE_8)
        if g is None:
            raise SystemExit(f"cannot decode {im['file_name']} inside the archive")
        t = cv2.resize(g, (THUMB_W, THUMB_H), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
        out[k] = t - t.mean()
        if k % 10000 == 0:
            print(f"  thumbnails {k}/{len(images)}", flush=True)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--zip", type=Path, default=paths.DATA_ROOT / "raw" / "kakadu_training_data.zip")
    ap.add_argument("--out", type=Path, default=paths.DATA_ROOT / "processed" / "kakadu")
    ap.add_argument("--threshold", type=float, default=4.0, help="grouping distance (see the docstring)")
    ap.add_argument("--val-frac", type=float, default=0.0,
                    help="0 (default) writes train only; >0 holds out groups but see the docstring: it leaks")
    ap.add_argument("--max-side", type=int, default=0,
                    help="resize so the longer side is at most N px (JPEG q92); 0 keeps the original bytes. "
                         "960 matches training at imgsz 960 and makes the set about 4x smaller to move")
    ap.add_argument("--max-per-group", type=int, default=15, help="cap on frames kept per scene group; 0 keeps all")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--regroup", action="store_true", help="recompute splits/kakadu_groups.json")
    ap.add_argument("--groups-only", action="store_true", help="stop after grouping and the leakage check")
    ap.add_argument("--limit", type=int, default=0, help="only the first N images (smoke test)")
    args = ap.parse_args()

    if not args.zip.exists():
        sys.exit(f"{args.zip} does not exist.\nFetch it first:\n  python scripts/fetch_url.py "
                 '"https://zenodo.org/records/7250921/files/202210-KakaduFishAI-TrainingData.zip?download=1" '
                 f"{paths.rel(args.zip)} --md5 c282e3f00844a7a4eeab5a02fa005cfa")
    need = args.zip.stat().st_size * 1.3
    free = shutil.disk_usage(args.zip.parent).free
    if not args.groups_only and free < need:
        sys.exit(f"not enough free disk to write the dataset: need about {need / 1e9:.1f} GB, have {free / 1e9:.1f} GB.")

    info = resolve_device("auto")
    torch_device = "cuda" if info.kind == "cuda" else "cpu"
    print(f"device for grouping: {torch_device}")

    with Zip64Reader(args.zip) as reader:
        coco = json.loads(reader.read(ANN_NAME))
        images = sorted(coco["images"], key=lambda i: i["id"])
        if args.limit:
            images = images[: args.limit]
        keep_ids = {i["id"] for i in images}
        boxes: dict[int, list] = {}
        for a in coco["annotations"]:
            if a["image_id"] in keep_ids:
                boxes.setdefault(a["image_id"], []).append(a["bbox"])
        print(f"{len(images)} images, {sum(len(v) for v in boxes.values())} boxes, "
              f"{len(coco['categories'])} species merged into one class")

        thumbs = thumbnails(reader, images, args.out.parent / ".cache" / f"kakadu_thumbs_{len(images)}.npy")

        ids = [i["id"] for i in images]
        use_saved = GROUPS_FILE.exists() and not args.regroup and not args.limit
        if use_saved:
            saved = json.loads(GROUPS_FILE.read_text())
            if saved["image_ids"] != ids or saved["threshold"] != args.threshold:
                sys.exit(f"{paths.rel(GROUPS_FILE)} does not match this archive or threshold. "
                         "Rerun with --regroup if that is intended.")
            group = np.array(saved["group"])
            print(f"grouping: loaded {len(set(group.tolist()))} groups from {paths.rel(GROUPS_FILE)}")
        else:
            print(f"grouping at threshold {args.threshold} ...", flush=True)
            group = group_frames(thumbs, args.threshold, torch_device)
            if not args.limit:
                GROUPS_FILE.parent.mkdir(parents=True, exist_ok=True)
                GROUPS_FILE.write_text(json.dumps({
                    "algorithm": ALGORITHM, "threshold": args.threshold, "source": "doi:10.5281/zenodo.7250921",
                    "image_ids": ids, "group": group.tolist()}))
                print(f"  wrote {paths.rel(GROUPS_FILE)}")
        sizes = np.bincount(group)
        print(f"groups: {len(sizes)}, largest {sizes.max()} frames ({100 * sizes.max() / len(group):.1f}%), "
              f"singletons {(sizes == 1).sum()}")
        if sizes.max() > 0.15 * len(group):
            sys.exit("one group holds over 15% of the frames: scenes have merged and a split would be "
                     "meaningless. Lower --threshold.")

        rng = np.random.default_rng(args.seed)
        keep = np.ones(len(group), bool)
        if args.max_per_group > 0:
            keep[:] = False
            for g in range(len(sizes)):
                members = np.where(group == g)[0]
                if len(members) > args.max_per_group:
                    members = rng.choice(members, args.max_per_group, replace=False)
                keep[members] = True
        print(f"keeping {int(keep.sum())} of {len(group)} frames (max {args.max_per_group} per group)")

        is_val = split_groups(group, args.val_frac, args.seed) if args.val_frac > 0 else np.zeros(len(group), bool)
        lo = None
        if args.val_frac > 0:
            if set(group[is_val].tolist()) & set(group[~is_val].tolist()):
                sys.exit("BUG: a group appears on both sides of the split.")
            lo, p1, _ = min_cross_distance(thumbs, is_val, torch_device)
            print(f"split: {int((~is_val).sum())} train / {int(is_val.sum())} val frames; closest train-val pair "
                  f"distance {lo:.2f} (grouping threshold {args.threshold}), 1st percentile {p1:.2f}")
            print("WARNING: this split leaks same-scene frames (checked by eye at threshold 4.0); "
                  "do not use it to compare models.")
            if lo < args.threshold:
                sys.exit(f"LEAKAGE: a val frame is within {lo:.2f} of a train frame, closer than the grouping "
                         f"threshold {args.threshold}. Refusing to write the dataset.")
        if args.groups_only:
            return 0

        n_boxes = {"train": 0, "val": 0}
        n_img = {"train": 0, "val": 0}
        for im, val, kept in zip(images, is_val, keep):
            if not kept:
                continue
            split = "val" if val else "train"
            ext = ".jpg" if args.max_side else Path(im["file_name"]).suffix.lower()
            stem = f"kak_{im['id']:05d}"
            img_dir, lab_dir = args.out / "images" / split, args.out / "labels" / split
            img_dir.mkdir(parents=True, exist_ok=True)
            lab_dir.mkdir(parents=True, exist_ok=True)
            raw_bytes = reader.read(im["file_name"])
            if args.max_side:
                import cv2
                frame = cv2.imdecode(np.frombuffer(raw_bytes, np.uint8), cv2.IMREAD_COLOR)
                scale = args.max_side / max(frame.shape[:2])
                if scale < 1:
                    frame = cv2.resize(frame, (round(frame.shape[1] * scale), round(frame.shape[0] * scale)),
                                       interpolation=cv2.INTER_AREA)
                raw_bytes = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 92])[1].tobytes()
            (img_dir / f"{stem}{ext}").write_bytes(raw_bytes)
            rows = []
            for bb in boxes.get(im["id"], []):
                yolo = coco_to_yolo(bb, im["width"], im["height"])
                if yolo:
                    rows.append("0 " + " ".join(f"{v:.6f}" for v in yolo))
            (lab_dir / f"{stem}.txt").write_text("\n".join(rows) + ("\n" if rows else ""))
            n_img[split] += 1
            n_boxes[split] += len(rows)

    (args.out / "prepare.json").write_text(json.dumps({
        "name": "kakadu", "source": "Kakadu Fish AI dataset, doi:10.5281/zenodo.7250921, CC BY 4.0",
        "grouping": "picture-similarity scene groups (no video or site field exists)", "threshold": args.threshold,
        "seed": args.seed, "val_frac": args.val_frac, "max_per_group": args.max_per_group, "max_side": args.max_side,
        "groups": int(len(sizes)),
        "images": n_img, "boxes": n_boxes, "closest_train_val_distance": lo,
        "split_warning": "a held-out split inside Kakadu leaks same-scene frames; train only by default",
        "note": "every frame contains a fish: no empty frames, so false alarms cannot be measured here",
    }, indent=2))
    print(f"wrote {paths.rel(args.out)}: images {n_img}, boxes {n_boxes}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
