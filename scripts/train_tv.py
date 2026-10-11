#!/usr/bin/env python
"""Train a detector from a second family (torchvision Faster R-CNN, MobileNetV3-Large FPN) on the same fish data.

    python scripts/train_tv.py --config tv_frcnn_deepfish_only --seed 0
    python scripts/train_tv.py --config tv_frcnn_combined --seed 0
    python scripts/train_tv.py --config tv_frcnn_combined --limit-images 64 --epochs 1    # smoke test

Why this exists: every result so far is a YOLOv8 result, and the Ultralytics code and weights are AGPL-3.0. A reviewer will ask whether the
main finding (more diverse data beats everything else) is specific to YOLO, and the licence question may need a permissively licensed
detector anyway. torchvision's Faster R-CNN is BSD-licensed code with COCO-pretrained weights, a different architecture (two stage, FPN,
anchors) and the same labels, so running the DeepFish-only and the combined recipe through it tests the data claim on a second family.

Same data, same validation split (DeepFish val), same AP routine as scripts/bootstrap_ci.py. The checkpoint is the epoch with the best
0.1 * AP50 + 0.9 * AP50-95 on DeepFish validation, as for the YOLO runs. One seed per recipe is a screen, not a result.
Writes runs/<date>-<name>-s<seed>/ with config.yaml, results.csv, metrics.json, env.txt and weights/best.pt (a state dict).
Guarded with `if __name__ == "__main__"` because DataLoader workers spawn on Windows and macOS.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import yaml  # noqa: E402

from fish import env as fenv  # noqa: E402
from fish import paths  # noqa: E402
from fish.device import resolve_device  # noqa: E402

IMG_EXT = {".jpg", ".jpeg", ".png"}


def read_yolo_labels(path: Path, w: int, h: int) -> np.ndarray:
    rows = []
    if path.exists():
        for line in path.read_text().splitlines():
            p = line.split()
            if len(p) >= 5:
                cx, cy, bw, bh = (float(x) for x in p[1:5])
                rows.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
    return np.array(rows, dtype=np.float32).reshape(-1, 4)


def list_images(dirs: list[Path]) -> list[Path]:
    out = []
    for d in dirs:
        out += sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXT)
    return out


def label_path(img: Path) -> Path:
    parts = list(img.parts)
    i = len(parts) - 1 - parts[::-1].index("images")        # the last "images" folder in the path
    parts[i] = "labels"
    return Path(*parts).with_suffix(".txt")


class YoloDetDataset:
    def __init__(self, images: list[Path], train: bool):
        self.images, self.train = images, train

    def __len__(self):
        return len(self.images)

    def __getitem__(self, i):
        import cv2
        import torch
        p = self.images[i]
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        boxes = read_yolo_labels(label_path(p), w, h)
        if self.train and random.random() < 0.5:
            img = img[:, ::-1].copy()
            boxes = boxes.copy()
            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]
        t = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
        target = {"boxes": torch.from_numpy(boxes), "labels": torch.ones((len(boxes),), dtype=torch.int64)}
        return t, target


def collate(batch):
    return tuple(zip(*batch))


def build_model(imgsz: int, pretrained: bool = True):
    from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn
    from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
    kw = dict(min_size=int(round(imgsz * 9 / 16 / 32) * 32), max_size=imgsz, box_score_thresh=0.005, box_detections_per_img=300)
    m = fasterrcnn_mobilenet_v3_large_fpn(weights="DEFAULT" if pretrained else None, **kw)
    m.roi_heads.box_predictor = FastRCNNPredictor(m.roi_heads.box_predictor.cls_score.in_features, 2)
    return m


def predict_images(model, images: list[Path], device, batch: int = 8):
    """Return per-image (boxes xyxy, scores) in original pixels."""
    import cv2
    import torch
    model.eval()
    boxes, confs = [], []
    with torch.no_grad():
        for i in range(0, len(images), batch):
            ims = []
            for p in images[i:i + batch]:
                im = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
                ims.append(torch.from_numpy(im).permute(2, 0, 1).float().div(255.0).to(device.torch))
            for o in model(ims):
                boxes.append(o["boxes"].cpu().numpy().astype(float).reshape(-1, 4))
                confs.append(o["scores"].cpu().numpy().astype(float))
    return boxes, confs


def score_val(model, images, gts, device):
    import bootstrap_ci as bc
    boxes, confs = predict_images(model, images, device)
    sc = bc.Scorer(gts, boxes, confs)
    n_gt = np.array([len(g) for g in gts], dtype=float)
    w = np.ones(len(images))
    return sc.ap(w, n_gt, [0]), sc.ap(w, n_gt, list(range(len(bc.IOUS))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--workers", type=int, default=None, help="override the config (0 is the safe choice on Windows)")
    ap.add_argument("--limit-images", type=int, default=0, help="smoke test: use only the first N train and val images")
    args = ap.parse_args()

    cfg_path = paths.CONFIG_DIR / "train" / f"{args.config}.yaml"
    if not cfg_path.exists():
        raise SystemExit(f"No training config {args.config!r} at {paths.rel(cfg_path)}")
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    epochs = args.epochs or cfg.get("epochs", 12)
    data_cfg = paths.CONFIG_DIR / "data" / f"{cfg['data']}.yaml"
    spec = yaml.safe_load(data_cfg.read_text(encoding="utf-8"))
    root = (paths.DATA_ROOT / spec["path"]).resolve()
    train_dirs = [root / d for d in (spec["train"] if isinstance(spec["train"], list) else [spec["train"]])]
    val_dirs = [root / d for d in (spec["val"] if isinstance(spec["val"], list) else [spec["val"]])]
    for d in train_dirs + val_dirs:
        if not d.exists():
            raise SystemExit(f"{d} does not exist. Run: python scripts/get_data.py")

    import cv2
    import torch
    from torch.utils.data import DataLoader

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = resolve_device(cfg.get("device") or fenv.load_profile().get("device", "auto"))
    train_imgs, val_imgs = list_images(train_dirs), list_images(val_dirs)
    if args.limit_images:
        train_imgs, val_imgs = train_imgs[: args.limit_images], val_imgs[: args.limit_images]
    gts = []
    for p in val_imgs:
        h, w = cv2.imread(str(p)).shape[:2]
        gts.append(read_yolo_labels(label_path(p), w, h))
    print(f"{cfg['name']} seed {args.seed}: {len(train_imgs)} train / {len(val_imgs)} val images, {epochs} epochs, device {device}")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    run_dir = paths.RUNS_ROOT / f"{stamp}-{cfg['name']}-s{args.seed}"
    (run_dir / "weights").mkdir(parents=True, exist_ok=True)
    (run_dir / "config.yaml").write_text(yaml.safe_dump({**cfg, "seed": args.seed, "epochs": epochs, "train_images": len(train_imgs),
                                                          "val_images": len(val_imgs)}, sort_keys=False), encoding="utf-8")
    fenv.write_env_file(run_dir)

    batch = int(cfg.get("batch", 8))
    workers = args.workers if args.workers is not None else int(cfg.get("workers", 0))
    loader = DataLoader(YoloDetDataset(train_imgs, True), batch_size=batch, shuffle=True, num_workers=workers, collate_fn=collate,
                        drop_last=True, persistent_workers=workers > 0)
    model = build_model(int(cfg.get("imgsz", 960))).to(device.torch)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=float(cfg.get("lr", 0.01)), momentum=0.9, weight_decay=float(cfg.get("weight_decay", 4e-5)))
    total = epochs * len(loader)
    warm = min(500, total // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: (s + 1) / warm if s < warm else 0.5 * (1 + np.cos(np.pi * (s - warm) / max(1, total - warm))))
    scaler = torch.amp.GradScaler("cuda", enabled=device.torch.type == "cuda")

    best, rows = -1.0, []
    for ep in range(epochs):
        model.train()
        t0, run = time.time(), 0.0
        for k, (ims, tgts) in enumerate(loader):
            ims = [i.to(device.torch) for i in ims]
            tgts = [{a: b.to(device.torch) for a, b in t.items()} for t in tgts]
            with torch.autocast(device_type=device.torch.type, enabled=device.torch.type == "cuda"):
                loss = sum(model(ims, tgts).values())
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            run += float(loss)
            if k % 200 == 0:
                print(f"  epoch {ep + 1}/{epochs} iter {k}/{len(loader)} loss {float(loss):.3f}", flush=True)
        ap50, ap5095 = score_val(model, val_imgs, gts, device)
        fit = 0.1 * ap50 + 0.9 * ap5095
        rows.append({"epoch": ep + 1, "train_loss": round(run / max(1, len(loader)), 4), "val_AP50": round(ap50, 4),
                     "val_AP50_95": round(ap5095, 4), "fitness": round(fit, 4), "seconds": round(time.time() - t0, 1)})
        print(f"  epoch {ep + 1}: AP50 {ap50:.4f} AP50-95 {ap5095:.4f} ({rows[-1]['seconds']} s)", flush=True)
        torch.save(model.state_dict(), run_dir / "weights" / "last.pt")
        if fit > best:
            best = fit
            torch.save(model.state_dict(), run_dir / "weights" / "best.pt")
            best_row = rows[-1]
        with (run_dir / "results.csv").open("w", newline="") as f:
            wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            wr.writeheader()
            wr.writerows(rows)

    (run_dir / "metrics.json").write_text(json.dumps({"ap50": best_row["val_AP50"], "ap50_95": best_row["val_AP50_95"], "best_epoch": best_row["epoch"],
                                                      "epochs": epochs, "seed": args.seed, "batch": batch,
                                                      "note": "DeepFish val, 101-point AP (scripts/bootstrap_ci.py), best epoch by 0.1*AP50+0.9*AP50-95"}, indent=2))
    print(f"written: {paths.rel(run_dir)}  best epoch {best_row['epoch']}  AP50 {best_row['val_AP50']}  AP50-95 {best_row['val_AP50_95']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
