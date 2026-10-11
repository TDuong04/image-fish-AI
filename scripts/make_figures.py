#!/usr/bin/env python
"""Draw the paper figures from committed run records (no new measurements).

    python scripts/make_figures.py            # writes docs/report/figures/*.png and *.pdf

Figures (each reads one committed run record, named in the caption in docs/report/paper.md):
  fig1_leakage      random split vs site split on the same six sites         runs/stats-20261011-leakage-control/leakage.json (F38)
  fig2_unseen       AP50 on the unseen Mediterranean source, eight arms      runs/unseen-20261011-deepecomar-labels/unseen.json (F39)
  fig3_adaptation   fine-tuning on n labelled frames of one fixed camera     runs/adaptation-20261011/adaptation.json (F41)
  fig4_device       frame rate and junction temperature over 30 minutes      runs/device-20261007/sustained_*.jsonl (F29, F34)
Error bars are the 95% cluster-bootstrap intervals stored in the records.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from fish import paths  # noqa: E402

RUNS = paths.RUNS_ROOT
OUT = paths.REPO_ROOT / "docs" / "report" / "figures"
BLUE, ORANGE, GREEN, GREY, RED = "#0072B2", "#E69F00", "#009E73", "#7f7f7f", "#D55E00"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 130, "savefig.bbox": "tight"})


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", OUT / f"{name}.png")


def err(point, ci):
    return [[point - ci[0]], [ci[1] - point]]


def fig_leakage():
    d = json.loads((RUNS / "stats-20261011-leakage-control" / "leakage.json").read_text())["AP50"]
    items = [("Random split\n(same six sites in training)", d["random_split_model_on_the_same_sites"], ORANGE),
             ("Site split\n(sites held out, 3 seeds)", d["site_split_controls_on_the_same_sites"], BLUE)]
    fig, ax = plt.subplots(figsize=(4.6, 3.4))
    for i, (lab, v, c) in enumerate(items):
        ax.bar(i, v["point"], color=c, width=0.55, yerr=err(v["point"], v["ci95"]), capsize=4)
        ax.text(i, v["point"] + 0.04 + (v["ci95"][1] - v["point"]) * (i == 1), f"{v['point']:.2f}", ha="center")
    ax.set_xticks([0, 1], [x[0] for x in items])
    ax.set_ylabel("AP50 on the six validation sites")
    ax.set_ylim(0, 1.12)
    ax.set_title("Same sites, same images: only the split differs", fontsize=10)
    save(fig, "fig1_leakage")


def fig_unseen():
    a = json.loads((RUNS / "unseen-20261011-deepecomar-labels" / "unseen.json").read_text())["arms"]
    order = [("yolov4_merge", "YOLO-Fish YOLOv4 (published)", GREY), ("yolov8s_combined", "YOLOv8s, ours (3 seeds)", BLUE), ("yolov8n_combined", "YOLOv8n, ours (3 seeds)", BLUE),
             ("yolofish_1_merge", "YOLO-Fish-1 (published)", GREY), ("yolov3_merge", "YOLOv3 (published)", GREY), ("yolov8n_plus_kakadu", "YOLOv8n, DeepFish + Kakadu", ORANGE),
             ("yolofish_2_merge", "YOLO-Fish-2 (published)", GREY), ("yolov8n_deepfish_only", "YOLOv8n, DeepFish only", ORANGE)]
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for i, (k, lab, c) in enumerate(order):
        v = a[k]["AP50"]
        ax.barh(i, v["point"], color=c, height=0.6, xerr=err(v["point"], v["ci95"]), capsize=3)
        ax.text(v["ci95"][1] + 0.01, i, f"{v['point']:.2f}", va="center")
    ax.set_yticks(range(len(order)), [o[1] for o in order])
    ax.invert_yaxis()
    ax.set_xlabel("AP50 on the unseen Mediterranean set (1,516 frames, 18 groups)")
    ax.set_xlim(0, 0.62)
    ax.set_title("Nothing trained elsewhere transfers well", fontsize=10)
    save(fig, "fig2_unseen")


def fig_adaptation():
    s = json.loads((RUNS / "adaptation-20261011" / "adaptation.json").read_text())["sets"]
    xs = {"zero-shot (3 seeds, mean)": 0, "fine-tuned on 25 frames": 25, "fine-tuned on 100 frames": 100, "fine-tuned on all early frames": 530}
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.3), sharey=True)
    for ax, (tname, title) in zip(axes, [("test_vivo", "Later frames, same camera"), ("test_other", "Other cameras")]):
        arms = s[tname]["arms"]
        pts = [(x, arms[k]["AP50"], arms[k]["AP50_ci95"]) for k, x in xs.items()]
        xpos = [0, 1, 2, 3]
        ax.errorbar(xpos, [p[1] for p in pts], yerr=[[p[1] - p[2][0] for p in pts], [p[2][1] - p[1] for p in pts]], fmt="o-", color=BLUE, capsize=3, label="ours (YOLOv8s seed 2 recipe)")
        c = arms["COCO-only control, all early frames"]
        ax.errorbar([3.12], [c["AP50"]], yerr=err(c["AP50"], c["AP50_ci95"]), fmt="s", color=RED, capsize=3, label="COCO-only control")
        ax.set_xticks(xpos, ["zero-shot", "25", "100", "all (~530)"])
        ax.set_xlabel("labelled frames used for fine-tuning")
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 0.9)
    axes[0].set_ylabel("AP50")
    axes[0].legend(frameon=False, loc="lower right", fontsize=8)
    save(fig, "fig3_adaptation")


def load_windows(name):
    rows = [json.loads(l) for l in (RUNS / "device-20261007" / name).read_text().splitlines()]
    return [r for r in rows if r.get("type") == "window"]


def fig_device():
    runs = [("sustained_yolov8s_960_20261007.jsonl", "YOLOv8s seed 0 (F29)", BLUE), ("sustained_yolov8s_s2_960_20261011.jsonl", "YOLOv8s seed 2, shipped (F34)", ORANGE)]
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(6.4, 4.4), sharex=True)
    for name, lab, c in runs:
        w = load_windows(name)
        t = [r["t_s"] / 60 for r in w]
        a1.plot(t, [r["fps"] for r in w], color=c, label=lab, lw=1.2)
        a2.plot(t, [r["tj_c"] for r in w], color=c, lw=1.2)
    a1.set_ylabel("frames per second")
    a1.set_ylim(0, 30)
    a1.legend(frameon=False, loc="lower right", fontsize=8)
    a2.set_ylabel("junction temperature (C)")
    a2.set_xlabel("minutes into the run (Orin Nano, 25 W, video loop)")
    a1.set_title("Thirty minutes of video: no throttling visible", fontsize=10)
    save(fig, "fig4_device")


def main() -> int:
    for f in (fig_leakage, fig_unseen, fig_adaptation, fig_device):
        f()
    return 0


if __name__ == "__main__":
    sys.exit(main())
