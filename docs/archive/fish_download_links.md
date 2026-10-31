# Fish Detection & Counting — Download / Clone Links

All repos, datasets, papers, and weights for the five resources.

---

## 1. Caltech Fish Counting (CFC)

**Code (clone this first — it has the download scripts):**
- `git clone https://github.com/visipedia/caltech-fish-counting.git`

**Dataset (sonar, ~132.8 GB full / ~1.4 GB tiny subset):**
- Record page: https://data.caltech.edu/records/1y23m-j8r69
- Download instructions live in the repo README (uses their `download.py` / listed URLs)

**CFC-DAOD extension (domain adaptation, 168k boxes / 29k frames):**
- Folder + links: https://github.com/visipedia/caltech-fish-counting/tree/main/CFC-DAOD

**Paper (ECCV 2022):**
- https://arxiv.org/abs/2207.09295  (PDF: https://arxiv.org/pdf/2207.09295)

---

## 2. Justin Kay — ALDI / Align & Distill + papers

**ALDI code (Detectron2-based DAOD framework):**
- `git clone https://github.com/justinkay/aldi.git`
- Models/weights: https://github.com/justinkay/aldi/blob/main/docs/MODELS.md
- Install: https://github.com/justinkay/aldi/blob/main/docs/INSTALL.md

**Papers:**
- Align & Distill (TMLR 2025): https://arxiv.org/abs/2403.12029  (PDF: https://arxiv.org/pdf/2403.12029v1)
- UDA in the Real World — Sonar (NeurIPS 2023 workshop): https://openreview.net/forum?id=Y08yLPFm1z  (PDF: https://openreview.net/pdf?id=Y08yLPFm1z)
- Project page: https://aldi-daod.github.io/

---

## 3. DeepFish

**Code:**
- `git clone https://github.com/alzayats/DeepFish.git`

**Dataset (~40k optical full-HD images; ~3 GB):**
- Project page (download button): https://alzayats.github.io/DeepFish/
- Direct (Google Drive, hosted by authors): https://drive.google.com/open?id=1iqzaJgT9Vq-Q92Acrz3Fbsl7d_agFkj1

**Paper (Scientific Reports 2020):**
- https://www.nature.com/articles/s41598-020-71639-x  (arXiv: https://arxiv.org/abs/2008.12603)

---

## 4. OzFish

**Code / metadata:**
- `git clone https://github.com/open-AIMS/ozfish.git`

**Dataset (~80k crops, ~45k boxes; hosted on AIMS/Pawsey, not in the git repo):**
- Download portal: https://apply.pawsey.org.au/p/ax3003/
- Record / DOI: https://doi.org/10.25845/5e28f062c5097
- Mirror (YOLO-format, Hugging Face): https://huggingface.co/datasets/Devi-Ayyagari/yolov7_OzFish

---

## 5. YOLO-Fish

**Code + cfg + pretrained weights:**
- `git clone https://github.com/tamim662/YOLO-Fish.git`
- Weights + processed datasets are linked from the repo README (Google Drive)

**Paper (Ecological Informatics 2022):**
- https://www.sciencedirect.com/science/article/abs/pii/S1574954122002977

---

## One-shot clone (all 5 repos)

```bash
mkdir -p fish-research && cd fish-research
git clone https://github.com/visipedia/caltech-fish-counting.git
git clone https://github.com/justinkay/aldi.git
git clone https://github.com/alzayats/DeepFish.git
git clone https://github.com/open-AIMS/ozfish.git
git clone https://github.com/tamim662/YOLO-Fish.git
```

> Note: the **repos** clone freely. The **large datasets** (CFC sonar, DeepFish images, OzFish crops) and the **weights** are hosted off-GitHub (Caltech DataCite, Google Drive, Pawsey/AIMS) and usually need a browser, `gdown`, or a portal request — a plain `git clone` won't pull them.
