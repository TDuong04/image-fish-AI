# Reference tracker

One row per source in `references.bib`. **Verification** says what was actually checked, so a claim is never
attributed to a paper on the strength of a search snippet. Update this file in the same change as any citation.

Levels: **Crossref** = title, authors, venue, year, DOI matched against the Crossref record. **Record** = checked against the
publisher/archive page. **Read** = the passage cited was read in the paper itself. **Snippet** = the cited claim comes
from a web-search result summary only; do not quote numbers from it in the final report until the paper is read.

| Key | Source | Verification | Used in `paper.md` for | Status |
|---|---|---|---|---|
| `saleh2020deepfish` | DeepFish dataset paper, Sci. Rep. 2020 | **Read** (Europe PMC full text, PMC7473859): about 40 thousand images from 20 habitats of tropical Australia (marine, cameras mounted on frames lowered to the seabed); original labels are classification only, point-level (3,200 images) and segmentation labels added; splits are not completely random but built so each split represents the variety of habitats (not site-disjoint as in ours); CC BY 4.0 article. Crossref for the first four authors; two later authors from memory | Datasets 3.1; related work | Check full author list; the detection boxes we use come from the YOLO-Fish authors' export, not from this paper |
| `muksit2022yolofish` | YOLO-Fish, Ecol. Inform. 2022 | Crossref (first 4 authors); remaining authors from memory; AP of about 76% from Snippet | Related work 2.1 | Read the paper before quoting figures; check authors |
| `vijayalakshmi2025aquayolo` | AquaYOLO, Sci. Rep. 2025 | **Read** (Europe PMC full text, PMC11842836): own pond dataset DePondFi (aquaponds, South India, 8,150 images, about 50k boxes, not public: data availability says all data are in the article); split 70/20/10 into 5,705 / 1,630 / 815 images with **no site- or video-level split described**; 25 epochs; DePondFi mAP@50 0.909, mAP@50-95 0.520, P 0.889, R 0.848; DeepFish+OzFish combined (7,763 images after combining and augmenting) P 0.885, R 0.857, mAP@50 0.814, mAP@50-95 0.473 | Related work 2.1; discussion of comparability | Quote only with the split caveat; their splits and metric details differ from ours |
| `zhang2024bssfish` | BSSFISH-YOLOv8, Animals 2024 | Record (PMC11273371): authors, journal, year, section structure and the three modifications (SPD-Conv, BiFormer attention, small-target layer) read | Related work 2.1; structure model for this report | Volume/pages missing |
| `marrable2023bruvs` | Generalised length measurement in stereo-BRUVS, Front. Mar. Sci. 2023 | Crossref only | Related work 2.2 (BRUVS context), title-level | Read before saying anything about its method or results |
| `jansen2022kakadu` | Kakadu dataset, Zenodo | Record (Zenodo API): authors, title, counts, licence | Datasets 3.2 | OK |
| `aims2019ozfish` | OzFish dataset, AIMS | Record as quoted from the AIMS catalogue in `docs/PROGRESS.md` section 8 | Datasets 3.3 | Re-check the catalogue page for the exact citation text |
| `redmon2016yolo` | YOLO, CVPR 2016 | Crossref | Method 4.1 | OK |
| `lin2014coco` | COCO, ECCV 2014 | Crossref (first 4 authors); rest from memory | Method 4.2 (pretraining) | Check full author list |
| `everingham2010pascal` | PASCAL VOC challenge, IJCV 2010 | Crossref (first 4 authors); last author from memory | Method 4.3 (AP, IoU 0.5 criterion) | Check |
| `neubeck2006nms` | Efficient NMS, ICPR 2006 | Crossref | Method 4.1 | OK |
| `akyon2022sahi` | SAHI, ICIP 2022 | Crossref | Method 4.4, results 5.6 (tiling) | Read to confirm how it merges tiles; our merge rule is our own |
| `kapoor2023leakage` | Leakage and reproducibility, Patterns 2023 | Crossref | Method 4.3 (leakage), discussion | OK |
| `jocher2023ultralytics` | Ultralytics YOLOv8 software | Not checked against the repository (version and author list as conventionally cited) | Method 4.1, licence | Check the repository's current citation text |
| `catalan2023deepecomar` | DEEPECOMAR Mediterranean fish dataset, Zenodo | Record (Zenodo API: authors, title, counts, licence) and the data itself (opened, converted, boxes drawn) | Datasets (unseen-source test); eval_protocol item 10 | OK |
| `beery2018terra` | Recognition in Terra Incognita, ECCV 2018 | Crossref | Related work 2.4 (generalising to new camera locations) | Read before stating its results |
| `ditria2020abundance` | Fish abundance with object detection, Front. Mar. Sci. 2020 | Crossref (first five authors) | Related work 2.1 | Read |
| `salman2020hybrid` | Fish detection in underwater video, ICES J. Mar. Sci. 2020 | Crossref (first five authors) | Related work 2.1 | Read |
| `villon2016coral` | Coral reef fish detection, ACIVS 2016 | Crossref (first five authors) | Related work 2.1 | Read |
| `liu2020generic` | Deep learning for generic object detection: a survey, IJCV 2020 | Crossref (first five authors) | Background 2.1 | Read before citing specifics |
| `ren2017fasterrcnn` | Faster R-CNN, TPAMI 2017 | Crossref | Method (second detector family) | OK |
| `ovalle2022onboard` | Deep learning for fish on board fishing vessels, Marine Policy 2022 | Crossref | Related work 2.1 | Read |
| `sun2022enhance` | Image enhancement for coral reef fish detection, Ecol. Inform. 2022 | Crossref | Related work 2.1 | Read |

## Sources considered and not used

- Several other fish-detection papers appeared in search results (EPANet, DeepFishNET+, an ETRI Journal ensemble benchmark,
  a Stanford CS231n report). They were **not read and are not cited**. Add them only after reading.
- "Accounting for Variance in Machine Learning Benchmarks" (on seed variance) was searched for by title and not found in
  Crossref, so it is not cited; our seed-variance statements rest on our own measurements.

## Rules

- Never cite from a snippet. Open the source, read the passage, then move the row to **Read**.
- Do not compare our numbers to a cited paper's without saying in the same sentence that splits, image sizes and metric
  definitions may differ (project rule, `CLAUDE.md`).
- A reference with an unchecked author list stays flagged here until checked.
