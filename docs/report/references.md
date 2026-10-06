# Reference tracker

One row per source in `references.bib`. **Verification** says what was actually checked, so a claim is never
attributed to a paper on the strength of a search snippet. Update this file in the same change as any citation.

Levels: **Crossref** = title, authors, venue, year, DOI matched against the Crossref record. **Record** = checked against the
publisher/archive page. **Read** = the passage cited was read in the paper itself. **Snippet** = the cited claim comes
from a web-search result summary only; do not quote numbers from it in the final report until the paper is read.

| Key | Source | Verification | Used in `paper.md` for | Status |
|---|---|---|---|---|
| `saleh2020deepfish` | DeepFish dataset paper, Sci. Rep. 2020 | Crossref (first 4 authors); two later authors from memory | Datasets 3.1; related work | Check full author list |
| `muksit2022yolofish` | YOLO-Fish, Ecol. Inform. 2022 | Crossref (first 4 authors); remaining authors from memory; AP of about 76% from Snippet | Related work 2.1 | Read the paper before quoting figures; check authors |
| `vijayalakshmi2025aquayolo` | AquaYOLO, Sci. Rep. 2025 | Crossref; evaluates on DeepFish and OzFish from Snippet; the publisher page redirected to a login, so not read | Related work 2.1; discussion of comparability | Snippets gave different mAP figures for what may be different configurations, so **no AquaYOLO number is used**; read the paper |
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
