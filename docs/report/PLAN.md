# Writing plan: thesis and publication

Decided 2026-10-11: write a **thesis** (full scope) and a **publication** (shorter, from the same evidence). Both draw on `docs/PROGRESS.md`
(findings F1 to F34), `docs/eval_protocol.md` and the draft in `paper.md`; no number is written anywhere without its finding id.

## 1. Thesis

Source of the text: the existing draft (`paper.md`) expanded, plus `docs/GUIDE.md` for the background chapter.

| Chapter | Content | Evidence | State |
|---|---|---|---|
| 1 Introduction | motivation, three research questions (what drives accuracy; are the scores trustworthy; what does it cost on a small device), contributions | | drafted in short form |
| 2 Background and related work | object detection and YOLO; evaluation metrics; underwater fish detection; BRUVS; small-object detection; edge inference (TensorRT, Jetson); leakage in ML evaluation | `GUIDE.md`, `references.md` | background drafted; related work **needs the papers read** |
| 3 Data | DeepFish, Kakadu, OzFish; grouping, splits, leak checks, the sealed set | F13, F17, F18, F20, F21, `data_audit.md` | drafted |
| 4 Methodology | models, training, protocol, seeds and controls, error analysis, device methods | `eval_protocol.md` | drafted |
| 5 Results | 5.1 baseline and resolution (F1 to F4); 5.2 where it fails (F10, F11, F19, F20); 5.3 what did not help (F12, F14 to F16, tiling F25 and F26, 1280 px F27); 5.4 data (F17 to F24, F32); 5.5 sealed test (F33); 5.6 on the Orin (F5 to F9, F28 to F31, F34) | | drafted; F34 pending |
| 6 Discussion | what limits accuracy; why scores are below published ones; deployment trade-offs | | drafted |
| 7 Limitations and threats to validity | | | drafted |
| 8 Conclusion and future work | | | **pending** (after F34) |
| Appendices | reproduction commands and environments; the evaluation protocol; the corrections table (PROGRESS section 6); licences and attribution; glossary | | to assemble |

Needs from the owner: institution template, length limit, deadline, supervisor.

## 2. Publication

Target: a **workshop or short paper (6 to 8 pages)**, then a journal extension if wanted. Working title: *Frame leakage and data diversity in underwater fish detection:
a leakage-controlled study with measured edge deployment*.

Claims, each with evidence in hand:
1. **The split protocol changes the score by more than any model change.** 0.98 mAP@50 on a random per-image split against about 0.47 by capture site (F20), same pipeline.
2. **Data diversity beats resolution, architecture and augmentation.** DeepFish-only 0.473, plus Kakadu 0.698, plus OzFish 0.750 (three seeds each, F18, F24); augmentation, stride-4 head, 1280 px and test-time tiling add nothing (F12, F23, F25 to F27).
3. **YOLOv8s on the combined data is the best model** (0.812 over three seeds, F32; sealed OzFish test 0.700, run once, F33) and runs at about 24 fps on video on a Jetson Orin Nano (F29, F34).

What stays out of the short paper: the long negative-results tables, the guide and most of the appendices (they go in the thesis).

Extras for a **journal version** (not needed for the thesis or the workshop paper):
- an unseen-source test (a dataset no training or selection data came from),
- an external comparison under our split (for example the YOLO-Fish weights, or another detector family; the licence decision may require one anyway),
- confidence intervals that resample whole clips or sites, since frames are not independent,
- a field test on live footage.

## 3. Work list, in order

| # | Task | Who | Depends on |
|---|---|---|---|
| 1 | Template, length, deadline, co-authors, target venue | owner | |
| 2 | Licence decision (AGPL), data availability and attribution statement, whether to publish weights | owner | |
| 3 | Read YOLO-Fish, AquaYOLO, the BRUVS paper; fix author lists; widen the references to 30 to 50 | assistant | |
| 4 | Clip-level bootstrap confidence intervals on the main comparisons (new script, existing results) | assistant | |
| 5 | Figures: pipeline diagram, dataset samples, score by data recipe with seeds, recall by size, example detections and failures, device timing | assistant | |
| 6 | Abstract and conclusion | assistant | F34 |
| 7 | Convert to the venue and thesis templates; proofread | both | 1 |
| 8 | Replace the stale `report_sections.md`; refresh the headline in `CLAUDE.md` | assistant | |

Tickets: FD-147 (thesis), FD-148 (publication) in `docs/tickets/BACKLOG.md`.
