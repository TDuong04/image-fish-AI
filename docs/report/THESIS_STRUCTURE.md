---
title: Thesis Structure — Fish Detection on a Jetson Orin Nano
created: 2026-10-11
tags: [thesis, fish-detection, structure, decision]
status: draft
---

# Thesis structure

> [!decision]
> Our own thesis is **one document with one argument**: a fish detector's score depends more on how the data are split and which data are used than on the model, and a
> small device can run the chosen model. The structure below puts the evidence in the order a reader needs it: data and protocol first, then what failed, then what worked,
> then the sealed test and the device. The FishIoT team-report note was used only as a layout template (a summary, the decision, a section tree), not as our content.
> Decided 2026-10-11.

## Summary

The project turned a first YOLOv8n baseline (0.50 mAP@50) into a three-seed YOLOv8s result (0.812 mAP@50, sealed test 0.700) and measured it on a Jetson Orin Nano (about 24 fps on video).
The most reusable findings are methodological (a random split inflates the score from about 0.47 to 0.98; frames are not independent; every comparison needs a same-machine control and several seeds)
and empirical (more diverse data helped far more than resolution, architecture, augmentation or tiling). Every number in the thesis carries a finding id (F#) that points to its run folder and its limits
in `docs/PROGRESS.md`.

## The layout

```text
Front matter    Title page, abstract (written last), acknowledgements, contents, lists of figures and tables

## 1. Introduction
   1.1  Motivation and problem                       (BRUVS footage, why a small device)
   1.2  Research questions                           RQ1 what drives accuracy; RQ2 can the scores be trusted;
                                                     RQ3 what does it cost on the device
   1.3  Contributions
   1.4  Scope and exclusions                         sonar, species classification, training on the board, live camera
   1.5  Outline of the thesis

## 2. Background and Related Work
   2.1  Object detection and the YOLO family         (from docs/GUIDE.md sections 2 and 3)
   2.2  Training, transfer learning and metrics      IoU, precision, recall, AP, mAP@50 and mAP@50-95 (GUIDE 4 and 5)
   2.3  Underwater fish detection and BRUVS          needs the papers read (references.md)
   2.4  Data splits, near-duplicate frames, leakage  (Kapoor and Narayanan, 2023)
   2.5  Small objects and tiling                     (Akyon et al., 2022)
   2.6  Edge inference: TensorRT and the Jetson      (GUIDE 10)

## 3. Data
   3.1  DeepFish                                     training and validation, split by site
   3.2  Kakadu                                       grouping by picture similarity; why it is train-only (F13)
   3.3  OzFish                                       acquisition, overlap with the sealed set, held-out deployments, the sealed set (F21, data_audit.md)
   3.4  Data not used                                CFC (sonar)
   3.5  The split protocol and the evidence for it   random split 0.98 against 0.47 by site (F20)
   3.6  Licences and attribution

## 4. Methodology
   4.1  Detector and training recipe                 YOLOv8 n, s, m; YOLO11s; COCO start; early stopping
   4.2  Evaluation protocol                          eval_protocol.md: site splits, sealed set, thresholds chosen on validation
   4.3  Experimental design                          same-machine controls, seed spread, three seeds per claim, rules fixed in advance
   4.4  Error analysis                               found and missed, false alarms by cause, recall by size
   4.5  Test-time tiling                             method and the four pre-set merge rules
   4.6  Device methodology                           engines built on the board, end to end, power mode stated, thermal runs
   4.7  Reproducibility infrastructure               paths, machine profiles, queue runner, the documentation rule

## 5. Results
   5.1  Baseline and resolution                      F1 to F4
   5.2  Cross-dataset behaviour of the baseline      F2
   5.3  Where the detector fails                     F10, F11, F19
   5.4  Why the scores are below published figures   F20
   5.5  What did not help                            F12 (P2 head), F14 to F16 and F23 (augmentation, size on DeepFish alone), F25 and F26 (tiling), F27 (1280 px)
   5.6  What helped: more diverse data               F17 to F24
   5.7  Model size and the seed study                F32
   5.8  The sealed OzFish test                       F33
   5.9  On the Jetson Orin Nano                      F5 to F9, F28 to F31, F34
   5.10 An informal out-of-domain check              the reef photograph (labelled as not measured)

## 6. Discussion
   6.1  What limits accuracy
   6.2  Validity of the evaluation                  what the sealed test does and does not show
   6.3  Deployment trade-offs                        accuracy against frame rate and heat
   6.4  Licence and reuse                            the AGPL position of the weights
   6.5  What this means for fish monitoring

## 7. Limitations and Threats to Validity
## 8. Conclusion and Future Work

Appendices
   A  Reproduction: environments, commands, run register (run folder to finding)
   B  The evaluation protocol (docs/eval_protocol.md)
   C  The corrections log (PROGRESS section 6) and the bug list (section 7)
   D  Glossary (from docs/GUIDE.md)
   E  Licences, attributions and data availability
References
```

## State of each part

| Part | Source text | State |
|---|---|---|
| 1 | `paper.md` section 1 | short form drafted; expand |
| 2.1, 2.2, 2.6 | `GUIDE.md` | drafted for the owner; rewrite for the reader |
| 2.3 to 2.5 | `paper.md` section 2 | stub; **papers must be read** |
| 3 | `paper.md` section 3, `data_audit.md` | drafted |
| 4 | `paper.md` section 4, `eval_protocol.md` | drafted |
| 5.1 to 5.9 | `paper.md` section 5 | drafted; **5.9 waits for F34** |
| 5.10 | | one paragraph to write (the finding must stay labelled as not measured) |
| 6, 7 | `paper.md` sections 6 and 7 | drafted |
| 8, abstract | | **pending** |
| Appendices | `eval_protocol.md`, `PROGRESS.md`, `GUIDE.md` | assemble |

## Writing order

1. **4 Methodology and 3 Data**: settled; write these first.
2. **5 Results**: the evidence; waits only for F34.
3. **2 Background and related work**: after the papers are read.
4. **6 Discussion and 7 Limitations**, then **1 Introduction**.
5. **Conclusion, abstract**, then the appendices and the front matter.

## What is not in the thesis

Chat history, the other team members' RL work, and the data-handling episodes (slow downloads, the manual transfers) beyond what the reproduction appendix needs.
