# Report workspace

The research report is drafted here, section by section, as results come in.

| File | What |
|---|---|
| `paper.md` | The manuscript draft. Sections marked **[PENDING]** wait on experiments. |
| `references.bib` | Machine-readable references. |
| `references.md` | How each reference was verified, where it is used and what is still to check. |

## Structure (model: Zhang et al. 2024, an open-access YOLOv8 fish-detection paper, plus the usual applied-vision layout)

| Section | State |
|---|---|
| Abstract | **Pending**: written last |
| 1 Introduction | Drafted |
| 2 Related work | Drafted from verified bibliographic data; two papers (YOLO-Fish, AquaYOLO) still to be **read** before their results are quoted |
| 3 Datasets | Drafted |
| 4 Method | Drafted |
| 5 Results | 5.1 to 5.6 and 5.8 drafted; **5.7 (1280 px) pending F27**; device results are YOLOv8n only |
| 6 Discussion | Drafted, with placeholders for F27 and literature comparison |
| 7 Limitations | Drafted |
| 8 Conclusion | **Pending** |

## Rules

- Every number carries its finding id (F#); `tests/test_report.py` checks that every cited finding exists in `docs/PROGRESS.md`
  and that every bibliography key is tracked in `references.md`.
- Never cite a paper that has not been opened. Record how it was verified in `references.md`.
- Update `paper.md` in the same change as a new finding that it depends on. If a finding is overturned, fix the paper and
  add the correction to `docs/PROGRESS.md` section 6.
- Do not compare our numbers with a cited paper's without stating that splits, sizes and metrics may differ.
