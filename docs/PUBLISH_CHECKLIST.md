# Publication checklist: making the repository and weights public under the AGPL-3.0

Decision recorded 2026-10-11: **share everything openly under the AGPL-3.0** (`docs/report/LICENCE_OPTIONS.md`, option 1). Nothing has been pushed or published yet. This list is what has
been done in the working tree and what still needs the owner, because some items are irreversible or reveal information once public.

## Done in the working tree (2026-10-11)

- [x] `LICENSE`: the official AGPL-3.0 text, unmodified.
- [x] `NOTICE.md`: project licence and why, third-party software, dataset credits with the changes made to each (OzFish labels are a derivative work), YOLO-Fish used for evaluation only.
- [x] `docs/MODEL_CARD.md`: intended use, data, performance with finding ids, limitations, weight checksums.
- [x] AGPL section 13: the Gradio apps (`demo/app.py`, `edge/app.py`) print an offer of the source, taken from `$FISH_SOURCE_URL` (set it to the public repository address when serving).
- [x] README licence section; the datasets are not in the repository (git ignores `data/`, `models/`, `*.pt`, `*.onnx`, `*.engine`, `*.zip`).
- [x] The demo text that claimed there were "no Jetson numbers" (written before the board was measured) was corrected.

## Needs the owner before anything is made public

1. **A personal file is in the git history** (`creativity_marks_appeal.md`, added in commit `b56ca8d`, later ignored). Pushing this history publishes it. Options: (a) publish from a **fresh filtered copy**
   (`git filter-repo --path creativity_marks_appeal.md --invert-paths` on a clone, which leaves your working repository untouched; recommended), (b) rewrite this repository's history and force-push, or (c) publish without history. I have not done any of these.
2. **Infrastructure and personal identifiers in tracked files:** the private server and board hostnames in `README.md` and `docs/TRAINING.md` (`grep grifufu`), the name of another person's project on the shared server in `docs/OVERNIGHT.md`,
   your Kaggle username in `kaggle/kernel-metadata.json`, and a Unix account name in about 48 `runs/*/env.json` files. Decide whether to keep, generalise or remove each; I can generalise them in a commit if you say so.
3. **The repository address and whether it is public** (the project has a Gitea and a GitHub remote). Then set `FISH_SOURCE_URL` for the apps and fill the address into the availability statement.
4. **Publishing the weights:** create a release with the six checkpoints of the model card (and ONNX files if wanted) and the `LICENSE`, `NOTICE.md` and `docs/MODEL_CARD.md`; the older release `v0.1.0-models` holds the first baselines under the same licence.
5. **A citation for the report** (DOI or arXiv number) once it exists, for the model card and the README.
6. **Copyright holder line:** `NOTICE.md` says "the authors of this repository (see the git history)"; replace it with names (and the institution, if the institution has a claim on the work, which your supervisor can tell you).

## Checks to repeat just before publishing

- `git grep -n -i -e password -e secret -e token -e "api[_-]key"` finds nothing sensitive; no dataset or weight file is tracked (`git ls-files | grep -E "\.(pt|onnx|engine|zip|mp4|jpg|png)$"` is empty).
- `python -m pytest` passes; `docs/PROGRESS.md` names every run folder (the tests enforce it).
- The numbers in the report match the findings list (`docs/report/paper.md` cites a finding for every number; `tests/test_report.py` checks the ids exist).
