# Publication checklist: making the repository and weights public under the AGPL-3.0

Decision recorded 2026-10-11: **share everything openly under the AGPL-3.0** (`docs/report/LICENCE_OPTIONS.md`, option 1). **Correction (2026-10-31):** the earlier line here said nothing had been pushed. In fact the GitHub repository `TDuong04/image-fish-AI` is public (created 2026-09-22, last push 2026-10-04, release `v0.1.0-models`), so its history up to commit `eeba2fc` is the unfiltered history, including the personal file removed in `4fb0893` and the private identifiers that the filtered public copy replaces. Decide whether to make it private, or replace it with the filtered copy, before pushing anything further.

## Done

- [x] `LICENSE`: the official AGPL-3.0 text, unmodified.
- [x] `NOTICE.md`: project licence and why, third-party software, dataset credits with the changes made to each (OzFish labels are a derivative work), YOLO-Fish used for evaluation only.
- [x] `docs/MODEL_CARD.md`: intended use, data, performance with finding ids, limitations, weight checksums.
- [x] AGPL section 13: the Gradio apps (`demo/app.py`, `edge/app.py`) print an offer of the source, taken from `$FISH_SOURCE_URL` (set it to the public repository address when serving).
- [x] README licence section; the datasets are not in the repository (git ignores `data/`, `models/`, `*.pt`, `*.onnx`, `*.engine`, `*.zip`).
- [x] The demo text that claimed there were "no Jetson numbers" (written before the board was measured) was corrected.
- [x] **Identifiers generalised in the working tree (owner's approval, 2026-10-11):** private hostnames and the team repository address became placeholders (`<repository-url>`, `git.example.org`, `example.org`),
  the Unix account path in the run records became `<repo>`, another person's project name became `another-project`, and the Kaggle username became `your-kaggle-username` (replace it with your own
  to run `kaggle/`). The same replacements were applied to the history of the public copy (below).
- [x] **Personal file kept out of the public copy (owner's approval):** a personal file added in one early commit was removed from the whole history of a **separate filtered clone**, built with `git filter-repo`;
  this repository was not rewritten. The clone's location and the old-to-new commit map are recorded in `docs/report/PUBLIC_COPY.md`.

## Still open (owner)

1. **The repository address and whether it is public** (decided: later). Then set `FISH_SOURCE_URL` for the apps and fill the address into the availability statement and the model card.
2. **Publishing the weights:** create a release with the six checkpoints of the model card (and ONNX files if wanted) and the `LICENSE`, `NOTICE.md` and `docs/MODEL_CARD.md`.
3. **A citation for the report** (DOI or arXiv number) once it exists, for the model card and the README.
4. **Copyright holder line:** `NOTICE.md` says "the authors of this repository (see the git history)"; replace it with names (and the institution, if the institution has a claim on the work, which your supervisor can tell you).
5. **Author identity in the history:** every commit carries the author name and e-mail from your git configuration, and they become public with the history. If you prefer not to publish the e-mail address, say so and the public copy can be rewritten
   with a mailmap (a different public address or the platform's no-reply address).
6. **Commit hashes differ in the public copy.** The run records name the commit they were produced at; in the public copy those hashes do not exist, so use the commit map to translate (`docs/report/PUBLIC_COPY.md`).

## Checks to repeat just before publishing

- `git grep -n -i -e password -e secret -e token -e "api[_-]key"` finds nothing sensitive; no dataset or weight file is tracked (`git ls-files | grep -E "\.(pt|onnx|engine|zip|mp4|jpg|png)$"` is empty).
- `python -m pytest` passes; `docs/PROGRESS.md` names every run folder (the tests enforce it).
- The numbers in the report match the findings list (`docs/report/paper.md` cites a finding for every number; `tests/test_report.py` checks the ids exist).
- Search the public copy for the placeholders that still need a real value: `git grep -n "<repository-url>" -e "<owner>" -e "your-kaggle-username"`.
