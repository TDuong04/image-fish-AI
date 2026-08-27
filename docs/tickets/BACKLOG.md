# Fish Detection on Jetson Orin Nano — Backlog

Status: `TODO` · `IN PROGRESS` · `BLOCKED` · `DONE` · `CUT`
Priority: **P0** blocks everything · **P1** critical path · **P2** needed for a complete result ·
**P3** stretch.

Revision 2 — reconciled against an adversarial review. Corrections from that review are marked
**[corrected]**; tickets it added are marked **[added]**; tickets it argued to drop are marked
`CUT` with the reason kept rather than deleted.

**Rules of order:** FD-113 starts before anything else. E2 needs E0 done. E3 needs FD-020's
checkpoint. E4 needs the board. **E6 does not wait for anything — start FD-060/FD-061 now.**

---

# E0 — Unblock, decide, bootstrap

## FD-113 · OzFish acquisition sprint — parallel routes, 5-day gate · P0 · TODO **[added]**
**Why:** Every ticket in E2–E5 and the whole results half of the report is blocked on box
training data, none is on disk, and the previously-recorded acquisition routes do not hold up
(see FD-010). An open-ended "try in order" on the only blocking dependency is not a plan.

**Do — all five in parallel, this week, not sequentially:**
1. Submit the Pawsey access request at `https://apply.pawsey.org.au/p/ax3003/` **today**.
   Turnaround is the long pole; starting it costs an hour.
2. Open `https://storage.pawsey.org.au/public/m/FDFML/` **in a real browser** and capture the
   actual file URLs from the network tab. See FD-010 for why curl cannot answer this.
3. Fetch the YOLO-Fish authors' `ozfish_test.zip` and `deepfish_darknet.zip` — the gdown IDs
   are already sitting in `fish-research/README.md`. Darknet-format boxes on the target domain,
   costs nothing, nobody has tried it.
4. Email the open-AIMS/ozfish maintainers and the Marrable et al. authors.
5. **Name the Plan B dataset in writing now**, before it is needed: a box-labelled underwater
   set with intact source-clip IDs so FD-013 still works.

**Done when:** by end of day 5, either images + boxes on disk with source-video grouping keys
verifiably intact, **or** a written pivot to Plan B. No third outcome.

---

## FD-115 · Schedule and person-hour budget · P0 · TODO **[added]**
**Why:** This backlog is roughly 2× oversized for one person. Honest sizing full-time:
E0 ~1wk · E1 ~2–3wk (dominated by FD-113 risk) · E2 ~3–4wk · E3 ~2–3wk · E4 ~3–4wk (first-time
Jetson work is where solo embedded projects lose a month) · E5 ~2wk · E6 ~3wk → **~4–5 months
full-time.** Part-time alongside coursework, that is 2–3× over budget and ends with a
half-finished E4 and no report.

**Do:** Estimate each ticket in hours against a real calendar. Cut to fit *before* starting.
The `CUT` markers below are the recommended first cuts.

**Done when:** a dated plan exists and the ticket list matches the available hours.

---

## FD-118 · Acquire the Jetson Orin Nano · P0 · TODO **[added]**
**Why:** FD-003 and FD-040 both presume possession. The board is not in hand, procurement lead
time is unticketed, and it blocks half the deliverable. If it slips four weeks, E4 slips four
weeks. Pair with FD-119, which removes that serialisation.

**Do:** Order the module + carrier + **adequate cooling** + NVMe (not SD — see FD-040) + a PSU
that supports the higher nvpmodel modes. Decide 4GB vs 8GB *now* (Open Question 1) — it changes
the power-mode table, the TOPS figure and FD-032's framing.

---

## FD-008 · Licence decision — was FD-035, now P0 in E0 · TODO **[corrected]**
**Why:** Previously P2 in E3, which is backwards — its own rationale said "discovering this
after training is expensive" and it was then scheduled after training. Its outcome decides which
model family FD-020 trains.

**Do:**
- Frame the stakeholder question correctly: **not** "will it be sold" but **"will any copy of
  this software or these weights ever leave the institution?"** Handing a unit to a partner
  fisheries agency is conveying under GPL/AGPL §5–6.
- **[corrected]** Ultralytics' AGPL-3.0 reaches **the COCO-pretrained `yolov8n.pt` weights that
  FD-020 initialises from**, so the resulting checkpoint is encumbered regardless of what code
  ships. This was missing from the original analysis.
- **[corrected]** If picking an Apache-2.0 alternative: YOLOX-nano or NanoDet-Plus. **Not
  Ultralytics' `RTDETR`** — that implementation is AGPL like the rest of the repo; only upstream
  `lyuwenyu/RT-DETR` is Apache. RT-DETR is also a poor Orin candidate on technical grounds
  (deformable attention has a weak TensorRT plugin story on Jetson).
- GPL-3.0 YOLO-Fish weights must not be linked into anything distributed.

**Done when:** the licence position is decided in writing, before FD-020.

---

## FD-001 · Initialise git repository · P0 · **DONE**
**Done:** repo initialised, `.gitattributes` + `.gitignore` with `runs/**` negations, pushed to `IoT-Group/image-fish-AI`. Splits tracked; weights and data excluded.
**Why:** No version control. No result can be tied to a commit.

**Do:**
- `git init`, `.gitattributes` (`* text=auto`, `*.sh text eol=lf` — otherwise shell scripts
  written on Windows fail on the Jetson with `\r: command not found`).
- `.gitignore` excludes `data/`, `*.pth`, `*.weights`, `*.engine`, `*.onnx`, `.venv/`,
  `__pycache__/`, archives.
- **[corrected]** It must **not** blanket-exclude `runs/`. CLAUDE.md requires every number be
  regenerable from a committed config; excluding `runs/` leaves every `config.yaml`,
  `metrics.json` and `env.txt` untracked and FD-051's matrix with no provenance. Use negation
  patterns: ignore `runs/**` but keep `runs/**/config.yaml`, `runs/**/metrics.json`,
  `runs/**/env.txt`. **`splits/` is tracked, explicitly.**
- Ignore `fish-research/` (vendored clones); record clone commands in `fish_download_links.md`.
- No LFS. Binaries stay out of git; provenance goes in `docs/data_manifest.md`.

---

## FD-002 · Portable training environment bootstrap · P0 · **DONE**
**Verified:** Python 3.11 venv, torch 2.6.0+cu124, `cuda.is_available()==True` on RTX 4060, ultralytics 8.4.130. `check_env.py` reports READY. Caches redirected into `data/.cache` by the env profile.
**Why:** Installed torch is `2.12.1+cpu` (`cuda.is_available() == False`) — training today runs
silently on CPU. System Python is 3.14.4, unsupported by Ultralytics and CUDA torch wheels. And
this must be fixable identically on Linux, a hosted notebook, and a collaborator's machine.

**Do:**
- Python 3.10/3.11 venv at `.venv/` inside the repo — not on a named drive.
- Three requirement sets: `requirements-train.txt` (x86 + CUDA), `requirements-edge.txt`
  (Jetson: TensorRT bindings, numpy, opencv — **no torch from PyPI**), `requirements-dev.txt`.
- `scripts/bootstrap.py` — one cross-platform entrypoint that detects OS/arch/CUDA, picks the
  right torch index, installs the right set, and refuses readably on a bad Python version.
- `scripts/check_env.py` — prints platform, arch, Python, torch, CUDA/JetPack, GPU, VRAM, free
  disk at `DATA_ROOT`, resolved device; runs a one-iteration GPU matmul; non-zero exit if the
  machine cannot train.

**Done when:** `check_env.py` reports an accelerator on the workstation **and** exits cleanly and
informatively on a CPU-only box and on the Jetson.

**Status:** scripts written and their failure paths verified. **The install path itself is
untested** — no 3.10 venv exists on the workstation yet, so `bootstrap.py` has never actually
installed anything. First teammate to run it should report back.

**Trap:** `pip install torch` gives the CPU wheel on Windows — needs the explicit
`--index-url .../whl/cu124`. On Jetson the PyPI wheel is wrong entirely: ARM64 + JetPack needs
NVIDIA's prebuilt wheels matched to the JetPack version. `bootstrap.py` branches on this.

---

## FD-006 · Portable path and config layer · P0 · **DONE** **[added]**
**Done:** `src/fish/{paths,device,env,datasets}.py`; 55 pytest checks enforce no absolute paths, no hardcoded cuda, `__main__` guards. The tests caught a real regression -- `configs/data/*.yaml` were being written with absolute paths -- now stored relative to `DATA_ROOT` and resolved per machine at run time.
**Done so far:** `src/fish/{paths,device,env}.py`, `.gitattributes`, `scripts/{bootstrap,check_env,train}.py`, `requirements-{train,edge,dev}.txt`. Remaining: the `pytest` guard in FD-110 that keeps absolute paths from reappearing.
**Why:** This project already spans a Windows workstation, a hosted Linux notebook, and an ARM64
Jetson. Any absolute path, drive letter, or hardcoded device breaks two of the three. Retrofitting
this after scripts exist is far more expensive than doing it first.

**Do:**
- `src/fish/paths.py` — resolves repo root from `__file__`; exposes `DATA_ROOT` (from
  `$FISH_DATA_ROOT`, default `<repo>/data`), `RUNS_ROOT`, `MODELS_ROOT`. Nothing else computes a
  path from scratch. `pathlib` only.
- `src/fish/device.py` — resolve once, `cuda` → `mps` → `cpu`, log the choice. No
  `torch.device("cuda")` anywhere else.
- Batch size / image size / workers / AMP come from config with a default that fits ~6 GB VRAM.
  Auto-scale or fail loudly; never silently OOM.
- `if __name__ == "__main__":` guard on anything using DataLoader workers — Windows and macOS
  `spawn` instead of `fork` and will otherwise fork-bomb.
- All dataset filenames lowercase, no spaces — Linux is case-sensitive, Windows is not, and a
  path that works here can 404 on the board.

**Done when:** `grep -rE '[A-Z]:\\|/home/|/kaggle/' src/ scripts/ edge/ configs/` returns nothing,
and every script runs from a different working directory.

---

## FD-007 · Machine profiles · P1 · **IN PROGRESS** **[added]**
**Done so far:** `configs/env/{workstation,colab,cpu,jetson}.yaml` + `$FISH_ENV` selection, cache redirection, `env.txt`/`env.json` per run. Remaining: `docs/environments.md`, and confirming the jetson profile on real hardware.
**Why:** Adding a fourth machine must mean adding a YAML file, not editing code. Cache
redirection in particular (`HF_HOME`, `TORCH_HOME`, `ULTRALYTICS_DIR`, pip cache, temp) is
machine-specific and must not be a one-off shell export that only exists in someone's history.

**Do:** `configs/env/{workstation,colab,jetson}.yaml` selected by `$FISH_ENV`, carrying data root,
cache redirection, batch/worker defaults and device hints. Document known-good profiles in
`docs/environments.md`. `env.txt` in every run records which profile produced it.

**Done when:** the same training command runs unchanged on two different machines via `$FISH_ENV`.

---

## FD-003 · Confirm and record board specification · P0 · TODO **[corrected]**
**Why:** TensorRT version, ONNX opset, precision strategy and model budget are all pinned by
facts not yet in hand.

**Do:** Record in `docs/hardware.md`: exact module (4GB vs 8GB), JetPack/L4T, CUDA, cuDNN,
**TensorRT version**, storage medium, cooling, PSU, available `nvpmodel` modes.

**[corrected]** Three errors in the original ticket:
- Super Mode requires **JetPack 6.2+** — it cannot exist on JetPack 5.1.x, so "5.1.x or 6.x" and
  "Super applied?" were mutually inconsistent.
- **67 TOPS is the 8GB figure.** Orin Nano **4GB** Super is **34 TOPS** (base 40 / 20). Under
  the "assume 4GB" rule, 67 must not appear.
- Add: **Orin Nano has no DLA** (Orin NX and AGX Orin do). No second inference engine to offload
  to — everything runs on the 1024-core Ampere GPU. And see FD-116: **no NVENC either.**

**Blocks:** FD-030 (opset), FD-031, FD-032, all of E4.

---

## FD-004 · Data layout and disk budget · P0 · TODO
**Why:** An unplanned download fills the volume mid-run and corrupts checkpoints. On this
workstation the system drive has ~7 GB free vs ~182 GB on the data drive — but the layout must
not encode that; the same code has to work where there is only one volume.

**Do:** `DATA_ROOT/{raw,interim,processed,eval}/` created by script. `docs/data_manifest.md` with
source URL, licence, size, **checksum**, required/optional/rejected — **relative paths only**.
Mark full CFC images out of scope (sonar + 132 GB). Cache redirection via FD-007's profile. Every
fetch script checks free space against the declared size and refuses up front.

---

## FD-005 · Evaluation protocol, written before training · **P0** · **DONE** **[corrected]**
**Done:** `docs/eval_protocol.md`, written before any real training run. Resolves the FD-005/FD-024 threshold contradiction and absorbs FD-103's seed rule.
**Why:** Defining metrics after seeing results is p-hacking. Was P1 while blocking a P0 — fixed.

**Do:** `docs/eval_protocol.md` specifying:
- Primary: mAP@50 / mAP@50-95, single class, COCO-style via pycocotools.
- Secondary: P/R/F1 at the operating threshold; **false positives per empty frame**.
- Latency: end-to-end (decode→preprocess→infer→NMS→post), warm-up discarded, median + p95 over
  ≥500 frames, power mode stated.
- Which split is sealed until final reporting.
- **[corrected]** The original said "thresholds fixed across all compared models," which
  directly contradicted FD-024's "tune the threshold." Resolution: **mAP is threshold-free by
  definition — fix nothing for it**; the operating threshold is selected on validation per
  FD-107 and the report says which split it came from.
- **[added]** Absorbs FD-103: the multi-seed rule below.

---

## FD-103 · Seeds, variance, and minimum detectable difference · P0 · TODO **[added]**
**Why:** FD-021, FD-022, FD-023 and FD-051 all compare **single runs**. Seed-to-seed mAP variance
for a nano detector on a few-thousand-image dataset is routinely ±1–2 points — larger than the
differences those tickets are meant to decide. As written, three E2 decisions are made on noise
and the report claims differences it cannot support. Must be settled *before* training, because
retrofitting seeds means re-running everything.

**Do:** ≥3 seeds for any comparison that drives a decision. Report mean ± range, never a bare
best. State the smallest difference you are willing to call real, in advance.

---

## FD-100 · Run registry · P1 · TODO **[added]**
**Why:** CLAUDE.md defines a `runs/<date>-<slug>/` convention but nothing enforces or aggregates
it. FD-051's matrix would otherwise be assembled by hand from directory listings — which is
exactly where transcription errors enter published tables.

**Do:** `scripts/collect_runs.py` walks `runs/`, validates each has config + metrics + env + git
SHA, emits one dataframe. MLflow local file backend optional. **Not W&B** — network dependency,
another account, no benefit here.

---

## FD-110 · Regression tests · P1 · **DONE** **[added]**
**Done:** `tests/`, 55 checks covering grouping, label parsing and the portability rules. Already caught one real bug. Remaining: wire a pre-commit hook.
**Why:** Four checks are load-bearing and all currently manual: label round-trip (FD-011),
leakage assertion (FD-013), ONNX parity (FD-030), benchmark repeatability (FD-050). FD-030 says
its check is "committed as a regression test" but nothing runs tests.

**Do:** `pytest` suite + a pre-commit hook. Full CI is not warranted solo; a hook is. Difference
between "the leakage check passed once" and "it passes on every commit."

---

## FD-102 · Backup and disaster recovery · P1 · TODO **[added]**
**Why:** FD-001 ignores `data/`, weights, engines and (originally) `runs/`. Git therefore protects
the markdown and nothing else. The entire result set lives on one drive in one laptop. That drive
failing at week 14 ends the project.

**Do:** Split files and run metadata into git (FD-001). Checkpoints and derived datasets to a
second drive or offsite, on a stated cadence. Raw downloads get a documented re-fetch path so they
need no backup.

---

# E1 — Data

## FD-010 · Acquire OzFish bounding boxes · P0 · TODO **[corrected]**
**Why:** The only real box-labelled optical fish dataset available. DeepFish's main split is
points, not boxes.

**[corrected] — two of the three previously-named routes do not hold up:**
- **The HuggingFace "mirror" is not a dataset.** `Devi-Ayyagari/yolov7_OzFish` contains exactly
  three files — `.gitattributes`, `README.md`, `yolov7_OzFish.pt` (74.7 MB of weights). Zero
  images, zero labels. HF dataset search for "ozfish" returns empty. Struck as a route.
- **"Pawsey URLs confirmed dead" is unsupported.** `storage.pawsey.org.au/public/m/FDFML/` and
  every subpath return **HTTP 200** — but so does a deliberately nonsensical path, serving a
  1.7 KB `text/html` shell. It is a JavaScript portal with a catch-all route, so **curl can
  establish neither presence nor absence.** Must be opened in a real browser (FD-113 step 2).

**Do:** execute FD-113. Verify whatever arrives: image count, box count, corrupt-file scan,
orphan-label scan. If via a third-party repackaging, record provenance — it is a citation risk
*and* an attribution-chain problem under CC BY (FD-112).

---

## FD-011 · Convert OzFish to YOLO format · P1 · TODO **[corrected]**
**[corrected] — the original ticket's central work item was misconceived.** It called for
"collapse all species to one class, keep species in a sidecar." Per `open-AIMS/ozfish` README
line 57, verbatim: *"Unlike the crops, frames and videos, these annotations are **fish/no-fish
only and have no species/genus/family labels**."* The ~45k boxes are **already single-class**.
The 507 species / 200 genera figure describes the ~80k **crops**, a different artefact. There is
no species collapse to perform and no sidecar to build. (A much smaller species-labelled box
subset exists at `labelled/speciesboxes`, annotated in VGG by an ecologist — separate, optional.)

**Consequences:** FD-063's planned limitation "single-class collapse from 507 species" describes
a limitation that does not exist and must be struck. **The same error is in
`report_sections.md` §2.2** — "45,000 bounding box annotations spanning 507 species" — and must
be fixed there (FD-060).

**Do:**
- Convert to normalised `cx cy w h` + `data.yaml`, single class `fish`.
- Clamp/drop degenerate boxes (zero area, out of bounds, < 2 px).
- **Acceptance criterion, non-negotiable:** the **source video / operation ID for every image
  survives** into the label path or a sidecar, verified by script. FD-013 splits on it; a
  third-party repackaging in YOLO layout may already have stripped it. Recovering it afterwards
  is not reliably possible, and without it the leakage-free split is impossible.
- Round-trip check: render 30 random images with boxes, inspect visually.

**Trap:** silent xyxy-vs-xywh / absolute-vs-normalised errors train to ~0 mAP with no error
message. The visual check is not optional.

---

## FD-105 · Annotation quality and label-noise audit · P1 · TODO **[added]**
**Why:** OzFish boxes came from SageMaker Ground Truth crowd consensus, and published work on the
dataset illustrates wrong annotations. **Missed fish are the specific failure that destroys
FD-024's headline metric** — an unlabelled fish in an "empty" frame scores as a false positive,
making FP-per-empty-frame uninterpretable. Nothing else in the backlog measures label noise.

**Do:** Re-label 200 random frames yourself. Report inter-annotator agreement and miss rate. Carry
it as an error bar on FD-024.

---

## FD-012 · DeepFish box test set · P2 · TODO **[corrected]**
**Why:** Independent, different-domain test set for FD-026.

**[corrected] — check before building.** The original assumed hand-deriving boxes from the 620
segmentation masks. But `report_sections.md` §2.3 itself cites AquaYOLO evaluating **detection on
DeepFish over 4,505 images**, and the YOLO-Fish authors' **`deepfish_darknet.zip`** — gdown ID
already in `fish-research/README.md` — is DeepFish in darknet **box** format. That is potentially
7× more data for none of the work.

**Do:** Try the existing derivative first (FD-113 step 3). Only if that fails, derive from masks
via connected components, spot-check 50, and document the touching-fish merge rate. Never
synthesise boxes from FishLoc **points** — a point has no extent and any box around it is
fabricated ground truth.

---

## FD-013 · Leakage-free splits · P0 · **DONE**
**Done:** site-level split in `scripts/prepare_dataset.py` with a hard assertion that no group appears on both sides. The risk was real: DeepFish clips run to 462 consecutive frames, so a random split would have scattered ~73 near-identical frames of one clip across train and val. Splits committed under `splits/`.
**Why:** OzFish frames come from video. Random per-image splitting puts near-identical adjacent
frames in train and test, inflating mAP by a large unknown margin. **Still the single most likely
way this project produces a wrong headline number.** Depends on FD-011 preserving video IDs.

**Do:** Split at video / deployment / site level. Assert no source video appears in two splits.
Perceptual-hash check for near-duplicates across boundaries. Commit split files; never regenerate
randomly per run.

---

## FD-014 · Dataset characterisation audit · **P0, before FD-020** · **DONE** **[corrected]**
**Done:** `scripts/audit_dataset.py` + `docs/data_audit.md`. Headline finding: at 640 px, 14% of DeepFish and 35% of OzFish boxes fall below ~16 px, the floor a stride-8 P3 head can resolve. Resolution is now a first-class variable, not a default.
**Why:** Was P1 running beside training. It gates more than FD-022's resolution list: **if the
median box is under ~32 px at training resolution, the project needs tiling** — which changes the
architecture of FD-030, FD-031, FD-034, FD-042 and FD-043. Discovering that after E2 invalidates
E2.

**Do:** `docs/data_audit.md`: box area distribution (px and as image fraction); boxes per image;
fraction of frames with **zero** fish; source resolutions; brightness/turbidity proxy.

---

## FD-015 · Smoke subset · P1 · **DONE**
**Done:** `--max-per-group 12` builds a 240-image subset spanning all 20 sites, sampled evenly within each clip rather than taking the first N (consecutive frames are near-identical). Full train->val cycle in ~2 min.
~200 train / 50 val, stratified across sites and count buckets, wired as `data/smoke.yaml`. Every
pipeline change smoke-tested first. **Done when** a full train→export→eval cycle runs in under
10 minutes.

## FD-101 · Checksum verification gate · P2 · TODO **[added]**
FD-004 writes checksums; nothing verifies them. `scripts/verify_data.py`, called by FD-020 before
training. Full DVC is overkill; a verify gate is not.

## FD-016 · Target-domain evaluation set · P2 · TODO
Hand-label 100–200 frames of real deployment footage; seal until final evaluation. If no such
footage exists (Open Question 3), use DeepFish as the domain-shift proxy and record the absence as
a stated limitation.

---

# E2 — Training

## FD-020 · YOLOv8n baseline · P0 · TODO
640px (or FD-014's recommendation), single class, fixed seeds per FD-103, trained **to
convergence** with early stopping on val mAP — not a fixed small epoch count. Logs to
`runs/<date>-yolov8n-base/` with config, metrics, env, git SHA.

**Trap:** the LCFCN result in `report_sections.md` was stopped at 10 epochs and is explicitly not
converged. Do not repeat that pattern — it makes the number uncomparable to anything published.

## FD-021 · `n` vs `s` · P1 · TODO **[shrunk]**
Two models only. **Dropped:** "also consider yolov11n/s" — doubles the grid to answer a question
FD-008 may make moot. Compare mAP, params, FLOPs, desktop latency, across FD-103's seeds.

## FD-022 · Resolution · P1 · TODO **[shrunk]**
Two resolutions (FD-014's recommendation + 640) and a decision — not a four-point curve. The curve
is a nice figure, not a load-bearing result. If small-object mAP binds, evaluate a P2 head or
tiling against plain upscaling.

## FD-023 · Augmentation ablation · P2 · **CUT** **[cut]**
Worst effort-to-evidence ratio in the backlog: a defensible ablation is n_configs × n_seeds runs
to resolve effects smaller than the seed variance FD-103 exposes. **Replace with one decision:**
Ultralytics defaults, close mosaic for the final 10 epochs, state it as a limitation.

## FD-024 · Empty-frame false-positive rate · P1 · TODO
BRUVS video is mostly empty water; a detector firing on kelp and particulates produces a useless
count, and mAP hides it. Include background images at a measured ratio; report FP-per-empty-frame
as a first-class metric; interpret against FD-105's miss rate and FD-107's threshold.

## FD-104 · Trivial baselines · P1 · TODO **[added]**
**Why:** The report has no floor — nothing to say "the detector beats X."
**Do:** (a) predict-nothing, giving the mAP and FP rate of doing nothing on mostly-empty footage;
(b) **classical frame-differencing / background subtraction** — genuinely competitive on a static
BRUVS camera and runs at hundreds of FPS on the Orin's CPU, which is an uncomfortable and
*interesting* result; (c) zero-shot COCO YOLOv8n, no fine-tuning.
Without (b) a reviewer will ask why a 3.2M-parameter network was needed for a fixed camera
pointed at a bait bag.

## FD-107 · Threshold selection and calibration · P1 · TODO **[added]**
**Why:** Beyond resolving the FD-005/FD-024 contradiction: **quantisation shifts the score
distribution**, so a threshold held fixed across FP32/FP16/INT8 is not a fixed operating point,
and FD-051's cross-precision comparison is then not like-for-like.
**Do:** Select on validation, never test. State the criterion. Re-select per precision or justify
not doing so. Reliability diagram / ECE so "confidence 0.5" means something.

## FD-026 · Cross-dataset generalisation · P1 · TODO
Evaluate every carried-forward checkpoint on FD-012's DeepFish set and FD-016's target set. Report
the in-domain → cross-domain mAP drop explicitly. **Do not cut** — this is a core contribution.

## FD-025 · YOLO-Fish baseline · **CUT — take the written-justification branch** **[cut]**
Building AlexeyAB darknet with CUDA on Windows is a multi-day yak shave for a comparison the
ticket already permitted satisfying in writing. Cite their published number with an explicit
statement that the splits differ. **[corrected]** Extend that caveat to **throughput**: their
"179 FPS" is a desktop-GPU number and must never appear near an Orin Nano FPS figure. Also
removes the GPL-3.0 linkage concern.

---

# E2b - Training quality

Everything here is about extracting the best achievable accuracy from the data,
*before* anything is traded away for the board. Ordered by expected value.

## FD-120 - Small-object architecture: P2 head - P1 - TODO
**Why:** The audit's central finding is that 14% (DeepFish) / 35% (OzFish) of boxes sit below
the ~16 px floor of a stride-8 P3 head at 640 px. A **P2 head (stride 4)** attacks that failure
mode directly and costs far less than doubling input resolution, which is the one thing the
Orin Nano can least afford.
**Do:** Train `yolov8n-p2` against the 640 and 960 baselines. Compare mAP overall *and*
restricted to boxes under 32 px, which is where the effect must appear if it is real. Record
parameter and FLOP cost, since it carries straight into E3.
**Done when:** the small-box mAP delta is measured, with the latency cost stated.

## FD-121 - Tiled inference vs upscaling - P2 - TODO
**Why:** The other route to small objects: slice 1920x1080 into overlapping tiles inferred at
native scale. Higher effective resolution without a larger network, at the cost of several
forward passes per frame - a very different point on the Jetson cost curve than a bigger
input, and possibly a better one in *batch* mode where latency is soft.
**Do:** SAHI-style tiling at inference against plain upscaling, matched on total compute.
Handle duplicate detections across tile seams explicitly.
**Done when:** both sit on the same accuracy-vs-compute plot.

## FD-122 - Hyperparameter search - P2 - TODO
**Why:** Ultralytics defaults are tuned for COCO: 80 classes, ~7 objects per image, large
objects. This is 1 class, 2.4 objects per image, mostly small. lr, warmup, box/cls loss
weights and the optimiser are unlikely to be at their best.
**Do:** `model.tune()` or a bounded random search on the val split, budget fixed in advance.
Validate the winner across FD-103's seeds - a search this small will otherwise select noise.
**Done when:** tuned settings beat defaults by more than the seed spread, or are shown not to.

## FD-123 - Dense-scene NMS and max_det - P1 - TODO
**Why:** OzFish reaches **232 boxes in one frame**, against an Ultralytics default of
`max_det=300`. Close enough to check rather than assume. Class-agnostic NMS and the IoU
threshold also behave differently on overlapping schooling fish than on COCO. The smoke run
already showed **postprocess 11.1 ms vs inference 3.3 ms** - NMS dominating even on a desktop
GPU, which is FD-034's problem arriving early.
**Do:** Sweep `iou` and `max_det`; confirm no truncation on the densest frames; measure the
postprocess share of end-to-end latency.
**Done when:** settings are fixed with evidence and no evaluation is silently truncating.

## FD-124 - Augmentation for underwater density - P2 - TODO
**Why:** Replaces the cut FD-023 with something narrower and better motivated. Two augments
plausibly matter here and are not defaults: `copy_paste` (raises object density, matching the
OzFish target domain) and HSV/turbidity jitter (the dominant real-world nuisance).
**Do:** Test exactly these two, evaluated on the **cross-dataset** set, not in-domain.
**Done when:** kept or rejected on cross-dataset evidence.

## FD-125 - Training schedule and checkpoint selection - P2 - TODO
**Why:** `epochs: 150, patience: 30` was a reasonable guess, not a measurement. The selection
metric matters too: mAP@50-95 favours tight boxes, F1 favours the operating point the device
actually runs at.
**Do:** Confirm from the loss/mAP curves that 150 epochs is enough. State the selection metric
in `eval_protocol.md` and apply it consistently.
**Done when:** the schedule is justified from a curve rather than assumed.

## FD-126 - Merge DeepFish + OzFish training - P2 - **blocked by FD-113** - TODO
**Why:** The YOLO-Fish authors' own best results came from a *merged* dataset, and their
`merge_*.weights` are already on disk. Given the sharp domain gap measured in the audit
(2.4 vs 23.7 boxes/image), merging is the obvious way to cover both regimes.
**Do:** Once OzFish training data exists, build a merged dataset with grouping preserved across
both sources and compare against DeepFish-only under the same protocol.
**Done when:** merged vs single-source is measured on both test sets.

## FD-127 - Zero-shot and trivial baselines - P1 - TODO
**Why:** FD-104's floor, made concrete now that data exists. Cheap, and the report has no lower
bound without it.
**Do:** (a) predict-nothing on the 30%-empty DeepFish val; (b) COCO-pretrained YOLOv8n with no
fine-tuning (COCO has no fish class - report what it confuses them for); (c) frame-differencing
on static BRUVS footage.
**Done when:** all three appear in the results table.

## FD-128 - Small-box stratified reporting - P1 - TODO
**Why:** A single mAP number hides the exact effect this project is about. If resolution
recovers small fish, that shows up in a size-stratified breakdown and is invisible in the
aggregate.
**Do:** Report mAP split by box area (<16 px, 16-32, 32-96, >96) for every model. Build it into
the eval harness so it is produced automatically, not by hand.
**Done when:** every results row carries a size breakdown.

---

# E3 — Edge optimisation

## FD-119 · Desktop TensorRT dry-run track · P1 · TODO **[added]**
**Why:** The cheapest schedule de-risk available, and it had no ticket. The RTX 4060 is also
Ampere and runs TensorRT. ONNX export, the parity harness, INT8 calibration code, the NMS
benchmark and FD-050 itself can all be written and debugged on x86 **before the board arrives**.
Without this, E3 and E4 are fully serialised behind FD-118.

**Do:** Build and debug the whole export/quantise/benchmark chain on the workstation. Treat the
board as the machine that produces final numbers, not the machine where the code is first written.

## FD-108 · DeepStream vs raw TensorRT — decision spike · P1 · **blocks FD-042, FD-043** · TODO **[added]**
**Why:** FD-042 originally said "GStreamer `nvv4l2decoder` (or DeepStream)" as a parenthetical.
That parenthetical is a multi-week architectural fork: DeepStream gives decode, batching, tracking
and NVMM plumbing free but imposes its own config world and pins JetPack versions; raw TensorRT +
PyGStreamer gives control at the cost of weeks of buffer management. FD-034 silently assumes the
raw path.
**Do:** Timebox to 2 days. Decide. Write it down. Doing FD-042 first risks discarding the work.

## FD-034 · NMS placement · **P1, before FD-030** · TODO **[corrected]**
Was P2 while blocking P1 FD-030, whose own steps say "decide and document (see FD-034)." With
hundreds of candidates per frame, CPU-side NMS on the Arm cores can cost more than the inference.
Benchmark `EfficientNMS_TRT` in-graph vs CPU NMS after the engine; verify detections match.

## FD-030 · ONNX export with parity check · P1 · TODO
Fixed input size, static batch 1, opset constrained by FD-003's TensorRT version. Parity test:
identical inputs through PyTorch and onnxruntime; assert max abs logit diff under tolerance and
identical detections on ≥100 images. Committed as a test that FD-110 actually runs.
**Trap:** dynamic shapes and in-graph NMS are the two most common Jetson conversion failures.

## FD-031 · TensorRT FP16 engine, built on-device · P1 · TODO
`trtexec --fp16` **on the Orin** — engines are not portable from x86. Record build time, engine
size, peak build memory (engine building can OOM a 4GB board; may need swap). Measure the
FP32→FP16 accuracy delta rather than assuming it.
**[added] Deliverable:** a committed **engine build script**, not a command typed at a prompt —
FD-064 depends on it existing.

## FD-032 · INT8 quantisation · **P2, reframed** · TODO **[corrected]**
**[corrected] — the original premise was wrong.** It claimed "on a 4GB Orin Nano, INT8 is likely
required to hit both the memory and latency budget." YOLOv8n is ~3.2M params — roughly 6 MB of
FP16 weights; memory is not the constraint. Published Orin measurements put YOLOv8n @640 INT8
around 3.5 ms, with FP16 comfortably real-time. **And in offline batch mode there is no latency
budget at all** — only throughput, which will be bound by video decode and CPU pre/post-processing,
not GPU inference.
**Reframe:** INT8 stays in the plan as a legitimate **research question about the accuracy cost of
quantisation** — which is a genuine report contribution. It is not a deployment necessity, and
calling it one leads to defending a threshold you were never near.

**Do:** Calibration set of 300–1000 images spanning sites/turbidity/density, **disjoint from
test**. Record its provenance — calibrating on test images inflates the result.
**[corrected] Method:** if the board runs JetPack 6.x it ships **TensorRT 10.3**, where implicit
quantisation and `IInt8EntropyCalibrator2` are **deprecated** in favour of explicit quantisation
(Q/DQ nodes) via TensorRT Model Optimizer. Do not write against the deprecated calibrator without
checking FD-003 first.
**Trap:** `trtexec --int8` **without** a calibration cache does not error — it silently applies
default dynamic ranges and produces a fast, badly wrong engine. Invisible unless you evaluate.
Easiest way in the world to publish a wrong INT8 number.

## FD-114 · Model and engine artefact registry · P2 · TODO **[added]**
FD-043 references "model hash, engine hash" and FD-051 needs engine size, but nothing defines
where engines live or how one traces back to (checkpoint, ONNX, opset, calibration cache, TRT
version, board, JetPack) — or what invalidates them. Otherwise: six `.engine` files, no memory of
which is which.

## FD-033 · Compression contingency · **CUT** **[cut]**
Contingency on a premise (INT8 insufficient) that is probably false per FD-032. Delete rather than
demote; reinstate only if FD-031/FD-032 actually miss a target.

---

# E4 — Jetson deployment

## FD-050 · Benchmark harness · **P0, moved to front of E4** · TODO **[corrected]**
Was "P0 for E5," but FD-042 and FD-044 both produce timing numbers and both come earlier —
building the instrument afterwards guarantees re-running them. Buildable on x86 first via FD-119.

**Do:** discard warm-up; time each stage separately (decode / preprocess / infer / NMS / post)
**and** end-to-end; report median, p95, stddev over ≥500 frames; record nvpmodel mode, clocks,
temperature, JetPack, engine hash; emit JSON.
**Done when:** repeated runs of one config agree within a few percent.

## FD-040 · Provision the board · P0 · TODO
Flash JetPack, `jetson-stats`, CUDA/TensorRT sanity check, headless SSH, enumerate `nvpmodel`
modes, configure swap. Record all in `docs/hardware.md`.
**[added] Rule: freeze the JetPack version for the study duration.** A minor upgrade invalidates
every engine **and every FPS number collected with it**. Nothing else in the plan said this.
**Trap:** SD-booted Orins are I/O bound — prefer NVMe, and if it boots from SD, expect decode to be
storage-limited and note that before blaming the model.

## FD-109 · Reproducible on-device environment · P2 · TODO **[added]**
No ticket said how `edge/` dependencies reach the board reproducibly, so FD-064's REPRODUCE.md
cannot cover the device half. Pin an `l4t-base` container, or at minimum a pinned apt/pip manifest
for aarch64.

## FD-041 · On-device smoke inference · P1 · TODO
Build the engine on-device, run 10 images, compare detections against desktop PyTorch on the same
images. Investigate any mismatch before proceeding.

## FD-042 · Hardware-accelerated decode · P1 · TODO · **blocked by FD-108**
Naive `cv2.VideoCapture` decodes H.264 on the Arm CPU and bottlenecks before the GPU saturates —
producing an FPS number that measures the decoder, not the model. Use `nvv4l2decoder` with NVMM
buffers; avoid device→host→device copies. **Benchmark decode-only throughput in isolation first**,
so the model's share of the budget is known.

## FD-116 · Video encode is CPU-only on this board · P2 · TODO **[added]**
**The Orin Nano has no NVENC** — it is the one Orin module where NVIDIA removed the hardware
encoder. NVDEC exists, so decode (FD-042) is fine, but **any annotated video output** — FD-046's
overlay and the qualitative figures FD-053 needs — is CPU x264 on the A78AE cores and will
dominate the pipeline. Nothing acknowledged this.
**Do:** Write detections to CSV/JSON as the primary output. Render annotated video **off-board**,
or only for short figure clips, and never inside a timed benchmark.

## FD-043 · Batch video CLI · P1 · TODO · **blocked by FD-108**
`edge/run_batch.py`: video directory → per-frame detections + per-video summary CSV. Resumable
(a crash mid-video must not lose hours), progress logging, run manifest with model hash, engine
hash, thresholds, power mode.

## FD-106 · Temporal consistency, tracking, and MaxN · P1 · TODO **[added]**
**Why — the largest unclaimed contribution in the plan.** The pipeline is per-frame with zero
temporal reasoning, while the primary deployment mode is *offline batch over video*: the one mode
where temporal information is free and complete (you can look forward as well as back). Three
consequences, none previously ticketed:
1. The ecological output a marine scientist actually wants from BRUVS is **MaxN**, not per-frame
   boxes — and nothing produces it.
2. Track-length filtering is the cheapest available suppressor of exactly the false positives
   FD-024 worries about: a kelp frond firing on one frame in forty is deleted for free.
3. Per-frame mAP on adjacent video frames is a **statistically dependent** measurement, and the
   report currently treats frames as independent samples.
**Do:** SORT / ByteTrack, no appearance model — near-free on CPU. Report MaxN per clip alongside
detection metrics, and the FP reduction from track filtering.

## FD-044 · Power mode and thermal characterisation · P1 · TODO **[corrected]**
**Why:** An FPS number without a power mode is meaningless, and a short benchmark hides
throttling — a passively cooled Orin can lose a large fraction of throughput after minutes.
**[corrected]** CLAUDE.md previously asserted "power mode changes FPS by 2–3×." That is
overstated for the assumed board: the **4GB** module's modes are **7W and 10W** (a 1.43× envelope,
which will not produce a 2–3× swing); the 2–3× figure is only reachable on the **8GB** module under
JetPack 6.2 Super (7W → 25W MAXN_SUPER). State the *requirement* — every FPS number cites its mode
— without asserting a magnitude the project has not measured. Inheriting a headline number from a
memo instead of from the board is precisely what CLAUDE.md's Honesty section forbids.
**Do:** Benchmark each available mode ± `jetson_clocks`. Run a **sustained ≥30 min** load logging
temperature, clocks and power. Report burst **and** sustained FPS; where they differ materially,
sustained is the real number.

## FD-117 · Energy per frame · P2 · TODO **[added]**
The claimed contribution is edge deployment, and energy is what edge deployment is *for*. FD-044
logs power only as a throttling proxy. Joules-per-frame across precisions and power modes is
cheap, novel and defensible — and currently not a reported metric.

## FD-045 · Memory profiling · P1 · TODO
Log peak RSS + GPU memory across a long batch run; check for leaks across videos; verify headroom
for decode buffers, not just the engine. **[corrected]** Note that **all** Jetsons use unified
memory — the 8GB module is not "8GB of dedicated VRAM"; JetPack plus a desktop session consumes
~1.5–2 GB before your process starts. The 4GB/8GB difference is quantity, not architecture.

## FD-111 · Unattended long-run safety · P2 · TODO **[added]**
FD-043 says "resumable" and FD-045 says "no OOM," but neither covers disk-full mid-run (an
SD-booted Orin writing per-frame detections will fill), thermal shutdown, power loss, watchdog
restart, or checkpoint cadence. **And nothing at all covers the dev-box side:** a multi-hour
training run on a Windows laptop that will happily sleep or take a Windows Update reboot at
hour six.

## FD-046 · Real-time camera path · **CUT → Future Work** **[cut]**
Stated stretch goal, and it drags in the NVENC problem (FD-116), systemd, autostart, recovery and
frame-drop policy. Becomes two sentences in FD-063.

---

# E5 — Evaluation

## FD-051 · Results matrix · P1 · TODO **[shrunk]**
**[corrected]** The written cross-product was ~48 on-device runs and will not finish. Also, the
FP32→FP16→INT8 chain is right as a **parity check** (FD-030, FD-041) but wrong as a **sweep axis**:
TensorRT FP32 on an Orin Nano is not a configuration anyone deploys, and measuring it burns scarce
board time for a cell no one reads.
**Do:** {2 models} × {FP16, INT8} × {2 power modes} = **8 cells**. Drop the FP32 column and the
resolution axis. Capture mAP@50, mAP@50-95, FP-per-empty-frame, end-to-end median/p95, memory,
engine size, J/frame. Sourced from FD-100, not hand-transcribed.

## FD-052 · Accuracy-vs-latency Pareto · P2 · TODO
mAP vs **sustained** on-device FPS; mark the frontier; recommend one operating point with the
trade-off stated plainly.

## FD-053 · Qualitative failure analysis · P2 · TODO
Categorise: turbidity, dense schools, occlusion, small/distant fish, motion blur, camouflage,
non-fish FPs. Compare FP16 vs INT8 failure cases to show *what kind* of detections quantisation
costs. Render figures off-board (FD-116).

---

# E6 — Report · **start now, do not queue behind E4**

## FD-060 · Rewrite Related Work for edge detection · P1 · **start now** · TODO
Needs no data, no GPU, no board. Keep and adapt the dataset and YOLO sections; add the missing
pillar — edge/embedded inference: TensorRT, PTQ/QAT, pruning, distillation, prior Jetson-deployed
ecological monitoring. Reframe LCFCN as a related-but-different paradigm.
**[corrected] Must fix §2.2's factual error:** "45,000 bounding box annotations spanning 507
species" — the boxes are class-agnostic; the species figure belongs to the crops (FD-011).

## FD-061 · Reconcile or retire the LCFCN experiment · P1 · **start now** · TODO
val MAE 0.3375 came from a 10-epoch, unconverged Kaggle run in a session that no longer exists,
measuring a different task. **Recommendation: cut it** unless supervision requires otherwise —
every hour reconciling a counting experiment is an hour not spent on the board.
**[corrected]** If it is kept, three things must be fixed, not just the labelling:
- §4.3 "confirming that continued training would yield substantial further gains" — speculation
  from a loss curve, stated as confirmed fact.
- §4.4 presents the **epoch-0** model's output as the qualitative result for a 10-epoch run.
- That figure is described but not present.

## FD-062 · Methods · P2 · TODO
Data provenance and splits, the leakage check, hyperparameters and seeds, the full
export/quantisation chain, benchmark methodology, exact hardware and JetPack versions.

## FD-063 · Results and threats to validity · P2 · TODO
**[corrected] Strike** "single-class collapse from 507 species" — that limitation does not exist
(FD-011). **Keep/add:** third-party mirror provenance, label noise (FD-105), small cross-dataset
set, sonar-vs-optical exclusion, single hardware unit (no across-board variance), thermal
conditions at measurement time, frames-as-dependent-samples (FD-106), and real-time deferred to
Future Work (FD-046).

## FD-112 · Dataset licensing and attribution · P2 · TODO **[added]**
FD-008 covers detector *code* only. Missing: CC BY 3.0 AU attribution on OzFish-derived labels
(the converted YOLO labels are a derivative work); that a third-party mirror **breaks the
attribution provenance chain** CC BY requires; and that FD-064 must redistribute checksums, not
images.

## FD-064 · Reproducibility package · P2 · TODO
Configs, split files, requirement sets, **FD-031's engine build script**, the benchmark harness,
and `REPRODUCE.md` from clean clone to headline number. Dataset checksums, never the data.
**Done when:** a clean run on a fresh machine reaches the training step with no manual step beyond
dataset download.

---

# Open questions — resolve in week one, not month three

These are stakeholder decisions with zero technical cost and total downstream leverage.

1. **Orin Nano 4GB or 8GB?** Sets the power-mode table, the TOPS figure, and FD-032's framing.
2. **Will any copy of the software or weights leave the institution?** (Not "will it be sold.")
   Decides FD-008, and FD-008 decides what FD-020 trains.
3. **Does real deployment footage exist?** Decides whether FD-016 is P1 work or a limitation
   paragraph.
4. **Deadline and weekly hours?** FD-115 cannot be done without this, and the backlog is ~2×
   oversized until it is.
5. **Must the LCFCN counting work appear in the report?** Drives FD-061.
