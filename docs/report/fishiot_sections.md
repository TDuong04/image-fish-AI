# Detection sections for the FishIoT team report

Draft text for the two slots in the team report's agreed structure (design in section 2.4, results in section 3.2), plus the two short insertions
in sections 1 and 4/5. Written 2026-10-11 from `docs/PROGRESS.md`; every number carries its finding id. Aim: about one page for 2.4 and
about one and a half for 3.2, trimmed to the report's page budget. Items marked **[CHECK]** need input from the RL half or the team.

---

## Insert in 1B (Existing solutions): one paragraph

Automated fish monitoring is usually done by detecting fish in underwater video with convolutional detectors, most often from the YOLO family
(Redmon et al., 2016; Muksit et al., 2022). The practical difficulties are small and camouflaged fish, differences between sites, and evaluation
protocols that let near-identical video frames appear on both sides of a train-test split, which inflates scores (Kapoor and Narayanan, 2023). **[CHECK: add 2 to 3 citations after the related-work reading, see `references.md`]**

## Insert in 1C (Proposed approach): two sentences

The system is a closed loop: a fish detector turns camera footage into a count, the RL agent uses that count as one entry of its observation, and the agent operates
the fish infrastructure. The detector is the only physical sensor for this observation; the other entries come from records and instruments.

---

## 2.4 Fish observation subsystem (design)

**Role.** The detector supplies the one observation the agent cannot get from records or instruments: how many fish are in view. It takes video frames and returns boxes with confidence scores; the
count for a frame is the number of boxes above a confidence threshold, and the clip-level figure is the maximum count over the clip (MaxN, the standard in baited-video work).
The board application already returns per-frame counts, MaxN and a per-frame CSV (`edge/app.py`). **[CHECK: how the RL agent consumes this value, its time step and units]**

**Model.** A single-class YOLOv8 detector (Jocher et al., 2023), initialised from COCO weights and fine-tuned on fish. The chosen model is YOLOv8s trained at 960 px (a 1080p frame is evaluated at 544x960);
YOLOv8n is the faster fallback. Earlier counting experiments (LCFCN) are not part of the final design: that run was stopped at 10 epochs and was not converged.

**Data and evaluation protocol.** Training uses three openly licensed datasets, all converted to one class: DeepFish (training and validation), Kakadu (extra training data) and OzFish (training frames plus a held-out set).
Frames from one video are near-duplicates, so splits are made over whole capture sites and checked for leakage. This matters: the same pipeline scores 0.98 mAP@50 on a random per-image split and
about 0.47 on a split by site (F20). Every comparison uses same-machine controls and at least three seeds, and one OzFish test set is sealed and was evaluated once, at the end (F33).

**Deployment.** The detector runs on a Jetson Orin Nano 8 GB: the model is exported to ONNX and compiled to a TensorRT FP16 engine on the board, because engines are specific to the GPU, TensorRT version and driver.
Latency is measured end to end (decode, resize, network, post-processing) and every figure states the power mode.

---

## 3.2 Detection subsystem results

**Accuracy.** On the DeepFish validation split (held-out sites), YOLOv8s trained on all three datasets scores a mean of **0.812 mAP@50** and **0.567 mAP@50-95** over three seeds (range 0.041), against 0.750 and 0.514 for
YOLOv8n; every YOLOv8s seed is above every YOLOv8n seed (F32). Training data was the largest lever: DeepFish alone 0.473, adding Kakadu 0.698, adding OzFish 0.750 for YOLOv8n (three seeds each, F18, F24), while higher resolution, stronger augmentation, a finer detection head and test-time tiling gave no further gain (F12, F23, F25 to F27).

**Cross-dataset.** On the sealed OzFish test set (352 frames, run once on the three YOLOv8s seeds) the mean is **0.700 mAP@50** and **0.380 mAP@50-95** (F33). This set comes from new deployments of the same OzFish footage, not an unrelated source, so it
measures the domain shift between deployments and not a new site type. The earlier DeepFish-only models scored 0.217 on the same set (F2).

**What the detector gets wrong.** The detector misses fish rather than mislocating them: about 72% of fish are found on DeepFish validation and about 61% on the crowded OzFish set. Large fish (over 96 px) are found almost always (99%), medium fish about 77%, small fish (16 to 32 px) about 29%,
and fish under 16 px are never found (F10, F32). **Counts from this detector are therefore biased, and the sign depends on the statistic**: per-frame totals are about a fifth too low on DeepFish validation (79 to 84% of fish counted by YOLOv8s, F36) and lower still in crowded scenes and for small fish, while the clip-level maximum is over-estimated (119 to 150% of the true MaxN) because false alarms inflate the largest frame. This is relevant for an agent that treats the count as an observation: a windowed median is safer than a maximum. The default 0.25 confidence threshold gives very
different false-alarm counts between seeds (80 to 601), so the operating threshold must be set per deployed model on validation footage (F32).

**On the Orin Nano** (25 W mode, clocks not locked, fan on). End to end, the YOLOv8n engine takes about 41 ms per frame (about 24 fps) with image decode, and about 34 fps on video; a 30-minute video run held the frame rate with no throttling
and a junction temperature near 57 C (F8, F9). The YOLOv8s engine takes about 50 ms per frame and runs at about 24 fps on video, also without throttling (F28, F29). The faster MAXN_SUPER mode did not speed up the network (F31). **[F34: seed 2 engine results pending]**
These are single short runs at one ambient temperature that was not recorded.

**Status and next steps.** The detector has not been evaluated on footage from the target deployment; its training data are tropical rivers and reefs, and an informal test on a reef photograph from outside the datasets showed it finds large fish and many small ones
but misses fish in dense, dark schools (not a measured result). **[CHECK: what footage the FishIoT site will produce; if it differs strongly, retraining or fine-tuning on a sample of it is the next step]**

---

## Insert in 4/5 (Discussion, Conclusion): one sentence

The detector's accuracy was measured on footage from other sites, so the observation it provides should be expected to degrade, and to under-count, when the camera and water conditions of the deployment differ from the training data.
