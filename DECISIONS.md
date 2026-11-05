# DECISIONS (append-only)

## 2026-11-05
- English only for the first paper; Indic languages after.
- IEEEtran journal format. CNN trained from scratch; knowledge distillation from frozen DINOv2-small allowed, student is the shipped model.
- Forgery proxies: other writers (content-matched for CVL), AI-generated pages, Firemaker disguise as a separate stress test.
- IAM single-page writers used for training only, as top/bottom halves.
- CVL German text 6 excluded (English scope).
- CORRECTION: earlier note that Firemaker page 3 is an imitation was wrong; the README says self-disguise.
- MIN_INK lowered 0.005 -> 0.0025 after 34 CVL pages with thin pens were wrongly marked blank.
- Label-shuffle test as first written (permuting writer IDs) was invalid because it preserves grouping; replaced by per-batch label shuffling.
- Novelty check (web search, 2026-11-05): teacher-student distillation already exists in writer verification (Priya et al. 2025, GCN) and signature verification (arXiv 2308.01136), so distillation is described as an adopted technique, not a contribution. Calibrated LR with an inconclusive outcome has forensic precedent (CEDAR-FOX, Chu NIJ); score-level fusion has older precedent. The claimed contribution is the combination (page-level, deep, multi-reference, content-matched negatives, cross-dataset and AI-forgery evaluation); no writer-ID work with content-matched negatives was found. Wording stays "to our knowledge".
- Gate G2 failed (2026-11-05 23:50): DiffusionPen picks a writer's style from IAM word-level images (7 GB `.pt` files) and needs the Stable Diffusion VAE and a new dependency layer, plus composing full forged pages; the GPU is also occupied by the seed/ablation queue. Skipped. No AI imitation of specific writers is evaluated; the paper says so. Fallback: the Kaggle real-vs-generated set, whose "generated" pages are font-rendered text (folder names such as ccursive, schoolbell2, better grade), used only to measure how often synthetic pages are accepted against real references (a synthetic-impostor false-accept rate). It must not be described as AI-forgery detection.
- Seed/ablation queue changed: seeds 1-2 at 120 epochs (early stopping), ablations at 40 epochs against a 40-epoch baseline (abl_main40).
- 2026-11-06 02:00: the remaining six ablations and the final evaluation are deferred to the morning at the user's request; the queue driver was killed after `abl_main40` finished and `train/queue_resume.sh` was written. The laptop is shut down overnight. No test-set or Firemaker number has been looked at yet.
- 2026-11-06: git repository initialised on branch `master` with no remote; initial commits are made directly on master because there is nothing to branch from. Data, model weights, checkpoints, third-party clones, paper PDFs and `.env` are git-ignored.
