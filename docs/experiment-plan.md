# Experiment plan and progress

Approved 2026-11-05 (about 22:10 IST). Budget: 10 hours wall-clock to a compiled paper. Source plan: `/home/lebi/.claude/plans/make-a-detailed-plan-encapsulated-matsumoto.md`. Legend: DONE, RUNNING, TODO, DROPPED.

## Phase 0: environment and data (0:00-0:45)  DONE
- DONE Reuse `laya-laya` image; build `handwriting-train:0.1` (15 s).
- DONE IAM, CVL (resumed after Zenodo throttling, md5 verified), Firemaker, Kaggle sets downloaded; Indic and CEDAR paused.
- DONE Cache and manifests (see `data.md`); leakage checks pass.

## Phase 1: code, tests, baselines (to 1:45)  DONE
- DONE `data.py`, `model.py`, `train.py`, `eval.py`, `baselines.py`; 6 unit tests pass.
- DONE Leak test: with labels shuffled inside each batch the loss stays at chance (ln 24 = 3.135) and nothing is learned. (A first version that permuted writer ids was invalid and replaced; see DECISIONS.md.)
- DONE Baselines on validation (EER IAM / CVL): random CNN 22.4% / 34.3%; random features with adapted BatchNorm 8.5% / 21.3%; HOG 14.1% / 23.6%; frozen DINOv2-small 5.0% / 16.0%.
- DROPPED Gate G2, AI-imitation forgery generation with DiffusionPen (needs IAM word images, SD VAE, page composition; GPU busy). Fallback is a synthetic-page false-accept test with the Kaggle generated set.

## Phase 2: main model (to 2:30)  DONE for seed 0
- DONE `main_s0`: ResNet-18 from scratch + DINOv2 distillation, 58 min, early stop at epoch 109, best epoch 79. Validation EER IAM 0.48% (AUC 0.9998), CVL 5.9% (AUC 0.984); mean 3.19%. Gate G3 (IAM EER < 35%) passed.
- DONE Evaluation code run on validation: on CVL, multi-reference helps (N=1: 4.6%, N=3 centroid: 2.95%).

## Phase 3: seeds and ablations (2:30-6:15)  PARTLY DONE; remaining runs resume in the morning (`train/queue_resume.sh`)
Order, one GPU job at a time:
| Run | Setting | Epochs | Status |
|---|---|---|---|
| main_s1, main_s2 | seeds 1, 2 | 120 (early stop) | DONE (val EER IAM/CVL 1.20/5.94 and 1.03/5.59) |
| abl_main40 | main config, 40 epochs (ablation reference) | 40 | DONE |
| abl_no_distill | distill off | 40 | TODO |
| abl_mean_pool | mean pooling | 40 | TODO |
| abl_no_cmatch | CVL batches without content matching | 40 | TODO |
| abl_no_mining | p_mine = 0 | 40 | TODO |
| abl_k8 | 8 patches per page | 40 | TODO |
| abl_no_aug | augmentation off | 40 | TODO |
Estimates: 120-epoch run about 58 min; 40-epoch ablation about 20 min. Not planned: ArcFace comparison, DINOv2-base teacher (dropped for time).
Inference-only ablations (free): fusion rules, N in {1,2,3,5}, TTA, K_eval.

## Phase 4: final evaluation (6:15-7:15)  TODO
- TODO Evaluate all seeds on **test + Firemaker** exactly once (`src.eval --test`), thresholds and calibrators from validation only.
- TODO Firemaker: p1 vs p4 (natural), p2 (upper case), p3 (disguise stress test, reported separately from forgery).
- TODO Synthetic-page false-accept rate with the Kaggle generated set.
- TODO Baselines on test (B0-B3), bootstrap 95% CIs over writers, calibration (ECE), 3-way verdict coverage and error, attention heatmaps and deletion test.
- TODO `src/report.py` to generate every table and figure from `runs/*.json` (no hand-typed numbers).

## Phase 5: paper (7:15-9:15, drafting overlaps phase 3)  IN PROGRESS
See `paper-plan.md`.

## Gates
G1 CVL available: passed. G2 generator: dropped. G3 IAM val EER < 35%: passed (0.48%). G4 freeze experiments at 6:15. G5 freeze text at 9:00.

## After the paper (not in the 10-hour budget)
- Resume the paused downloads (IIIT Indic Hindi/Tamil/Telugu/Malayalam about 19 GB, PHDIndic_11 via IEEE login, CEDAR via HF token). Pick the four languages with Lebi.
- Retrain/finetune for Indic scripts, cross-script tests, and own volunteer-copied forgery collection per language.
- Optional: real AI-imitation forgeries with DiffusionPen / VATr++ using IAM word images.
