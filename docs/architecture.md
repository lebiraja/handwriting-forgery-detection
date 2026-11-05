# Architecture

Project: **Page-level handwriting forgery detection** (English first, Indian languages later).
Authors: Lincy Jenita J (URK23CS5079); guide S D Lalitha, Associate Professor.
Last updated: 2026-11-05 23:55 IST.

## What the system does
Given N reference pages of a claimed writer and one questioned page, return a similarity score and a verdict: `consistent`, `inconclusive` or `likely forged`. The unit of evidence is a **full page**, not a word, sentence or signature.

## Data flow
```
raw pages (IAM, CVL, Firemaker, Kaggle)           datasets/<source>/
   | src/prep_cache.py   crop printed parts, background-normalise, resize to width 1600, uint8 .npy
   v
datasets/cache/<ds>/*.npy  +  cache/manifests/<ds>_pages.csv   (status ok | blank | duplicate | error)
   | src/manifests.py    writer-disjoint splits, leakage assertions
   v
datasets/cache/manifests/master.csv   (path, writer, split, dataset, cond, view, page)
   | src/data.py         ink-centred patches, augmentation, content-matched batch sampler
   v
src/train.py  --  student ResNet-18 (from scratch) + attention pooling -> 256-d page embedding
   |                loss = SupCon + 0.5*feature-distill + 0.5*relational-distill  (frozen DINOv2-small teacher)
   v
runs/<name>/{best.pt,last.pt,metrics.jsonl,config.yaml}
   | src/eval.py         single/multi-reference, fusion rules, calibration, 3-way verdict, Firemaker, synthetic test
   v
runs/<name>/eval*.json  -> src/report.py (planned) -> paper tables and figures
```

## Components
| Component | Path | Role |
|---|---|---|
| Page cache builder | `train/src/prep_cache.py` | IAM header/footer crop by long horizontal lines; CVL printed-block crop; background normalisation; blank and duplicate filtering |
| Manifest builder | `train/src/manifests.py` | Writer-disjoint train/val/test; IAM single-page writers become top/bottom half views (training only); asserts no writer overlap, Firemaker test-only, eval writers have >= 2 pages |
| Data/sampler | `train/src/data.py` | Patches centred on ink pixels with progressive ink threshold; deterministic eval patches seeded per page; TTA shift; augmentation flag; `BatchSampler` with IAM batches, content-matched CVL batches and kNN writer mining |
| Model | `train/src/model.py` | `PageEncoder` (ResNet-18 1-channel, attention or mean pooling, distillation head), `supcon_loss`, `feature_distill`, `relational_distill` |
| Training | `train/src/train.py` | AMP, EMA, cosine schedule, resume, OOM/NaN handling, SIGTERM-safe checkpoints, JSONL log, optional label-shuffle leak test |
| Metrics | `train/src/metrics.py` | EER, AUC on all page pairs |
| Evaluation | `train/src/eval.py` | Single-ref and multi-ref (N = 1,2,3,5) with fusion rules mean/max/top2/centroid; Platt and isotonic calibration; 3-way verdict; Firemaker stress tests |
| Baselines | `train/src/baselines.py` | B0 random CNN, B2 frozen DINOv2-small, B3 HOG |
| Queue | `train/run_train.sh`, `train/queue.sh` | One GPU job at a time (flock), niced, sleep-inhibited, GPU temperature log |

## Model details
- Patch 128x128, K = 16 patches per page for training, K = 64 deterministic patches for evaluation.
- ResNet-18 (timm, random init, 1 channel) -> 256-d; attention pooling `softmax(w^T tanh(V z))`; L2-normalised page embedding.
- Batches of 12 writers x 2 pages. 60% IAM batches; 40% CVL batches where all writers write the same two texts (content-matched negatives). With p = 0.5 a batch is built from nearest-neighbour writers (centroids recomputed every 10 epochs).
- Teacher: frozen DINOv2-small (Apache-2.0), used only in training. The shipped model is the student alone.
- Optimiser: AdamW lr 3e-4, wd 1e-4, 5-epoch warm-up, cosine, up to 120 epochs, grad clip 1.0, EMA 0.999, patience 30 epochs.

## Decision logic (evaluation)
- Fusion over N references: mean, max, top-2 mean, centroid cosine.
- Calibration (Platt, isotonic) and thresholds are fitted on **validation only**.
- `consistent` if score >= 99th percentile of validation impostor scores; `likely forged` if <= 1st percentile of validation genuine scores; else `inconclusive`.

## Known limits of the design
- No public data of one person imitating another. Proxies: other writers (content-matched on CVL), Firemaker self-disguise (stress test, not forgery), font-rendered synthetic pages (false-accept rate only).
- IAM validation EER is very low (0.48% for `main_s0`), partly because pages of one writer share scanner and pen cues. Firemaker (different scanner, never used for tuning) is the cross-dataset check.
- English only. Indic scripts follow after the first paper.
