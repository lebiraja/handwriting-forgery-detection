# VERSIONS

SemVer for the code in `train/`. Unreleased log below; no tagged release yet.

## Unreleased (0.1.0 in progress)
- Added: page cache builder, manifests with leakage checks, ink-centred patch sampler with augmentation, content-matched CVL batches, kNN writer mining.
- Added: ResNet-18 attention-pooling page encoder, SupCon loss, DINOv2 feature and relational distillation, EMA, resume, OOM/NaN handling.
- Added: baselines (random CNN, frozen DINOv2, HOG), evaluation (multi-reference fusion, calibration, 3-way verdict, Firemaker stress tests).
- Fixed: `--set` override bug that resumed another run's checkpoint (see FIXES.md).
- Image: `handwriting-train:0.1` on `laya-laya:latest`.
- Added: `src/report.py` (figures and tables from runs/), TikZ architecture figure, eval raw-score dump and eligible-writer counts.
- Added: `train/queue.sh` / `queue_resume.sh`, repository initialised (master).
- Trained: main seeds 0-2 and the 40-epoch reference run; six ablations pending.
- Planned for 0.1.0: `src/report.py`, synthetic-page false-accept test in `eval.py`, final test evaluation.
- Added: `src/verdict.py` (three-way verdict from saved scores), `src/multiref.py` (controlled multi-reference study), synthetic-page false-accept test in `eval.py`, `train/final_eval.sh`.
- Fixed: `eval.three_way` overlapping thresholds on separable corpora; multi-reference trials confounded with writer set (see FIXES.md).
- Added: `demo/` (FastAPI verification demo, CPU) and `train/dashboard/` (live training dashboard).
