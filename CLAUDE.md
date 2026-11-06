# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Final-year research project (Lincy Jenita J, URK23CS5079): **page-level handwriting forgery detection**. Given N reference pages of a claimed writer and one questioned full page, output a similarity score and a verdict (`consistent` / `inconclusive` / `likely forged`). The deliverable is an IEEEtran paper (6 pages including references), not a product. English only for the first paper; Indic languages come after.

Read first: `HANDOVER.md` (newest entry on top, has the "rules that bite" list), top of `STATUS.md`, `docs/README.md` (doc index), `BLOCKERS.md`. `FIXES.md` and `DECISIONS.md` (append-only) record past bugs and deviations.

## Layout

- `train/` – all code (Python, runs only in Docker). `train/src/` modules, `train/tests/`, `train/runs/` (per-run `config.yaml`, `metrics.jsonl`, `best_val.json`, `eval_*.json`; logs and JSON are committed, checkpoints are git-ignored).
- `paper/` – LaTeX (`main.tex`, `sections/`, `tables/`, `figures/`). `tables/*.tex` and `figures/*.pdf` are **generated** by `train/src/report.py`; never hand-edit them.
- `datasets/`, `models/weights/`, `code/`, `papers/` – third-party data, weights, clones, PDFs; git-ignored. `.env` holds tokens (mode 600, never print or commit).
- `docs/paper-claims-map.md` – every number in the paper must map to a result file here.

## Commands

All run from `train/` unless noted. Image `handwriting-train:0.1` (built on local `laya-laya:latest`); `src/` and `tests/` are bind-mounted, so code edits need no rebuild. Full command set: `docs/docker.md`.

```
docker compose run --rm train python -m pytest -q -p no:cacheprovider tests                     # all tests
docker compose run --rm train python -m pytest -q -p no:cacheprovider tests -k test_name        # single test
./run_train.sh <run_name> --set seed=1 num_workers=12      # one training run (flock'd, niced, resumes if run_name exists)
nohup ./queue_resume.sh > runs/queue.log 2>&1 &            # remaining ablations; done when runs/queue.done exists
docker compose run --rm train python -m src.baselines --split val --names b0 b2 b3
docker compose run --rm train python -m src.eval --run /app/runs/main_s0 --out /app/runs/main_s0/eval_val.json
/tmp/rv/bin/python train/src/report.py --runs train/runs --out paper   # from repo root; venv recipe in docs/docker.md
cd paper && latexmk -pdf main.tex
```

Tests read the real cached data (`/data/cache/manifests/master.csv`), so they need `datasets/cache` to exist. No linter is configured.

## Architecture (needs several files to see)

Pipeline: `prep_cache.py` (crop printed parts, normalise, width 1600, `.npy` + status CSV) → `manifests.py` (writer-disjoint splits, leakage assertions, writes `master.csv`) → `data.py` (ink-centred 128px patches, augmentation, custom `BatchSampler`) → `train.py` + `model.py` → `runs/<name>/` → `eval.py` / `baselines.py` → `report.py` → `paper/`.

- **Model:** `PageEncoder` = ResNet-18 (timm, random init, 1 channel) over K patches per page, attention pooling, 256-d L2-normalised page embedding. Loss = SupCon + 0.5 feature-distill + 0.5 relational-distill from a frozen DINOv2-small teacher (`/models`). Only the student ships; distillation is an adopted technique, not a claimed contribution.
- **Batches:** 12 writers x 2 pages; 60% IAM, 40% CVL where all writers copy the same texts (content-matched negatives); half the batches use kNN-mined nearest-neighbour writers (centroids refreshed every 10 epochs).
- **Evaluation:** multi-reference (N = 1,2,3,5) with fusion rules mean/max/top2/centroid; Platt/isotonic calibration and verdict thresholds (1st/99th percentile) fitted on **validation only**; Firemaker is a test-only cross-dataset stress test; Kaggle "generated" pages measure synthetic-impostor false-accept rate.
- **Single GPU:** RTX 4050 (6 GB). `run_train.sh` serialises jobs with `flock /tmp/handwriting_gpu.lock`; take the same lock for any other GPU job (e.g. `src.explain`).

## Rules that bite

- Test sets and Firemaker are evaluated **once, at the end** (`--test`); never tune after seeing them. Firemaker is never used for training or tuning.
- Firemaker p3 is self-disguise, not impersonation; Kaggle "generated" pages are fonts. Never call either AI forgery.
- Paper numbers come only from `runs/*/eval*.json` via `report.py`; none hand-typed. `paper/tables/tab_main.tex` currently holds validation placeholders and is not included.
- `--resume` is always on in `run_train.sh`: use a new run name for a new run; resume refuses a different seed. `--set` takes repeated `key=value` (`distill.enabled=false` for nested).
- IAM validation EER is optimistic (shared scanner/pen); lead with Firemaker and CVL in discussion. IAM data is a third-party Kaggle copy (see BLOCKERS.md).
- `explain.py` has never been run; expect to debug it.
- Pandas rows: use `r["view"]`, not `r.view`. zsh: use `$=VAR` to split. Never `pkill -f` a pattern that appears in your own command.
- Per global preferences: docs in `docs/` stay in sync with code; prepend `STATUS.md` / `FIXES.md` / `HANDOVER.md` entries after real changes.
