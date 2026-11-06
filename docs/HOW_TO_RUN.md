# How to run (GPU and CPU)

All commands are from the repository root unless stated. Training and evaluation run in Docker (`handwriting-train:0.1`, built from `train/Dockerfile`).

## What runs where
| Task | GPU | CPU |
|---|---|---|
| Train (`src.train`), evaluate (`src.eval`, `multiref`, `baselines`, `explain`) | **yes (required)** | not supported: the device is hard-coded to CUDA in these modules |
| Unit tests, cache and manifest building (`prep_cache`, `manifests`) | not needed | yes |
| `verdict.py`, `report.py`, paper build | not needed | yes |
| **Demo service (verify two pages)** | not needed | **yes (default, CPU only)** |

## GPU (training and evaluation)
Requirements: NVIDIA driver, Docker with the NVIDIA container runtime, image `handwriting-train:0.1` (`cd train && docker compose build`).
```
cd train
docker compose run --rm train python -m pytest -q -p no:cacheprovider tests          # tests (need datasets/cache)
./run_train.sh main_s0 --set seed=0                                                    # one training run (resumes if the run exists)
nohup ./queue_resume.sh > runs/queue.log 2>&1 &                                        # the six ablations
docker compose run --rm train python -m src.baselines --split val --names b0 b2 b3
docker compose run --rm train python -m src.eval --run /app/runs/main_s0 --out /app/runs/main_s0/eval_val.json
./final_eval.sh                                                                        # FINAL test evaluation: run once, do not tune after
```
Single GPU job at a time (`run_train.sh` uses a lock). About 3.5 GB VRAM and 26 s per epoch on an RTX 4050 (6 GB). Live view: `python3 train/dashboard/serve.py` then http://localhost:8765.

## CPU (demo, tests, reports)
Demo (upload a reference page and a questioned page, get a verdict):
```
cd demo && docker compose up -d                       # http://localhost:8010 (CUDA hidden, 3 CPUs)
docker compose run --rm demo python -m demo.calibrate # optional: refit thresholds on VALIDATION pages
```
API: `GET /api/health`, `GET /api/info`, `POST /api/verify` (multipart: `reference` x1-5, `questioned`). About 2 to 3 s per comparison.

Reports and paper (no GPU):
```
python3 -m venv /tmp/rv && /tmp/rv/bin/pip install matplotlib scipy scikit-learn numpy
/tmp/rv/bin/python train/src/verdict.py --runs train/runs
/tmp/rv/bin/python train/src/report.py --runs train/runs --out paper
cd paper && latexmk -pdf main.tex
```
To make training itself run on CPU, the `"cuda"` device and `torch.autocast("cuda")` calls in `train.py`, `eval.py`, `baselines.py`, `explain.py`, `multiref.py` would need a device switch; this has not been done or tested, and CPU training would be far slower.

## Data
Datasets are not in git (`datasets/`, third-party licences): IAM and CVL need registration, Firemaker a request (see `datasets/DATASETS.md`). Pages are cached with `src.prep_cache`, then `src.manifests` writes `datasets/cache/manifests/master.csv`. Model weights go in `models/weights/` (DINOv2-small teacher). Secrets live in `.env` (git-ignored).

## Pitfalls
- Never re-run `src.eval --test` to change numbers; the multi-reference and verdict results come from `multiref.py` and `verdict.py` (see `FIXES.md`).
- `--resume` is always on in `run_train.sh`: use a new run name for a new run. Do not `pkill -f` a pattern that appears in your own command.
