# Docker and run commands

Last updated: 2026-11-05 23:55 IST.

## Image
`handwriting-train:0.1`, built from `train/Dockerfile` on top of the local image **`laya-laya:latest`** (torch 2.6.0+cu124, torchvision, PIL; CUDA verified on the RTX 4050). Added only: `timm==1.0.11`, `opencv-python-headless==4.10.0.84`, `scikit-learn==1.6.1`, `pandas==2.2.3`, `pytest==8.3.3`. Torch is not reinstalled. Entry point is cleared, default command is `python -m src.train --config src/config.yaml`.

## Compose (`train/docker-compose.yml`)
| Item | Value |
|---|---|
| user | `1000:1000`, `HOME=/tmp` (no root-owned files in `runs/`) |
| GPU | `deploy.resources.reservations.devices` nvidia x1 |
| shm_size | 4 GB (DataLoader workers) |
| `../datasets` -> `/data` | read-only |
| `../models/weights` -> `/models` | read-only (DINOv2 teacher) |
| `./runs` -> `/app/runs` | read-write |
| `./src`, `./tests` | mounted so code edits need no rebuild |
| env | `PYTORCH_ALLOC_CONF=expandable_segments:True` |

No ports, no network services, no secrets. Training is a batch job, so there is no healthcheck. Tokens live only in `final-year/.env` (mode 600, git-ignored) and are not mounted into the container.

## Hardware
RTX 4050 Laptop (6 GB), i7-13650HX (20 threads), 15 GB RAM. Measured: about 3.5 GB VRAM and 25-28 s per epoch (70 steps) with distillation; GPU waits on CPU data loading at times.

## Commands
Run from `final-year/train/`.
```
# unit tests (CPU)
docker compose run --rm train python -m pytest -q -p no:cacheprovider tests

# build caches (CPU, niced)  -- datasets: iam | cvl | fm | kg
docker run --rm --user 1000:1000 -e HOME=/tmp -v $PWD/../datasets:/data -v $PWD/src:/app/src handwriting-train:0.1 \
  python -m src.prep_cache --dataset iam --root /data/kaggle_iam-handwritten-forms-dataset --out /data/cache --workers 8
# same docker run form as above (write access to /data needed):  python -m src.manifests --cache /data/cache

# one training run (queued behind any other GPU job)
./run_train.sh <run_name> --set seed=1 num_workers=12

# whole queue (seeds 1-2 then ablations)
nohup ./queue.sh > runs/queue.log 2>&1 &

# remaining ablations (morning resume)
nohup ./queue_resume.sh > runs/queue.log 2>&1 &

# baselines and evaluation
docker compose run --rm train python -m src.baselines --split val --names b0 b2 b3
docker compose run --rm train python -m src.eval --run /app/runs/main_s0 --out /app/runs/main_s0/eval_val.json          # val only
docker compose run --rm train python -m src.eval --run /app/runs/main_s0 --out /app/runs/main_s0/eval_test.json --test  # FINAL only
```
`--set` accepts several `key=value` pairs and may be repeated (`nested.key=value` for `distill.enabled=false`). A resume refuses a checkpoint from a different seed.

## Plots and tables
```
python3 -m venv /tmp/rv && /tmp/rv/bin/pip install matplotlib scipy scikit-learn numpy
/tmp/rv/bin/python train/src/report.py --runs train/runs --out paper      # reads train/runs/*, writes paper/figures and paper/tables
cd paper && latexmk -pdf main.tex
```
Architecture diagram is TikZ (`paper/figures/architecture.tex`), compiled with the paper.

## Live logs
```
tail -f train/runs/<run>.log                 # raw
tail -f train/runs/<run>_gpu.log             # temperature, throttle flags, VRAM, utilisation per minute
watch -n 1 nvidia-smi
```
Compose note: with `--set seed=1` and `--resume` already in `run_train.sh`, re-running the same `<run_name>` continues that run.

## Gotchas
- zsh does not word-split unquoted variables: use `$=VAR` or write the command out.
- Never `pkill -f` a pattern that also appears in your own command line (it kills the shell). Use `ps ... | awk '/[p]attern/'`.
- `r.view` on a pandas row is the method `Series.view`; use `r["view"]`.
- `--resume` is always on in `run_train.sh`; use a new `<run_name>` for a new run.

## Demo service (`demo/docker-compose.yml`)
Built from `demo/Dockerfile` (`handwriting-train:0.1` + pinned `huggingface_hub`). The model is **downloaded from the public HF repo `lebiraja/handwriting-forgery-pageencoder`** (`HF_REPO`) on first start and cached in `demo/.hf_cache` (git-ignored); a local `MODEL_DIR` with `best.pt` + `config.yaml` takes precedence. Device: `DEMO_DEVICE=auto` (default) uses CUDA if torch sees a GPU, else CPU; `DEMO_DEVICE=cpu` forces CPU. GPU in Docker: add `-f docker-compose.gpu.yml`. `cpus: 3`, port `127.0.0.1:8010`, healthcheck on `/api/health`. Mounts: `../train/src` (ro), `../datasets` (ro, calibration only), `./app` (rw), `./.hf_cache`.
```
cd demo && docker compose up -d --build                      # http://localhost:8010 (auto GPU/CPU)
DEMO_DEVICE=cpu docker compose up -d                          # force CPU
docker compose run --rm demo python -m demo.calibrate         # refit thresholds on VALIDATION pages, rewrite samples
```
API: `GET /api/info`, `GET /api/health`, `POST /api/verify` (multipart `reference` x1-5, `questioned`). Live training dashboard: `python3 train/dashboard/serve.py` -> http://localhost:8765.
