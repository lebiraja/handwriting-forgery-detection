# FIXES

## 2026-11-05 23:44 - seed-1 run silently resumed seed-0's checkpoint
- **Symptom:** `main_s1.log` printed "resumed at epoch 110" and continued `main_s0` with seed 1; `main_s0/config.yaml` got `seed: 1`, `last.pt` and `metrics.jsonl` were extended.
- **Root cause:** `train/src/train.py:79` declared `--set` with `nargs="*"`. `run_train.sh` passes `--set out_dir=...` and the queue passes a second `--set seed=1 ...`; argparse keeps only the last `--set`, so `out_dir` fell back to `main_s0` from `config.yaml` and `--resume` loaded its `last.pt`.
- **Fix:** `action="extend"` on `--set` (`train.py:79`), plus an assertion that a resumed checkpoint has the same seed (`train.py`, resume block) and the seed is now stored in `last.pt`.
- **Repair:** `main_s0/best.pt` and `best_val.json` were untouched (no validation improvement). Removed 2 extra log lines, restored `seed: 0` and `num_workers: 8` in its `config.yaml`, renamed the overwritten `last.pt` to `last_epoch111_contaminated.pt`.
- **Verification:** argparse check returns all three `--set` values; relaunched `main_s1` shows no "resumed", writes to `runs/main_s1`, starts at epoch 0.
