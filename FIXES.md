# FIXES

## 2026-11-06 - three-way verdict overlapped on IAM; multi-reference curve confounded with writer set
- **Symptom 1:** `eval_test.json["three_way"]["iam"]` had tau_high 0.85 < tau_low 0.92 and accepted + rejected shares summing to over 100% (50.2% + 52.2%).
- **Root cause 1:** `train/src/eval.py` `three_way`: IAM is separable on validation, so the 99th percentile of impostor scores lies below the 1st percentile of genuine scores; the accept and reject regions overlap.
- **Fix 1:** if `hi < lo`, use one midpoint threshold and no inconclusive band (`eval.py` `three_way`); saved test scores re-scored by new `train/src/verdict.py` (no model re-run; thresholds from validation only). IAM verdict error 0.7%, CVL unchanged (8.5% inconclusive, 2.8% error).
- **Symptom 2:** IAM test EER at N=1 was 6.3% in the multi-reference trials but 2.2% over all page pairs; the N curve fell steeply (6.3, 2.4, 0.6, 0.7).
- **Root cause 2:** `eval.py` `make_trials` draws genuine queries only from writers with >= N+1 pages, so the writer set shrinks with N (99, 56, 45, 17 on IAM), confounding N with writer difficulty.
- **Fix 2:** new `train/src/multiref.py` evaluates every N on one fixed writer set with nested references and shared impostor queries (IAM 45 writers, CVL 62; extra N=5 set of 17 IAM writers). IAM centroid EER 2.25, 1.57, 1.22 for N = 1, 2, 3. The paper uses only this version.
- **Verification:** IAM N=1 on the fixed set (2.25) now agrees with the all-pairs EER (2.35); accepted + rejected + inconclusive sums to 100%. See DECISIONS.md.

## 2026-11-05 23:44 - seed-1 run silently resumed seed-0's checkpoint
- **Symptom:** `main_s1.log` printed "resumed at epoch 110" and continued `main_s0` with seed 1; `main_s0/config.yaml` got `seed: 1`, `last.pt` and `metrics.jsonl` were extended.
- **Root cause:** `train/src/train.py:79` declared `--set` with `nargs="*"`. `run_train.sh` passes `--set out_dir=...` and the queue passes a second `--set seed=1 ...`; argparse keeps only the last `--set`, so `out_dir` fell back to `main_s0` from `config.yaml` and `--resume` loaded its `last.pt`.
- **Fix:** `action="extend"` on `--set` (`train.py:79`), plus an assertion that a resumed checkpoint has the same seed (`train.py`, resume block) and the seed is now stored in `last.pt`.
- **Repair:** `main_s0/best.pt` and `best_val.json` were untouched (no validation improvement). Removed 2 extra log lines, restored `seed: 0` and `num_workers: 8` in its `config.yaml`, renamed the overwritten `last.pt` to `last_epoch111_contaminated.pt`.
- **Verification:** argparse check returns all three `--set` values; relaunched `main_s1` shows no "resumed", writes to `runs/main_s1`, starts at epoch 0.
