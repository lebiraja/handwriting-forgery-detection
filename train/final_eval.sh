#!/bin/bash
# Final evaluation, run ONCE: test + Firemaker + synthetic pages per seed, test baselines, attention/deletion test.
# Thresholds and calibrators come from validation only. Do not tune anything after this runs.
cd /home/lebi/Projects/lincy/final-year/train || exit 1
run() { flock /tmp/handwriting_gpu.lock nice docker compose run --rm train "$@"; }
for s in 0 1 2; do
  run python -m src.eval --run /app/runs/main_s$s --out /app/runs/main_s$s/eval_test.json --test || echo "FAILED eval s$s"
done
run python -m src.baselines --split test --names b0 b2 b3 || echo "FAILED baselines"
run python -m src.explain --run /app/runs/main_s0 --out /app/runs/main_s0/explain --split val || echo "FAILED explain"
echo FINAL_EVAL_DONE > runs/final_eval.done
