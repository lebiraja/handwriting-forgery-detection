#!/bin/bash
# usage: ./run_train.sh <run_name> [--set k=v ...]   one GPU job at a time (flock), niced, sleep-inhibited
name=$1; shift
cd /home/lebi/Projects/lincy/final-year/train || exit 1
mkdir -p runs
( while true; do echo "$(date +%T),$(nvidia-smi --query-gpu=temperature.gpu,clocks_throttle_reasons.active,memory.used,utilization.gpu --format=csv,noheader)"; sleep 60; done ) >> runs/${name}_gpu.log 2>&1 &
MON=$!
flock /tmp/handwriting_gpu.lock nice systemd-inhibit --what=sleep:idle --why="handwriting training" \
  docker compose run --rm train python -m src.train --config src/config.yaml --resume --set out_dir=/app/runs/$name "$@" >> runs/${name}.log 2>&1
echo "exit $?" >> runs/${name}.log
kill $MON
