#!/usr/bin/env bash
# CartPole hyper-parameter sweep (configs A-C x 3 seeds), run on CPU.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
mkdir -p results logs
BASE="--game cartpole --episodes 1000 --capacity 100000 --warmup 1000 --eps-mid 0.05 --eps-end 0.01
      --lr-final-frac 0.1 --check-every 10 --val-episodes 10 --device cpu"
declare -A CFG=(
  [A]="--lr 5e-4 --lr-decay-steps 2e5 --batch 128 --target-every 1000 --eps-steps 20000"
  [B]="--lr 2.5e-4 --lr-decay-steps 3e5 --batch 128 --target-every 2000 --eps-steps 20000"
  [C]="--lr 1e-3 --lr-decay-steps 5e4 --batch 64 --target-every 1000 --eps-steps 10000 --updates-per-step 2"
  [D]="--lr 3e-4 --lr-decay-steps 4e5 --lr-final-frac 0.05 --batch 256 --target-every 500 --eps-steps 20000 --episodes 1500"
  [E]="--lr 1e-4 --lr-final-frac 1.0 --batch 128 --target-every 250 --eps-steps 20000 --episodes 1500"
  [F]="--lr 1e-4 --lr-final-frac 1.0 --batch 128 --target-every 250 --eps-steps 20000 --episodes 1500 --dueling"
)
CONFIGS=${CONFIGS:-"${!CFG[@]}"}
for c in $CONFIGS; do
  for seed in 0 1 2; do
    OMP_NUM_THREADS=2 nohup $PY dqn.py train $BASE ${CFG[$c]} --seed $seed --out results/sweep_${c}_s$seed \
      > logs/sweep_${c}_s$seed.log 2>&1 &
  done
done
echo launched
