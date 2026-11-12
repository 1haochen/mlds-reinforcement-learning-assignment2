#!/usr/bin/env bash
# Launch every training run used in the report (run on a deepdish node).
# Each MsPacman run is pinned to ONE GPU card via CUDA_VISIBLE_DEVICES.
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
GPU_MAIN=${GPU_MAIN:-0}
GPU_ABL=${GPU_ABL:-1}
TIME=${TIME:-9000}          # wall-clock budget (seconds) for the MsPacman runs
mkdir -p results logs

# ---- MsPacman (main): Double DQN + target net + Huber + lr & epsilon schedules
MSP="--game mspacman --num-envs 8 --batch 32 --updates-per-step 2 --lr 1e-4 --lr-final-frac 0.2
     --lr-decay-steps 6e6 --capacity 200000 --warmup 20000 --eps-steps 1000000 --eps-mid 0.1
     --eps-end 0.01 --target-every 2500 --check-every 100 --val-episodes 5 --episodes 8000
     --time-limit $TIME --device cuda"
CUDA_VISIBLE_DEVICES=$GPU_MAIN nohup $PY dqn.py train $MSP --out results/mspacman --seed 0 \
  > logs/mspacman.log 2>&1 &
# ---- MsPacman ablation: no target network (bootstrap from the online network)
CUDA_VISIBLE_DEVICES=$GPU_ABL nohup $PY dqn.py train $MSP --out results/mspacman_notarget --seed 0 --no-target \
  > logs/mspacman_notarget.log 2>&1 &

# ---- CartPole: 3 seeds of the main config + two ablations (CPU beats GPU for a tiny MLP)
CP="--game cartpole --episodes 600 --lr 1e-3 --lr-final-frac 0.1 --lr-decay-steps 1e5 --batch 64
    --capacity 50000 --warmup 1000 --eps-steps 10000 --eps-mid 0.05 --eps-end 0.01 --target-every 500
    --check-every 10 --val-episodes 10 --device cpu"
for seed in 0 1 2; do
  for variant in "" "--no-target" "--no-double"; do
    name=cartpole${variant//--no-/_no}_s$seed
    OMP_NUM_THREADS=2 nohup $PY dqn.py train $CP --seed $seed $variant --out results/$name > logs/$name.log 2>&1 &
  done
done
echo launched
