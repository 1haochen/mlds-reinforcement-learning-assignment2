# MLDS Assignment 2: Deep Q-Network (CartPole-v1, MsPacman-v5)

My own DQN implementation in PyTorch. It has an online Q-network, a uniform replay buffer and a hard-copied **target network**. On top of that it uses Double-DQN targets, Huber loss, gradient clipping, a linear learning-rate schedule and a piecewise-linear ε schedule.
CartPole-v1 uses γ = 0.95, and MsPacman-v5 uses γ = 0.99 with the assignment's `preprocess_observation` plus a 4-frame stack.

The full write-up with every figure and the discussion is in **`Assignment2_Report.pdf`**.

## Results

| | CartPole-v1 (γ = 0.95) | MsPacman-v5 (γ = 0.99) |
|---|---|---|
| Training | 600 episodes (main config, seed 1) | 4,614 episodes / 3.13M steps, 2.5 h on one RTX 2080 Ti |
| 500 held-out greedy episodes | **497.3 ± 34.3** (99.4% reach 500, solved) | **1,664.8 ± 497.1** (random policy: 233) |
| Checkpoint | `results/cartpole_s1/best.pt` | `results/mspacman/best.pt` |

## Files

| File | Purpose |
|---|---|
| `dqn.py` | DQN: networks, replay buffer, training loop (vectorised envs), evaluation (`train` / `evaluate` sub-commands) |
| `validate.py` | Validation experiments with expected outcomes: analytic chain MDP, terminal masking, pipeline edge cases, robustness / perturbation tests |
| `plots.py` | Builds every figure in `figures/` from the logged CSV/JSON files |
| `run_all.sh` | Launches the runs used in the report: MsPacman (main + no-target ablation, one GPU each) and CartPole (3 seeds × {main, no-double, no-target}) |
| `sweep_cartpole.sh` | CartPole hyper-parameter sweep (configs A–F × 3 seeds) |
| `summarize.py` | CartPole stability table across all runs |
| `finalize.sh` | Post-training: 500-episode evaluations, behavioural tests, figures |
| `build_report.py` | Fills every number in the report from the result files and writes `report_full.md` |
| `results/` | Training logs (`train.csv`, `validation.jsonl`), configs, best checkpoints, 500-episode evaluations, validation JSON |
| `figures/` | All report figures |
| `report.md`, `report_body_*.md` | Report source: `python3 build_report.py && pandoc report_full.md -o Assignment2_Report.pdf --pdf-engine=xelatex` |

## How to run

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt

# 1) Analytic / pipeline tests (CPU, ~15 s)
.venv/bin/python validate.py analytic

# 2) Training (on a deepdish node; each MsPacman run is pinned to ONE GPU)
GPU_MAIN=0 GPU_ABL=1 TIME=9000 bash run_all.sh   # MsPacman main + ablation, CartPole seeds/ablations
bash sweep_cartpole.sh                            # CartPole hyper-parameter sweep

# single runs, e.g.
.venv/bin/python dqn.py train --game cartpole --out results/my_cp --episodes 1000
CUDA_VISIBLE_DEVICES=0 .venv/bin/python dqn.py train --game mspacman --out results/my_mp --device cuda --num-envs 8

# 3) 500-episode evaluation of a checkpoint (held-out seeds 100000-100499, greedy policy)
.venv/bin/python dqn.py evaluate --checkpoint results/cartpole_s1/best.pt --episodes 500
.venv/bin/python dqn.py evaluate --checkpoint results/mspacman/best.pt --episodes 500 --workers 20

# 4) Behavioural tests on the trained agents
.venv/bin/python validate.py cartpole --checkpoint results/cartpole_s1/best.pt
.venv/bin/python validate.py mspacman --checkpoint results/mspacman/best.pt
.venv/bin/python validate.py failure_modes --checkpoint results/cartpole_s{0,1,2}/best.pt results/cartpole_s{0,1,2}/last.pt

# 5) Figures
.venv/bin/python plots.py --cartpole results/cartpole_s1 --mspacman results/mspacman --out figures

# 6) Report
python3 build_report.py && pandoc report_full.md -o Assignment2_Report.pdf --pdf-engine=xelatex
```

## What each validation test checks

| Test | What it does | Expected outcome if the algorithm is correct |
|---|---|---|
| T1 chain MDP (`validate.py analytic`) | Trains the *same* replay/target/update code on a 4-state corridor with a closed-form Q* for γ ∈ {0, 0.5, 0.95, 0.99} | Learned Q equals γ^(3-s) etc. to < 1e-2; greedy action is optimal |
| T1b broken terminal mask | Same MDP but the terminal flag is ignored | Values drift toward 1/(1−γ); shows the mask is what makes T1 pass |
| T2 terminated vs truncated | TD target with identical inputs but different flags | terminal → r; truncated (time limit) → r + γV(s') |
| T3 pipeline edge cases | Pac-Man colour removed, no uint8 overflow, stack initialisation, vector-env autoreset uses the *true* final frame, replay reconstructs s' exactly | all `true` |
| T4 policy noise (`validate.py cartpole/mspacman`) | Inject ε-random actions at test time | return decreases monotonically with ε; ε = 1 matches a random policy |
| T5 CartPole physics perturbation | Change pole length / mass / force / gravity | best near the training physics, gradual degradation away from it |
| T6 mirror symmetry | Q(s) vs Q(−s) | confident actions flip left ↔ right |
| T7 Q-level vs theory | CartPole Q of a never-failing policy | → 1/(1−0.95) = 20 |
| Ablations | no target network; vanilla max target | no-target is unstable, Double DQN reduces over-estimation |
