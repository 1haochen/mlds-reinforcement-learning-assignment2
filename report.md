---
title: "Assignment 2: Deep Q-Network"
subtitle: "MLDS Advanced Algorithms for AI: CartPole-v1 and MsPacman-v5"
author: "Yihao Chen"
date: "November 11, 2026"
geometry: margin=0.9in
fontsize: 10pt
colorlinks: true
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

# Submission information

| | |
|---|---|
| GitHub username | `1haochen` |
| Repository | `mlds-reinforcement-learning-assignment2` |
| Public link | <https://github.com/1haochen/mlds-reinforcement-learning-assignment2> |
| Hardware | deepdish3, **one** NVIDIA RTX 2080 Ti per MsPacman run (`CUDA_VISIBLE_DEVICES`), CPU for CartPole |
| Software | Python 3.10, PyTorch 2.5.1, Gymnasium 1.2.0, ale-py 0.11.2 |

## How to run the code

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
.venv/bin/python validate.py analytic                        # T1-T3 analytic & pipeline tests (~15 s, CPU)
GPU_MAIN=0 GPU_ABL=1 TIME=9000 bash run_all.sh                # all MsPacman + CartPole runs (1 GPU per MsPacman run)
bash sweep_cartpole.sh                                       # CartPole hyper-parameter sweep A-E x 3 seeds
.venv/bin/python dqn.py evaluate --checkpoint <run>/best.pt --episodes 500 [--workers 16]
.venv/bin/python validate.py cartpole --checkpoint <cartpole best.pt>   # T4-T7
.venv/bin/python validate.py mspacman --checkpoint results/mspacman/best.pt
.venv/bin/python plots.py --cartpole <cartpole run> --mspacman results/mspacman --out figures
```

Each test's purpose and expected outcome is listed in `README.md` and in Section 4.

%%BODY%%
