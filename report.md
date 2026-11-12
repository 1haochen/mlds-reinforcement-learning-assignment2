---
title: "Assignment 2: Deep Q-Network"
subtitle: "MLDS Advanced Algorithms for AI: CartPole-v1 and MsPacman-v5"
author: "Yihao Chen"
date: "November 11, 2026"
geometry: margin=0.9in
fontsize: 10pt
colorlinks: true
mainfont: "Times New Roman"
mathfont: "STIX Two Math"
monofont: "Menlo"
header-includes:
  - \usepackage{float}
  - \floatplacement{figure}{H}
---

# Submission information

| Item | Value |
|---|---|
| GitHub username | `1haochen` |
| Repository | `mlds-reinforcement-learning-assignment2` |
| Public link | <https://github.com/1haochen/mlds-reinforcement-learning-assignment2> |
| Hardware | deepdish3, **one** NVIDIA RTX 2080 Ti per MsPacman run (`CUDA_VISIBLE_DEVICES`), CPU for CartPole |
| Software | Python 3.10, PyTorch 2.5.1, Gymnasium 1.2.0, ale-py 0.11.2 |

## How to run the code

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt
# T1-T3 analytic & pipeline tests (~15 s, CPU)
.venv/bin/python validate.py analytic
# all MsPacman + CartPole runs (one GPU per MsPacman run), then the CartPole sweep A-F
GPU_MAIN=0 GPU_ABL=1 TIME=9000 bash run_all.sh
bash sweep_cartpole.sh
# 500-episode greedy evaluation on held-out seeds 100000-100499
.venv/bin/python dqn.py evaluate --checkpoint results/cartpole_s1/best.pt --episodes 500
.venv/bin/python dqn.py evaluate --checkpoint results/mspacman/best.pt --episodes 500 \
    --workers 20
# behavioural tests T4-T8
.venv/bin/python validate.py cartpole --checkpoint results/cartpole_s1/best.pt
.venv/bin/python validate.py mspacman --checkpoint results/mspacman/best.pt
.venv/bin/python validate.py failure_modes --checkpoint results/cartpole_s{0,1,2}/best.pt \
    results/cartpole_s{0,1,2}/last.pt
# figures, then this report
.venv/bin/python plots.py --cartpole results/cartpole_s1 --mspacman results/mspacman
python3 build_report.py && pandoc report_full.md -o Assignment2_Report.pdf --pdf-engine=xelatex
```

Each test's purpose and expected outcome is listed in `README.md` and in Section 5.

%%BODY%%
