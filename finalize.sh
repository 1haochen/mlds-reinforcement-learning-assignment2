#!/usr/bin/env bash
# After training: 500-episode evaluations, behavioural tests and figures.
#   bash finalize.sh <cartpole run dir>
set -euo pipefail
cd "$(dirname "$0")"
PY=${PY:-.venv/bin/python}
CP=${1:?cartpole run dir, e.g. results/sweep_E_s1}
export OMP_NUM_THREADS=1
$PY dqn.py evaluate --checkpoint "$CP/best.pt" --episodes 500 --workers 16
$PY validate.py analytic
$PY validate.py cartpole --checkpoint "$CP/best.pt"
if [[ -f results/mspacman/best.pt ]]; then
  $PY dqn.py evaluate --checkpoint results/mspacman/best.pt --episodes 500 --workers 24
  $PY validate.py mspacman --checkpoint results/mspacman/best.pt
fi
$PY plots.py --cartpole "$CP" --mspacman results/mspacman --out figures
