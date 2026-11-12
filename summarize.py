"""Print the CartPole stability table (markdown) used in the report from results/*/validation.jsonl."""
import glob
import json
from pathlib import Path

import numpy as np

GROUPS = [('main (lr 1e-3, tgt 500)', 'cartpole_s'), ('no Double DQN', 'cartpole_nodouble_s'),
          ('no target network', 'cartpole_notarget_s'), ('A: lr 5e-4, batch 128, tgt 1000', 'sweep_A_s'),
          ('B: lr 2.5e-4, batch 128, tgt 2000', 'sweep_B_s'), ('C: lr 1e-3, 2 updates/step', 'sweep_C_s'),
          ('D: lr 3e-4, batch 256, 1500 ep', 'sweep_D_s'), ('E: lr 1e-4 const, tgt 250, 1500 ep', 'sweep_E_s'),
          ('F: E + dueling head', 'sweep_F_s')]

print('| Configuration | seeds with a 500 checkpoint | first 500 (episode, per seed) | '
      'validation checks at 500 | best val. (per seed) | mean val., 2nd half |')
print('|---|---|---|---|---|---|')
for label, prefix in GROUPS:
    firsts, hits, checks, bests, late = [], 0, 0, [], []
    runs = sorted(glob.glob(f'results/{prefix}[0-9]'))
    for d in runs:
        v = [json.loads(x) for x in open(Path(d) / 'validation.jsonl')]
        m = np.array([x['mean'] for x in v])
        first = next((x['episode'] for x in v if x['mean'] >= 500), None)
        firsts.append(str(first) if first else '–')
        hits += first is not None
        checks += int((m >= 500).sum())
        bests.append(f'{m.max():.0f}')
        late.append(m[len(m) // 2:].mean())
    print(f'| {label} | {hits}/{len(runs)} | {", ".join(firsts)} | {checks} | {", ".join(bests)} | '
          f'{np.mean(late):.0f} |')
