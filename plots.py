"""Render every figure in the report from the logged CSV/JSON files (no data is synthesised).

  python plots.py --cartpole results/cartpole_s0 --mspacman results/mspacman --out figures
"""
import argparse
import csv
import glob
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    '#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948')
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'
plt.rcParams.update({
    'figure.facecolor': 'white', 'axes.facecolor': '#fcfcfb', 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
    'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'xtick.color': INK2, 'ytick.color': INK2, 'font.size': 9,
    'axes.titlesize': 10, 'axes.titleweight': 'bold', 'legend.frameon': False, 'lines.linewidth': 1.6})
MSPACMAN_ACTIONS = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN', 'UPRIGHT', 'UPLEFT', 'DOWNRIGHT', 'DOWNLEFT']


def read_csv(path):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return {k: np.array([float(r[k]) if r[k] not in ('', 'True', 'False') else
                         (np.nan if r[k] == '' else float(r[k] == 'True')) for r in rows]) for k in rows[0]}


def read_jsonl(path):
    return [json.loads(line) for line in open(path)] if Path(path).exists() else []


def moving_average(x, n=100):
    c = np.cumsum(np.insert(x, 0, 0.0))
    out = np.empty_like(x, dtype=float)
    for i in range(len(x)):  # trailing window; shorter at the start
        lo = max(0, i + 1 - n)
        out[i] = (c[i + 1] - c[lo]) / (i + 1 - lo)
    return out


def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    print('wrote', path)


def training_plots(run, out, name, gamma):
    t = read_csv(Path(run) / 'train.csv')
    ep = t['episode']
    # (i) max Q-values
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(ep, t['qmax_max'], color=BLUE, alpha=0.25, lw=0.7)
    ax.plot(ep, moving_average(t['qmax_max'], 50), color=BLUE, label='max over episode of max$_a$ Q(s,a)  (50-ep avg)')
    ax.plot(ep, moving_average(t['qmax_mean'], 50), color=ORANGE,
            label='mean over episode of max$_a$ Q(s,a)  (50-ep avg)')
    if name == 'CartPole-v1':
        ax.axhline(1 / (1 - gamma), color=INK2, ls='--', lw=1)
        ax.text(ep[-1], 1 / (1 - gamma) + 0.3, r'$1/(1-\gamma)=20$', ha='right', va='bottom', color=INK2)
    ax.set(xlabel='Training episode', ylabel='Q-value', title=f'{name}: maximum Q-value during training')
    ax.legend(loc='lower right')
    save(fig, out / f'{name.split("-")[0].lower()}_qvalues.png')
    # (ii) rewards + moving average
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    ax.plot(ep, t['reward'], color=BLUE, alpha=0.3, lw=0.6, label='Episode reward')
    ax.plot(ep, moving_average(t['reward'], 100), color=ORANGE, lw=2, label='Moving average (last 100 episodes)')
    v = read_jsonl(Path(run) / 'validation.jsonl')
    if v:
        ax.scatter([x['episode'] for x in v], [x['mean'] for x in v], s=10, color=INK2, zorder=3,
                   label=f'Greedy validation mean ({len(v[0]["scores"])} episodes, ε=0)')
    ax.set(xlabel='Training episode', ylabel='Episode reward (raw score)', title=f'{name}: training rewards')
    ax.legend(loc='upper left')
    save(fig, out / f'{name.split("-")[0].lower()}_rewards.png')
    return t


def histogram(run, out, name):
    p = Path(run) / 'evaluation' / 'evaluation.csv'
    if not p.exists():
        return
    s = read_csv(p)['reward']
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    bins = np.arange(s.min() - 5, s.max() + 15, 10) if s.max() - s.min() < 200 else 30
    ax.hist(s, bins=bins, color=BLUE, edgecolor='white', linewidth=1)
    if 'Cart' in name:
        ax.set_yscale('log')
        ax.text(0.02, 0.6, f'{int((s >= 500).sum())} of {len(s)} episodes reach the 500-step limit',
                transform=ax.transAxes, color=INK2)
    ax.axvline(s.mean(), color=ORANGE, ls='--', label=f'mean = {s.mean():.1f}, std = {s.std(ddof=1):.1f}')
    ax.set(xlabel='Episode reward', ylabel='Number of episodes' + (' (log scale)' if 'Cart' in name else ''),
           title=f'{name}: {len(s)} evaluation episodes with the trained greedy policy')
    ax.legend(loc='upper left')
    save(fig, out / f'{name.split("-")[0].lower()}_histogram.png')


def calibration(run, out, name):
    """Non-standard plot: learned Q(s, a*) against the realised discounted return along held-out episodes."""
    p = Path(run) / 'evaluation' / 'calibration.csv'
    if not p.exists():
        return
    c = read_csv(p)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    first = c['episode'] == 1
    ax.plot(c['t'][first], c['qmax'][first], color=BLUE, label='predicted max$_a$ Q(s$_t$, a)')
    ax.plot(c['t'][first], c['mc_return'][first], color=ORANGE,
            label='realised discounted return Σ$_k$ γ$^k$ r$_{t+k}$' + (' (clipped rewards)' if 'Pac' in name else ''))
    if 'Cart' in name:
        ax.annotate('time limit truncates the realised return;\nQ correctly keeps bootstrapping (not terminal)',
                    xy=(485, 6), xytext=(230, 13), color=INK2, fontsize=8, arrowprops=dict(arrowstyle='->', color=INK2))
    ax.set(xlabel='Time step in held-out evaluation episode 1', ylabel='Discounted value',
           title=f'{name}: value calibration along an evaluation episode')
    ax.legend(loc='lower left' if 'Cart' in name else 'upper right')
    save(fig, out / f'{name.split("-")[0].lower()}_calibration.png')


def action_heatmap(t, out, n_actions=9, bins=60):
    """Non-standard plot: how the action distribution of the behaviour policy evolves."""
    acts = np.stack([t[f'act{i}'] for i in range(n_actions)], 1)
    edges = np.linspace(0, len(acts), bins + 1).astype(int)
    frac = np.stack([acts[a:b].sum(0) / max(1, acts[a:b].sum()) for a, b in zip(edges[:-1], edges[1:])], 1)
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    im = ax.imshow(frac, aspect='auto', cmap='Blues', origin='upper',
                   extent=[0, t['episode'][-1], n_actions - 0.5, -0.5], vmin=0)
    ax.set_yticks(range(n_actions), MSPACMAN_ACTIONS)
    ax.grid(False)
    ax.set(xlabel='Training episode', title='MsPacman-v5: share of each action taken (behaviour policy)')
    cb = fig.colorbar(im, ax=ax, pad=0.01)
    cb.set_label('fraction of steps in episode bin')
    eps = t['epsilon']
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ticks = [e for e in np.linspace(0, t['episode'][-1], 6)]
    ax2.set_xticks(ticks, [f'ε={eps[min(int(x), len(eps) - 1)]:.2f}' for x in ticks], fontsize=7, color=INK2)
    ax2.grid(False)
    save(fig, out / 'mspacman_action_heatmap.png')


def ablations(out):
    groups = {'Double DQN + target net (main)': ('results/cartpole_s*', BLUE),
              'Vanilla max target (no double)': ('results/cartpole_nodouble_s*', AQUA),
              'No target network': ('results/cartpole_notarget_s*', RED)}
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for label, (pat, col) in groups.items():
        curves = []
        for d in sorted(glob.glob(pat)):
            v = read_jsonl(Path(d) / 'validation.jsonl')
            curves.append((np.array([x['episode'] for x in v]), np.array([x['mean'] for x in v])))
        if not curves:
            continue
        n = min(len(c[0]) for c in curves)
        x = curves[0][0][:n]
        y = np.stack([c[1][:n] for c in curves])
        ax.fill_between(x, y.min(0), y.max(0), color=col, alpha=0.15, lw=0)
        ax.plot(x, y.mean(0), color=col, label=f'{label} ({len(curves)} seeds)')
    ax.set(xlabel='Training episode', ylabel='Greedy validation return (10 ep.)',
           title='CartPole-v1 ablation: mean across seeds, band = min to max')
    ax.legend(loc='upper left')
    save(fig, out / 'cartpole_ablation.png')

    runs = {'Main (target network)': ('results/mspacman', BLUE), 'No target network': ('results/mspacman_notarget', RED)}
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    for label, (d, col) in runs.items():
        if (Path(d) / 'train.csv').exists():
            t = read_csv(Path(d) / 'train.csv')
            ax.plot(t['episode'], moving_average(t['reward'], 100), color=col, label=label)
    ax.set(xlabel='Training episode', ylabel='Reward, 100-episode moving avg',
           title='MsPacman-v5 ablation: target network vs. none')
    ax.legend(loc='upper left')
    save(fig, out / 'mspacman_ablation.png')


def validation_plots(out):
    p = Path('results/validation/cartpole.json')
    if p.exists():
        d = json.loads(p.read_text())
        fig, axes = plt.subplots(1, 5, figsize=(12, 3), sharey=True)
        e = d['epsilon_noise']
        xs = [float(k) for k in e]
        axes[0].plot(xs, [e[k]['mean'] for k in e], marker='o', color=BLUE)
        axes[0].set(xlabel='random-action prob. ε', ylabel='mean reward (episodes)', title='policy noise')
        for ax, param in zip(axes[1:], ['length', 'masspole', 'force_mag', 'gravity']):
            rows = [r for r in d['physics'] if r['param'] == param]
            ax.errorbar([r['value'] for r in rows], [r['mean'] for r in rows], yerr=[r['std'] for r in rows],
                        marker='o', color=BLUE, capsize=3)
            default = {'length': 0.5, 'masspole': 0.1, 'force_mag': 10.0, 'gravity': 9.8}[param]
            ax.axvline(default, color=INK2, ls='--', lw=1)
            ax.set(xlabel=param, title=f'perturb {param}')
        fig.suptitle('CartPole-v1: robustness of the trained greedy policy '
                     '(dashed line = value used in training; error bars = 1 std)', fontweight='bold', fontsize=10)
        save(fig, out / 'cartpole_robustness.png')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--cartpole', default='results/cartpole_s0')
    ap.add_argument('--mspacman', default='results/mspacman')
    ap.add_argument('--out', default='figures')
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    training_plots(a.cartpole, out, 'CartPole-v1', 0.95)
    histogram(a.cartpole, out, 'CartPole-v1')
    calibration(a.cartpole, out, 'CartPole-v1')
    if (Path(a.mspacman) / 'train.csv').exists():
        t = training_plots(a.mspacman, out, 'MsPacman-v5', 0.99)
        action_heatmap(t, out)
        histogram(a.mspacman, out, 'MsPacman-v5')
        calibration(a.mspacman, out, 'MsPacman-v5')
    ablations(out)
    validation_plots(out)
