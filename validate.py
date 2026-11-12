"""Behavioural validation experiments (each has an analytical / a-priori expected outcome).

  python validate.py analytic                          # T1-T3: chain MDP, terminal masking, pipeline edge cases
  python validate.py cartpole --checkpoint CKPT        # T4-T6: noise, physics perturbation, mirror symmetry
  python validate.py mspacman --checkpoint CKPT        # T4/T7: trained agent vs random / noisy policy
  python validate.py failure_modes --checkpoint A B .. # T8: why CartPole episodes end
Results are written as JSON to results/validation/.
"""
import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch

from dqn import (FrameStacker, QNet, ReplayBuffer, dqn_update, greedy_scores, load, make_env,
                 make_vec_env, preprocess_observation, seed_all, td_target, MSPACMAN_COLOR)

OUT = Path('results/validation')


# ----------------------------------------------------------------------------- T1 chain MDP
def chain_mdp(gamma, mask_terminal=True, updates=3000, seed=0):
    """4-state corridor, one-hot states. 'right' (a=1) from s=3 pays +1 and terminates;
    every other move pays 0.  Closed form: Q*(s,R) = g^(3-s), Q*(s,L) = g * V*(max(s-1,0)).
    Trained with the exact ReplayBuffer / dqn_update / target-network code used for the games.
    With mask_terminal=False the terminal flag is ignored (a deliberate bug) -> values must blow up
    towards 1/(1-g) since the terminal state (s=0 encoding reused) keeps bootstrapping."""
    seed_all(seed)
    eye = np.eye(4, dtype=np.float32)
    buf = ReplayBuffer('cartpole', 2000, seed)
    for _ in range(250):
        for s in range(4):
            for a in (0, 1):
                if a == 1 and s == 3:
                    ns, r, term = eye[3], 1.0, 1.0   # terminal: next state is irrelevant if masked
                else:
                    ns, r, term = eye[min(s + 1, 3) if a == 1 else max(s - 1, 0)], 0.0, 0.0
                buf.add([eye[s]], [a], [r], [ns], [term if mask_terminal else 0.0])
    net = QNet('cartpole', 2)
    target = copy.deepcopy(net)
    opt = torch.optim.Adam(net.parameters(), lr=1e-3)
    for k in range(updates):
        dqn_update(net, target, opt, buf.sample(64), gamma, 'cpu')
        if (k + 1) % 100 == 0:
            target.load_state_dict(net.state_dict())
    with torch.no_grad():
        q = net(torch.tensor(eye)).numpy()
    v = np.array([gamma ** (3 - s) for s in range(4)])
    expected = np.stack([gamma * v[np.maximum(np.arange(4) - 1, 0)], v], 1)
    return q, expected


def analytic():
    out = {'chain_mdp': []}
    for gamma in (0.0, 0.5, 0.95, 0.99):
        q, exp = chain_mdp(gamma)
        out['chain_mdp'].append({'gamma': gamma, 'learned': q.round(4).tolist(), 'closed_form': exp.round(4).tolist(),
                                 'max_abs_err': float(np.abs(q - exp).max()),
                                 'greedy_matches_optimal': bool((q.argmax(1) == 1).all())})
    q_bug, exp = chain_mdp(0.95, mask_terminal=False)
    out['chain_mdp_without_terminal_mask'] = {'gamma': 0.95, 'learned': q_bug.round(3).tolist(),
                                              'closed_form': exp.round(4).tolist(),
                                              'max_abs_err': float(np.abs(q_bug - exp).max())}

    # T2 terminal masking vs. time-limit truncation: same reward/next-value, different flag.
    y = td_target(torch.tensor([1., 1.]), torch.tensor([1., 0.]), torch.tensor([20., 20.]), 0.95)
    out['td_target_terminal_vs_truncated'] = {'terminated': float(y[0]), 'truncated': float(y[1]),
                                              'expected': [1.0, 20.0]}

    # T3 pipeline edge cases on the real MsPacman environment
    obs = np.zeros((210, 160, 3), np.uint8)
    obs[1, 0] = [210, 164, 74]      # Ms. Pac-Man colour -> must be blacked out
    obs[3, 0] = [255, 255, 255]     # white: RGB sum 765 would overflow uint8 if summed naively
    f = preprocess_observation(obs)
    edge = {'pacman_pixel_zeroed': int(f[0, 0]) == 0, 'white_pixel_255_no_overflow': int(f[1, 0]) == 255,
            'frame_shape': list(f.shape), 'mspacman_color': MSPACMAN_COLOR}
    # Real env: verify the stored transition at an episode boundary uses the TRUE final frame
    # (SAME_STEP autoreset returns the reset obs), and the new episode's stack is 4 copies of its first frame.
    envs = make_vec_env('mspacman', 2)
    o, _ = envs.reset(seed=3)
    fs = FrameStacker('mspacman', 2)
    s = fs.reset(o)
    buf = ReplayBuffer('mspacman', 50_000)
    edge['stack_after_reset_is_4_copies'] = bool((s[:, 0] == s[:, 3]).all())
    boundary_ok, ns_reconstruct_ok, steps = None, True, 0
    rng = np.random.default_rng(0)
    while boundary_ok is None and steps < 20_000:
        a = rng.integers(9, size=2)
        o, r, term, trunc, info = envs.step(a)
        frames = fs.encode(o)
        done = term | trunc
        for i in np.flatnonzero(done):
            frames[i] = preprocess_observation(info['final_obs'][i])
        ns = fs.push(frames)
        buf.add(s, a, np.sign(r), ns, term.astype(np.float32))
        # replay buffer must reconstruct next_state exactly from (state, newest frame)
        p = (buf.pos - 2) % buf.capacity
        rec = np.concatenate([buf.s[p, 1:], buf.nf[p][None]])
        ns_reconstruct_ok &= bool((rec == ns[0]).all())
        s = ns
        if done.any():
            i = int(np.flatnonzero(done)[0])
            final = preprocess_observation(info['final_obs'][i])
            reset_frame = preprocess_observation(o[i])
            p = (buf.pos - 2 + i) % buf.capacity
            boundary_ok = {'stored_next_frame_is_final_obs': bool((buf.nf[p] == final).all()),
                           'final_differs_from_reset_obs': bool((final != reset_frame).any()),
                           'terminal_flag_stored': float(buf.term[p]), 'step': steps}
            s = fs.reset(o, np.flatnonzero(done))
            edge['new_episode_stack_is_4_copies'] = bool((s[i, 0] == s[i, 3]).all())
        steps += 1
    envs.close()
    edge['episode_boundary'] = boundary_ok
    edge['next_state_reconstruction_exact'] = ns_reconstruct_ok
    out['pipeline_edge_cases'] = edge
    return out


# ----------------------------------------------------------------------------- T4-T6 CartPole
def cartpole(checkpoint, n=100):
    net, ck = load(checkpoint, 'cpu')
    out = {'checkpoint_episode': ck['episode']}
    seeds = [300_000 + i for i in range(n)]
    out['epsilon_noise'] = {}
    for eps in (0.0, 0.05, 0.1, 0.25, 0.5, 1.0):
        sc = [x['reward'] for x in greedy_scores(net, 'cartpole', seeds, 'cpu', epsilon=eps)]
        out['epsilon_noise'][str(eps)] = {'mean': float(np.mean(sc)), 'std': float(np.std(sc)),
                                          'fraction_500': float(np.mean(np.array(sc) >= 500))}
    # Physics perturbation: the policy was trained at default physics; change one constant at a time.
    out['physics'] = []
    for name, values in {'length': [0.25, 0.5, 1.0, 2.0], 'masspole': [0.05, 0.1, 0.5, 1.0],
                         'force_mag': [5.0, 10.0, 15.0, 20.0], 'gravity': [4.9, 9.8, 14.7, 19.6]}.items():
        for v in values:
            env = make_env('cartpole')
            setattr(env.unwrapped, name, v)
            u = env.unwrapped
            u.total_mass = u.masspole + u.masscart
            u.polemass_length = u.masspole * u.length
            sc = []
            with torch.no_grad():
                for sd in seeds[:30]:
                    o, _ = env.reset(seed=sd)
                    tot = 0
                    while True:
                        o, r, te, tr, _ = env.step(int(net(torch.tensor(o)[None]).argmax()))
                        tot += r
                        if te or tr:
                            break
                    sc.append(tot)
            env.close()
            out['physics'].append({'param': name, 'value': v, 'mean': float(np.mean(sc)), 'std': float(np.std(sc))})
    # Mirror symmetry: CartPole dynamics are symmetric under s -> -s with left<->right.
    rng = np.random.default_rng(1)
    s = rng.uniform([-2, -1, -0.15, -1], [2, 1, 0.15, 1], (5000, 4)).astype(np.float32)
    with torch.no_grad():
        q, qm = net(torch.tensor(s)), net(torch.tensor(-s))
    gap = (q[:, 0] - q[:, 1]).abs() > 0.05
    out['mirror_symmetry_uniform_box'] = {
        'action_flip_rate_confident_states': float((q.argmax(1) != qm.argmax(1))[gap].float().mean()),
        'n_confident_states': int(gap.sum()),
        'mean_abs_value_asymmetry': float((q.max(1).values - qm.max(1).values).abs().mean())}
    # Same test on states the greedy policy actually visits (on-distribution).
    env, visited = make_env('cartpole'), []
    with torch.no_grad():
        for sd in seeds[:10]:
            o, _ = env.reset(seed=sd)
            while True:
                visited.append(o)
                o, _, te, tr, _ = env.step(int(net(torch.tensor(o)[None]).argmax()))
                if te or tr:
                    break
        s = torch.tensor(np.array(visited))
        q, qm = net(s), net(-s)
    gap = (q[:, 0] - q[:, 1]).abs() > 0.01
    out['mirror_symmetry_visited'] = {
        'action_flip_rate_confident_states': float((q.argmax(1) != qm.argmax(1))[gap].float().mean()),
        'n_confident_states': int(gap.sum()), 'n_states': len(visited),
        'mean_abs_value_asymmetry': float((q.max(1).values - qm.max(1).values).abs().mean())}
    out['qmax_vs_theory'] = {'theory_never_terminating': 1 / (1 - 0.95),
                             'mean_qmax_at_start_states': float(net(torch.tensor(np.array(
                                 [make_env('cartpole').reset(seed=sd)[0] for sd in seeds]))).max(1).values.mean())}
    return out


def failure_modes(checkpoints, n=20):
    """T8: why do CartPole episodes end? pole angle vs. cart leaving the track vs. time limit."""
    out = []
    env = make_env('cartpole')
    for ckpt in checkpoints:
        net, ck = load(ckpt, 'cpu')
        causes, lengths = [], []
        with torch.no_grad():
            for sd in range(500, 500 + n):
                o, _ = env.reset(seed=sd)
                steps = 0
                while True:
                    o, _, te, tr, _ = env.step(int(net(torch.tensor(o)[None]).argmax()))
                    steps += 1
                    if te or tr:
                        break
                causes.append('truncated' if tr else ('cart_x' if abs(o[0]) > 2.4 else 'pole_angle'))
                lengths.append(steps)
        out.append({'checkpoint': ckpt, 'episode': ck['episode'], 'mean_length': float(np.mean(lengths)),
                    **{c: causes.count(c) for c in ('pole_angle', 'cart_x', 'truncated')}})
    return out


# ----------------------------------------------------------------------------- T7 MsPacman
def mspacman(checkpoint, n=30):
    net, ck = load(checkpoint, 'cpu')
    seeds = [300_000 + i for i in range(n)]
    out = {'checkpoint_episode': ck['episode']}
    for eps in (0.0, 0.05, 0.25, 1.0):
        res = greedy_scores(net, 'mspacman', seeds, 'cpu', epsilon=eps)
        sc = [x['reward'] for x in res]
        out[f'epsilon_{eps}'] = {'mean': float(np.mean(sc)), 'std': float(np.std(sc)),
                                 'mean_length': float(np.mean([x['length'] for x in res]))}
    return out


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('which', choices=['analytic', 'cartpole', 'mspacman', 'failure_modes'])
    p.add_argument('--checkpoint', nargs='+')
    a = p.parse_args()
    torch.set_num_threads(2)
    OUT.mkdir(parents=True, exist_ok=True)
    if a.which == 'analytic':
        res = analytic()
    elif a.which == 'failure_modes':
        res = failure_modes(a.checkpoint)
    else:
        res = globals()[a.which](a.checkpoint[0])
    (OUT / f'{a.which}.json').write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1)[:4000])
