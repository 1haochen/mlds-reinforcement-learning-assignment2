"""Deep Q-Network for CartPole-v1 and ALE/MsPacman-v5 (MLDS Assignment 2).

Components required by the assignment:
  * neural network      -> QNet (MLP for CartPole, Nature-style CNN for MsPacman)
  * replay buffer       -> ReplayBuffer (uniform sampling, ring storage)
  * target network      -> `target` in train(), hard-copied every --target-every updates

Extras: Double-DQN targets, Huber loss, gradient clipping, linear epsilon schedule,
linear learning-rate decay, and vectorised environments (N actors, one learner).

Usage:
  python dqn.py train    --game cartpole --out results/cartpole
  python dqn.py train    --game mspacman --out results/mspacman --device cuda --num-envs 8
  python dqn.py evaluate --checkpoint results/cartpole/best.pt --episodes 500
"""
import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from torch import nn

GAMMA = {'cartpole': 0.95, 'mspacman': 0.99}
ENV_ID = {'cartpole': 'CartPole-v1', 'mspacman': 'ALE/MsPacman-v5'}
MSPACMAN_COLOR = 210 + 164 + 74


# ----------------------------------------------------------------------------- environment
def preprocess_observation(obs):
    """Assignment preprocessing. Returns uint8 in [0, 255] (= assignment's int8 + 128).

    obs.sum promotes uint8 -> uint64, so the RGB sum (max 765) cannot overflow.
    Storing uint8 instead of int8 is lossless; QNet maps it back to [-1, 1).
    """
    img = obs[1:176:2, ::2]
    img = img.sum(axis=2)
    img[img == MSPACMAN_COLOR] = 0
    return (img // 3).astype(np.uint8)  # (88, 80)


def make_env(game, **kwargs):
    if game == 'mspacman':
        import ale_py
        gym.register_envs(ale_py)
    return gym.make(ENV_ID[game], **kwargs)


def make_vec_env(game, n):
    fns = [lambda: make_env(game) for _ in range(n)]
    mode = gym.vector.AutoresetMode.SAME_STEP  # final obs delivered in info, reset obs returned
    if n == 1 or game == 'cartpole':
        return gym.vector.SyncVectorEnv(fns, autoreset_mode=mode)
    return gym.vector.AsyncVectorEnv(fns, autoreset_mode=mode)


class FrameStacker:
    """Keeps the last 4 preprocessed frames for each of N environments."""

    def __init__(self, game, n, k=4):
        self.game, self.k = game, k
        self.stack = None if game == 'cartpole' else np.zeros((n, k, 88, 80), np.uint8)

    def encode(self, obs):
        return np.stack([preprocess_observation(o) for o in obs])

    def reset(self, obs, idx=None):
        """Start of an episode: the first frame is repeated k times (no stale frames)."""
        if self.game == 'cartpole':
            return np.asarray(obs, np.float32)
        idx = np.arange(len(obs)) if idx is None else idx
        frames = self.encode(obs[idx])
        self.stack[idx] = frames[:, None]
        return self.stack.copy()

    def push(self, frames):
        self.stack = np.concatenate([self.stack[:, 1:], frames[:, None]], axis=1)
        return self.stack.copy()


# ----------------------------------------------------------------------------- model / buffer
class QNet(nn.Module):
    def __init__(self, game, n_actions, dueling=False):
        super().__init__()
        self.game, self.dueling = game, dueling
        if game == 'cartpole' and dueling:  # Q = V + A - mean(A): action gaps learned separately
            self.net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU())
            self.value, self.adv = nn.Linear(128, 1), nn.Linear(128, n_actions)
        elif game == 'cartpole':
            self.net = nn.Sequential(nn.Linear(4, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU(),
                                     nn.Linear(128, n_actions))
        else:  # 4x88x80 -> 32x21x19 -> 64x9x8 -> 64x7x6
            self.net = nn.Sequential(
                nn.Conv2d(4, 32, 8, 4), nn.ReLU(), nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
                nn.Conv2d(64, 64, 3, 1), nn.ReLU(), nn.Flatten(),
                nn.Linear(64 * 7 * 6, 512), nn.ReLU(), nn.Linear(512, n_actions))

    def forward(self, x):
        if self.game != 'cartpole':
            x = (x.float() - 128.0) / 128.0  # assignment's int8 range -> [-1, 1)
        if self.dueling:
            h = self.net(x)
            a = self.adv(h)
            return self.value(h) + a - a.mean(1, keepdim=True)
        return self.net(x)


class ReplayBuffer:
    """Ring buffer. For images, next_state is not stored separately: it is
    state[1:] + the new frame, which cuts memory ~45% and is exact for a frame stack."""

    def __init__(self, game, capacity, seed=0):
        self.game, self.capacity, self.size, self.pos = game, capacity, 0, 0
        self.rng = np.random.default_rng(seed)
        if game == 'cartpole':
            self.s = np.zeros((capacity, 4), np.float32)
            self.ns = np.zeros((capacity, 4), np.float32)
        else:
            self.s = np.zeros((capacity, 4, 88, 80), np.uint8)
            self.nf = np.zeros((capacity, 88, 80), np.uint8)
        self.a = np.zeros(capacity, np.int64)
        self.r = np.zeros(capacity, np.float32)
        self.term = np.zeros(capacity, np.float32)
        self.added = np.zeros(capacity, np.int64)  # global insertion counter (age diagnostics)
        self.count = 0

    def add(self, s, a, r, ns, term):
        for i in range(len(a)):
            p = self.pos
            self.s[p] = s[i]
            if self.game == 'cartpole':
                self.ns[p] = ns[i]
            else:
                self.nf[p] = ns[i][-1]
            self.a[p], self.r[p], self.term[p] = a[i], r[i], term[i]
            self.added[p] = self.count
            self.count += 1
            self.pos = (p + 1) % self.capacity
            self.size = min(self.size + 1, self.capacity)

    def sample(self, n):
        idx = self.rng.integers(0, self.size, n)
        s = self.s[idx]
        if self.game == 'cartpole':
            ns = self.ns[idx]
        else:
            ns = np.concatenate([s[:, 1:], self.nf[idx][:, None]], axis=1)
        return s, self.a[idx], self.r[idx], ns, self.term[idx]

    def __len__(self):
        return self.size


def td_target(reward, terminated, next_value, gamma):
    """y = r + gamma * (1 - terminated) * V(s'). Truncation (time limit) is NOT terminal."""
    return reward + gamma * (1.0 - terminated) * next_value


def dqn_update(net, target, opt, batch, gamma, device, double=True, use_target=True):
    s, a, r, ns, term = (torch.as_tensor(x, device=device) for x in batch)
    q = net(s).gather(1, a[:, None]).squeeze(1)
    with torch.no_grad():
        bootstrap = target if use_target else net
        nq = bootstrap(ns)
        if double:
            v = nq.gather(1, net(ns).argmax(1, keepdim=True)).squeeze(1)
        else:
            v = nq.max(1).values
        y = td_target(r, term, v, gamma)
    loss = nn.functional.smooth_l1_loss(q, y)  # Huber
    opt.zero_grad(set_to_none=True)
    loss.backward()
    nn.utils.clip_grad_norm_(net.parameters(), 10.0)
    opt.step()
    return loss.item(), q.detach().mean().item()


# ----------------------------------------------------------------------------- training
def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def linear(start, end, frac):
    return start + (end - start) * min(1.0, max(0.0, frac))


def epsilon_at(step, args):
    """Piecewise linear: 1 -> eps_mid over eps_steps, then -> eps_end over the next 2*eps_steps."""
    if step < args.eps_steps:
        return linear(1.0, args.eps_mid, step / args.eps_steps)
    return linear(args.eps_mid, args.eps_end, (step - args.eps_steps) / (2 * args.eps_steps))


@torch.no_grad()
def greedy_scores(net, game, seeds, device, epsilon=0.0, max_steps=None, trace_first=0):
    """Play one episode per seed with an (epsilon-)greedy policy.

    Returns per-episode dicts with raw return, length, the max-Q trajectory and the
    realised discounted (clipped) return for Q calibration."""
    out = []
    env = make_env(game)
    rng = np.random.default_rng(seeds[0] + 777)
    for k, seed in enumerate(seeds):
        obs, _ = env.reset(seed=int(seed))
        fs = FrameStacker(game, 1)
        s = fs.reset(obs[None])
        total, rewards, qmax, trace = 0.0, [], [], []
        while True:
            q = net(torch.as_tensor(s, device=device))[0]
            a = int(rng.integers(env.action_space.n)) if rng.random() < epsilon else int(q.argmax())
            obs, r, term, trunc, _ = env.step(a)
            if k < trace_first and len(trace) < 10:
                trace.append({'t': len(trace) + 1,
                              'state': s[0].round(4).tolist() if game == 'cartpole' else 'frame stack',
                              'q': q.cpu().numpy().round(4).tolist(), 'action': a, 'reward': float(r),
                              'terminated': bool(term), 'truncated': bool(trunc)})
            total += r
            rewards.append(r if game == 'cartpole' else float(np.sign(r)))
            qmax.append(float(q.max()))
            s = fs.reset(obs[None]) if game == 'cartpole' else fs.push(fs.encode(obs[None]))
            if term or trunc or (max_steps and len(rewards) >= max_steps):
                break
        g, returns = 0.0, []
        for rr in reversed(rewards):
            g = rr + GAMMA[game] * g
            returns.append(g)
        out.append({'seed': int(seed), 'reward': float(total), 'length': len(rewards), 'qmax': qmax,
                    'return': returns[::-1], 'trace': trace, 'truncated': bool(trunc)})
    env.close()
    return out


def save_checkpoint(path, net, args, episodes, steps):
    torch.save({'model': net.state_dict(), 'game': args.game, 'episode': episodes, 'steps': steps,
                'config': {k: v for k, v in vars(args).items() if k != 'func'}}, path)


def train(args):
    seed_all(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.backends.cudnn.benchmark = True
    gamma = GAMMA[args.game]
    envs = make_vec_env(args.game, args.num_envs)
    n_actions = envs.single_action_space.n
    net = QNet(args.game, n_actions, args.dueling).to(device)
    target = copy.deepcopy(net).eval()
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1.5e-4 if args.game == 'mspacman' else 1e-8)
    buffer = ReplayBuffer(args.game, args.capacity, args.seed)
    (out / 'config.json').write_text(json.dumps({k: v for k, v in vars(args).items() if k != 'func'}, indent=2))

    obs, _ = envs.reset(seed=args.seed)
    fs = FrameStacker(args.game, args.num_envs)
    s = fs.reset(obs)
    ep_ret = np.zeros(args.num_envs)
    ep_len = np.zeros(args.num_envs, np.int64)
    ep_qmax = [[] for _ in range(args.num_envs)]
    ep_actions = np.zeros((args.num_envs, n_actions), np.int64)
    steps = updates = episodes = 0
    best, start, losses, qpred = -np.inf, time.time(), [], []
    fields = ['episode', 'env_steps', 'updates', 'reward', 'length', 'epsilon', 'lr', 'qmax_max',
              'qmax_mean', 'qmax_start', 'loss', 'q_pred', 'seconds'] + [f'act{i}' for i in range(n_actions)]
    f = (out / 'train.csv').open('w', buffering=1)
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()
    vf = (out / 'validation.jsonl').open('w', buffering=1)
    done_training = False
    while not done_training:
        eps = epsilon_at(steps, args)
        with torch.no_grad():
            q = net(torch.as_tensor(s, device=device)).cpu().numpy()
        actions = q.argmax(1)
        explore = np.random.random(args.num_envs) < eps
        actions[explore] = np.random.randint(n_actions, size=explore.sum())
        obs, r, term, trunc, info = envs.step(actions)
        done = term | trunc
        # SAME_STEP autoreset: `obs` is already the reset obs for finished envs;
        # the true last observation lives in info['final_obs'].
        if args.game == 'cartpole':
            real_next = obs.copy()
            if done.any():
                for i in np.flatnonzero(done):
                    real_next[i] = info['final_obs'][i]
            ns = real_next.astype(np.float32)
        else:
            frames = fs.encode(obs)
            if done.any():
                for i in np.flatnonzero(done):
                    frames[i] = preprocess_observation(info['final_obs'][i])
            ns = fs.push(frames)
        reward = r if args.game == 'cartpole' else np.sign(r)  # clip Atari rewards to {-1,0,1}
        buffer.add(s, actions, reward, ns, term.astype(np.float32))
        for i in range(args.num_envs):
            ep_qmax[i].append(q[i].max())
            ep_actions[i, actions[i]] += 1
        ep_ret += r
        ep_len += 1
        steps += args.num_envs
        s = ns
        if done.any():
            s = fs.reset(obs, np.flatnonzero(done)) if args.game != 'cartpole' else obs.astype(np.float32)

        if len(buffer) >= args.warmup:
            for _ in range(args.updates_per_step):
                lr = linear(args.lr, args.lr * args.lr_final_frac, steps / args.lr_decay_steps)
                for g in opt.param_groups:
                    g['lr'] = lr
                loss, qp = dqn_update(net, target, opt, buffer.sample(args.batch), gamma, device,
                                      double=not args.no_double, use_target=not args.no_target)
                losses.append(loss)
                qpred.append(qp)
                updates += 1
                if updates % args.target_every == 0:
                    target.load_state_dict(net.state_dict())

        for i in np.flatnonzero(done):
            episodes += 1
            qm = np.array(ep_qmax[i])
            row = dict(episode=episodes, env_steps=steps, updates=updates, reward=ep_ret[i], length=ep_len[i],
                       epsilon=round(eps, 5), lr=opt.param_groups[0]['lr'], qmax_max=qm.max(),
                       qmax_mean=qm.mean(), qmax_start=qm[0],
                       loss=np.mean(losses) if losses else '', q_pred=np.mean(qpred) if qpred else '',
                       seconds=round(time.time() - start, 1))
            row.update({f'act{j}': ep_actions[i, j] for j in range(n_actions)})
            writer.writerow(row)
            losses, qpred = [], []
            ep_ret[i], ep_len[i], ep_qmax[i] = 0, 0, []
            ep_actions[i] = 0

            if episodes % args.check_every == 0:
                net.eval()
                res = greedy_scores(net, args.game, [10_000 + j for j in range(args.val_episodes)], device,
                                    epsilon=args.val_epsilon)
                net.train()
                scores = [x['reward'] for x in res]
                mean = float(np.mean(scores))
                vf.write(json.dumps({'episode': episodes, 'env_steps': steps, 'mean': mean, 'scores': scores}) + '\n')
                print(f'{args.game} ep={episodes} steps={steps} eps={eps:.3f} val={mean:.1f} '
                      f'best={max(best, mean):.1f} t={time.time() - start:.0f}s', flush=True)
                if mean >= best:
                    best = mean
                    save_checkpoint(out / 'best.pt', net, args, episodes, steps)
                save_checkpoint(out / 'last.pt', net, args, episodes, steps)
            if (episodes >= args.episodes or time.time() - start > args.time_limit
                    or (args.stop_at_solve and args.game == 'cartpole' and best >= 500
                        and episodes >= args.min_episodes)):
                done_training = True
                break
    save_checkpoint(out / 'last.pt', net, args, episodes, steps)
    f.close()
    vf.close()
    envs.close()
    print(f'finished: {episodes} episodes, {steps} env steps, {updates} updates, best val {best:.1f}', flush=True)


# ----------------------------------------------------------------------------- evaluation
def load(checkpoint, device):
    ck = torch.load(checkpoint, map_location=device, weights_only=False)
    n_actions = 2 if ck['game'] == 'cartpole' else 9
    net = QNet(ck['game'], n_actions, ck['config'].get('dueling', False)).to(device)
    net.load_state_dict(ck['model'])
    return net.eval(), ck


def evaluate(args):
    seed_all(args.seed)
    net, ck = load(args.checkpoint, args.device)
    game = ck['game']
    out = Path(args.out or Path(args.checkpoint).parent / 'evaluation')
    out.mkdir(parents=True, exist_ok=True)
    seeds = [args.seed + i for i in range(args.episodes)]
    if args.workers > 1:  # split seeds over CPU processes (pure inference)
        import multiprocessing as mp
        chunks = [seeds[i::args.workers] for i in range(args.workers)]
        with mp.get_context('spawn').Pool(args.workers) as pool:
            parts = pool.starmap(_eval_chunk, [(args.checkpoint, c, args.epsilon) for c in chunks])
        res = sorted([x for p in parts for x in p], key=lambda x: x['seed'])
        res[0] = greedy_scores(net, game, [seeds[0]], args.device, args.epsilon, trace_first=1)[0]
    else:
        res = greedy_scores(net, game, seeds, args.device, args.epsilon, trace_first=1)
    scores = np.array([x['reward'] for x in res])
    with (out / 'evaluation.csv').open('w') as f:
        w = csv.writer(f)
        w.writerow(['episode', 'seed', 'reward', 'length', 'truncated'])
        for i, x in enumerate(res):
            w.writerow([i + 1, x['seed'], x['reward'], x['length'], x['truncated']])
    with (out / 'calibration.csv').open('w') as f:  # Q(s,a*) vs realised discounted return
        w = csv.writer(f)
        w.writerow(['episode', 't', 'remaining', 'qmax', 'mc_return'])
        for i, x in enumerate(res[:20]):
            for t, (qv, g) in enumerate(zip(x['qmax'], x['return'])):
                w.writerow([i + 1, t, x['length'] - t, round(qv, 4), round(g, 4)])
    (out / 'first10.json').write_text(json.dumps(res[0]['trace'], indent=1))
    summary = {'checkpoint': str(args.checkpoint), 'checkpoint_episode': ck['episode'], 'n': len(scores),
               'mean': float(scores.mean()), 'std': float(scores.std(ddof=1)), 'min': float(scores.min()),
               'max': float(scores.max()), 'median': float(np.median(scores)), 'epsilon': args.epsilon,
               'seeds': [seeds[0], seeds[-1]]}
    if game == 'cartpole':
        summary['fraction_500'] = float(np.mean(scores >= 500))
    (out / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


def _eval_chunk(checkpoint, seeds, epsilon):
    torch.set_num_threads(1)
    net, ck = load(checkpoint, 'cpu')
    res = greedy_scores(net, ck['game'], seeds, 'cpu', epsilon)
    for x in res[20:]:
        x['qmax'], x['return'] = [], []
    return res


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='mode', required=True)
    t = sub.add_parser('train')
    t.add_argument('--game', choices=['cartpole', 'mspacman'], required=True)
    t.add_argument('--out', required=True)
    t.add_argument('--episodes', type=int, default=1000)
    t.add_argument('--time-limit', type=float, default=1e9, help='wall-clock seconds')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--device', default='cpu')
    t.add_argument('--num-envs', type=int, default=1)
    t.add_argument('--lr', type=float, default=1e-3)
    t.add_argument('--lr-final-frac', type=float, default=0.1)
    t.add_argument('--lr-decay-steps', type=float, default=1e5)
    t.add_argument('--capacity', type=int, default=50_000)
    t.add_argument('--batch', type=int, default=64)
    t.add_argument('--warmup', type=int, default=1_000)
    t.add_argument('--eps-steps', type=int, default=10_000)
    t.add_argument('--eps-mid', type=float, default=0.05)
    t.add_argument('--eps-end', type=float, default=0.01)
    t.add_argument('--updates-per-step', type=int, default=1)
    t.add_argument('--target-every', type=int, default=500, help='gradient updates between hard copies')
    t.add_argument('--check-every', type=int, default=20, help='episodes between validation runs')
    t.add_argument('--val-episodes', type=int, default=10)
    t.add_argument('--val-epsilon', type=float, default=0.0)
    t.add_argument('--min-episodes', type=int, default=300)
    t.add_argument('--stop-at-solve', action='store_true')
    t.add_argument('--no-double', action='store_true', help='vanilla max target (ablation)')
    t.add_argument('--no-target', action='store_true', help='bootstrap from online net (ablation)')
    t.add_argument('--dueling', action='store_true', help='dueling value/advantage head (CartPole only)')
    t.set_defaults(func=train)
    e = sub.add_parser('evaluate')
    e.add_argument('--checkpoint', required=True)
    e.add_argument('--out')
    e.add_argument('--episodes', type=int, default=500)
    e.add_argument('--seed', type=int, default=100_000, help='disjoint from training/validation seeds')
    e.add_argument('--epsilon', type=float, default=0.0)
    e.add_argument('--device', default='cpu')
    e.add_argument('--workers', type=int, default=1)
    e.set_defaults(func=evaluate)
    args = p.parse_args()
    if args.mode == 'train' and args.warmup < args.batch:
        p.error('--warmup must be >= --batch')
    args.func(args)


if __name__ == '__main__':
    main()
