# 1. Implementation (`dqn.py`)

**Algorithm.** Standard DQN with experience replay and a target network, plus Double-DQN targets.
For a transition $(s,a,r,s',d)$ the loss is the Huber loss between $Q_\theta(s,a)$ and

$$y = r + \gamma\,(1-d_{\text{terminated}})\;Q_{\bar\theta}\big(s',\arg\max_{a'} Q_\theta(s',a')\big),$$

where $\bar\theta$ is the **target network**. It is a hard copy of $\theta$, refreshed every $C$ gradient updates.
Gradients are clipped to norm 10 and optimised with Adam.

| Component | CartPole-v1 | MsPacman-v5 |
|---|---|---|
| Network | MLP 4-128-128-2 (ReLU) | Nature CNN on 4x88x80: conv 32@8x8/4, 64@4x4/2, 64@3x3/1, FC 512, 9 outputs |
| State | raw 4-d observation | `preprocess_observation` (assignment function) with the last 4 frames stacked |
| Discount $\gamma$ | 0.95 | 0.99 |
| Replay buffer | 50k transitions, uniform; 1k warm-up | 200k transitions, uniform, 20k warm-up; $s'$ stored as one frame and rebuilt from $s$ |
| Target network | hard copy every %%CP_TARGET%% updates | hard copy every 2,500 updates (= 10k env steps) |
| Loss / optimiser | Huber, Adam %%CP_LR%% | Huber, Adam lr $10^{-4}\to2\cdot10^{-5}$ (linear LR scheduler over 6M steps; the 2.5 h budget ended at 3.1M steps, lr $\approx5.8\cdot10^{-5}$) |
| Exploration | $\varepsilon$: $1\to0.05$ over 10k steps, then $\to0.01$ | $\varepsilon$: $1\to0.1$ over 1M steps, then $\to0.01$ over 2M steps |
| Batch / replay ratio | %%CP_BATCH%%, 1 update per env step | 32, 2 updates per 8 env steps (8 parallel envs) |
| Reward used for learning | +1 per step | sign-clipped to {−1, 0, +1}; raw game score is what we plot |

**Engineering choices.**
(i) *Vectorised actors.* MsPacman uses 8 `AsyncVectorEnv` workers that feed one learner on one GPU. The replay ratio stays the Nature value (one 32-sample update per 4 env steps).
(ii) *Memory.* The buffer stores $s$ (4 frames) and only the newest frame of $s'$, which cuts memory by about 45%. `validate.py` T3 checks that $s'$ is rebuilt exactly.
(iii) *Checkpoint selection.* Every $k$ episodes the greedy policy plays validation seeds 10000+ and we keep `best.pt`. The final 500-episode evaluation uses **disjoint** seeds 100000–100499, so the selection is not scored on its own data.
(iv) *Logging.* Every episode we log reward, length, $\varepsilon$, learning rate, loss, the mean and max of $\max_a Q(s_t,a)$ along the episode, and the action counts.

# 2. Edge cases and how they are handled

1. **Termination vs. truncation (time limit).** CartPole-v1 stops at 500 steps with `truncated=True`, and MsPacman-v5 has a frame cap. A truncated state is *not* terminal, so the target must still bootstrap. We store only `terminated` as the done-mask. T2 checks the target values directly, and the CartPole calibration plot (Fig. 5) shows Q staying near 20 right up to step 500.
2. **Autoreset in vectorised envs.** After an episode ends, Gymnasium's default autoreset returns the *reset* observation. Storing that as $s'$ would silently corrupt every terminal transition. We use `AutoresetMode.SAME_STEP` and take $s'$ from `info["final_obs"]`. T3 checks on the real game that the stored frame equals the true final frame and differs from the reset frame.
3. **Frame stack at episode start.** A new episode's stack is filled with 4 copies of its first frame, so frames never leak from the previous episode (T3).
4. **uint8 overflow in preprocessing.** The RGB sum can reach 765. `ndarray.sum` promotes to a wider integer, so it does not wrap. We store the frame as uint8 `img//3`, which is the assignment's int8 value + 128 and therefore lossless; the network maps it back to $[-1,1)$. T3 checks that a white pixel gives 255 and a Ms. Pac-Man pixel gives 0.
5. **Replay buffer wrap-around and warm-up.** Insertion uses a ring index. Sampling only starts once the buffer holds `warmup ≥ batch` transitions (enforced by the argument parser), so we never sample from empty slots.
6. **Unbounded / large rewards.** MsPacman rewards range from 10 to 1,600 per step. They are sign-clipped for learning, which bounds $|Q|\le 1/(1-\gamma)=100$, and the raw scores are reported.
7. **Loss of lives in MsPacman.** Losing a life is not treated as terminal. The env's `terminated` flag (game over) is the only terminal signal, so the agent optimises the real objective.
8. **Short-horizon discounting in CartPole ($\gamma=0.95$).** The effective horizon is $1/(1-\gamma)=20$ steps. A slow drift of the cart toward $|x|=2.4$ that ends the episode 150 steps later changes $Q$ by only $20\cdot0.95^{150}\approx 0.009$. We observed exactly this failure mode (Section 5, T8). It is the main reason CartPole needs a small learning rate, a large buffer and checkpoint selection.
9. **Greedy loops / ties.** Evaluation is greedy ($\varepsilon=0$). MsPacman-v5 keeps its sticky actions (p = 0.25), so a deterministic policy cannot loop forever, and the frame cap bounds episode length in any case.
10. **Single GPU.** Each MsPacman process sees exactly one card via `CUDA_VISIBLE_DEVICES`.
