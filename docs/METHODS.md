# Methods: pipelines and math

Two learned components sit on top of `small_av_world`:

1. **A driving policy**, trained with **PPO** in a JAX copy of one city segment and deployed in Rust inside a classical
   safety envelope (`avrl/`).
2. **A generative world model**, which samples possible 3 s futures of an AV's surroundings, trained with
   **diffusion** or **flow matching** on transitions logged in the city (`dwm/`).

```
                   ┌──────────────── avrl (policy) ────────────────┐
 small_av_world ──▶│ JAX env  ─▶  PPO  ─▶  MLP policy  ─▶ export   │──▶ Rust AV: features ─▶ MLP ─▶ envelope ─▶ LQR + vehicle
   (Rust city)     └───────────────────────────────────────────────┘                 │
        ▲                                                                            │ AV_TRANSITIONS (10 Hz log)
        │                  ┌──────────────── dwm (world model) ────────────────┐     ▼
        └──── benchmark ───│ windows ─▶ kinematic residual ─▶ basis ─▶ diffusion / flow ─▶ sampled futures ─▶ CRPS │
                           └────────────────────────────────────────────────────┘
```

---

## 1. The driving problem as an MDP (`avrl/env.py`)

Inside the city the AV never changes lane, so control is one-dimensional: choose the acceleration.

**State.** The state $s_t$ holds:
- the ego: front position $x$, speed $v$, actual acceleration $a$;
- up to 8 vehicles ahead and one follower, each with an IDM parameter set;
- the signal phase;
- the next segment's tail vehicle and queue;
- the tracker memory.

**Observation.** $o_t \in \mathbb{R}^{22}$, exactly what the Rust agent can compute:
- ego speed and acceleration, the speed limit, the distance $d$ to the line;
- the light (one-hot), room at the next segment, "must stop";
- the leader's gap $g$, its speed $v_\ell$, the closing speed and the gap rate;
- the second car;
- the last command;
- the required decelerations $v^2/2(d-1)$ and $(v - v_\ell)_+^2 / 2(g-2)$;
- the time headway;
- the standing time.

**Action.** $u \in [-1, 1]$ maps to the command $a_{cmd} = 3u$ for $u \ge 0$ and $8u$ for $u < 0$ (m/s²).

**Dynamics** (10 Hz, $\Delta t = 0.1$ s):
- Ego, first-order actuator lag $\tau = 0.25$ s, as in `vehicle.rs`:

$$a_{t+1} = a_t + \tfrac{\Delta t}{\tau + \Delta t}(a_{cmd} - a_t), \qquad v_{t+1} = \max(0, v_t + a_{t+1}\Delta t)$$

- Other vehicles follow the **Intelligent Driver Model** with per-driver $(v_0, T, a_{max}, b, s_0)$:

$$\dot v = a_{max}\left[1 - \left(\tfrac{v}{v_0}\right)^4 - \left(\tfrac{s^*(v, \Delta v)}{s}\right)^2\right], \qquad s^* = s_0 + \max\!\left(0,\ vT + \tfrac{v\,\Delta v}{2\sqrt{a_{max} b}}\right)$$

- Signals use the world's timing (60–70 s cycle, 3 s yellow, 1 s all-red).
- The AV is handed over to the next segment only on green or yellow with room, $g_{tail} \ge 5.5 + v^2/6$.
- Other vehicles are seen through 2 Hz snapshots, extrapolated at constant speed.
- A new track starts at $\hat v_\ell = 0$ and is then filtered.

**Reward** (per step):

$$r_t = 0.01\,\Delta x - 5\!\cdot\!10^{-4} a^2 - 0.1\,(-a - 3)_+ - 2\!\cdot\!10^{-5}\,\dot a^2 - 0.05\,(0.8 - \text{THW})_+ - 0.1\,(2.5 - g)_+ - 0.05\,(v - v_{max})_+ - 0.02\,(a_\pi - a)$$
$$\quad - 1\cdot[\text{past the line without permission}] - 0.01\cdot[\text{idle}] - 30\cdot[\text{collision}] - 20\cdot[\text{red run}] - 15\cdot[\text{blocked junction}]$$

Episodes last 2048 steps, or end at a collision. Driving more than 8 m past the line without permission counts as a collision.

## 2. PPO (`avrl/ppo.py`)

**Networks.**
- Policy: $\pi_\theta(u\mid o) = \mathcal N(\mu_\theta(o), \sigma^2)$, where $\mu_\theta$ is an MLP $22 \to 64 \to 64 \to 1$ (tanh) and $\log\sigma$ is a learned scalar.
- Value: $V_\phi$, an MLP of the same shape.

**Rollout.** 4096 parallel envs × 64 steps (`jax.vmap` plus `lax.scan`). Each env's next segment and next episode start are pre-sampled once per rollout, which makes the env about 10× faster on CPU.

**Advantages** (GAE, with $\gamma = 0.995$ and $\lambda = 0.95$; $d_t$ marks an episode end):

$$\delta_t = r_t + \gamma (1 - d_t) V_\phi(o_{t+1}) - V_\phi(o_t), \qquad \hat A_t = \sum_{l \ge 0} (\gamma\lambda)^l \prod_{j<l}(1-d_{t+j})\, \delta_{t+l}$$

**Update.** 4 epochs × 8 minibatches; Adam with a linearly decaying learning rate from $3\cdot10^{-4}$; gradient-norm clipping at 0.5. Each minibatch minimises:

$$\rho_t = \frac{\pi_\theta(u_t\mid o_t)}{\pi_{\theta_{old}}(u_t\mid o_t)}, \qquad
\mathcal L = -\mathbb E\big[\min(\rho_t \tilde A_t,\ \mathrm{clip}(\rho_t, 0.8, 1.2)\,\tilde A_t)\big] + 0.5\,\mathbb E\big[(V_\phi(o_t) - \hat R_t)^2\big]$$

Here $\tilde A$ is the per-minibatch standardised advantage and $\hat R_t = \hat A_t + V(o_t)$. Training runs 120 M steps, about 9 min on 4 CPU cores.

**Safety envelope** (applied in training and in deployment). The policy's acceleration is capped:

$$a = \max\!\Big(\min\big(a_\pi,\ a^{IDM}_{stop},\ a^{IDM}_{lead},\ \max(1.5(v_{max} - v), -2)\big), -8\Big)$$

- $a^{IDM}_{stop}$ is the IDM acceleration toward a stopped obstacle 1 m before the line, when a stop is required ($a_{max} = 2$, $b = 3$, $T = 0.5$, $s_0 = 1$).
- $a^{IDM}_{lead}$ is a tight IDM bound behind the leader ($T = 0.6$, $s_0 = 2$).

**Deployment.** The Rust agent computes the same 22 features from its confirmed tracks, runs the MLP at 10 Hz, applies the envelope and follows the lane centre with LQR. Tests check the forward pass and the envelope against JAX.

## 3. The world model (`dwm/`)

**Data.** The city logs every AV decision at 10 Hz (`AV_TRANSITIONS`). A training example is:
- the condition $c$: 1 s of history $h \in \mathbb R^{10 \times 22}$ and the AV's next 3 s of actions $u_{1:30}$;
- the target $y \in \mathbb R^{30 \times 6}$: the next 3 s of AV speed, distance to the line, leader seen, gap, leader speed and must-stop.

Windows never cross a segment hand-over. Train and test are split by vehicle.

**Physics residual.** The AV's own motion follows from its actions. The kinematic prediction $y^{kin}$ pushes $u_{1:30}$ through the vehicle model and holds the leader at constant speed:

$$v^{kin}_{k} = v_0 + \textstyle\sum a_j \Delta t \ \ (\text{lagged } a), \qquad g^{kin}_k = g_0 + v_{\ell,0}\, t_k - \textstyle\sum_{j \le k} v^{kin}_j \Delta t$$

The models learn only the normalised residual $r = (y - y^{kin}) / \sigma_{f,c}$, where $\sigma$ is the per-(step, channel) standard deviation.

**Output bases** (`dwm/basis.py`). The model generates a latent $z$, decoded linearly: $r = D z$. The encoder is ridge-regularised least squares,

$$z = (D^\top D + \lambda\,\tfrac{\mathrm{tr}(D^\top D)}{n} I)^{-1} D^\top r$$

and $z$ is standardised per dimension. The bases:

| basis | $D$ per channel |
|---|---|
| steps | identity (30 values) |
| control points | piecewise-linear interpolation of 10 knots |
| DCT | $D_{fk} \propto \cos\big(\pi (f + \tfrac12) k / 30\big)$, $k < 8$ |
| leader physics | leader acceleration knots $\alpha$; leader-speed residual $= I_1 U \alpha$, gap residual $= I_1 I_1 U \alpha - I_1 U z_v$, plus down-weighted free terms; $I_1$ = cumulative sum × $\Delta t$; ridge $\lambda = 10^{-4}$ (condition number $3\cdot10^6$ otherwise) |

**Diffusion** (DDPM training, DDIM sampling).
- Cosine schedule $\bar\alpha_t$, $T = 100$ noise levels.
- Forward process: $x_t = \sqrt{\bar\alpha_t}\,x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$.
- The network predicts $v = \sqrt{\bar\alpha_t}\,\epsilon - \sqrt{1 - \bar\alpha_t}\,x_0$ and is trained with $\mathbb E\,\|f_\theta(x_t, t, c) - v\|^2$.
- Sampling: DDIM with 5 steps, $\eta = 0$:

$$\hat x_0 = \sqrt{\bar\alpha_t}x_t - \sqrt{1 - \bar\alpha_t}f_\theta, \quad \hat\epsilon = \sqrt{1 - \bar\alpha_t}x_t + \sqrt{\bar\alpha_t}f_\theta, \quad x_{t'} = \sqrt{\bar\alpha_{t'}}\hat x_0 + \sqrt{1 - \bar\alpha_{t'}}\hat\epsilon$$

**Flow matching** (rectified flow).
- Training path: $x_\tau = (1 - \tau) z + \tau x_1$ with $z \sim \mathcal N(0, I)$ and $\tau \sim U(0, 1)$.
- Loss: $\mathbb E\,\|f_\theta(x_\tau, \tau, c) - (x_1 - z)\|^2$.
- Sampling: integrate $\dot x = f_\theta(x, \tau, c)$ from noise with 10 Euler steps.

**Denoiser.** An MLP with FiLM conditioning, width 256, 3 residual blocks, about 1 M parameters. Each block computes $h \leftarrow h + W_2\,\mathrm{silu}(W_1\,\mathrm{silu}(\mathrm{LN}(h)(1 + \gamma) + \beta))$, where $(\gamma, \beta)$ come from an embedding of the condition and the noise level. Also implemented, but too slow to judge on CPU:
- a DiT-style transformer over the history and future tokens, with adaLN-zero;
- a 1-D convolution.

**Evaluation.** With $K$ samples $X_k$ and the true value $y$, the score is the CRPS:

$$\mathrm{CRPS} = \frac1K\sum_k |X_k - y| - \frac{1}{2K^2}\sum_{k,l} |X_k - X_l|$$

For a single deterministic prediction this is the absolute error. The other measures:
- **Skill:** the mean CRPS ratio against the kinematic baseline over speed and gap at 1, 2 and 3 s (below 1 beats physics).
- **Coverage:** the share of real outcomes inside the 5–95 % sample band.
- **Roughness:** the mean $|\Delta^2 g|$ of sampled gaps divided by that of real gaps.
