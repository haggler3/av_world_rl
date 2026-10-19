# av_world_rl

Learning on top of [`small_av_world`](https://github.com/haggler3/small_av_world), the Rust city simulator:

1. **A learned driver.** A pure-JAX copy of one city road segment, PPO, and an export path that runs the policy inside the
   Rust AV stack (`small_av_world` and [`small_av`](https://github.com/haggler3/small_av)'s `rust/` crate). It replaces the
   AV's sampling planner, which was 75-97 % of its CPU time and caused most of its hard braking.
2. **Generative world models.** Diffusion and flow-matching models that sample possible 3 s futures of the traffic around an
   AV, trained on transitions logged in the city (`dwm/`).

The intersection simulator with turn lanes, risky human drivers and calibrated crash rates is in its own repo:
[`av_world`](https://github.com/haggler3/av_world).

![the learned driver vs a rule-based driver](docs/img/compare.gif)

## Headline results

**Learned driver in the city** (Rust, against the original planner stack):

| | planner stack | learned driver |
|---|---|---|
| CPU per AV step (LiDAR / oracle perception) | 20.1 / 13.2 µs | **5.0 / 2.0 µs** |
| 64 benchmark trips vs an ordinary driver (paired) | -0.04 min | **-1.00 min** (SE 0.41) |
| collisions / red-light runs / hard brakes per trip | 0 / 0 / 25.5 | 0 / 0 / **0.5** |
| 30 % AVs, morning peak: AV / human trip time, hard brakes per AV-hour | 10.5 / 10.7 min, 94.8 | **9.5 / 10.1 min, 2.3** |
| 100 % AVs | gridlock | **flows** (8.6 min average trip) |

The policy runs inside a classical safety envelope (it changes about 0.3 % of decisions); without it the policy alone has
10.8 collisions per 1000 km in the env. At 100 % AVs it had 2 collisions and 36 red-light entries in 15,492 trips: it was
trained against human-like drivers only.

**World models** (3 s futures, on the 64-trip benchmark; skill = error relative to a physics model, lower is better):

| | trips with the driver it was trained on | trips with the planner stack (unseen driving style) |
|---|---|---|
| diffusion, 30 steps (7.5 min of training) | **0.32** | 2.83 |
| flow matching, DCT basis (4k steps) | 0.38 | 2.14 |
| diffusion, leader-physics decoder (4k steps) | 0.44 | **1.06** |

With the driver they were trained on, all models beat physics by 2-3x. With a different driver only the physics-structured
model holds up: a world model trained on one policy's data does not transfer to another.

## Quick start

```bash
pip install -r requirements.txt                 # on a GPU: pip install "jax[cuda12]" optax scipy matplotlib pillow
export JAX_DEFAULT_PRNG_IMPL=unsafe_rbg         # much cheaper random numbers on CPU

python -m avrl.ppo --steps 120e6 --envs 4096 --out results/final        # train the driver (~9 min on 4 CPU cores)
python -m avrl.compare results/final/policy.npz --out results/final      # metrics, figures, GIF
python -m avrl.export results/final/policy.npz --world ../small_av_world # weights + parity vectors for Rust
python -m pytest -q tests

python -m dwm.gifs2d                            # train diffusion and flow world models (~5 min), 2D GIFs
python -m dwm.benchmark                         # score them on the 64-trip benchmark

# in small_av_world
cargo run --release -- trips --seeds 8 --driver learned
cargo run --release -- fleet --driver learned --share 0.3 --start 0 --hours 10 --av-from 7.3 --cohort-start 7.5 --cohort-end 9.0
```

## 1. The learned driver (`avrl/`)

Inside the city the AV never changes lane, so the decision is one-dimensional: how hard to accelerate. `avrl/env.py` copies
one road segment of the world as a fixed-shape JAX function (IDM traffic with the world's driver mix, signals, the
junction-entry rule, 2 Hz world snapshots, tracker quirks), so thousands of copies run under `jax.vmap` (about 400k steps/s
on 4 CPU cores). `avrl/ppo.py` trains a 22 → 64 → 64 → 1 MLP; `avrl/export.py` writes it into the Rust crate, where tests
check the forward pass and the envelope against JAX. The trainer draws live curves and a time-space diagram of a fixed
scenario at every checkpoint:

![learning progress](docs/img/learning_progress.png)

It took six versions to get safe; every fix came from the city's collision and red-light logs ([`docs/HISTORY.md`](docs/HISTORY.md)).

## 2. World models (`dwm/`)

Given 1 s of an AV's history and its next 3 s of planned actions, the models sample possible futures of its speed, the gap
to the car ahead, the leader's speed and the stop-line state. The key design choice is a **physics residual**: the AV's own
motion follows from its actions, so the models only learn how the world deviates from a kinematic prediction.

| | |
|---|---|
| ![DCT](docs/img/dwm_dct.gif) | ![flow matching](docs/img/dwm_flow_dct.gif) |

What the experiments showed (all tables in [`docs/WORLD_MODELS.md`](docs/WORLD_MODELS.md)):
* the physics residual and conditioning on the planned actions are essential;
* diffusion beats a deterministic regressor with the same network once trained long enough, and gives calibrated uncertainty;
* smooth output bases (DCT) fix jagged samples; flow matching beats diffusion at equal training;
* all models are over-confident on short budgets, and none transfers to a driving style it never saw.

## Documentation

* [`docs/METHODS.md`](docs/METHODS.md): pipelines and math (the MDP, PPO, the safety envelope, diffusion, flow matching, bases, CRPS)
* [`docs/HISTORY.md`](docs/HISTORY.md): the six versions of the learned driver and what is still wrong
* [`docs/WORLD_MODELS.md`](docs/WORLD_MODELS.md): every world-model experiment, ablation and benchmark table

## Layout

```
avrl/       driving env, PPO, baselines, evaluation, visuals, export, one-step world model
dwm/        diffusion / flow world models: data, model, bases, training, ablations, benchmark, GIFs
tests/      env rules and physics, determinism, safety envelope, PPO update
results/    final/ (the shipped policy), dwm/ (world-model results and weights), wm/
docs/       METHODS.md, HISTORY.md, WORLD_MODELS.md, img/
```
