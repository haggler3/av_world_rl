"""PPO for the corridor env, in pure JAX (no Flax: the weights are plain arrays so Rust/NumPy can run them as-is).

Sized for a T4 by default (4096 envs x 64 steps per update); on a CPU use fewer envs. Everything inside one update
(rollout, GAE, epochs, minibatches) is a single jitted function; the Python loop around it only logs and draws.
"""
import argparse
import csv
import json
import os
import time
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import env as E

HIDDEN = 64


def init_mlp(key, sizes):
    params = []
    for i, (a, b) in enumerate(zip(sizes[:-1], sizes[1:])):
        key, k = jax.random.split(key)
        scale = 0.01 if i == len(sizes) - 2 else np.sqrt(2.0)
        w = jax.nn.initializers.orthogonal(scale)(k, (a, b))
        params.append({"w": w, "b": jnp.zeros(b)})
    return params


def mlp(params, x):
    for layer in params[:-1]:
        x = jnp.tanh(x @ layer["w"] + layer["b"])
    return x @ params[-1]["w"] + params[-1]["b"]


def init_params(key):
    k1, k2 = jax.random.split(key)
    return {"pi": init_mlp(k1, [E.OBS_DIM, HIDDEN, HIDDEN, 1]), "v": init_mlp(k2, [E.OBS_DIM, HIDDEN, HIDDEN, 1]),
            "log_std": jnp.full((1,), -0.5)}


def policy_mean(params, obs):
    return mlp(params["pi"], obs)[..., 0]


def value(params, obs):
    return mlp(params["v"], obs)[..., 0]


def log_prob(params, obs, u):
    mu, ls = policy_mean(params, obs), params["log_std"][0]
    return -0.5 * ((u - mu) / jnp.exp(ls)) ** 2 - ls - 0.5 * jnp.log(2 * jnp.pi)


class Config:
    def __init__(self, **kw):
        self.n_envs = 4096
        self.n_steps = 64
        self.epochs = 4
        self.minibatches = 8
        self.lr = 3e-4
        self.gamma = 0.995
        self.lam = 0.95
        self.clip = 0.2
        self.ent = 0.0
        self.vf = 0.5
        self.max_grad = 0.5
        self.__dict__.update(kw)


def make_update(cfg: Config, env_cfg: E.EnvConfig, opt):
    vstep = jax.vmap(E.step, in_axes=(0, 0, None, 0))
    vspare = jax.vmap(E.make_spare, in_axes=(0, None))
    stat_keys = ("dist", "hard", "coll", "red", "blocked", "segs", "shield")

    def rollout(params, st, obs, key):
        k_sp, k_act = jax.random.split(key)
        spare = vspare(jax.random.split(k_sp, cfg.n_envs), env_cfg)

        def body(c, k):
            st, obs = c
            mu = policy_mean(params, obs)
            u = mu + jnp.exp(params["log_std"][0]) * jax.random.normal(k, mu.shape)
            v = value(params, obs)
            st2, obs2, r, d, info = vstep(st, u, env_cfg, spare)
            ep = {n: info["s_" + n].sum() for n in stat_keys} | {"episodes": d.sum()}
            return (st2, obs2), (obs, u, log_prob(params, obs, u), v, r, d, ep)

        (st, obs), traj = jax.lax.scan(body, (st, obs), jax.random.split(k_act, cfg.n_steps))
        return st, obs, traj

    def gae(params, traj, last_obs):
        obs, u, lp, v, r, d, _ = traj
        last_v = value(params, last_obs)

        def body(c, x):
            adv_next, v_next = c
            r_t, d_t, v_t = x
            nd = 1.0 - d_t
            delta = r_t + cfg.gamma * v_next * nd - v_t
            adv = delta + cfg.gamma * cfg.lam * nd * adv_next
            return (adv, v_t), adv

        _, adv = jax.lax.scan(body, (jnp.zeros_like(last_v), last_v), (r, d.astype(jnp.float32), v), reverse=True)
        return adv, adv + v

    def loss_fn(params, b):
        obs, u, lp_old, adv, ret = b
        lp = log_prob(params, obs, u)
        ratio = jnp.exp(lp - lp_old)
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv).mean()
        vl = ((value(params, obs) - ret) ** 2).mean()
        ent = params["log_std"][0] + 0.5 * jnp.log(2 * jnp.pi * jnp.e)
        return pg + cfg.vf * vl - cfg.ent * ent, (pg, vl, ((ratio - 1) - jnp.log(ratio)).mean())

    @jax.jit
    def update(params, opt_state, st, obs, key):
        k_roll, k_perm = jax.random.split(key)
        st, obs_last, traj = rollout(params, st, obs, k_roll)
        adv, ret = gae(params, traj, obs_last)
        o, u, lp = traj[0], traj[1], traj[2]
        n = cfg.n_envs * cfg.n_steps
        flat = jax.tree.map(lambda x: x.reshape((n,) + x.shape[2:]), (o, u, lp, adv, ret))
        mb = n // cfg.minibatches

        def epoch(c, k):
            params, opt_state = c
            perm = jax.random.permutation(k, n)

            def minibatch(c, idx):
                params, opt_state = c
                b = jax.tree.map(lambda x: x[idx], flat)
                (l, aux), g = jax.value_and_grad(loss_fn, has_aux=True)(params, b)
                upd, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), aux

            (params, opt_state), aux = jax.lax.scan(minibatch, (params, opt_state), perm.reshape(cfg.minibatches, mb))
            return (params, opt_state), aux

        (params, opt_state), aux = jax.lax.scan(epoch, (params, opt_state), jax.random.split(k_perm, cfg.epochs))
        ep = jax.tree.map(lambda x: x.sum(), traj[6])
        logs = ep | {"reward": traj[4].mean(), "pg": aux[0].mean(), "vloss": aux[1].mean(), "kl": aux[2].mean(),
                     "std": jnp.exp(params["log_std"][0])}
        return params, opt_state, st, obs_last, logs

    return update


def save_params(params, path):
    flat = {}
    for net in ("pi", "v"):
        for i, layer in enumerate(params[net]):
            flat[f"{net}_w{i}"] = np.asarray(layer["w"])
            flat[f"{net}_b{i}"] = np.asarray(layer["b"])
    flat["log_std"] = np.asarray(params["log_std"])
    np.savez(path, **flat)


def load_params(path):
    z = np.load(path)
    p = {net: [{"w": jnp.asarray(z[f"{net}_w{i}"]), "b": jnp.asarray(z[f"{net}_b{i}"])} for i in range(3)] for net in ("pi", "v")}
    p["log_std"] = jnp.asarray(z["log_std"])
    return p


def deterministic_policy(params):
    f = lambda obs: jnp.clip(policy_mean(params, obs), -1.0, 1.0)  # noqa: E731
    f.params = params  # lets viz trace params through jit instead of recompiling per checkpoint
    return f


def train(steps=30e6, out="results/run", seed=0, viz_every=10, **kw):
    from . import viz  # matplotlib only needed here

    cfg = Config(**kw)
    env_cfg = E.EnvConfig()
    os.makedirs(out, exist_ok=True)
    n_updates = int(steps // (cfg.n_envs * cfg.n_steps))
    sched = optax.linear_schedule(cfg.lr, cfg.lr * 0.1, n_updates * cfg.epochs * cfg.minibatches)
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad), optax.adam(sched, eps=1e-5))
    key = jax.random.PRNGKey(seed)
    key, kp, ke = jax.random.split(key, 3)
    params = init_params(kp)
    opt_state = opt.init(params)
    st = jax.vmap(E.reset, in_axes=(0, None))(jax.random.split(ke, cfg.n_envs), env_cfg)
    obs = jax.vmap(E.observe)(st)
    update = make_update(cfg, env_cfg, opt)
    rows, t0 = [], time.time()
    snapshots = []
    print(f"{n_updates} updates x {cfg.n_envs * cfg.n_steps} steps")
    for it in range(n_updates):
        if it % viz_every == 0 or it == n_updates - 1:
            snapshots.append((it, viz.scenario_trace(deterministic_policy(params))))
            viz.progress_figure(snapshots, os.path.join(out, "learning_progress.png"))
        key, k = jax.random.split(key)
        params, opt_state, st, obs, logs = update(params, opt_state, st, obs, k)
        logs = {n: float(v) for n, v in logs.items()}
        km = max(logs["dist"] / 1000.0, 1e-6)
        row = {"update": it, "env_steps": (it + 1) * cfg.n_envs * cfg.n_steps, "wall_s": round(time.time() - t0, 1),
               "reward": logs["reward"], "episodes": logs["episodes"],
               "speed_kmh": 3.6 * logs["dist"] / (cfg.n_envs * cfg.n_steps * E.DT),
               "hard_per_km": logs["hard"] / km,
               "coll_per_1000km": 1000 * logs["coll"] / km,
               "red_per_100seg": 100 * logs["red"] / max(logs["segs"], 1),
               "shield_pct": 100 * logs["shield"] / (cfg.n_envs * cfg.n_steps),
               "kl": logs["kl"], "vloss": logs["vloss"], "std": logs["std"]}
        rows.append(row)
        if it % 5 == 0 or it == n_updates - 1:
            sps = row["env_steps"] / max(row["wall_s"], 1e-6)
            print(f"[{it:4d}] {row['env_steps'] / 1e6:6.1f}M steps {sps / 1e3:6.0f}k/s  reward {row['reward']:+.4f}  "
                  f"speed {row['speed_kmh']:5.1f} km/h  hard/km {row['hard_per_km']:.3f}  coll/1000km {row['coll_per_1000km']:.2f}  "
                  f"red/100seg {row['red_per_100seg']:.2f}  shield {row['shield_pct']:.2f}%  std {row['std']:.3f}", flush=True)
            viz.training_curves(rows, os.path.join(out, "training_curves.png"))
            save_params(params, os.path.join(out, "policy.npz"))
    with open(os.path.join(out, "log.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    save_params(params, os.path.join(out, "policy.npz"))
    viz.training_curves(rows, os.path.join(out, "training_curves.png"))
    return params, rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=float, default=30e6)
    ap.add_argument("--envs", type=int, default=4096)
    ap.add_argument("--out", default="results/run")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--viz-every", type=int, default=10)
    a = ap.parse_args()
    train(a.steps, a.out, a.seed, a.viz_every, n_envs=a.envs)
