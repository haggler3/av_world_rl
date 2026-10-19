"""Multi-agent PPO in a whole simulated city area (av_world's `xnet`: Oakland, Pittsburgh).

Every AV in the area is driven by one shared policy, so one simulator step of E copies of the area gives hundreds of
transitions (about 1,000 for 8 copies). The policy outputs

  u[0]  an acceleration command in [-1, 1] -> [-8, 3] m/s^2, capped by the classical safety envelope (IDM bounds toward
        the stop target and the leader, and the speed limit), as in the corridor env;
  u[1]  a gap-acceptance scale for its next junction (tanh -> 0.8 .. 1.5 x the human critical gaps; the rule-based AV
        uses 1.25): how cautiously to turn across or merge into traffic.

The rest of the area (humans with the calibrated risky behaviour, pedestrians, buses, vans, signals, right of way) is the
xnet simulator, unchanged, and it reacts to the AVs. Each copy gets its own AV share, demand and human risk level.

Training: behaviour cloning of the rule-based AV for a quick start, then PPO. One update (rollout of all copies, GAE per
vehicle slot with masks, epochs) is one jitted function.

    python -m avrl.xppo --out results/xnet            # ~10 min on 4 CPU cores
"""
import argparse
import json
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from xnet import network as NW
from xnet import sim as X

from .ppo import init_mlp, mlp

HIDDEN = 64
U_DIM = 2
U_RULE = jnp.array([0.0, float(np.arctanh((1.25 - 1.15) / 0.35))])  # the rule-based AV's gap acceptance


def init_params(key):
    k1, k2 = jax.random.split(key)
    return {"pi": init_mlp(k1, [X.OBS_DIM, HIDDEN, HIDDEN, U_DIM]), "v": init_mlp(k2, [X.OBS_DIM, HIDDEN, HIDDEN, 1]),
            "log_std": jnp.array([-0.7, -1.0])}


def mean(params, obs):
    return mlp(params["pi"], obs)


def value(params, obs):
    return mlp(params["v"], obs)[..., 0]


def log_prob(params, obs, u):
    mu, ls = mean(params, obs), params["log_std"]
    return (-0.5 * ((u - mu) / jnp.exp(ls)) ** 2 - ls - 0.5 * jnp.log(2 * jnp.pi)).sum(-1)


def u_to_accel(u0):
    u0 = jnp.clip(u0, -1.0, 1.0)
    return jnp.where(u0 >= 0, 3.0 * u0, 8.0 * u0)


def accel_to_u(a):
    a = jnp.clip(a, -8.0, 3.0)
    return jnp.where(a >= 0, a / 3.0, a / 8.0)


def stochastic(params):
    def f(obs, key):
        mu = mean(params, obs)
        u = mu + jnp.exp(params["log_std"]) * jax.random.normal(key, mu.shape)
        return u, jnp.stack([log_prob(params, obs, u), value(params, obs)], -1)
    return f


def deterministic(params):
    def f(obs, key):
        return mean(params, obs), jnp.zeros((obs.shape[0], 2))
    return f


class Config:
    def __init__(self, **kw):
        self.area = "oakland"
        self.n_envs = 8
        self.n_slots = 1024
        self.n_steps = 64  # 12.8 s per update
        self.epochs = 4
        self.minibatches = 4
        self.lr = 3e-4
        self.gamma = 0.99  # per 0.2 s
        self.lam = 0.95
        self.clip = 0.2
        self.vf = 0.5
        self.max_grad = 0.5
        self.warm = 300.0
        self.bc_updates = 60
        self.reward = {}  # overrides of xnet.sim.Reward
        self.reset_every = 15  # updates (a copy restarts every 60 updates = 13 simulated minutes)
        self.__dict__.update(kw)


def behaviours(n, seed=0, knobs=None):
    """Per-copy domain randomisation: AV share, demand, human risk (calibrated rates x 1, 5 or 15)."""
    rng = np.random.default_rng(seed)
    knobs = knobs or {}
    rows = []
    for e in range(n):
        A = [1.0, 5.0, 15.0][e % 3]
        rows.append(X.Behaviour(**{k: min(v * A, 0.95) if k != "distract_rate" else v * A for k, v in knobs.items()},
                                p_av=[0.15, 0.4, 1.0][(e // 3) % 3] if n > 1 else 0.3, demand=float(rng.uniform(0.8, 1.3))))
    return jax.tree.map(lambda *x: jnp.asarray(x, jnp.float32), *rows)


def make_fns(cfg: Config, net, beh, opt):
    nh = jnp.asarray(net["nh"])
    J = X.to_jnp(net)
    E = cfg.n_envs

    def env_step(params, st, key, pol):
        keys = jax.random.split(key, E)
        return jax.vmap(lambda s, b, k: X.step(s, J, b, nh, rl=(pol(params), k)))(st, beh, keys)

    def rollout(params, st, key, pol):
        def body(st, k):
            st, tr = env_step(params, st, k, pol)
            return st, tr
        return jax.lax.scan(body, st, jax.random.split(key, cfg.n_steps + 1))

    @jax.jit
    def warm(params, st, key):
        return rollout(params, st, key, deterministic)[0]

    def bc_loss(params, tr):
        target = jnp.stack([accel_to_u(tr["a_rule"]), jnp.broadcast_to(U_RULE[1], tr["a_rule"].shape)], -1)
        err = ((mean(params, tr["obs"]) - target) ** 2).sum(-1)
        w = tr["valid"].astype(jnp.float32)
        return (err * w).sum() / jnp.maximum(w.sum(), 1.0)

    @jax.jit
    def bc_update(params, opt_state, st, key):
        st, tr = rollout(params, st, key, deterministic)  # the rule-based shield caps it, close to the rule driver

        def one(c, _):
            params, opt_state = c
            loss, g = jax.value_and_grad(bc_loss)(params, tr)
            upd, opt_state = opt.update(g, opt_state, params)
            return (optax.apply_updates(params, upd), opt_state), loss

        (params, opt_state), loss = jax.lax.scan(one, (params, opt_state), None, length=8)
        return params, opt_state, st, loss[-1]

    def gae(tr):
        r, d, v_ = tr["reward"], tr["done"].astype(jnp.float32), tr["aux"][..., 1]
        valid = tr["valid"].astype(jnp.float32)
        v_next = v_[1:]
        r, d, v_, valid = r[:-1], d[:-1], v_[:-1], valid[:-1]

        def body(adv_next, x):
            r_t, d_t, v_t, vn_t, ok = x
            nd = (1.0 - d_t) * ok
            delta = r_t + cfg.gamma * vn_t * nd - v_t
            adv = (delta + cfg.gamma * cfg.lam * nd * adv_next) * ok
            return adv, adv

        _, adv = jax.lax.scan(body, jnp.zeros_like(r[0]), (r, d, v_, v_next, valid), reverse=True)
        return adv, adv + v_, valid

    def ppo_loss(params, b):
        obs, u, lp_old, adv, ret, w = b
        lp = log_prob(params, obs, u)
        ratio = jnp.exp(lp - lp_old)
        n = jnp.maximum(w.sum(), 1.0)
        mu_a = (adv * w).sum() / n
        sd_a = jnp.sqrt(((adv - mu_a) ** 2 * w).sum() / n) + 1e-8
        adv = (adv - mu_a) / sd_a
        pg = -(jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * adv) * w).sum() / n
        vl = (((value(params, obs) - ret) ** 2) * w).sum() / n
        kl = ((((ratio - 1) - jnp.log(ratio)) * w).sum() / n)
        return pg + cfg.vf * vl, (pg, vl, kl)

    @jax.jit
    def update(params, opt_state, st, key):
        k_roll, k_perm = jax.random.split(key)
        m0 = st.m
        st, tr = rollout(params, st, k_roll, stochastic)
        adv, ret, w = gae(tr)
        T = cfg.n_steps
        flat = lambda x: x[:T].reshape((T * E * cfg.n_slots,) + x.shape[3:])
        data = (flat(tr["obs"]), flat(tr["u"]), flat(tr["aux"][..., 0]), adv.reshape(-1), ret.reshape(-1), w.reshape(-1))
        n = data[0].shape[0]
        mb = n // cfg.minibatches

        def epoch(c, k):
            params, opt_state = c
            perm = jax.random.permutation(k, n)[: mb * cfg.minibatches].reshape(cfg.minibatches, mb)

            def one(c, idx):
                params, opt_state = c
                b = tuple(x[idx] for x in data)
                (loss, aux), g = jax.value_and_grad(ppo_loss, has_aux=True)(params, b)
                upd, opt_state = opt.update(g, opt_state, params)
                return (optax.apply_updates(params, upd), opt_state), aux

            return jax.lax.scan(one, (params, opt_state), perm)

        (params, opt_state), aux = jax.lax.scan(epoch, (params, opt_state), jax.random.split(k_perm, cfg.epochs))
        nv = jnp.maximum(w.sum(), 1.0)
        dm = (st.m - m0).sum(0)
        stats = {"reward_per_step": (tr["reward"][:T] * tr["valid"][:T]).sum() / nv, "transitions": w.sum(),
                 "pg": aux[0].mean(), "vl": aux[1].mean(), "kl": aux[2].mean(), "m": dm}
        return params, opt_state, st, stats

    return warm, bc_update, update


def summarize(m):
    d = dict(zip(X.METRICS, np.asarray(m)))
    av_h = max(d["av_veh_s"] / 3600, 1e-9)
    d = {k: float(v) for k, v in d.items()}
    return {"av_hours": round(d["av_veh_s"] / 3600, 2), "av_speed_kmh": 3.6 * d["av_veh_m"] / max(d["av_veh_s"], 1),
            "av_hard_per_h": d["av_hard"] / av_h, "av_jerk_rms": float(np.sqrt(d["av_jerk2"] / max(d["av_veh_s"], 1))),
            "av_crashes": int(d["av_crash"]), "av_red": int(d["av_red"]), "av_near": int(d["av_near"]),
            "av_shield_share": d["av_shield"] / max(d["av_veh_s"] / X.DT, 1),
            "av_trip_min": d["av_trip_s"] / max(d["av_finished"], 1) / 60,
            "human_trip_min": (d["trip_s"] - d["av_trip_s"]) / max(d["finished"] - d["av_finished"], 1) / 60,
            "trips_finished": int(d["finished"]), "teleports": int(d["teleports"]),
            "crashes": int(sum(d[k] for k in X.CRASH_NAMES))}


def save(params, path):
    flat = {"log_std": np.asarray(params["log_std"])}
    for name in ("pi", "v"):
        for i, l in enumerate(params[name]):
            flat[f"{name}{i}_w"], flat[f"{name}{i}_b"] = np.asarray(l["w"]), np.asarray(l["b"])
    np.savez(path, **flat)


def load(path):
    z = np.load(path)
    p = {"log_std": jnp.asarray(z["log_std"])}
    for name in ("pi", "v"):
        p[name] = [{"w": jnp.asarray(z[f"{name}{i}_w"]), "b": jnp.asarray(z[f"{name}{i}_b"])} for i in range(3)]
    return p


def train(out="results/xnet", updates=200, seed=0, init=None, **kw):
    if init is not None:
        kw.setdefault("bc_updates", 0)
    cfg = Config(**kw)
    X.R = X.Reward(**cfg.reward)  # reward weights (read when the step is traced)
    os.makedirs(out, exist_ok=True)
    net, geo, info = NW.build(cfg.area)
    knobs = json.load(open(os.path.join(os.path.dirname(NW.__file__), "..", "results", "calibration.json")))["calibrated"]
    beh = behaviours(cfg.n_envs, seed, knobs)
    key = jax.random.PRNGKey(seed)
    params = init_params(key) if init is None else load(init)
    J = X.to_jnp(net)
    st = jax.vmap(lambda k: X.reset(k, J, cfg.n_slots))(jax.random.split(key, cfg.n_envs))
    warm, bc_update, update = make_fns(cfg, net, beh, optax.chain(optax.clip_by_global_norm(cfg.max_grad), optax.adam(cfg.lr)))
    t0 = time.time()
    for i in range(int(cfg.warm / (X.DT * (cfg.n_steps + 1)))):
        st = warm(params, st, jax.random.fold_in(key, 10_000 + i))
    jax.block_until_ready(st)
    print(f"warm-up {time.time() - t0:.0f} s", flush=True)
    log = []
    bc_opt = optax.adam(1e-3)
    bc_state = bc_opt.init(params)
    _, bc_update, _ = make_fns(cfg, net, beh, bc_opt)
    for i in range(cfg.bc_updates):
        params, bc_state, st, loss = bc_update(params, bc_state, st, jax.random.fold_in(key, 20_000 + i))
        if i % 25 == 0:
            print(f"bc {i:4d} loss {float(loss):.4f}  {time.time() - t0:.0f} s", flush=True)
    if cfg.bc_updates:
        save(params, os.path.join(out, "policy_bc.npz"))
    opt = optax.chain(optax.clip_by_global_norm(cfg.max_grad), optax.adam(cfg.lr))
    opt_state = opt.init(params)
    fresh = jax.vmap(lambda k: X.reset(k, J, cfg.n_slots))
    for i in range(updates):
        if cfg.reset_every and i > 0 and i % cfg.reset_every == 0:  # restart a quarter of the copies (jams never stick)
            sel = (jnp.arange(cfg.n_envs) % 4) == (i // cfg.reset_every) % 4
            new = fresh(jax.random.split(jax.random.fold_in(key, 30_000 + i), cfg.n_envs))
            st = jax.tree.map(lambda a, b: jnp.where(sel.reshape((-1,) + (1,) * (a.ndim - 1)), b, a), st, new)
        params, opt_state, st, s = update(params, opt_state, st, jax.random.fold_in(key, i))
        row = {"update": i, "time_s": round(time.time() - t0, 1), "reward_per_step": float(s["reward_per_step"]),
               "transitions": int(s["transitions"]), "kl": float(s["kl"]), "vl": float(s["vl"]),
               "log_std": np.asarray(params["log_std"]).round(3).tolist(), **summarize(s["m"])}
        log.append(row)
        if i % 10 == 0 or i == updates - 1:
            print(json.dumps({k: (round(v, 3) if isinstance(v, float) else v) for k, v in row.items()}, default=float), flush=True)
            save(params, os.path.join(out, "policy.npz"))
    with open(os.path.join(out, "train_log.json"), "w") as f:
        json.dump(log, f, default=float)
    save(params, os.path.join(out, "policy.npz"))
    return params


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/xnet")
    ap.add_argument("--updates", type=int, default=200)
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--bc", type=int, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--init", default=None, help="continue from a saved policy (skips behaviour cloning)")
    ap.add_argument("--reward", default="{}", help='JSON overrides of xnet.sim.Reward, e.g. \'{"progress": 0.04}\'')
    a = ap.parse_args()
    train(a.out, a.updates, a.seed, init=a.init, n_envs=a.envs, reward=json.loads(a.reward), **({} if a.init else {"bc_updates": a.bc}))
