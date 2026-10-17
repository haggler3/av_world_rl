"""A learned world model in the policy's own feature space: (obs_t, u_t) -> obs_{t+1}, trained on transitions logged
by the AVs inside `small_av_world` (AV_TRANSITIONS=file cargo run --release -- fleet --driver learned ...).

The hand-written corridor env (`env.py`) is fast but only as faithful as its assumptions. This model is learned from the
city itself, so comparing the two on held-out city data measures the sim-to-world gap, and the model can serve as a
"dream" environment for fine-tuning the policy on what the city actually does.

    python -m avrl.worldmodel collect-env --out data/env.bin          # same format, from the JAX env (for comparison)
    python -m avrl.worldmodel train --world data/world.bin --env data/env.bin --out results/wm
"""
import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import env as E
from . import ppo

ROW = E.OBS_DIM + 4  # [vehicle id, t, segment, obs.., u]
LIGHT = [4, 5, 6]
BINARY = [7, 8, 9, 10, 15]
CONT = [i for i in range(E.OBS_DIM) if i not in LIGHT + BINARY]
NAMES = ["v", "accel", "vmax", "d_line", "red", "yellow", "green", "room", "arrives", "must_stop", "lead_seen", "gap",
         "lead_v", "dv", "dgap", "lead2_seen", "gap2", "last_cmd", "req_line", "req_lead", "thw", "idle_t"]


def load_rows(path):
    a = np.fromfile(path, dtype="<f4")
    return a.reshape(-1, ROW)


def transitions(rows):
    """Consecutive decisions of the same vehicle on the same segment, 0.1 s apart."""
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    same = (rows[1:, 0] == rows[:-1, 0]) & (rows[1:, 2] == rows[:-1, 2]) & (np.abs(rows[1:, 1] - rows[:-1, 1] - 0.1) < 0.02)
    i = np.where(same)[0]
    o, u, o2 = rows[i, 3:3 + E.OBS_DIM], rows[i, 3 + E.OBS_DIM], rows[i + 1, 3:3 + E.OBS_DIM]
    return o.astype(np.float32), u.astype(np.float32), o2.astype(np.float32), rows[i, 0]


def collect_env(policy_path, out, n_envs=256, steps=3000, seed=0):
    """Log the JAX env driven by the same policy, in the Rust logger's row format (segment index = m_segs)."""
    params = ppo.load_params(policy_path)
    cfg = E.EnvConfig()
    st = jax.vmap(E.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(seed), n_envs), cfg)
    obs = jax.vmap(E.observe)(st)

    @jax.jit
    def body(c, xs):
        st, obs, segc = c
        k, i = xs
        u = jnp.clip(ppo.policy_mean(params, obs), -1, 1)
        sp = jax.vmap(E.make_spare, in_axes=(0, None))(jax.random.split(k, n_envs), cfg)
        st2, obs2, r, d, info = jax.vmap(E.step, in_axes=(0, 0, None, 0))(st, u, cfg, sp)
        row = jnp.concatenate([jnp.arange(n_envs, dtype=jnp.float32)[:, None], jnp.full((n_envs, 1), i * E.DT),
                               segc[:, None], obs, u[:, None]], axis=1)
        # a new segment or a new episode both break the transition chain
        return (st2, obs2, segc + info["s_segs"] + d), row

    keys = jax.random.split(jax.random.PRNGKey(seed + 1), steps)
    (_, _, _), rows = jax.lax.scan(body, (st, obs, jnp.zeros(n_envs)), (keys, jnp.arange(steps, dtype=jnp.float32)))
    rows = np.asarray(rows).reshape(-1, ROW)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    rows.astype("<f4").tofile(out)
    print(f"wrote {len(rows)} rows to {out}")


# ---------------------------------------------------------------- model
def init(key, hidden=128, n_models=3):
    keys = jax.random.split(key, n_models)
    return [ppo.init_mlp(k, [E.OBS_DIM + 1, hidden, hidden, len(CONT) + len(BINARY) + len(LIGHT)]) for k in keys]


def predict(model, norm, o, u):
    """Mean over the ensemble: next continuous features, P(binary flags), P(light)."""
    mu, sd = norm
    x = jnp.concatenate([o, u[..., None]], -1)
    outs = [ppo.mlp(m, x) for m in model]
    dc = jnp.mean(jnp.stack([y[..., :len(CONT)] for y in outs]), 0)
    pb = jnp.mean(jnp.stack([jax.nn.sigmoid(y[..., len(CONT):len(CONT) + len(BINARY)]) for y in outs]), 0)
    pl = jnp.mean(jnp.stack([jax.nn.softmax(y[..., -3:]) for y in outs]), 0)
    cont = o[..., jnp.array(CONT)] + dc * sd + mu
    return cont, pb, pl


def assemble(o, cont, pb, pl):
    """Write predictions back into a full observation vector (flags thresholded, light argmax)."""
    o2 = o.at[..., jnp.array(CONT)].set(cont)
    o2 = o2.at[..., jnp.array(BINARY)].set((pb > 0.5).astype(o.dtype))
    o2 = o2.at[..., jnp.array(LIGHT)].set(jax.nn.one_hot(jnp.argmax(pl, -1), 3))
    # no leader -> the placeholder values observe() uses (gap 60 m, leader speed = own speed, ...)
    no = o2[..., 10] < 0.5
    for i, val in ((11, 1.2), (13, 0.0), (14, 0.0), (15, 0.0), (16, 1.0), (19, 0.0), (20, 2.0)):
        o2 = o2.at[..., i].set(jnp.where(no, val, o2[..., i]))
    return o2.at[..., 12].set(jnp.where(no, o2[..., 0], o2[..., 12]))


def loss_fn(m, norm, o, u, o2):
    mu, sd = norm
    y = ppo.mlp(m, jnp.concatenate([o, u[:, None]], -1))
    target = (o2[:, jnp.array(CONT)] - o[:, jnp.array(CONT)] - mu) / sd
    l_c = ((y[:, :len(CONT)] - target) ** 2).mean()
    lb = y[:, len(CONT):len(CONT) + len(BINARY)]
    l_b = optax.sigmoid_binary_cross_entropy(lb, o2[:, jnp.array(BINARY)]).mean()
    l_l = optax.softmax_cross_entropy(y[:, -3:], o2[:, jnp.array(LIGHT)]).mean()
    return l_c + l_b + l_l


def fit(data, key, epochs=30, batch=4096, lr=1e-3, n_models=3):
    o, u, o2 = (jnp.asarray(x) for x in data[:3])
    d = np.asarray(o2[:, jnp.array(CONT)] - o[:, jnp.array(CONT)])
    norm = (jnp.asarray(d.mean(0)), jnp.asarray(d.std(0) + 1e-3))
    model = init(key, n_models=n_models)
    opt = optax.adam(lr)
    n = o.shape[0]
    steps = epochs * (n // batch)

    @jax.jit
    def train_one(m, k):
        st = opt.init(m)

        def body(c, k):
            m, st = c
            idx = jax.random.randint(k, (batch,), 0, n)  # bootstrap sampling per member
            l, g = jax.value_and_grad(loss_fn)(m, norm, o[idx], u[idx], o2[idx])
            upd, st = opt.update(g, st, m)
            return (optax.apply_updates(m, upd), st), l

        (m, _), ls = jax.lax.scan(body, (m, st), jax.random.split(k, steps))
        return m, ls

    out, curves = [], []
    for i, m in enumerate(model):
        m, ls = train_one(m, jax.random.fold_in(key, i))
        out.append(m)
        curves.append(np.asarray(ls))
    return out, norm, np.mean(curves, 0)


def one_step_errors(model, norm, data):
    o, u, o2 = (jnp.asarray(x) for x in data[:3])
    cont, pb, pl = jax.jit(predict)(model, norm, o, u)
    err = np.asarray(jnp.abs(cont - o2[:, jnp.array(CONT)])).mean(0)
    acc_b = np.asarray(((pb > 0.5) == (o2[:, jnp.array(BINARY)] > 0.5)).mean(0))
    acc_l = float((jnp.argmax(pl, -1) == jnp.argmax(o2[:, jnp.array(LIGHT)], -1)).mean())
    return {NAMES[i]: float(e) for i, e in zip(CONT, err)} | {NAMES[i] + "_acc": float(a) for i, a in zip(BINARY, acc_b)} | {"light_acc": acc_l}


def open_loop(model, norm, rows, horizon=30, n=400, seed=0, need_lead=False):
    """Feed the logged actions for `horizon` steps and compare the imagined trajectory with what the city did."""
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    rng = np.random.default_rng(seed)
    starts = []
    for i in rng.permutation(len(rows) - horizon):
        w = rows[i:i + horizon + 1]
        ok = (w[:, 0] == w[0, 0]).all() and (w[:, 2] == w[0, 2]).all() and np.allclose(np.diff(w[:, 1]), 0.1, atol=0.02)
        if ok and need_lead:
            ok = (w[:, 3 + 10] > 0.5).all() and np.abs(w[-1, 3] - w[0, 3]) * E.V_SCALE > 1.0  # a leader all along, speed changes
        if ok:
            starts.append(i)
        if len(starts) >= n:
            break
    idx = np.array(starts)[:, None] + np.arange(horizon + 1)
    real = jnp.asarray(rows[idx][..., 3:3 + E.OBS_DIM])
    acts = jnp.asarray(rows[idx][..., 3 + E.OBS_DIM])

    @jax.jit
    def roll(o0, acts):
        def body(o, u):
            o2 = assemble(o, *predict(model, norm, o, u))
            return o2, o2

        _, traj = jax.lax.scan(body, o0, acts.T[:-1])
        return jnp.concatenate([o0[None], traj], 0).transpose(1, 0, 2)

    imag = roll(real[:, 0], acts)
    return np.asarray(real), np.asarray(imag)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    c = sub.add_parser("collect-env")
    c.add_argument("--policy", default="results/main/policy.npz")
    c.add_argument("--out", default="data/env.bin")
    t = sub.add_parser("train")
    t.add_argument("--world", required=True)
    t.add_argument("--env", required=True)
    t.add_argument("--out", default="results/wm")
    a = ap.parse_args()
    if a.cmd == "collect-env":
        collect_env(a.policy, a.out)
        return
    from . import viz_wm

    os.makedirs(a.out, exist_ok=True)
    key = jax.random.PRNGKey(0)
    wrows, erows = load_rows(a.world), load_rows(a.env)
    W, Ev = transitions(wrows), transitions(erows)
    # split the city data by vehicle so test vehicles are never seen in training
    ids = np.unique(W[3])
    test_ids = set(np.random.default_rng(0).choice(ids, len(ids) // 5, replace=False).tolist())
    te = np.array([i in test_ids for i in W[3]])
    W_tr, W_te = tuple(x[~te] for x in W), tuple(x[te] for x in W)
    print(f"city transitions: {len(W[0])} ({(~te).sum()} train / {te.sum()} test, {len(ids)} vehicles); env transitions: {len(Ev[0])}")
    m_world, n_world, c_world = fit(W_tr, key)
    m_env, n_env, c_env = fit(Ev, key)
    res = {"n_city_train": int((~te).sum()), "n_city_test": int(te.sum()), "n_env": int(len(Ev[0])),
           "learned_from_city": one_step_errors(m_world, n_world, W_te),
           "learned_from_jax_env": one_step_errors(m_env, n_env, W_te),
           "learned_from_jax_env_on_env": one_step_errors(m_env, n_env, Ev)}
    test_rows = wrows[np.isin(wrows[:, 0], list(test_ids))]
    for subset, need in (("all", False), ("following", True)):
        real, imag_w = open_loop(m_world, n_world, test_rows, need_lead=need)
        _, imag_e = open_loop(m_env, n_env, test_rows, need_lead=need)
        for name, imag in (("city_model", imag_w), ("env_model", imag_e)):
            for f, i in (("speed_kmh", 0), ("gap_m", 11)):
                scale = E.V_SCALE * 3.6 if f == "speed_kmh" else 50.0
                res[f"open_loop_3s_{subset}_{name}_{f}_mae"] = round(float(np.abs(imag[:, -1, i] - real[:, -1, i]).mean() * scale), 3)
    with open(os.path.join(a.out, "wm_results.json"), "w") as f:
        json.dump(res, f, indent=1)
    viz_wm.figure(real, imag_w, imag_e, c_world, c_env, os.path.join(a.out, "world_model.png"))
    np.savez(os.path.join(a.out, "world_model.npz"), mu=np.asarray(n_world[0]), sd=np.asarray(n_world[1]),
             **{f"m{k}_{nm}{i}": np.asarray(l[nm]) for k, m in enumerate(m_world) for i, l in enumerate(m) for nm in ("w", "b")})
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
