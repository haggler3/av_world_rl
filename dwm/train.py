"""Train and evaluate the diffusion world model under a wall-clock budget.

    python -m dwm.train --budget 540 --out results/dwm/final            # the full run (< 10 min including eval)
"""
import argparse
import json
import os
import pickle
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import model as M
from .data import CHANNELS, F, H, NC, load

DT = 0.1
SCALE = np.array([c[2] for c in CHANNELS], np.float32)  # feature units -> physical units
EVAL_CH = {"speed": 0, "gap": 3, "lead_v": 4, "d_line": 1}  # channel index in the future chunk
HORIZONS = {"1s": 9, "2s": 19, "3s": 29}


class Stats:
    def __init__(self, d, n=200_000, seed=0):
        tr = np.where(~d["test"])[0]
        tr = np.sort(np.random.default_rng(seed).choice(tr, min(n, len(tr)), replace=False))  # stats from a sample
        self.h_mu = d["hist"][tr].reshape(-1, d["hist"].shape[-1]).mean(0)
        self.h_sd = d["hist"][tr].reshape(-1, d["hist"].shape[-1]).std(0) + 1e-3
        delta = (d["fut"][tr] - d["cur"][tr][:, None]).std(0) + 1e-3
        resid = (d["fut"][tr] - d["kin"][tr]).std(0)
        self.sd = {"delta": delta, "residual": np.maximum(resid, 0.05 * delta)}  # [F, C]; floor keeps tiny residuals sane


def kinematic(cur_feat, hist, act):
    """Kinematic prediction [N, F, C] in feature units: own actions through the Rust vehicle model (0.25 s actuator lag,
    2 x 0.05 s substeps), the leader at constant speed, flags held."""
    cur = cur_feat * SCALE
    v0, dl0, gap0, vl0 = cur[:, 0], cur[:, 1], cur[:, 3], cur[:, 4]
    out = np.repeat(cur[:, None], F, 1)
    v, a = v0.copy(), hist[:, -1, 1] * 3.0
    s_ego = np.zeros_like(v)
    alpha = 0.05 / (0.25 + 0.05)
    for k in range(F):
        cmd = np.where(act[:, k] >= 0, 3.0 * act[:, k], 8.0 * act[:, k])
        for _ in range(2):
            a = a + alpha * (cmd - a)
            vn = np.maximum(v + a * 0.05, 0.0)
            s_ego += 0.5 * (v + vn) * 0.05
            v = vn
        out[:, k, 0] = v
        out[:, k, 1] = dl0 - s_ego
        out[:, k, 3] = gap0 + vl0 * DT * (k + 1) - s_ego
    return (out / SCALE).astype(np.float32)


def base_of(cfg: M.Cfg, d, idx=None):
    sl = slice(None) if idx is None else idx
    if cfg.target == "residual":
        return d["kin"][sl]
    return np.repeat(d["cur"][sl][:, None], F, 1)


def make_cond(cfg: M.Cfg, st: Stats, hist, act):
    h = (hist - st.h_mu) / st.h_sd
    h = h[:, H - cfg.hist_len:].reshape(h.shape[0], -1)
    if not cfg.use_actions:
        return h.astype(np.float32)
    parts = [h, act] + ([np.ones((len(h), 1))] if cfg.cond_drop > 0 else [])  # last column: "actions present"
    return np.concatenate(parts, 1).astype(np.float32)


def get_codec(cfg: M.Cfg, d, st: Stats):
    """(Dec, Enc, z_sd) for the non-legacy bases; z_sd normalises the latent to unit scale."""
    from . import basis as Bz

    key = ("codec", cfg.basis, cfg.nb, cfg.target)
    if key not in _CACHE:
        Dec, Enc = Bz.codec(cfg.basis, cfg.nb, st.sd[cfg.target], SCALE)
        tr = np.where(~d["test"])[0][:100_000]
        r = ((d["fut"][tr] - base_of(cfg, d, tr)) / st.sd[cfg.target]).reshape(len(tr), -1)
        z = r @ Enc.T
        z_sd = z.std(0) + 1e-3
        # what the basis cannot represent (tracker noise, leader switches): its spread per (step, channel)
        res = (r - z @ Dec.T).reshape(len(tr), F, NC)
        mad = 1.4826 * np.median(np.abs(res - np.median(res, 0)), 0)  # robust: leader switches make the std meaningless
        _CACHE[("obs_sd",) + key[1:]] = mad.astype(np.float32)
        _CACHE[key] = (Dec, Enc, z_sd.astype(np.float32))
    return _CACHE[key]


def with_latent(cfg: M.Cfg):
    """Fill in the latent length for a basis config."""
    import dataclasses

    from . import basis as Bz

    if cfg.basis == "legacy":
        return cfg
    return dataclasses.replace(cfg, lat=Bz.physical_decoder(cfg.basis, cfg.nb).shape[1])


_CACHE = {}


def prepared(cfg: M.Cfg, d, st: Stats):
    """Training arrays on device, cached across runs that share the conditioning and the target."""
    tr = ~d["test"]
    kc = (cfg.hist_len, cfg.use_actions, cfg.cond_drop > 0)
    if kc not in _CACHE:
        _CACHE[kc] = jnp.asarray(make_cond(cfg, st, d["hist"][tr], d["act"][tr]))
    kt = (cfg.target, cfg.knots, cfg.basis, cfg.nb)
    if kt not in _CACHE and cfg.basis != "legacy":
        Dec, Enc, z_sd = get_codec(cfg, d, st)
        r = ((d["fut"][tr] - base_of(cfg, d, tr)) / st.sd[cfg.target]).reshape(int(tr.sum()), -1)
        _CACHE[kt] = jnp.asarray(((r @ Enc.T) / z_sd)[..., None])
    if kt not in _CACHE:
        _, P = M.upsampler(cfg)
        _CACHE[kt] = jnp.asarray(np.einsum("lf,nfc->nlc", P, (d["fut"][tr] - base_of(cfg, d, tr)) / st.sd[cfg.target]))
    return _CACHE[kc], _CACHE[kt]


def train(cfg: M.Cfg, d, st: Stats, budget_s, log_every=100, verbose=True, max_steps=None):
    cond, x0 = prepared(cfg, d, st)
    n = cond.shape[0]
    ab = M.alphas_bar(cfg.T)
    key = jax.random.PRNGKey(cfg.seed)
    key, k0 = jax.random.split(key)
    p = M.init(k0, cfg)
    ema = p
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.scale_by_adam(), optax.add_decayed_weights(1e-4))
    o = opt.init(p)

    @jax.jit
    def chunk(p, ema, o, key, frac0, frac1, cond, x0):
        def body(c, xs):
            p, ema, o = c
            k, frac = xs
            k1, k2 = jax.random.split(k)
            idx = jax.random.randint(k1, (cfg.batch,), 0, n)
            l, g = jax.value_and_grad(M.loss_fn)(p, cfg, ab, x0[idx], cond[idx], k2)
            lr = cfg.lr * 0.5 * (1 + jnp.cos(jnp.pi * jnp.clip(frac, 0, 1)))  # cosine decay over the budget
            upd, o = opt.update(g, o, p)
            upd = jax.tree.map(lambda u: -lr * u, upd)
            p = optax.apply_updates(p, upd)
            ema = jax.tree.map(lambda e, q: cfg.ema * e + (1 - cfg.ema) * q, ema, p)
            return (p, ema, o), l

        fr = jnp.linspace(frac0, frac1, log_every)
        (p, ema, o), ls = jax.lax.scan(body, (p, ema, o), (jax.random.split(key, log_every), fr))
        return p, ema, o, ls.mean()

    t0 = time.time()
    steps, curve, frac = 0, [], 0.0
    while True:
        key, k = jax.random.split(key)
        el = time.time() - t0
        if max_steps:
            f0, f1 = steps / max_steps, min(1.0, (steps + log_every) / max_steps)
        else:
            f0 = el / budget_s
            f1 = min(1.0, f0 + (el / max(steps, 1)) * log_every / budget_s) if steps else 0.02
        p, ema, o, l = chunk(p, ema, o, k, f0, f1, cond, x0)
        l = float(l)
        steps += log_every
        curve.append((steps, time.time() - t0, l))
        if verbose and len(curve) % 5 == 1:
            print(f"  step {steps:6d}  {time.time() - t0:6.1f}s  loss {l:.4f}", flush=True)
        if time.time() - t0 > budget_s or (max_steps and steps >= max_steps):
            break
    return {"params": ema, "steps": steps, "train_s": time.time() - t0, "curve": curve,
            "n_params": M.n_params(p)}


# ------------------------------------------------------------------ evaluation
def crps(samples, y):
    """samples [N, K], y [N] -> [N]   (CRPS = E|X - y| - 0.5 E|X - X'|; equals MAE for one sample)."""
    a = np.abs(samples - y[:, None]).mean(1)
    b = np.abs(samples[:, :, None] - samples[:, None, :]).mean((1, 2))
    return a - 0.5 * b


def baselines(d, idx):
    """Constant velocity, and kinematic (own actions through the vehicle model, leader at constant speed). [N, 1, F, C]."""
    cur = d["cur"][idx]
    t = (np.arange(F) + 1) * DT
    cv = np.repeat(cur[:, None] * SCALE, F, 1)
    v0, vl0 = cur[:, 0] * SCALE[0], cur[:, 4] * SCALE[4]
    cv[:, :, 1] = cur[:, None, 1] * SCALE[1] - v0[:, None] * t
    cv[:, :, 3] = cur[:, None, 3] * SCALE[3] + (vl0 - v0)[:, None] * t
    return {"constant velocity": cv[:, None], "kinematic": (d["kin"][idx] * SCALE)[:, None]}


def load_data(path):
    d = load(path)
    if "kin" not in d:
        d["kin"] = kinematic(d["cur"], d["hist"], d["act"])
        np.save(os.path.join(path, "kin.npy"), d["kin"])
    return d


def decode(cfg, d, st, x):
    """Model output [B, L, C'] -> normalised residual [B, F, C]."""
    if cfg.basis == "legacy":
        return np.einsum("fl,blc->bfc", M.upsampler(cfg)[0], x)
    Dec, _, z_sd = get_codec(cfg, d, st)
    return ((x[..., 0] * z_sd) @ Dec.T).reshape(len(x), F, NC)


def sample_physical(res, cfg, st, d, idx, K, seed=0, steps=None, guidance=None, eta=None):
    cond = jnp.asarray(make_cond(cfg, st, d["hist"][idx], d["act"][idx]))
    ab = M.alphas_bar(cfg.T)
    N = cond.shape[0]
    cond_k = jnp.repeat(cond, K, 0)
    fn = jax.jit(lambda p, c, k: M.sample(p, cfg, ab, c, k, steps, guidance, eta))
    t0 = time.time()
    x = np.asarray(jax.block_until_ready(fn(res["params"], cond_k, jax.random.PRNGKey(seed))))
    sample_s = time.time() - t0
    x = decode(cfg, d, st, x)
    if cfg.obs_noise and cfg.basis != "legacy":
        get_codec(cfg, d, st)
        obs_sd = _CACHE[("obs_sd", cfg.basis, cfg.nb, cfg.target)]
        x = x + obs_sd * np.random.default_rng(seed + 1).normal(size=x.shape).astype(np.float32)
    x = x.reshape(N, K, F, NC) * st.sd[cfg.target] + base_of(cfg, d, idx)[:, None]
    return x * SCALE, sample_s


def metrics(pred, d, idx):
    """pred [N, K, F, C] physical. CRPS per channel/horizon (gap & lead speed only where a leader is seen), coverage, flags."""
    real = d["fut"][idx] * SCALE
    seen_now = d["cur"][idx][:, 2] > 0.5
    out = {}
    for name, c in EVAL_CH.items():
        for hn, h in HORIZONS.items():
            m = np.ones(len(idx), bool)
            if name in ("gap", "lead_v"):
                m = seen_now & (real[:, h, 2] > 0.5)
            out[f"crps_{name}_{hn}"] = float(crps(pred[m, :, h, c], real[m, h, c]).mean())
    for name in ("speed", "gap"):
        c, h = EVAL_CH[name], 29
        m = np.ones(len(idx), bool) if name == "speed" else seen_now & (real[:, h, 2] > 0.5)
        if pred.shape[1] > 1:
            lo, hi = np.percentile(pred[m, :, h, c], 5, 1), np.percentile(pred[m, :, h, c], 95, 1)
            out[f"coverage90_{name}_3s"] = float(((real[m, h, c] >= lo) & (real[m, h, c] <= hi)).mean())
        out[f"mae_mean_{name}_3s"] = float(np.abs(pred[m, :, h, c].mean(1) - real[m, h, c]).mean())
    # roughness: mean |second difference| of the gap per step, sampled futures relative to the real ones (1 = as smooth as reality)
    m = seen_now & (real[:, :, 2] > 0.5).all(1)
    r_pred = np.abs(np.diff(pred[m, :, :, 3], 2, axis=-1)).mean()
    r_real = np.abs(np.diff(real[m, :, 3], 2, axis=-1)).mean()
    out["roughness_gap_vs_real"] = float(r_pred / max(r_real, 1e-9))
    for name, c in (("lead_seen", 2), ("must_stop", 5)):
        out[f"acc_{name}_3s"] = float(((pred[:, :, 29, c].mean(1) > 0.5) == (real[:, 29, c] > 0.5)).mean())
    return out


def skill(m, ref):
    """Mean CRPS ratio vs a reference over speed and gap at 1/2/3 s (lower is better; < 1 beats the reference)."""
    ks = [f"crps_{n}_{h}" for n in ("speed", "gap") for h in HORIZONS]
    return float(np.mean([m[k] / ref[k] for k in ks]))


def eval_idx(d, n, seed=1):
    te = np.where(d["test"])[0]
    return np.sort(np.random.default_rng(seed).choice(te, n, replace=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dwm")
    ap.add_argument("--budget", type=float, default=540)
    ap.add_argument("--out", default="results/dwm/final")
    ap.add_argument("--cfg", default="{}", help="JSON overrides of model.Cfg")
    ap.add_argument("--eval-n", type=int, default=4000)
    ap.add_argument("--K", type=int, default=32)
    a = ap.parse_args()
    t_all = time.time()
    os.makedirs(a.out, exist_ok=True)
    d = load_data(a.data)
    st = Stats(d)
    cfg = M.Cfg(**json.loads(a.cfg))
    print(cfg)
    res = train(cfg, d, st, a.budget)
    idx = eval_idx(d, a.eval_n)
    pred, sample_s = sample_physical(res, cfg, st, d, idx, a.K if cfg.kind != "regress" else 1)
    out = {"cfg": cfg.__dict__ | {"extra": None}, "steps": res["steps"], "train_s": round(res["train_s"], 1),
           "n_params": res["n_params"], "sample_ms_per_future": round(1000 * sample_s / (len(idx) * pred.shape[1]), 4)}
    base = {k: metrics(v, d, idx) for k, v in baselines(d, idx).items()}
    out["model"] = metrics(pred, d, idx)
    out["baselines"] = base
    out["skill_vs_kinematic"] = skill(out["model"], base["kinematic"])
    out["total_wall_s"] = round(time.time() - t_all, 1)
    with open(os.path.join(a.out, "metrics.json"), "w") as f:
        json.dump(out, f, indent=1)
    with open(os.path.join(a.out, "model.pkl"), "wb") as f:
        pickle.dump({"params": jax.tree.map(np.asarray, res["params"]), "cfg": cfg, "stats": st.__dict__}, f)
    np.save(os.path.join(a.out, "curve.npy"), np.array(res["curve"]))
    print(json.dumps({k: v for k, v in out.items() if k != "baselines"}, indent=1))


if __name__ == "__main__":
    main()
