"""Diffusion world model: p(next 3 s of state | last 1 s of features, next 3 s of actions), in pure JAX.

The future is a [F, C] chunk (30 steps x 6 channels). With `target="residual"` the model only learns how the world
deviates from a kinematic prediction (the AV's own actions through its vehicle model, the leader at constant speed): the
AV's own motion is almost fully determined by its actions, the leader's is not. Targets are divided by their
per-(step, channel) standard deviation. A denoiser (residual MLP or 1-D temporal conv, both FiLM-conditioned on the
history, the actions and the noise level) is trained with the DDPM objective; sampling uses deterministic DDIM, so the
diversity of futures comes from the initial noise. `kind="regress"` is the same network trained to output the future
directly (the deterministic ablation).
"""
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

from .data import F, H, NC, OBS_DIM


@dataclass(frozen=True)
class Cfg:
    kind: str = "diffusion"  # "diffusion" | "flow" (rectified flow / flow matching) | "regress"
    target: str = "residual"  # "residual": future minus the kinematic prediction | "delta": future minus the current value
    net: str = "mlp"  # "mlp" | "conv" | "attn" (DiT-style transformer over history + future tokens)
    heads: int = 4
    pred: str = "v"  # diffusion target: "eps" | "x0" | "v"
    width: int = 256
    depth: int = 3
    T: int = 100  # training noise levels
    sample_steps: int = 10  # DDIM steps
    use_actions: bool = True
    hist_len: int = H  # how many history steps the model sees (1 = only the current features)
    knots: int = 0  # > 0: generate this many control points over the 3 s and interpolate linearly (smooth futures)
    basis: str = "legacy"  # "legacy" (steps / knots above) | "steps" | "knots" | "dct" | "leader_phys" (see basis.py)
    nb: int = 10  # latent size per channel for the bases
    lat: int = 0  # latent length, set by train.py for the non-legacy bases (the latent is [lat, 1])
    cond_drop: float = 0.0  # > 0: hide the actions this often in training (classifier-free guidance)
    guidance: float = 1.0  # sampling: out = uncond + guidance * (cond - uncond)
    eta: float = 0.0  # DDIM stochasticity: 0 deterministic, 1 = DDPM-like (more spread)
    obs_noise: bool = False  # add measurement noise with the spread of what the basis cannot represent (an observation model)
    ema: float = 0.999
    lr: float = 1e-3
    batch: int = 512
    seed: int = 0
    extra: dict = field(default_factory=dict, hash=False, compare=False)


# ------------------------------------------------------------------ schedule
def alphas_bar(T):
    """Cosine schedule (Nichol & Dhariwal)."""
    s = 0.008
    t = np.arange(T + 1) / T
    f = np.cos((t + s) / (1 + s) * np.pi / 2) ** 2
    ab = f / f[0]
    return jnp.asarray(np.clip(ab[1:], 1e-5, 1.0), jnp.float32)


# ------------------------------------------------------------------ layers
def dense_init(key, n_in, n_out, scale=1.0):
    return {"w": jax.random.normal(key, (n_in, n_out)) * scale / np.sqrt(n_in), "b": jnp.zeros(n_out)}


def dense(p, x):
    return x @ p["w"] + p["b"]


def layernorm(x, eps=1e-5):
    m = x.mean(-1, keepdims=True)
    v = ((x - m) ** 2).mean(-1, keepdims=True)
    return (x - m) / jnp.sqrt(v + eps)


def time_embed(t, dim=128):
    half = dim // 2
    freqs = jnp.exp(-np.log(1000.0) * jnp.arange(half) / half)
    a = t[:, None] * freqs[None]
    return jnp.concatenate([jnp.sin(a), jnp.cos(a)], -1)


def conv1d(p, x):
    """x [B, L, Cin] -> [B, L, Cout], 'SAME' padding, kernel p['w'] [K, Cin, Cout]. Written as shifted copies + one
    matmul: XLA's convolution is ~100x slower on CPU for these small shapes."""
    K = p["w"].shape[0]
    L = x.shape[1]
    xp = jnp.pad(x, ((0, 0), (K // 2, K // 2), (0, 0)))
    cols = jnp.concatenate([xp[:, k:k + L] for k in range(K)], -1)  # [B, L, K*Cin]
    return cols @ p["w"].reshape(-1, p["w"].shape[-1]) + p["b"]


def seq_len(cfg: Cfg):
    return cfg.lat or cfg.knots or F


def n_ch(cfg: Cfg):
    return 1 if cfg.lat else NC


def upsampler(cfg: Cfg):
    """[F, L] linear-interpolation matrix from the generated control points to the 30 future steps, and its pseudo-inverse."""
    L = seq_len(cfg)
    if L == F:
        return np.eye(F, dtype=np.float32), np.eye(F, dtype=np.float32)
    pos = np.linspace(0, F - 1, L)
    U = np.zeros((F, L), np.float32)
    for f in range(F):
        j = min(np.searchsorted(pos, f, side="right") - 1, L - 2)
        w = (f - pos[j]) / (pos[j + 1] - pos[j])
        U[f, j], U[f, j + 1] = 1 - w, w
    return U, np.linalg.pinv(U).astype(np.float32)


def cond_dim(cfg: Cfg):
    return cfg.hist_len * OBS_DIM + (F if cfg.use_actions else 0) + (1 if cfg.cond_drop > 0 else 0)


def init(key, cfg: Cfg):
    ks = iter(jax.random.split(key, 64))
    W = cfg.width
    p = {"cond": [dense_init(next(ks), cond_dim(cfg), W), dense_init(next(ks), W, W)],
         "temb": [dense_init(next(ks), 128, W), dense_init(next(ks), W, W)]}
    if cfg.net == "attn":
        D, L = W, seq_len(cfg)
        p["hist_in"] = dense_init(next(ks), OBS_DIM, D)
        p["fut_in"] = dense_init(next(ks), NC + (1 if cfg.use_actions else 0), D)
        p["pos"] = jax.random.normal(next(ks), (cfg.hist_len + L, D)) * 0.02
        p["blocks"] = [{"ada": dense_init(next(ks), W, 6 * D, 0.0), "qkv": dense_init(next(ks), D, 3 * D),
                        "o": dense_init(next(ks), D, D), "m1": dense_init(next(ks), D, 4 * D), "m2": dense_init(next(ks), 4 * D, D)}
                       for _ in range(cfg.depth)]
        p["ada_out"] = dense_init(next(ks), W, 2 * D, 0.0)
        p["out"] = dense_init(next(ks), D, NC, 0.1)
        del p["cond"]  # history and actions enter as tokens; only the noise level is global (p["temb"])
    elif cfg.net == "mlp":
        p["inp"] = dense_init(next(ks), seq_len(cfg) * n_ch(cfg), W)
        p["blocks"] = [{"film": dense_init(next(ks), W, 2 * W, 0.1), "l1": dense_init(next(ks), W, W),
                        "l2": dense_init(next(ks), W, W, 0.1)} for _ in range(cfg.depth)]
        p["out"] = dense_init(next(ks), W, seq_len(cfg) * n_ch(cfg), 0.1)
    else:
        C = W // 4
        cv = lambda k, ci, co, s=1.0: {"w": jax.random.normal(k, (5, ci, co)) * s / np.sqrt(5 * ci), "b": jnp.zeros(co)}  # noqa: E731
        p["inp"] = cv(next(ks), NC, C)
        p["blocks"] = [{"film": dense_init(next(ks), W, 2 * C, 0.1), "c1": cv(next(ks), C, C), "c2": cv(next(ks), C, C, 0.1)}
                       for _ in range(cfg.depth)]
        p["out"] = cv(next(ks), C, NC, 0.1)
    return p


def _attn_apply(p, cfg: Cfg, x, t, cond):
    """DiT-style denoiser: tokens = history steps (22 features each) + future steps (noisy state + that step's action);
    full self-attention; the noise level modulates every block through adaLN with zero-initialised gates."""
    B, L = x.shape[0], x.shape[1]
    D, nh = cfg.width, cfg.heads
    hist = cond[:, :cfg.hist_len * OBS_DIM].reshape(B, cfg.hist_len, OBS_DIM)
    fut_in = x
    if cfg.use_actions:
        act = cond[:, -F:]
        U = jnp.asarray(upsampler(cfg)[0])
        a_tok = act @ (U / U.sum(0, keepdims=True))  # each token's action (the mean over the steps it covers)
        fut_in = jnp.concatenate([x, a_tok[..., None]], -1)
    h = jnp.concatenate([dense(p["hist_in"], hist), dense(p["fut_in"], fut_in)], 1) + p["pos"][None]
    c = dense(p["temb"][1], jax.nn.silu(dense(p["temb"][0], time_embed(t * 1000.0)))) if cfg.kind != "regress" else jnp.zeros((B, cfg.width))
    c = jax.nn.silu(c)
    for blk in p["blocks"]:
        s1, b1, g1, s2, b2, g2 = jnp.split(dense(blk["ada"], c)[:, None], 6, -1)
        y = layernorm(h) * (1 + s1) + b1
        q, k, v = jnp.split(dense(blk["qkv"], y), 3, -1)
        sh = lambda z: z.reshape(B, -1, nh, D // nh).transpose(0, 2, 1, 3)  # noqa: E731
        att = jax.nn.softmax(sh(q) @ sh(k).transpose(0, 1, 3, 2) / np.sqrt(D // nh), -1)
        y = (att @ sh(v)).transpose(0, 2, 1, 3).reshape(B, -1, D)
        h = h + g1 * dense(blk["o"], y)
        y = layernorm(h) * (1 + s2) + b2
        h = h + g2 * dense(blk["m2"], jax.nn.gelu(dense(blk["m1"], y)))
    s, b = jnp.split(dense(p["ada_out"], c)[:, None], 2, -1)
    return dense(p["out"], layernorm(h[:, -L:]) * (1 + s) + b)


def apply(p, cfg: Cfg, x, t, cond):
    """x [B, L, C] noisy future (zeros for the regressor), t [B] in [0, 1], cond [B, cond_dim] -> [B, L, C]."""
    if cfg.net == "attn":
        return _attn_apply(p, cfg, x, t, cond)
    c = dense(p["cond"][1], jax.nn.silu(dense(p["cond"][0], cond)))
    if cfg.kind != "regress":
        c = c + dense(p["temb"][1], jax.nn.silu(dense(p["temb"][0], time_embed(t * 1000.0))))
    c = jax.nn.silu(c)
    B = x.shape[0]
    if cfg.net == "mlp":
        h = dense(p["inp"], x.reshape(B, -1))
        for blk in p["blocks"]:
            g, b = jnp.split(dense(blk["film"], c), 2, -1)
            y = layernorm(h) * (1 + g) + b
            y = dense(blk["l2"], jax.nn.silu(dense(blk["l1"], jax.nn.silu(y))))
            h = h + y
        return dense(p["out"], jax.nn.silu(layernorm(h))).reshape(B, seq_len(cfg), n_ch(cfg))
    h = conv1d(p["inp"], x)
    for blk in p["blocks"]:
        g, b = jnp.split(dense(blk["film"], c), 2, -1)
        y = layernorm(h) * (1 + g[:, None]) + b[:, None]
        y = conv1d(blk["c2"], jax.nn.silu(conv1d(blk["c1"], jax.nn.silu(y))))
        h = h + y
    return conv1d(p["out"], jax.nn.silu(layernorm(h)))


# ------------------------------------------------------------------ objectives
def loss_fn(p, cfg: Cfg, ab, x0, cond, key):
    B = x0.shape[0]
    if cfg.kind == "regress":
        return ((apply(p, cfg, jnp.zeros_like(x0), jnp.zeros(B), cond) - x0) ** 2).mean()
    k1, k2, k3 = jax.random.split(key, 3)
    if cfg.kind == "flow":  # straight path noise -> data, regress its constant velocity
        t = jax.random.uniform(k1, (B, 1, 1))
        z = jax.random.normal(k2, x0.shape)
        xt = (1 - t) * z + t * x0
        return ((apply(p, cfg, xt, t[:, 0, 0], cond) - (x0 - z)) ** 2).mean()
    if cfg.cond_drop > 0:  # hide the actions (and their presence flag) for a random share of the batch
        keep = (jax.random.uniform(k3, (B, 1)) >= cfg.cond_drop).astype(cond.dtype)
        cond = jnp.concatenate([cond[:, :-(F + 1)], cond[:, -(F + 1):] * keep], 1)
    i = jax.random.randint(k1, (B,), 0, cfg.T)
    a = ab[i][:, None, None]
    eps = jax.random.normal(k2, x0.shape)
    xt = jnp.sqrt(a) * x0 + jnp.sqrt(1 - a) * eps
    out = apply(p, cfg, xt, (i + 1) / cfg.T, cond)
    target = {"eps": eps, "x0": x0, "v": jnp.sqrt(a) * eps - jnp.sqrt(1 - a) * x0}[cfg.pred]
    return ((out - target) ** 2).mean()


def to_x0_eps(cfg, out, xt, a):
    if cfg.pred == "eps":
        eps = out
        x0 = (xt - jnp.sqrt(1 - a) * eps) / jnp.sqrt(a)
    elif cfg.pred == "x0":
        x0 = out
        eps = (xt - jnp.sqrt(a) * x0) / jnp.sqrt(1 - a)
    else:  # v
        x0 = jnp.sqrt(a) * xt - jnp.sqrt(1 - a) * out
        eps = jnp.sqrt(1 - a) * xt + jnp.sqrt(a) * out
    return x0, eps


def sample(p, cfg: Cfg, ab, cond, key, steps=None, guidance=None, eta=None):
    """DDIM (eta = 0) from pure noise: one sample per row of `cond`. Returns normalised futures [B, L, C] (L = seq_len)."""
    B = cond.shape[0]
    if cfg.kind == "regress":
        return apply(p, cfg, jnp.zeros((B, seq_len(cfg), n_ch(cfg))), jnp.zeros(B), cond)
    S = steps or cfg.sample_steps
    if cfg.kind == "flow":
        return trace(p, cfg, ab, cond, key, S)[-1]
    w = cfg.guidance if guidance is None else guidance
    cond_u = jnp.concatenate([cond[:, :-(F + 1)], jnp.zeros_like(cond[:, -(F + 1):])], 1) if cfg.cond_drop > 0 else cond
    ts = jnp.linspace(cfg.T, 0, S + 1).round().astype(jnp.int32)  # T .. 0
    eta = cfg.eta if eta is None else eta
    key, knoise = jax.random.split(key)
    x = jax.random.normal(key, (B, seq_len(cfg), n_ch(cfg)))

    def body(x, k):
        i, j = ts[k], ts[k + 1]
        a = ab[i - 1]
        a_next = jnp.where(j > 0, ab[jnp.maximum(j - 1, 0)], 1.0)
        tt = jnp.full((B,), i / cfg.T)
        out = apply(p, cfg, x, tt, cond)
        if cfg.cond_drop > 0 and w != 1.0:
            out_u = apply(p, cfg, x, tt, cond_u)
            out = out_u + w * (out - out_u)
        x0, eps = to_x0_eps(cfg, out, x, a)
        x0 = jnp.clip(x0, -8.0, 8.0)
        sig = eta * jnp.sqrt((1 - a_next) / (1 - a) * (1 - a / a_next))
        z = jax.random.normal(jax.random.fold_in(knoise, k), x.shape)
        return jnp.sqrt(a_next) * x0 + jnp.sqrt(jnp.maximum(1 - a_next - sig ** 2, 0.0)) * eps + sig * z, None

    x, _ = jax.lax.scan(body, x, jnp.arange(S))
    return x


def trace(p, cfg: Cfg, ab, cond, key, steps):
    """Every intermediate state of the sampler, [S + 1, B, L, C']: Euler steps of the flow ODE, or DDIM steps."""
    B = cond.shape[0]
    x = jax.random.normal(key, (B, seq_len(cfg), n_ch(cfg)))
    if cfg.kind == "flow":
        def body(x, k):
            xn = x + apply(p, cfg, x, jnp.full((B,), k / steps), cond) / steps
            return xn, xn
    else:
        ts = jnp.linspace(cfg.T, 0, steps + 1).round().astype(jnp.int32)

        def body(x, k):
            i, j = ts[k], ts[k + 1]
            a = ab[i - 1]
            a_next = jnp.where(j > 0, ab[jnp.maximum(j - 1, 0)], 1.0)
            x0, eps = to_x0_eps(cfg, apply(p, cfg, x, jnp.full((B,), i / cfg.T), cond), x, a)
            xn = jnp.sqrt(a_next) * jnp.clip(x0, -8.0, 8.0) + jnp.sqrt(1 - a_next) * eps
            return xn, xn
    _, xs = jax.lax.scan(body, x, jnp.arange(steps))
    return jnp.concatenate([x[None], xs], 0)


def n_params(p):
    return sum(int(np.prod(a.shape)) for a in jax.tree.leaves(p))
