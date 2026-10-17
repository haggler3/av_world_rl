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
    kind: str = "diffusion"  # "diffusion" | "regress"
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
    return cfg.knots or F


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
    return cfg.hist_len * OBS_DIM + (F if cfg.use_actions else 0)


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
        p["inp"] = dense_init(next(ks), seq_len(cfg) * NC, W)
        p["blocks"] = [{"film": dense_init(next(ks), W, 2 * W, 0.1), "l1": dense_init(next(ks), W, W),
                        "l2": dense_init(next(ks), W, W, 0.1)} for _ in range(cfg.depth)]
        p["out"] = dense_init(next(ks), W, seq_len(cfg) * NC, 0.1)
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
    c = dense(p["temb"][1], jax.nn.silu(dense(p["temb"][0], time_embed(t * 1000.0)))) if cfg.kind == "diffusion" else jnp.zeros((B, cfg.width))
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
    if cfg.kind == "diffusion":
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
        return dense(p["out"], jax.nn.silu(layernorm(h))).reshape(B, seq_len(cfg), NC)
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
    k1, k2 = jax.random.split(key)
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


def sample(p, cfg: Cfg, ab, cond, key, steps=None):
    """DDIM (eta = 0) from pure noise: one sample per row of `cond`. Returns normalised futures [B, L, C] (L = seq_len)."""
    B = cond.shape[0]
    if cfg.kind == "regress":
        return apply(p, cfg, jnp.zeros((B, seq_len(cfg), NC)), jnp.zeros(B), cond)
    S = steps or cfg.sample_steps
    ts = jnp.linspace(cfg.T, 0, S + 1).round().astype(jnp.int32)  # T .. 0
    x = jax.random.normal(key, (B, seq_len(cfg), NC))

    def body(x, k):
        i, j = ts[k], ts[k + 1]
        a = ab[i - 1]
        a_next = jnp.where(j > 0, ab[jnp.maximum(j - 1, 0)], 1.0)
        out = apply(p, cfg, x, jnp.full((B,), i / cfg.T), cond)
        x0, eps = to_x0_eps(cfg, out, x, a)
        x0 = jnp.clip(x0, -8.0, 8.0)
        return jnp.sqrt(a_next) * x0 + jnp.sqrt(1 - a_next) * eps, None

    x, _ = jax.lax.scan(body, x, jnp.arange(S))
    return x


def n_params(p):
    return sum(int(np.prod(a.shape)) for a in jax.tree.leaves(p))
