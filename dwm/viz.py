"""An animated GIF of the diffusion world model at work, on held-out city scenes.

Each scene has two acts:
  1. denoising: K futures of the gap to the car ahead start as pure noise and become plausible trajectories over the
     DDIM steps (what really happened is drawn in white at the end);
  2. playback: the next 3 s on a top-down road strip that follows the AV. The real leader is solid; every sampled
     future is a translucent ghost leader, so the spread of the ghosts is the model's uncertainty.

    python -m dwm.viz --model results/dwm/final/model.pkl --out docs/img/diffusion_world_model.gif
"""
import argparse
import pickle

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from . import model as M  # noqa: E402
from . import train as T  # noqa: E402
from .data import F, NC  # noqa: E402

BG, PANEL, GRID, TXT = "#0d1117", "#161b22", "#30363d", "#e6edf3"
EGO, GHOST, REAL, ACCENT = "#58a6ff", "#f778ba", "#ffffff", "#7ee787"
LIGHT = {0: "#f85149", 1: "#d29922", 2: "#3fb950"}


def sample_trace(p, cfg, ab, cond, key, steps):
    """DDIM like model.sample, but returns x_t after every step: [S + 1, B, F, C]."""
    B = cond.shape[0]
    ts = jnp.linspace(cfg.T, 0, steps + 1).round().astype(jnp.int32)
    x = jax.random.normal(key, (B, M.seq_len(cfg), NC))

    def body(x, k):
        i, j = ts[k], ts[k + 1]
        a = ab[i - 1]
        a_next = jnp.where(j > 0, ab[jnp.maximum(j - 1, 0)], 1.0)
        x0, eps = M.to_x0_eps(cfg, M.apply(p, cfg, x, jnp.full((B,), i / cfg.T), cond), x, a)
        x0 = jnp.clip(x0, -8.0, 8.0)
        xn = jnp.sqrt(a_next) * x0 + jnp.sqrt(1 - a_next) * eps
        return xn, xn

    _, xs = jax.lax.scan(body, x, jnp.arange(steps))
    return jnp.concatenate([x[None], xs], 0)


def pick_scenes(d, n=3, seed=4):
    """Held-out windows where the leader is seen throughout and something happens: a braking leader, a leader pulling
    away, and an approach to a stop line behind a queue."""
    te = np.where(d["test"])[0]
    cur, fut = d["cur"][te] * T.SCALE, d["fut"][te] * T.SCALE
    seen = (cur[:, 2] > 0.5) & (fut[:, :, 2] > 0.5).all(1) & (cur[:, 3] < 45) & (cur[:, 0] > 3)
    dv_lead = fut[:, -1, 4] - cur[:, 4]
    rng = np.random.default_rng(seed)
    picks = []
    for cond in (seen & (dv_lead < -4), seen & (dv_lead > 3), seen & (cur[:, 5] > 0.5) & (cur[:, 1] < 60) & (dv_lead < -2)):
        c = np.where(cond)[0]
        if len(c):
            picks.append(te[rng.choice(c)])
    return np.array(picks[:n])


def render(model_path, data_path, out, K=24, steps=None, fps=14, data=None, interp=4, hold=10, dpi=100):
    with open(model_path, "rb") as f:
        blob = pickle.load(f)
    cfg, p = blob["cfg"], jax.tree.map(jnp.asarray, blob["params"])
    d = data if data is not None else T.load_data(data_path)
    st = T.Stats.__new__(T.Stats)
    st.__dict__.update(blob["stats"])
    idx = pick_scenes(d)
    steps = steps or cfg.sample_steps
    ab = M.alphas_bar(cfg.T)
    cond = jnp.repeat(jnp.asarray(T.make_cond(cfg, st, d["hist"][idx], d["act"][idx])), K, 0)
    xs = np.asarray(jax.jit(lambda c, k: sample_trace(p, cfg, ab, c, k, steps))(cond, jax.random.PRNGKey(3)))
    xs = np.einsum("fl,sblc->sbfc", M.upsampler(cfg)[0], xs)
    base = T.base_of(cfg, d, idx)  # [S, F, C] feature units
    sd = st.sd[cfg.target]
    traj = (xs.reshape(steps + 1, len(idx), K, F, NC) * sd + base[None, :, None]) * T.SCALE  # physical
    # smooth the denoising act: `interp` frames between consecutive sampler steps (the end points are the real states)
    w = np.linspace(0, 1, interp, endpoint=False)
    traj = np.stack([(1 - a) * traj[i] + a * traj[i + 1] for i in range(steps) for a in w] + [traj[-1]])
    n_den = len(traj) - 1
    real = d["fut"][idx] * T.SCALE
    cur = d["cur"][idx] * T.SCALE
    light = np.argmax(d["hist"][idx][:, -1, 4:7], -1)
    t = (np.arange(F) + 1) * T.DT
    titles = ["the car ahead brakes", "the car ahead pulls away", "a queue at a red light"]

    # frame plan: per scene, denoise (steps + 1 frames, eased) + hold + playback (F frames) + hold
    plan = []
    for s in range(len(idx)):
        plan += [(s, "noise", k) for k in range(n_den + 1)] + [(s, "noise", n_den)] * hold
        plan += [(s, "play", k) for k in range(F)] + [(s, "play", F - 1)] * hold

    fig = plt.figure(figsize=(9.6, 5.4), facecolor=BG)
    gs = fig.add_gridspec(2, 2, width_ratios=[1.15, 1], height_ratios=[1, 1], left=0.07, right=0.98, top=0.80, bottom=0.1,
                          wspace=0.22, hspace=0.55)
    ax_n = fig.add_subplot(gs[:, 0])
    ax_r = fig.add_subplot(gs[0, 1])
    ax_v = fig.add_subplot(gs[1, 1])
    for ax in (ax_n, ax_r, ax_v):
        ax.set_facecolor(PANEL)
        for sp in ax.spines.values():
            sp.set_color(GRID)
        ax.tick_params(colors=TXT, labelsize=8)
    title = fig.text(0.07, 0.935, "", color=TXT, fontsize=14, weight="bold")
    sub = fig.text(0.07, 0.885, "", color="#8b949e", fontsize=9.5)

    def draw(fr):
        s, act, k = plan[fr]
        tr, rl = traj[:, s], real[s]
        for ax in (ax_n, ax_r, ax_v):
            ax.cla()
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=TXT, labelsize=8)
            for sp in ax.spines.values():
                sp.set_color(GRID)
        title.set_text(f"Diffusion world model  ·  {titles[s]}")
        # ---- left: futures of the gap
        k_n = k if act == "noise" else n_den
        g = tr[k_n, :, :, 3]
        lo, hi = min(rl[:, 3].min(), tr[-1, :, :, 3].min()) - 6, max(rl[:, 3].max(), tr[-1, :, :, 3].max()) + 6
        segs = [np.column_stack([np.r_[0, t], np.r_[cur[s, 3], gi]]) for gi in g]
        noise_level = 1 - k_n / n_den
        lc = LineCollection(segs, colors=GHOST, linewidths=1.1, alpha=0.25 + 0.35 * (1 - noise_level))
        ax_n.add_collection(lc)
        if act == "noise" and k_n == n_den or act == "play":
            ax_n.plot(np.r_[0, t], np.r_[cur[s, 3], rl[:, 3]], color=REAL, lw=2.4, label="what really happened")
            q5, q95 = np.percentile(g, 5, 0), np.percentile(g, 95, 0)
            ax_n.fill_between(t, q5, q95, color=GHOST, alpha=0.12, lw=0)
        if act == "play":
            ax_n.axvline(t[k], color=ACCENT, lw=1, alpha=0.8)
        ax_n.set_xlim(0, 3.0)
        ax_n.set_ylim(lo, hi)
        ax_n.set_xlabel("seconds ahead", color=TXT, fontsize=9)
        ax_n.set_ylabel("gap to the car ahead [m]", color=TXT, fontsize=9)
        ax_n.grid(color=GRID, lw=0.5)
        if act == "noise":
            ax_n.set_title(f"denoising  step {k_n / interp:.0f}/{steps}" if k_n < n_den else f"denoised in {steps} steps",
                           color=GHOST if k_n < n_den else ACCENT, fontsize=10, loc="left")
            sub.set_text(f"{K} possible futures, sampled from noise, conditioned on 1 s of history and the AV's planned actions")
        else:
            ax_n.set_title(f"{K} sampled futures  ·  t = {t[k]:.1f} s", color=ACCENT, fontsize=10, loc="left")
            sub.set_text("playback: every pink ghost is one future the model imagined; white is the real car")
        if act == "noise" and k_n == n_den or act == "play":
            ax_n.legend(loc="upper left", fontsize=7.5, facecolor=PANEL, edgecolor=GRID, labelcolor=TXT)
        # ---- right top: road strip following the AV
        kk = k if act == "play" else 0
        ego_travel = cur[s, 1] - rl[kk, 1] if act == "play" else 0.0
        ax_r.set_xlim(-12, 75)
        ax_r.set_ylim(-3, 3)
        ax_r.set_yticks([])
        ax_r.axhline(0, color=GRID, lw=0.8, ls=(0, (6, 6)))
        d_line = rl[kk, 1] if act == "play" else cur[s, 1]
        if d_line < 75:
            ax_r.plot([d_line, d_line], [-2.6, 2.6], color=LIGHT[int(light[s])], lw=4, solid_capstyle="butt")
        for j in range(K):
            gx = tr[-1, j, kk, 3] if act == "play" else cur[s, 3]
            ax_r.add_patch(plt.Rectangle((gx, -1.0), 4.6, 2.0, color=GHOST, alpha=0.10 if act == "play" else 0.0, lw=0))
        gr = rl[kk, 3] if act == "play" else cur[s, 3]
        ax_r.add_patch(plt.Rectangle((gr, -1.0), 4.6, 2.0, fill=False, ec=REAL, lw=1.8))
        ax_r.add_patch(plt.Rectangle((-4.5, -1.0), 4.5, 2.0, color=EGO))
        ax_r.text(-11, 2.15, f"AV {rl[kk, 0] * 3.6 if act == 'play' else cur[s, 0] * 3.6:4.1f} km/h", color=EGO, fontsize=8.5)
        ax_r.text(30, 2.15, f"leader {rl[kk, 4] * 3.6 if act == 'play' else cur[s, 4] * 3.6:4.1f} km/h", color=REAL, fontsize=8.5)
        ax_r.set_xlabel("metres ahead of the AV", color=TXT, fontsize=8)
        ax_r.set_title("top-down, following the AV", color=TXT, fontsize=9.5, loc="left")
        # ---- right bottom: leader speed futures
        lv = tr[k_n, :, :, 4] * 3.6
        ax_v.add_collection(LineCollection([np.column_stack([t, li]) for li in lv], colors=GHOST, linewidths=1, alpha=0.3))
        if act == "noise" and k_n == n_den or act == "play":
            ax_v.plot(t, rl[:, 4] * 3.6, color=REAL, lw=2)
        ax_v.plot(t, real[s][:, 0] * 3.6 if act == "play" else np.full(F, np.nan), color=EGO, lw=1.6, alpha=0.9)
        if act == "play":
            ax_v.axvline(t[k], color=ACCENT, lw=1, alpha=0.8)
        ax_v.set_xlim(0, 3)
        vv = np.r_[rl[:, 4], tr[-1, :, :, 4].ravel(), rl[:, 0]] * 3.6
        ax_v.set_ylim(max(vv.min() - 8, -2), vv.max() + 8)
        ax_v.set_title("leader speed futures (pink), AV speed (blue) [km/h]", color=TXT, fontsize=9.5, loc="left")
        ax_v.set_xlabel("seconds ahead", color=TXT, fontsize=8)
        ax_v.grid(color=GRID, lw=0.5)
        return []

    anim = FuncAnimation(fig, draw, frames=len(plan), blit=False)
    anim.save(out, writer=PillowWriter(fps=fps), dpi=dpi, savefig_kwargs={"facecolor": BG})
    plt.close(fig)
    print("wrote", out, len(plan), "frames")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="results/dwm/final/model.pkl")
    ap.add_argument("--data", default="data/dwm")
    ap.add_argument("--out", default="docs/img/diffusion_world_model.gif")
    a = ap.parse_args()
    render(a.model, a.data, a.out)
