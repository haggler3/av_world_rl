"""Training visuals: live training curves, time-space diagrams of the policy on a fixed scenario, and an animated GIF.

The time-space diagram is the standard picture of longitudinal driving: time on x, distance on y, one line per vehicle.
The ego is the thick line; stop lines are horizontal bands coloured by the signal (red / amber / green). A good driver's
line is smooth: no kinks (hard brakes), never crosses a red band, never touches the line of the car ahead.
"""
import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from . import env as E  # noqa: E402

TRACE_STEPS = 900  # 90 s
TRACE_CFG = E.EnvConfig(p_congested_exit=0.5, p_queue=0.6, p_stall=0.0, p_arrive=0.0)
LIGHT_COLORS = {0: "#d64545", 1: "#e0a030", 2: "#3fa34d"}


def _trace(policy_fn, key):
    st = E.reset(key, TRACE_CFG)

    def body(c, k):
        st = c
        o = E.observe(st)
        u = policy_fn(o[None])[0]
        light = E.light_state(st.seg, st.t)
        offset = st.m_dist - st.ego_s  # absolute position of this segment's start
        rec = dict(t=st.step * E.DT, ego=st.m_dist, v=st.ego_v, a=st.ego_a, cars=offset + st.pos, cars_len=st.length,
                   alive=st.alive, line=offset + st.seg.L, light=light, fol=offset + st.fol_pos)
        st2, _, r, d, _ = E.step(st, u, TRACE_CFG)
        return st2, rec | {"r": r, "done": d}

    _, rec = jax.lax.scan(body, st, jax.random.split(key, TRACE_STEPS))
    return rec


def scenario_trace(policy_fn, seed=7):
    """Drive the fixed scenario (same key every call) with `policy_fn(obs[B, D]) -> u[B]`; returns numpy arrays."""
    from . import ppo

    if isinstance(policy_fn, dict):  # params
        policy_fn = ppo.deterministic_policy(policy_fn)
    rec = _trace_params(policy_fn, seed)
    return jax.tree.map(np.asarray, rec)


def _trace_params(policy_fn, seed):
    params = getattr(policy_fn, "params", None)
    key = jax.random.PRNGKey(seed)
    if params is not None:
        return _trace_learned(params, key)
    return _trace(policy_fn, key)


@jax.jit
def _trace_learned(params, key):
    from . import ppo

    return _trace(lambda o: jnp.clip(ppo.policy_mean(params, o), -1.0, 1.0), key)


def time_space(ax, tr, title=None, show_cars=True):
    t = tr["t"]
    end = np.argmax(tr["done"]) + 1 if tr["done"].any() else len(t)
    t = t[:end]
    # stop lines coloured by the signal: draw short horizontal segments per step
    line, light = tr["line"][:end], tr["light"][:end]
    for c in (0, 1, 2):
        m = light == c
        ax.scatter(t[m], line[m], s=2, color=LIGHT_COLORS[c], marker="s", linewidths=0, zorder=1)
    if show_cars:
        cars, alive = tr["cars"][:end], tr["alive"][:end]
        for i in range(cars.shape[1]):
            y = np.where(alive[:, i], cars[:, i], np.nan)
            ax.plot(t, y, color="#8a94a6", lw=0.8, zorder=2)
    ax.plot(t, tr["ego"][:end], color="#1f6feb", lw=2.2, zorder=3, label="ego")
    ax.set_xlabel("time [s]")
    ax.set_ylabel("distance [m]")
    lo, hi = tr["ego"][:end].min() - 10, tr["ego"][:end].max() + 60
    ax.set_ylim(lo, hi)
    if title:
        ax.set_title(title, fontsize=9)


def progress_figure(snapshots, path, max_panels=6):
    """One time-space diagram + speed trace per checkpoint, same scenario: watch the policy learn."""
    if len(snapshots) > max_panels:
        idx = np.unique(np.linspace(0, len(snapshots) - 1, max_panels).round().astype(int))
        snapshots = [snapshots[i] for i in idx]
    n = len(snapshots)
    fig, axes = plt.subplots(2, n, figsize=(3.2 * n, 6), squeeze=False, gridspec_kw={"height_ratios": [3, 1.2]})
    for j, (it, tr) in enumerate(snapshots):
        hard = int(((tr["a"][1:] < -4) & (tr["a"][:-1] >= -4) & (tr["v"][1:] > 3)).sum())
        crashed = bool(tr["done"].any())
        time_space(axes[0, j], tr, f"update {it}: {tr['ego'][-1]:.0f} m, {hard} hard brakes{', CRASH' if crashed else ''}")
        end = np.argmax(tr["done"]) + 1 if tr["done"].any() else len(tr["t"])
        axes[1, j].plot(tr["t"][:end], tr["v"][:end] * 3.6, color="#1f6feb", lw=1)
        axes[1, j].set_ylabel("km/h")
        axes[1, j].set_xlabel("time [s]")
        axes[1, j].set_ylim(0, 65)
    fig.suptitle("Same 90 s scenario at each checkpoint (grey: other cars, bands: stop line by signal)", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def training_curves(rows, path):
    x = np.array([r["env_steps"] for r in rows]) / 1e6
    panels = [("reward", "reward / step"), ("speed_kmh", "mean speed [km/h]"), ("hard_per_km", "hard brakes / km"),
              ("coll_per_1000km", "collisions / 1000 km"), ("red_per_100seg", "red runs / 100 segments"), ("std", "policy std")]
    fig, axes = plt.subplots(2, 3, figsize=(12, 6))
    for ax, (k, lab) in zip(axes.flat, panels):
        y = np.array([r[k] for r in rows], dtype=float)
        ax.plot(x, y, color="#1f6feb", alpha=0.35, lw=0.8)
        if len(y) >= 5:
            w = max(len(y) // 20, 3)
            ys = np.convolve(np.nan_to_num(y, nan=np.nanmean(y) if np.isfinite(y).any() else 0), np.ones(w) / w, mode="valid")
            ax.plot(x[w - 1:], ys, color="#1f6feb", lw=1.8)
        ax.set_title(lab, fontsize=10)
        ax.set_xlabel("env steps [M]")
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def compare_figure(traces, path, title):
    """Several drivers (name -> trace) on the same scenario: time-space, speed and acceleration."""
    n = len(traces)
    fig, axes = plt.subplots(3, n, figsize=(4.2 * n, 8), squeeze=False, gridspec_kw={"height_ratios": [3, 1, 1]})
    for j, (name, tr) in enumerate(traces.items()):
        hard = int(((tr["a"][1:] < -4) & (tr["a"][:-1] >= -4) & (tr["v"][1:] > 3)).sum())
        time_space(axes[0, j], tr, f"{name}: {tr['ego'][-1]:.0f} m in {tr['t'][-1]:.0f} s, {hard} hard brakes")
        axes[1, j].plot(tr["t"], tr["v"] * 3.6, color="#1f6feb", lw=1)
        axes[1, j].set_ylabel("speed [km/h]")
        axes[1, j].set_ylim(0, 65)
        axes[2, j].plot(tr["t"], tr["a"], color="#b34ad6", lw=1)
        axes[2, j].axhline(-4, color="#d64545", lw=0.8, ls="--")
        axes[2, j].set_ylabel("accel [m/s²]")
        axes[2, j].set_ylim(-8.5, 3.5)
        axes[2, j].set_xlabel("time [s]")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def animate(traces, path, seconds=60, fps=10):
    """Top-down strips (one per driver) following the ego: cars, stop line coloured by the signal. Writes a GIF."""
    from matplotlib.animation import FuncAnimation, PillowWriter

    names = list(traces)
    n = len(names)
    fig, axes = plt.subplots(n, 1, figsize=(9, 1.5 * n + 0.4), squeeze=False)
    steps = min(int(seconds / E.DT), len(next(iter(traces.values()))["t"]))
    frames = range(0, steps, max(int(1 / (fps * E.DT)), 1))
    artists = []
    for ax, name in zip(axes[:, 0], names):
        ax.set_facecolor("#20242c")
        ax.set_yticks([])
        ax.set_xlim(-40, 110)
        ax.set_ylim(-3, 3)
        ax.set_title(name, fontsize=9, loc="left")
        ax.set_xlabel("metres relative to the ego (blue; red while braking harder than 4 m/s²)", fontsize=7)
        artists.append(ax)

    def draw(k):
        for ax, name in zip(artists, names):
            tr = traces[name]
            for p in list(ax.patches) + list(ax.texts) + list(ax.lines):
                p.remove()
            ego = tr["ego"][k]
            ax.axhline(0, color="#555c69", lw=0.6)
            line = tr["line"][k] - ego
            ax.plot([line, line], [-2.5, 2.5], color=LIGHT_COLORS[int(tr["light"][k])], lw=4)
            for i in range(tr["cars"].shape[1]):
                if tr["alive"][k, i]:
                    front = tr["cars"][k, i] - ego
                    ax.add_patch(plt.Rectangle((front - tr["cars_len"][k, i], -1.0), tr["cars_len"][k, i], 2.0, color="#8a94a6"))
            fol = tr["fol"][k] - ego
            ax.add_patch(plt.Rectangle((fol - 4.5, -1.0), 4.5, 2.0, color="#8a94a6", alpha=0.6))
            a = tr["a"][k]
            ax.add_patch(plt.Rectangle((-4.5, -1.0), 4.5, 2.0, color="#d64545" if a < -4 else "#1f6feb"))
            ax.text(-38, 1.9, f"t {tr['t'][k]:5.1f} s   {tr['v'][k] * 3.6:4.1f} km/h   a {a:+.1f} m/s²", color="white", fontsize=8)
        return []

    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    fig.tight_layout()
    anim.save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
