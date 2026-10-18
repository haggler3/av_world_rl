"""Train each output representation (and a flow-matching model) for a short budget, score it, and draw one small 2D
top-down GIF per model on the same held-out scene.

    python -m dwm.gifs2d            # ~7 min: results/dwm/gifs2d/results.md, docs/img/dwm_<variant>.gif
"""
import json
import os
import time

import jax
import jax.numpy as jnp
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.animation import FuncAnimation, PillowWriter  # noqa: E402

from . import model as M  # noqa: E402
from . import train as T  # noqa: E402
from .data import F  # noqa: E402
from .quick import small_data  # noqa: E402
from .viz import pick_scenes  # noqa: E402

VARIANTS = [
    ("control_points", "diffusion · 10 control points", dict(basis="knots", nb=10, sample_steps=5)),
    ("dct", "diffusion · DCT basis", dict(basis="dct", nb=8, sample_steps=5)),
    ("leader_physics", "diffusion · leader physics decoder", dict(basis="leader_phys", nb=10, sample_steps=5)),
    ("flow_dct", "flow matching · DCT basis", dict(kind="flow", basis="dct", nb=8, sample_steps=10)),
]
BG, ROAD, LANE, TXT = "#0d1117", "#1c2128", "#8b949e", "#e6edf3"
EGO, GHOST, REAL = "#58a6ff", "#f778ba", "#ffffff"
LIGHT = {0: "#f85149", 1: "#d29922", 2: "#3fb950"}


def scene_paths(res, cfg, st, d, i, K=16, interp=3):
    """Physical gap futures [frames, K, F] for the imagining act, and the final samples [K, F, C]."""
    cond = jnp.repeat(jnp.asarray(T.make_cond(cfg, st, d["hist"][[i]], d["act"][[i]])), K, 0)
    xs = np.asarray(jax.jit(lambda c, k: M.trace(res["params"], cfg, M.alphas_bar(cfg.T), c, k, cfg.sample_steps))(cond, jax.random.PRNGKey(7)))
    phys = np.stack([(T.decode(cfg, d, st, x) * st.sd[cfg.target] + T.base_of(cfg, d, [i])) * T.SCALE for x in xs])  # [S+1, K, F, C]
    w = np.linspace(0, 1, interp, endpoint=False)
    frames = np.stack([(1 - a) * phys[s] + a * phys[s + 1] for s in range(len(phys) - 1) for a in w] + [phys[-1]])
    return frames, phys[-1]


def render(frames, final, d, i, title, sub, out, fps=12):
    real = d["fut"][i] * T.SCALE
    cur = d["cur"][i] * T.SCALE
    light = int(np.argmax(d["hist"][i][-1, 4:7]))
    plan = [("imagine", k) for k in range(len(frames))] + [("imagine", len(frames) - 1)] * 6 + [("play", k) for k in range(F)] + [("play", F - 1)] * 8
    fig, ax = plt.subplots(figsize=(6.4, 2.5), facecolor=BG)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.74, bottom=0.17)

    def car(x, y, color, fill=True, alpha=1.0, lw=1.6):
        ax.add_patch(plt.Rectangle((x, y - 0.95), 4.6, 1.9, color=color if fill else None, fill=fill, ec=color, lw=lw, alpha=alpha))

    def draw(n):
        act, k = plan[n]
        ax.cla()
        ax.set_facecolor(BG)
        ax.set_xlim(-8, 62)
        ax.set_ylim(-4.2, 4.2)
        ax.axis("off")
        ax.add_patch(plt.Rectangle((-8, -3.6), 70, 7.2, color=ROAD))
        ax.plot([-8, 62], [0, 0], color=LANE, lw=1, ls=(0, (5, 5)))
        for y in (-3.6, 3.6):
            ax.plot([-8, 62], [y, y], color=LANE, lw=1.2)
        if act == "imagine":
            gaps, gr, dl, t = frames[k][:, -1, 3], cur[3], cur[1], 0.0
            label = "imagining where the car ahead will be in 3 s" if k < len(frames) - 1 else "16 futures imagined; now play them out"
            v_av, v_lead = cur[0], cur[4]
        else:
            gaps, gr, dl, t = final[:, k, 3], real[k, 3], real[k, 1], (k + 1) * 0.1
            label = f"t = {t:.1f} s   ·   pink: imagined leaders   ·   white: the real one"
            v_av, v_lead = real[k, 0], real[k, 4]
        if -8 < dl < 62:
            ax.plot([dl, dl], [-3.6, 0], color=LIGHT[light], lw=4)
        for g in gaps:
            car(g, -1.8, GHOST, alpha=0.16)
        car(gr, -1.8, REAL, fill=False)
        car(-4.6, -1.8, EGO)
        ax.text(-7.5, 2.0, f"AV {v_av * 3.6:4.1f} km/h", color=EGO, fontsize=8)
        ax.text(gr, 2.0, f"leader {v_lead * 3.6:4.1f} km/h", color=REAL, fontsize=8)
        ax.text(-7.5, -4.15, label, color=LANE, fontsize=7.5)
        return []

    fig.text(0.02, 0.9, title, color=TXT, fontsize=11, weight="bold")
    fig.text(0.02, 0.8, sub, color=LANE, fontsize=8)
    FuncAnimation(fig, draw, frames=len(plan)).save(out, writer=PillowWriter(fps=fps), dpi=80, savefig_kwargs={"facecolor": BG})
    plt.close(fig)


def main(budget=60, out="results/dwm/gifs2d"):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    d = small_data()
    st = T.Stats(d)
    idx = T.eval_idx(d, 800)
    ref = T.metrics(T.baselines(d, idx)["kinematic"], d, idx)
    scene = int(pick_scenes(d)[0])  # a held-out scene where the car ahead brakes
    rows = []
    for key, title, over in VARIANTS:
        cfg = T.with_latent(M.Cfg(**over))
        r = T.train(cfg, d, st, budget, verbose=False)
        pred, ss = T.sample_physical(r, cfg, st, d, idx, 8)
        m = T.metrics(pred, d, idx)
        row = {"variant": title, "steps": r["steps"], "skill_vs_kinematic": round(T.skill(m, ref), 3), "crps_gap_3s": round(m["crps_gap_3s"], 3),
               "crps_lead_v_3s": round(m["crps_lead_v_3s"], 3), "coverage90_gap_3s": round(m["coverage90_gap_3s"], 3),
               "roughness_vs_real": round(m["roughness_gap_vs_real"], 2), "sample_ms_per_future": round(1000 * ss / (len(idx) * 8), 3)}
        rows.append(row)
        print(json.dumps(row), flush=True)
        frames, final = scene_paths(r, cfg, st, d, scene)
        sub = f"score vs physics {row['skill_vs_kinematic']:.2f} (lower is better) · gap error 3 s {row['crps_gap_3s']:.2f} m · 90% band holds {100 * row['coverage90_gap_3s']:.0f}% · roughness {row['roughness_vs_real']:.1f}x real"
        render(frames, final, d, scene, title, sub, f"docs/img/dwm_{key}.gif")
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    cols = list(rows[0])
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(md)
    print(md, f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
