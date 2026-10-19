"""Figures for the city-area driver: training curves and a street-view GIF of learned AVs in Oakland.

    python -m avrl.xviz      # -> docs/img/xnet_training.png, docs/img/xnet_learned_avs.gif
"""
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

SURFACE, INK, INK2, GRID, SERIES = "#1a1a19", "#f0efec", "#c3c2b7", "#3a3a37", "#3987e5"


def curves(logs=("results/xnet/train_log_250.json", "results/xnet3/train_log.json"), out="docs/img/xnet_training.png"):
    rows = []
    for k, path in enumerate(logs):
        for r in json.load(open(path)):
            rows.append((len(rows), k, r))
    x = np.array([i for i, _, _ in rows])
    split = next(i for i, k, _ in rows if k == 1)
    panels = [("reward_per_step", "reward per AV step"), ("av_speed_kmh", "AV speed while training [km/h]"),
              ("av_hard_per_h", "AV hard brakes per hour")]
    fig, axs = plt.subplots(1, 3, figsize=(12, 3.4), facecolor=SURFACE)
    for ax, (key, title) in zip(axs, panels):
        y = np.array([r[key] for _, _, r in rows], float)
        sm = np.convolve(y, np.ones(10) / 10, mode="valid")
        ax.set_facecolor(SURFACE)
        ax.plot(x, y, color=SERIES, lw=1, alpha=0.3)
        ax.plot(x[9:], sm, color=SERIES, lw=2)
        ax.axvline(split, color=INK2, lw=1, ls=(0, (3, 3)))
        ax.set_title(title, color=INK, fontsize=10, loc="left")
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.grid(color=GRID, lw=0.6)
        ax.set_xlabel("PPO update (64 steps x 8 copies of Oakland)", color=INK2, fontsize=8)
        if key == "av_hard_per_h":
            ax.set_yscale("log")
    axs[0].text(split + 4, axs[0].get_ylim()[0] + 0.05 * np.ptp(axs[0].get_ylim()), "fine-tune:\nprogress 0.04/m", color=INK2, fontsize=8)
    fig.suptitle("Multi-agent PPO in Oakland (training statistics, with exploration noise)", color=INK, fontsize=11, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(out, dpi=110, facecolor=SURFACE)
    print("wrote", out)


def gif(policy="results/xnet3/policy.npz", out="docs/img/xnet_learned_avs.gif"):
    from xnet import network as NW
    from xnet import render as R
    from xnet import sim as X

    from . import xppo as P

    net, geo, info = NW.build("oakland")
    sim = X.Sim(net, n_envs=1, beh=X.Behaviour(p_av=0.5, demand=1.2), policy=P.deterministic(P.load(policy)), seed=4)
    sim.run(600)
    frames = sim.run(75, record=(0, 2))
    nd = [n for n in np.nonzero(net["node_signal"])[0]
          if {"Forbes Avenue", "South Craig Street"} <= {geo.link_name[l] for l in np.nonzero(net["link_end"] == n)[0]}][0]
    R.animate(net, geo, info, frames, out, zoom=(geo.node_xy[nd], 100.0),
              title="Learned AVs (blue) in Oakland: one shared PPO policy drives half of the vehicles", fps=8, dpi=72)


if __name__ == "__main__":
    curves()
    gif()
