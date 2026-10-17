"""World-model figures: imagined vs real open-loop rollouts and how the error grows with the horizon."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .env import DT, V_SCALE  # noqa: E402


def figure(real, imag_city, imag_env, curve_city, curve_env, path, n_examples=3):
    t = np.arange(real.shape[1]) * DT
    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, n_examples + 1)
    # pick examples where something happens (speed changes by > 2 m/s)
    dv = np.abs(real[:, -1, 0] - real[:, 0, 0]) * V_SCALE
    ex = np.argsort(-dv)[: n_examples * 3 : 3]
    for j, i in enumerate(ex):
        for row, (f, k, scale, unit) in enumerate((("speed", 0, V_SCALE * 3.6, "km/h"), ("gap to leader", 11, 50.0, "m"))):
            ax = fig.add_subplot(gs[row, j])
            ax.plot(t, real[i, :, k] * scale, color="black", lw=2, label="city (real)")
            ax.plot(t, imag_city[i, :, k] * scale, color="#1f6feb", lw=1.5, label="model learned from city")
            ax.plot(t, imag_env[i, :, k] * scale, color="#e0a030", lw=1.5, ls="--", label="model learned from JAX env")
            ax.set_title(f"{f}, example {j + 1}", fontsize=9)
            ax.set_ylabel(unit)
            ax.set_xlabel("imagined horizon [s]")
            if row == 0 and j == 0:
                ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[0, -1])
    for imag, c, lab in ((imag_city, "#1f6feb", "learned from city"), (imag_env, "#e0a030", "learned from JAX env")):
        ax.plot(t, np.abs(imag[:, :, 0] - real[:, :, 0]).mean(0) * V_SCALE * 3.6, color=c, label=lab)
    ax.set_title("open-loop speed error vs horizon", fontsize=9)
    ax.set_ylabel("mean |error| [km/h]")
    ax.set_xlabel("horizon [s]")
    ax.legend(fontsize=7)
    ax = fig.add_subplot(gs[1, -1])
    w = 50
    for cur, c, lab in ((curve_city, "#1f6feb", "city data"), (curve_env, "#e0a030", "JAX env data")):
        ax.plot(np.convolve(cur, np.ones(w) / w, mode="valid"), color=c, label=lab)
    ax.set_yscale("log")
    ax.set_title("training loss", fontsize=9)
    ax.set_xlabel("step")
    ax.legend(fontsize=7)
    fig.suptitle("World model in the policy's feature space: does it predict what the city does next?", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)
