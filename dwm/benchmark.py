"""World models on the original small_av_world benchmark: the 64 trips of `av_world trips --seeds 8` (4 routes x 2 peaks
x 8 seeds), logged once with the original planner stack driving and once with the learned driver. The world models were
trained on a 10 % fleet run with the learned driver only, so the planner-driven trips test generalisation to another
driving style. Also draws the 2D GIFs on a benchmark scene.

    AV_TRANSITIONS=data/bench_planner.bin cargo run --release -- trips --seeds 8 --driver planner   (and --driver learned)
    python -m dwm.benchmark
"""
import json
import os
import pickle

import jax
import jax.numpy as jnp
import numpy as np

from . import data as D
from . import model as M
from . import train as T
from .gifs2d import VARIANTS, render, scene_paths
from .quick import small_data
from .viz import pick_scenes

MODELS = [("final", "diffusion · 30 steps (full data, 7.5 min)", "results/dwm/final/model.pkl")] + \
         [(k, t, f"results/dwm/gifs2d/{k}.pkl") for k, t, _ in VARIANTS]


def load_model(path):
    with open(path, "rb") as f:
        b = pickle.load(f)
    st = T.Stats.__new__(T.Stats)
    st.__dict__.update(b["stats"])
    return {"params": jax.tree.map(jnp.asarray, b["params"])}, b["cfg"], st


def main(n_eval=3000, K=16, out="results/dwm/benchmark"):
    os.makedirs(out, exist_ok=True)
    benches = {}
    for drv in ("planner", "learned"):
        path = f"data/bench_{drv}"
        if not os.path.exists(path):
            D.build(f"data/bench_{drv}.bin", path, test_frac=1.0)
        benches[drv] = T.load_data(path)
    d_train = small_data()  # the codecs (latent scales) come from the training data the models saw
    rows = []
    for drv, d in benches.items():
        idx = T.eval_idx(d, min(n_eval, int(d["test"].sum())))
        base = {k: T.metrics(v, d, idx) for k, v in T.baselines(d, idx).items()}
        ref = base["kinematic"]
        for name, m in base.items():
            rows.append({"benchmark driver": drv, "model": name, "skill_vs_kinematic": round(T.skill(m, ref), 3), **pick(m)})
        for key, title, path in MODELS:
            if not os.path.exists(path):
                continue
            res, cfg, st = load_model(path)
            if cfg.basis != "legacy":
                T.get_codec(cfg, d_train, st)
            pred, _ = T.sample_physical(res, cfg, st, d, idx, K)
            m = T.metrics(pred, d, idx)
            rows.append({"benchmark driver": drv, "model": title, "skill_vs_kinematic": round(T.skill(m, ref), 3), **pick(m)})
            print(json.dumps(rows[-1]), flush=True)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump(rows, f, indent=1)
    cols = list(rows[0])
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "".join("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n" for r in rows)
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(md)
    print(md)
    # 2D GIFs on a benchmark scene (learned-driver trips: the car ahead brakes)
    d = benches["learned"]
    scene = int(pick_scenes(d)[0])
    by = {r["model"]: r for r in rows if r["benchmark driver"] == "learned"}
    for key, title, path in MODELS[1:]:
        res, cfg, st = load_model(path)
        T.get_codec(cfg, d_train, st)
        frames, final = scene_paths(res, cfg, st, d, scene)
        r = by[title]
        sub = (f"64-trip benchmark scene · score vs physics {r['skill_vs_kinematic']:.2f} · gap error 3 s {r['crps_gap_3s']:.2f} m · "
               f"90% band holds {100 * r['coverage90_gap_3s']:.0f}% · roughness {r['roughness_vs_real']:.1f}x")
        render(frames, final, d, scene, title, sub, f"docs/img/dwm_{key}.gif")
        print("gif", key, flush=True)


def pick(m):
    keys = {"crps_speed_3s": "crps_speed_3s", "crps_gap_1s": "crps_gap_1s", "crps_gap_3s": "crps_gap_3s", "crps_lead_v_3s": "crps_lead_v_3s",
            "coverage90_gap_3s": "coverage90_gap_3s", "roughness_gap_vs_real": "roughness_vs_real"}
    return {v: round(m[k], 3) for k, v in keys.items() if k in m}


if __name__ == "__main__":
    main()
