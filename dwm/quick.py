"""A small, fast pass: subsample the data, compare two small denoisers for a fixed time, train the winner briefly, redraw the GIF.

    python -m dwm.quick            # ~8 min on 4 CPU cores
"""
import json
import os
import time

import numpy as np

from . import model as M
from . import train as T
from . import viz


def small_data(path="data/dwm", n_train=200_000, n_test=5_000, seed=0):
    d = {f[:-4]: np.load(os.path.join(path, f), mmap_mode="r") for f in os.listdir(path) if f.endswith(".npy")}
    rng = np.random.default_rng(seed)
    te, tr = np.where(d["test"])[0], np.where(~d["test"])[0]
    keep = np.sort(np.concatenate([rng.choice(tr, n_train, replace=False), rng.choice(te, n_test, replace=False)]))
    return {k: np.asarray(v[keep]) for k, v in d.items()}


def main(out="results/dwm/quick", budget=60, final_budget=180):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    d = small_data()
    st = T.Stats(d)
    idx = T.eval_idx(d, 800)
    ref = T.metrics(T.baselines(d, idx)["kinematic"], d, idx)
    print(f"data {time.time() - t0:.0f} s", flush=True)
    cands = {"MLP 256x3, 10 control points": M.Cfg(sample_steps=5, knots=10),
             "attention 64x2, 10 control points": M.Cfg(sample_steps=5, knots=10, net="attn", width=64, depth=2)}
    rows = {}
    for name, cfg in cands.items():
        r = T.train(cfg, d, st, budget, verbose=False)
        pred, _ = T.sample_physical(r, cfg, st, d, idx, 8)
        m = T.metrics(pred, d, idx)
        rows[name] = {"steps": r["steps"], "params": r["n_params"], "skill_vs_kinematic": round(T.skill(m, ref), 3),
                      "crps_gap_3s": round(m["crps_gap_3s"], 3), "coverage90_gap_3s": round(m["coverage90_gap_3s"], 3),
                      "roughness_gap_vs_real": round(m["roughness_gap_vs_real"], 2)}
        print(name, json.dumps(rows[name]), flush=True)
    best = min(rows, key=lambda k: rows[k]["skill_vs_kinematic"])
    cfg = cands[best]
    r = T.train(cfg, d, st, final_budget, verbose=False)
    pred, _ = T.sample_physical(r, cfg, st, d, idx, 16)
    m = T.metrics(pred, d, idx)
    final = {"model": best, "steps": r["steps"], "skill_vs_kinematic": round(T.skill(m, ref), 3),
             **{k: round(v, 3) for k, v in m.items()}, "kinematic": {k: round(v, 3) for k, v in ref.items()}}
    import pickle

    import jax
    with open(os.path.join(out, "model.pkl"), "wb") as f:
        pickle.dump({"params": jax.tree.map(np.asarray, r["params"]), "cfg": cfg, "stats": st.__dict__}, f)
    viz.render(os.path.join(out, "model.pkl"), None, "docs/img/diffusion_world_model.gif", K=16, fps=12, data=d, hold=4, dpi=80)
    final["total_wall_s"] = round(time.time() - t0, 1)
    with open(os.path.join(out, "results.json"), "w") as f:
        json.dump({"comparison_60s": rows, "final": final}, f, indent=1)
    print("final", best, json.dumps({k: final[k] for k in ("steps", "skill_vs_kinematic", "crps_gap_3s", "coverage90_gap_3s", "roughness_gap_vs_real", "total_wall_s")}))


if __name__ == "__main__":
    main()
