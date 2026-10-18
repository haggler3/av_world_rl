"""Equal-budget ablations of the diffusion world model (every variant gets the same training seconds and the same
held-out windows), written to results/dwm/ablations.{json,md}.

    python -m dwm.ablate --budget 90
"""
import argparse
import json
import os
import time

import numpy as np

from . import model as M
from . import train as T

BASE = dict()
VARIANTS = [
    ("base: residual target, MLP 256x3, v-pred, 10 DDIM steps", {}),
    ("target = change from current value (no physics prior)", {"target": "delta"}),
    ("deterministic regressor, same network", {"kind": "regress"}),
    ("predict eps", {"pred": "eps"}),
    ("predict x0", {"pred": "x0"}),
    ("1-D temporal conv denoiser", {"net": "conv"}),
    ("bigger MLP 512x4", {"width": 512, "depth": 4}),
    ("no future actions in the condition", {"use_actions": False}),
    ("history = current step only", {"hist_len": 1}),
]
SAMPLE_STEPS = [2, 5, 20]  # extra evaluations of the base model, no retraining
KEYS = ["crps_speed_3s", "crps_gap_1s", "crps_gap_3s", "crps_lead_v_3s", "coverage90_gap_3s", "acc_lead_seen_3s", "acc_must_stop_3s"]


def row(name, m, ref, extra):
    return {"variant": name, "skill_vs_kinematic": round(T.skill(m, ref), 3), **{k: round(m[k], 3) for k in KEYS if k in m}, **extra}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/dwm")
    ap.add_argument("--budget", type=float, default=90)
    ap.add_argument("--eval-n", type=int, default=1500)
    ap.add_argument("--K", type=int, default=16)
    ap.add_argument("--out", default="results/dwm")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    d = T.load_data(a.data)
    st = T.Stats(d)
    idx = T.eval_idx(d, a.eval_n)
    base = {k: T.metrics(v, d, idx) for k, v in T.baselines(d, idx).items()}
    ref = base["kinematic"]
    rows = [row(k, m, ref, {"steps": 0, "params": 0}) for k, m in base.items()]
    print(f"data ready in {time.time() - t0:.0f} s", flush=True)
    base_res = None
    for name, over in VARIANTS:
        cfg = M.Cfg(**over)
        res = T.train(cfg, d, st, a.budget, verbose=False)
        K = a.K if cfg.kind != "regress" else 1
        pred, ss = T.sample_physical(res, cfg, st, d, idx, K)
        r = row(name, T.metrics(pred, d, idx), ref, {"steps": res["steps"], "params": res["n_params"],
                                                     "ms_per_future": round(1000 * ss / (len(idx) * K), 3)})
        rows.append(r)
        print(json.dumps(r), flush=True)
        if not over:
            base_res = (cfg, res)
    cfg, res = base_res
    for s in SAMPLE_STEPS:
        pred, ss = T.sample_physical(res, cfg, st, d, idx, a.K, steps=s)
        r = row(f"base, {s} DDIM steps at sampling", T.metrics(pred, d, idx), ref, {"steps": res["steps"], "params": res["n_params"],
                                                                                    "ms_per_future": round(1000 * ss / (len(idx) * a.K), 3)})
        rows.append(r)
        print(json.dumps(r), flush=True)
    with open(os.path.join(a.out, "ablations.json"), "w") as f:
        json.dump({"budget_s": a.budget, "eval_n": a.eval_n, "K": a.K, "rows": rows}, f, indent=1)
    cols = ["variant", "skill_vs_kinematic"] + KEYS + ["steps", "params"]
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for r in rows:
        md += "| " + " | ".join(str(r.get(c, "")) for c in cols) + " |\n"
    with open(os.path.join(a.out, "ablations.md"), "w") as f:
        f.write(md)
    print(md)
    print(f"total {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
