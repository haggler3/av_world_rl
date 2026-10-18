"""Quick CPU architecture experiments (60 s of training each on a 200k-window sample, same 800 held-out windows):
output bases (control points, DCT, a physics decoder for the leader) and classifier-free guidance on the actions.

    python -m dwm.explore           # ~6 min
"""
import json
import os
import time

import numpy as np

from . import model as M
from . import train as T
from .quick import small_data

EXPS = [
    ("control points (10/channel)", dict(basis="knots", nb=10)),
    ("DCT (8 coefficients/channel)", dict(basis="dct", nb=8)),
    ("leader physics decoder (accel knots, integrated)", dict(basis="leader_phys", nb=10)),
    ("leader physics + classifier-free guidance", dict(basis="leader_phys", nb=10, cond_drop=0.1)),
    ("DCT + observation noise", dict(basis="dct", nb=8, obs_noise=True)),
    ("leader physics + observation noise", dict(basis="leader_phys", nb=10, obs_noise=True)),
]
GUIDANCE = [1.0, 1.5]


def projection_floor(cfg, d, st, idx):
    """Gap error at 3 s left after encoding and decoding the real future: the best this basis can do."""
    if cfg.basis == "legacy":
        return float("nan")
    Dec, Enc, _ = T.get_codec(cfg, d, st)
    r = ((d["fut"][idx] - T.base_of(cfg, d, idx)) / st.sd[cfg.target]).reshape(len(idx), -1)
    rec = ((r @ Enc.T) @ Dec.T).reshape(len(idx), -1, 6) * st.sd[cfg.target] + T.base_of(cfg, d, idx)
    m = (d["cur"][idx][:, 2] > 0.5) & (d["fut"][idx][:, -1, 2] > 0.5)
    return float(np.abs((rec[m, -1, 3] - d["fut"][idx][m, -1, 3]) * T.SCALE[3]).mean())


def main(budget=60, out="results/dwm/explore", only=None):
    t0 = time.time()
    os.makedirs(out, exist_ok=True)
    d = small_data()
    st = T.Stats(d)
    idx = T.eval_idx(d, 800)
    ref = T.metrics(T.baselines(d, idx)["kinematic"], d, idx)
    rows = []
    for name, over in [e for e in EXPS if only is None or any(o in e[0] for o in only)]:
        cfg = T.with_latent(M.Cfg(sample_steps=5, **over))
        r = T.train(cfg, d, st, budget, verbose=False)
        for w in (GUIDANCE if cfg.cond_drop > 0 else [1.0]):
            pred, ss = T.sample_physical(r, cfg, st, d, idx, 8, guidance=w)
            m = T.metrics(pred, d, idx)
            row = {"variant": name + (f", guidance {w}" if cfg.cond_drop > 0 else ""), "steps": r["steps"], "latent": cfg.lat or "-",
                   "skill_vs_kinematic": round(T.skill(m, ref), 3), "crps_gap_1s": round(m["crps_gap_1s"], 3),
                   "crps_gap_3s": round(m["crps_gap_3s"], 3), "crps_lead_v_3s": round(m["crps_lead_v_3s"], 3),
                   "coverage90_gap_3s": round(m["coverage90_gap_3s"], 3), "roughness_vs_real": round(m["roughness_gap_vs_real"], 2),
                   "projection_floor_gap_3s_m": round(projection_floor(cfg, d, st, idx), 3)}
            rows.append(row)
            print(json.dumps(row), flush=True)
    path = os.path.join(out, "results.json")
    if only and os.path.exists(path):  # merge with the earlier rows, replacing re-run variants
        old = [r for r in json.load(open(path)) if not any(r["variant"].startswith(o) for o in {x["variant"].split(",")[0] for x in rows})]
        rows = old + rows
    with open(path, "w") as f:
        json.dump(rows, f, indent=1)
    cols = list(rows[0])
    md = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n" + "".join("| " + " | ".join(str(r[c]) for c in cols) + " |\n" for r in rows)
    with open(os.path.join(out, "results.md"), "w") as f:
        f.write(md)
    print(md, f"\ntotal {time.time() - t0:.0f} s")


if __name__ == "__main__":
    import sys

    main(only=sys.argv[1:] or None)
