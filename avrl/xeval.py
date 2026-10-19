"""Evaluate a policy trained with avrl.xppo in the whole simulated area, against the rule-based AV and humans.

Same seeds for every row; 8 copies of the area x 30 simulated minutes after a 5-minute warm-up; humans with the
calibrated risky behaviour (and a x10 riskier world to measure AV crash involvement).

    python -m avrl.xeval results/xnet/policy.npz        # -> results/xnet/eval.json, eval.md
"""
import json
import os
import sys
import time

import jax
import numpy as np

from xnet import network as NW
from xnet import sim as X

from . import xppo as P


def run(net, beh, policy, n_envs=8, seconds=1800.0, warm=300.0, seed=11):
    sim = X.Sim(net, n_slots=1024, n_envs=n_envs, beh=beh, seed=seed, policy=policy)
    sim.run(warm)
    sim.st = sim.st._replace(m=jax.numpy.zeros_like(sim.st.m))
    t0 = time.time()
    sim.run(seconds)
    s = P.summarize(np.asarray(sim.st.m).sum(0))
    s["wall_s"] = round(time.time() - t0, 1)
    d = dict(zip(X.METRICS, np.asarray(sim.st.m).sum(0)))
    s["mean_speed_kmh"] = float(3.6 * d["veh_m"] / max(d["veh_s"], 1))
    s["junction_entries"] = int(d["entered"])
    return s


def main(path, area="oakland", only=None):
    out = os.path.dirname(path)
    net, geo, info = NW.build(area)
    knobs = json.load(open(os.path.join(os.path.dirname(NW.__file__), "..", "results", "calibration.json")))["calibrated"]
    params = P.load(path)
    learned = P.deterministic(params)
    rows = []
    cases = [("humans only", 0.0, 1.0, None)]
    for share in (0.25, 1.0):
        cases += [(f"{share:.0%} AVs, rule-based", share, 1.0, None), (f"{share:.0%} AVs, learned", share, 1.0, learned)]
    cases += [("25% AVs, rule-based, humans x10 riskier", 0.25, 10.0, None), ("25% AVs, learned, humans x10 riskier", 0.25, 10.0, learned)]
    if only:
        cases = [c for c in cases if only in c[0]]
    for name, share, A, pol in cases:
        beh = X.Behaviour(**{k: min(v * A, 0.95) for k, v in knobs.items()}, p_av=share, demand=1.0)
        s = run(net, beh, pol)
        s["case"] = name
        rows.append(s)
        print(json.dumps(s, default=float), flush=True)
    if only:
        return rows
    with open(os.path.join(out, "eval.json"), "w") as f:
        json.dump(rows, f, indent=1, default=float)
    cols = [("trips_finished", "trips", "{:.0f}"), ("mean_speed_kmh", "speed km/h", "{:.1f}"), ("av_trip_min", "AV trip min", "{:.2f}"),
            ("human_trip_min", "human trip min", "{:.2f}"), ("av_hard_per_h", "AV hard brakes /h", "{:.1f}"),
            ("av_jerk_rms", "AV jerk rms", "{:.2f}"), ("av_near", "AV near misses", "{:.0f}"), ("av_crashes", "AV crashes", "{:.0f}"),
            ("av_red", "AV red runs", "{:.0f}"), ("av_shield_share", "shield active", "{:.1%}"), ("teleports", "teleports", "{:.0f}")]
    lines = ["| case | " + " | ".join(c[1] for c in cols) + " |", "|---|" + "---|" * len(cols)]
    for r in rows:
        lines.append(f"| {r['case']} | " + " | ".join(fmt.format(r[k]) for k, _, fmt in cols) + " |")
    with open(os.path.join(out, "eval.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1], only="learned" if "--quick" in sys.argv else None)
