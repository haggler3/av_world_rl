"""Learned policy vs the rule driver on identical env seeds: metrics table, comparison figure and GIF.

    python -m avrl.compare results/main/policy.npz --out results/main
"""
import argparse
import json
import os

import jax

from . import baselines, evaluate, ppo, viz
from . import env as E


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--out", default="results/main")
    ap.add_argument("--envs", type=int, default=1024)
    ap.add_argument("--steps", type=int, default=2048)
    a = ap.parse_args()
    params = ppo.load_params(a.policy)
    learned = ppo.deterministic_policy(params)
    key = jax.random.PRNGKey(12345)  # held-out seeds: training never used this key
    res = {}
    runs = (("rule (IDM + stop-line governor)", baselines.rule_policy, E.EnvConfig()),
            ("learned (PPO) inside the safety envelope", learned, E.EnvConfig()),
            ("learned (PPO), envelope switched off", learned, E.EnvConfig(shield=False)))
    for name, fn, cfg in runs:
        res[name] = evaluate.summary(evaluate.rollout(fn, key, a.envs, a.steps, cfg))
        print(name, res[name])
    with open(os.path.join(a.out, "compare.json"), "w") as f:
        json.dump(res, f, indent=1)
    traces = {"rule driver": viz.scenario_trace(baselines.rule_policy), "learned policy": viz.scenario_trace(learned)}
    viz.compare_figure(traces, os.path.join(a.out, "compare_scenario.png"), "Same scenario, two drivers (held-out evaluation seed)")
    viz.animate(traces, os.path.join(a.out, "compare.gif"), seconds=60)


if __name__ == "__main__":
    main()
