"""Export a trained policy for the Rust stack (`small_av_world/src/av/policy.json`) and NumPy (`small_av`), plus
test vectors (observations from real env rollouts -> expected policy output) for the Rust parity test.

    python -m avrl.export results/main/policy.npz --world ../small_av_world --small-av ../small_av
"""
import argparse
import json
import os

import jax
import jax.numpy as jnp
import numpy as np

from . import env as E
from . import ppo


def policy_json(params):
    return {"obs_dim": E.OBS_DIM, "activation": "tanh", "action": "u in [-1,1] -> accel = 3u (u>=0) or 8u (u<0)",
            "layers": [{"w": np.asarray(l["w"]).tolist(), "b": np.asarray(l["b"]).tolist()} for l in params["pi"]]}


def test_vectors(params, n=64, seed=3):
    """Observations sampled along rule-driver rollouts (covering queues, lights, leaders) and the policy's raw output."""
    from .baselines import rule_policy

    cfg = E.EnvConfig()
    st = jax.vmap(E.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(seed), 16), cfg)
    obs = jax.vmap(E.observe)(st)
    out = []
    step = jax.jit(jax.vmap(E.step, in_axes=(0, 0, None)), static_argnums=2)
    for i in range(200):
        st, obs, *_ = step(st, rule_policy(obs), cfg)
        if i % 50 == 49:
            out.append(np.asarray(obs))
    o = np.concatenate(out)[:n]
    u = np.asarray(ppo.policy_mean(params, jnp.asarray(o)))
    shield = np.asarray(jax.vmap(E.shield_bound)(jnp.asarray(o)))
    return {"obs": o.tolist(), "u": u.tolist(), "shield": shield.tolist()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("policy")
    ap.add_argument("--world", default=None, help="small_av_world checkout")
    ap.add_argument("--small-av", default=None, help="small_av checkout")
    a = ap.parse_args()
    params = ppo.load_params(a.policy)
    pj = policy_json(params)
    if a.world:
        with open(os.path.join(a.world, "src/av/policy.json"), "w") as f:
            json.dump(pj, f)
        with open(os.path.join(a.world, "tests/policy_vectors.json"), "w") as f:
            json.dump(test_vectors(params), f)
        print("wrote", os.path.join(a.world, "src/av/policy.json"), "and tests/policy_vectors.json")
    if a.small_av:
        path = os.path.join(a.small_av, "avsim/models/longitudinal_policy.npz")
        np.savez(path, **{f"w{i}": np.asarray(l["w"]) for i, l in enumerate(params["pi"])},
                 **{f"b{i}": np.asarray(l["b"]) for i, l in enumerate(params["pi"])})
        print("wrote", path)


if __name__ == "__main__":
    main()
