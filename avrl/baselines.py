"""Hand-written drivers that only use the policy's observation vector (so they are comparable with the learned policy).

`rule_policy` is the longitudinal part of the Rust AV stack without its sampling planner: the IDM car-following governor
(agent.rs `follow_governor`: a 1.6, b 2.0, T 1.3 s, s0 2.5 m) and a stop-line governor towards a virtual stopped car 1 m
before the line whenever a stop is required (red, a yellow we can stop for, or no room at the next segment).
"""
import jax.numpy as jnp

from .env import V_SCALE, SENSOR_RANGE, idm, idm_free


def unpack(o):
    return dict(v=o[..., 0] * V_SCALE, vmax=o[..., 2] * V_SCALE, d=o[..., 3] * 100.0, must=o[..., 9] > 0.5,
                seen=o[..., 10] > 0.5, gap=o[..., 11] * 50.0, vl=o[..., 12] * V_SCALE)


def accel_to_u(a):
    return jnp.where(a >= 0.0, a / 3.0, a / 8.0)


def rule_policy(o, a_max=1.6, b=2.0, thw=1.3, s0=2.5):
    f = unpack(o)
    v, v0 = f["v"], jnp.maximum(f["vmax"], 1.0)
    a_follow = jnp.where(f["seen"], idm(v, v0, f["gap"], jnp.maximum(f["vl"], 0.0), a_max, b, thw, s0), idm_free(v, v0, a_max))
    a_stop = jnp.where(f["must"], idm(v, v0, f["d"] - 1.0, 0.0, a_max, b, thw, 1.0), 99.0)
    return jnp.clip(accel_to_u(jnp.minimum(a_follow, a_stop)), -1.0, 1.0)
