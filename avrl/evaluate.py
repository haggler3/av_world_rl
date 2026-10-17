"""Roll a policy in the batched env and report per-km safety/comfort metrics."""
import jax
import jax.numpy as jnp

from . import env as E


def rollout(policy_fn, key, n_envs=1024, steps=2048, cfg=E.EnvConfig()):
    """policy_fn(obs [B, OBS_DIM]) -> u [B]. Returns totals summed over all finished + running episodes."""
    k1, k2 = jax.random.split(key)
    st = jax.vmap(E.reset, in_axes=(0, None))(jax.random.split(k1, n_envs), cfg)
    obs = jax.vmap(E.observe)(st)
    vstep = jax.vmap(E.step, in_axes=(0, 0, None, 0))
    keys = ("dist", "hard", "coll", "red", "blocked", "segs", "jerk", "stopped", "steps")

    def body(c, k):
        st, obs, acc = c
        sp = jax.vmap(E.make_spare, in_axes=(0, None))(jax.random.split(k, n_envs), cfg)
        st, obs, r, d, info = vstep(st, policy_fn(obs), cfg, sp)
        acc = {n: acc[n] + jnp.where(d, info[n], 0.0).sum() for n in keys} | {"reward": acc["reward"] + r.sum(),
                                                                          "shield": acc["shield"] + info["s_shield"].sum()}
        return (st, obs, acc), None

    acc0 = {n: jnp.zeros(()) for n in keys + ("reward", "shield")}
    (st, obs, acc), _ = jax.lax.scan(body, (st, obs, acc0), jax.random.split(k2, steps))
    # add the episodes still running
    run = {"dist": st.m_dist, "hard": st.m_hard, "coll": st.m_coll, "red": st.m_red, "blocked": st.m_blocked,
           "segs": st.m_segs, "jerk": st.m_jerk, "stopped": st.m_stall_time, "steps": st.step.astype(jnp.float32)}
    return {n: float(acc[n] + (run[n].sum() if n in run else 0.0)) for n in acc}


def summary(t):
    km = t["dist"] / 1000.0
    hours = t["steps"] * E.DT / 3600.0
    return {
        "km": round(km, 1),
        "mean_speed_kmh": round(km / hours, 2),
        "hard_brakes_per_km": round(t["hard"] / km, 3),
        "collisions_per_1000km": round(1000 * t["coll"] / km, 2),
        "red_runs_per_100_segments": round(100 * t["red"] / max(t["segs"], 1), 3),
        "blocked_entries_per_100_segments": round(100 * t["blocked"] / max(t["segs"], 1), 3),
        "rms_jerk": round((t["jerk"] / (t["steps"] * E.DT)) ** 0.5, 3),
        "reward_per_step": round(t["reward"] / t["steps"], 4),
        "envelope_interventions_pct_of_steps": round(100 * t["shield"] / t["steps"], 3),
    }
