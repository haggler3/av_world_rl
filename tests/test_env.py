import jax
import jax.numpy as jnp
import numpy as np
import pytest

from avrl import env as E
from avrl import baselines, evaluate, ppo

CFG = E.EnvConfig()


def _batch(n=64, seed=0):
    st = jax.vmap(E.reset, in_axes=(0, None))(jax.random.split(jax.random.PRNGKey(seed), n), CFG)
    return st, jax.vmap(E.observe)(st)


def test_shapes_and_finite():
    st, obs = _batch()
    assert obs.shape == (64, E.OBS_DIM)
    step = jax.jit(jax.vmap(E.step, in_axes=(0, 0, None)), static_argnums=2)
    for _ in range(50):
        st, obs, r, d, info = step(st, jnp.zeros(64), CFG)
        assert np.isfinite(np.asarray(obs)).all() and np.isfinite(np.asarray(r)).all()


def test_deterministic():
    a, _ = _batch(seed=3)
    b, _ = _batch(seed=3)
    sa, *_ = jax.vmap(E.step, in_axes=(0, 0, None))(a, jnp.full(64, 0.5), CFG)
    sb, *_ = jax.vmap(E.step, in_axes=(0, 0, None))(b, jnp.full(64, 0.5), CFG)
    assert all(np.array_equal(np.asarray(x), np.asarray(y)) for x, y in zip(jax.tree.leaves(sa), jax.tree.leaves(sb)))


def test_light_matches_world_timing():
    # network.rs: green until 4 s before the end of the approach's phase, 3 s yellow, 1 s all-red, then red
    seg = E.Seg(L=100.0, vmax=11.1, cycle=60.0, green=24.0, offset=0.0, signal=True, arrives=False, cap=9.0)
    lights = [int(E.light_state(seg, t)) for t in (0.0, 19.9, 20.1, 22.9, 23.1, 30.0, 59.9)]
    assert lights == [2, 2, 1, 1, 0, 0, 0]


def test_spawned_traffic_respects_entry_room():
    st, _ = _batch(512)
    need = 4.5 + 1.0 + np.minimum(np.asarray(st.ego_v), 9.0) ** 2 / 6.0
    rear = np.where(np.asarray(st.alive), np.asarray(st.pos - st.length), np.inf).min(1)
    assert (rear >= need + E.ENTRY_S - E.EGO_LEN - 1e-4).all()


def test_full_throttle_crashes_and_rule_driver_does_not():
    raw = E.EnvConfig(shield=False)
    bad = evaluate.summary(evaluate.rollout(lambda o: jnp.full(o.shape[0], 1.0), jax.random.PRNGKey(0), 128, 600, raw))
    good = evaluate.summary(evaluate.rollout(baselines.rule_policy, jax.random.PRNGKey(0), 128, 600))
    assert bad["collisions_per_1000km"] > 500 and bad["red_runs_per_100_segments"] > 5
    assert good["collisions_per_1000km"] < 20 and good["red_runs_per_100_segments"] < 0.5


def test_ppo_update_runs():
    cfg = ppo.Config(n_envs=32, n_steps=8, minibatches=2, epochs=1)
    import optax

    opt = optax.adam(1e-3)
    params = ppo.init_params(jax.random.PRNGKey(0))
    st, obs = _batch(32)
    upd = ppo.make_update(cfg, CFG, opt)
    params2, _, _, _, logs = upd(params, opt.init(params), st, obs, jax.random.PRNGKey(1))
    assert np.isfinite(float(logs["reward"]))
    assert not np.allclose(np.asarray(params2["pi"][0]["w"]), np.asarray(params["pi"][0]["w"]))


def test_envelope_alone_keeps_full_throttle_safe():
    m = evaluate.summary(evaluate.rollout(lambda o: jnp.full(o.shape[0], 1.0), jax.random.PRNGKey(0), 128, 600))
    assert m["collisions_per_1000km"] < 20 and m["red_runs_per_100_segments"] < 0.5
    assert m["envelope_interventions_pct_of_steps"] > 10  # it is doing all the work here


def test_no_handover_without_permission():
    # red light, ego 1 m before the line at 5 m/s, full throttle: it must not be handed over; deep in the junction = crash
    st = E.reset(jax.random.PRNGKey(5), CFG)
    seg = st.seg._replace(signal=jnp.asarray(True), arrives=jnp.asarray(False), offset=jnp.asarray(30.0), cycle=jnp.asarray(60.0), green=jnp.asarray(24.0))
    st = st._replace(seg=seg, t=jnp.asarray(0.0), ego_s=seg.L - 1.0, ego_v=jnp.asarray(5.0), alive=jnp.zeros(E.N, bool))
    assert int(E.light_state(st.seg, st.t)) == 0
    rewards, crashed = [], False
    raw = E.EnvConfig(shield=False)
    for _ in range(40):
        st, _, r, d, info = E.step(st, jnp.asarray(1.0), raw)
        rewards.append(float(r))
        if bool(d):
            crashed = True
            break
        assert float(info["s_segs"]) == 0.0
    assert crashed and min(rewards) <= E.R_COLLISION
