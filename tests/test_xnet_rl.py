"""The multi-agent PPO on av_world's area simulator (needs av_world installed and its map data fetched)."""
import os

import jax
import jax.numpy as jnp
import numpy as np
import pytest

xnet = pytest.importorskip("xnet")
from xnet import network as NW  # noqa: E402
from xnet import sim as X  # noqa: E402

pytestmark = pytest.mark.skipif(not os.path.exists(os.path.join(os.path.dirname(NW.__file__), "..", "data", "pgh_segments.parquet")),
                                reason="needs av_world's python -m pgh.fetch")


@pytest.fixture(scope="module")
def net():
    return NW.build("oakland")[0]


def test_rl_hook_shapes_and_masks(net):
    from avrl import xppo as P

    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    params = P.init_params(jax.random.PRNGKey(0))
    st = X.reset(jax.random.PRNGKey(1), J, 512)
    f = jax.jit(lambda s, k: X.step(s, J, X.Behaviour(p_av=0.5), nh, rl=(P.stochastic(params), k)))
    for i in range(300):
        st, tr = f(st, jax.random.PRNGKey(i))
    assert tr["obs"].shape == (512, X.OBS_DIM) and tr["u"].shape == (512, 2)
    v = np.asarray(tr["valid"])
    assert v.sum() > 10 and np.all(np.asarray(tr["reward"])[~v] == 0)
    assert np.all(np.isfinite(np.asarray(tr["obs"])[v]))


def test_shield_keeps_a_full_throttle_policy_safe(net):
    """A policy that always floors it is capped by the envelope: no AV red runs, AVs stop for signals."""
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    clean = X.Behaviour(p_red=0, p_nolook=0, p_nosignal=0, distract_rate=0, p_ped_violate=0, p_av=0.5)
    floor = lambda o, k: (jnp.stack([jnp.ones(o.shape[0]), jnp.zeros(o.shape[0])], 1), jnp.zeros((o.shape[0], 2)))
    sim = X.Sim(net, n_envs=2, beh=clean, policy=floor, seed=2)
    sim.run(600)
    d = dict(zip(X.METRICS, np.asarray(sim.st.m).sum(0)))
    assert d["av_red"] == 0 and d["av_finished"] > 50


def test_ppo_update_runs(tmp_path, net):
    from avrl import xppo as P

    P.train(str(tmp_path), updates=2, n_envs=2, n_slots=512, n_steps=8, bc_updates=1, warm=10.0)
    assert os.path.exists(tmp_path / "policy.npz")
