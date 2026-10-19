"""World-model training data from the whole-area simulator (av_world's xnet) instead of small_av_world logs.

Every AV in E copies of the area is logged at every 0.2 s step: [vehicle id, t, 0, obs (28), u0] (u0 = the acceleration
command of the policy that drove it, as in the small_av_world AV_TRANSITIONS format). The AVs are driven by the policy
trained with avrl.xppo, with its exploration noise (so the model sees a spread of actions); the humans, pedestrians,
buses and signals around them are the full simulator.

    python -m dwm.xnet_data results/xnet2/policy.npz          # -> data/xnet_av.bin, data/dwm_xnet/
    DWM_SOURCE=xnet python -m dwm.train --data data/dwm_xnet --budget 240 --out results/dwm/xnet
"""
import os
import sys
import time

os.environ.setdefault("DWM_SOURCE", "xnet")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from xnet import network as NW  # noqa: E402
from xnet import sim as X  # noqa: E402

from avrl import xppo as P  # noqa: E402

from . import data as D  # noqa: E402


def collect(policy_path, out="data/xnet_av.bin", n_envs=4, minutes=15.0, warm=300.0, p_av=0.25, seed=5, chunk=250):
    assert D.SOURCE == "xnet" and D.OBS_DIM == X.OBS_DIM
    net, geo, info = NW.build("oakland")
    J, nh = X.to_jnp(net), jnp.asarray(net["nh"])
    params = P.load(policy_path)
    beh = X.Behaviour(p_av=p_av, demand=1.0)
    pol = P.stochastic(params)

    @jax.jit
    def run(st, key):
        def body(st, k):
            st, tr = jax.vmap(lambda s, kk: X.step(s, J, beh, nh, rl=(pol, kk)))(st, jax.random.split(k, n_envs))
            return st, (tr["obs"], tr["u"][..., 0], tr["valid"], tr["done"])
        return jax.lax.scan(body, st, jax.random.split(key, chunk))

    st = jax.vmap(lambda k: X.reset(k, J, 1024))(jax.random.split(jax.random.PRNGKey(seed), n_envs))
    key = jax.random.PRNGKey(seed + 1)
    for i in range(int(warm / (X.DT * chunk))):
        st, _ = run(st, jax.random.fold_in(key, i))
    ids = -np.ones((n_envs, 1024), np.int64)
    alive_prev = np.zeros((n_envs, 1024), bool)
    next_id, t = 0, 0.0
    n_rows = 0
    t0 = time.time()
    with open(out, "wb") as f:
        for c in range(int(minutes * 60 / (X.DT * chunk))):
            st, (obs, u, valid, done) = run(st, jax.random.fold_in(key, 1000 + c))
            obs, u, valid, done = (np.asarray(x) for x in (obs, u, valid, done))
            for k in range(chunk):
                new = valid[k] & ~alive_prev
                ids[new] = next_id + np.arange(new.sum())
                next_id += int(new.sum())
                v = valid[k]
                rows = np.concatenate([ids[v][:, None], np.full((v.sum(), 1), t), np.zeros((v.sum(), 1)), obs[k][v], u[k][v][:, None]], 1)
                f.write(rows.astype("<f4").tobytes())
                n_rows += len(rows)
                alive_prev = v & ~done[k]
                t += X.DT
            print(f"chunk {c}: {n_rows} rows, {next_id} vehicles, {time.time() - t0:.0f} s", flush=True)
    return out


if __name__ == "__main__":
    path = collect(sys.argv[1])
    D.build(path, "data/dwm_xnet")
