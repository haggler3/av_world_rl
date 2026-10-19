"""Windows of city AV logs for the diffusion world model.

A sample is (history, future actions) -> future state: the last H decisions of one AV (all 22 policy features), the
actions it takes over the next F decisions, and what then happens to six state channels. Windows never cross a segment
hand-over (tracks and the stop-line frame restart there) and train / test are split by vehicle.
"""
import os

import numpy as np

SOURCE = os.environ.get("DWM_SOURCE", "city")  # "city": small_av_world logs (10 Hz) | "xnet": av_world's area simulator (5 Hz)
if SOURCE == "xnet":
    OBS_DIM = 28  # xnet.sim.OBS_DIM
    DT = 0.2
    H, F = 5, 15  # 1 s of history, 3 s of future at 5 Hz
    # speed, distance to the stop target, leader seen, gap, leader speed, stop required (xnet observation indices)
    CHANNELS = [(0, "speed", 15.0), (7, "d_line", 100.0), (8, "lead_seen", 1.0), (9, "gap", 50.0), (10, "lead_v", 15.0), (6, "must_stop", 1.0)]
    LAG = False  # xnet applies the acceleration directly
else:
    OBS_DIM = 22
    DT = 0.1
    H, F = 10, 30  # 1 s of history, 3 s of future at 10 Hz
    # predicted channels: (feature index, name, physical scale of the feature)
    CHANNELS = [(0, "speed", 15.0), (3, "d_line", 100.0), (10, "lead_seen", 1.0), (11, "gap", 50.0), (12, "lead_v", 15.0), (9, "must_stop", 1.0)]
    LAG = True  # the Rust vehicle model: 0.25 s actuator lag
ROW = OBS_DIM + 4  # [vehicle id, t, segment, obs.., u]  (small_av_world AV_TRANSITIONS format)
CH = np.array([c[0] for c in CHANNELS])
NC = len(CHANNELS)


def load_rows(path):
    return np.fromfile(path, dtype="<f4").reshape(-1, ROW)


def windows(rows, stride=3):
    """Index arrays of valid windows: H + F consecutive decisions (DT apart) of one vehicle on one segment."""
    rows = rows[np.lexsort((rows[:, 1], rows[:, 0]))]
    L = H + F
    same = (rows[1:, 0] == rows[:-1, 0]) & (rows[1:, 2] == rows[:-1, 2]) & (np.abs(rows[1:, 1] - rows[:-1, 1] - DT) < 0.02)
    # run length of consecutive "same" links ending at each row
    ok = np.concatenate([[0], same.astype(np.int32)])
    run = np.zeros(len(rows), np.int32)
    for i in range(1, len(rows)):  # numba-free but fast enough (a few M rows: ~2 s)
        run[i] = run[i - 1] + 1 if ok[i] else 0
    ends = np.where(run >= L - 1)[0]
    ends = ends[::stride]
    return rows, ends - (L - 1)


def build(path, out, stride=3, test_frac=0.2, seed=0):
    rows = load_rows(path)
    rows, starts = windows(rows, stride)
    idx = starts[:, None] + np.arange(H + F)
    obs = rows[idx][..., 3:3 + OBS_DIM]
    act = rows[idx][..., 3 + OBS_DIM]
    vid = rows[starts, 0]
    hist = obs[:, :H]  # [N, H, 22]
    fut_act = act[:, H - 1:H - 1 + F]  # the action at the current step and the next F-1
    cur = obs[:, H - 1, CH]  # [N, C]
    fut = obs[:, H:, :][:, :, CH]  # [N, F, C]
    ids = np.unique(vid)
    test_ids = np.random.default_rng(seed).choice(ids, int(len(ids) * test_frac), replace=False)
    te = np.isin(vid, test_ids)
    os.makedirs(out, exist_ok=True)
    for k, v in dict(hist=hist.astype(np.float32), act=fut_act.astype(np.float32), cur=cur.astype(np.float32),
                     fut=fut.astype(np.float32), test=te, vid=vid).items():
        np.save(os.path.join(out, k + ".npy"), v)  # plain .npy: loads in seconds (an .npz took minutes)
    print(f"{len(starts)} windows ({(~te).sum()} train / {te.sum()} test) from {len(ids)} vehicles -> {out}")


def load(path):
    return {f[:-4]: np.load(os.path.join(path, f)) for f in os.listdir(path) if f.endswith(".npy")}


if __name__ == "__main__":
    import sys

    src = sys.argv[1] if len(sys.argv) > 1 else "data/city10.bin"
    build(src, sys.argv[2] if len(sys.argv) > 2 else "data/dwm")
