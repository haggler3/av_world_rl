"""A pure-JAX longitudinal driving environment that mirrors one AV in `small_av_world`.

Inside the city the AV never leaves its lane: on each road segment it follows the car ahead, stops at the stop line for a
red light or a full junction exit, and is handed over to the next segment. This env reproduces exactly that problem
(same IDM drivers, signal timing, entry-room rule, hand-over speed cap and the tracker's quirks) as a fixed-shape JAX
function, so thousands of copies run under `jax.vmap` and a policy can be trained with PPO in minutes.

Positions are metres along the current segment; the ego position `ego_s` is its front bumper (as in the Rust agent).
Every array in `State` has a fixed shape, `step` auto-resets, and nothing here allocates per step.
"""
from typing import NamedTuple

import jax
import jax.numpy as jnp

DT = 0.1  # policy and physics step [s] (the Rust agent calls the policy every 2 control steps = 10 Hz)
N = 8  # same-lane vehicle slots ahead of the ego
EGO_LEN = 4.5
ENTRY_S = 4.25  # ego front bumper after a hand-over (`enter_segment(.., 2.0 + 2.25)` in fleet.rs)
MAX_ACCEL, MAX_DECEL, TAU = 3.0, 8.0, 0.25  # vehicle.rs
SENSOR_RANGE = 60.0  # lidar.rs MAX_RANGE
EP_STEPS = 2048
SEG_TIMEOUT = 120.0  # a stalled vehicle blocks a segment for 10-25 min in the world; cut the episode's segment short
OBS_DIM = 22
V_SCALE = 15.0

# reward weights (per 0.1 s step)
W_PROGRESS = 0.01  # per metre driven (0.04 per step at a typical 4 m/s)
W_COMFORT = 0.0005  # per (m/s^2)^2
W_HARD = 0.1  # per (m/s^2 beyond 3) per step while moving
W_JERK = 0.00002  # per (m/s^3)^2
W_HEADWAY = 0.05
W_IDLE = 0.01  # standing still with nothing to wait for (no stop required, no car within 8 m)
R_COLLISION = -30.0
R_RED = -20.0
R_BLOCKED = -15.0  # entering a full junction: the world holds the AV at the line and the car behind can run into it
W_SPEED = 0.05  # per (m/s above the speed limit) per step
W_GAP = 0.1  # per metre the standstill/queue gap is below 2.5 m (world drivers keep ~2 m; circles touch below ~0.2 m)
W_SHIELD = 0.02  # per m/s^2 the safety envelope had to take off the policy's command
W_PAST = 1.0  # per step past the stop line without permission to enter the junction (red or full exit)
DEEP = 8.0  # driving this far past the line without permission = entering the junction into cross traffic (a crash)
SNAP_EVERY = 5  # the world updates every 0.5 s; the AV sees other cars extrapolated at constant velocity in between


class EnvConfig(NamedTuple):
    """Domain-randomisation knobs (static: changing one recompiles)."""

    p_stall: float = 0.04  # a stalled vehicle (incident) on a new segment
    p_arrive: float = 0.05  # this segment is the last of the trip (no stop line)
    p_unsignalised: float = 0.08
    p_congested_exit: float = 0.35  # the next segment's queue reaches back towards its entry
    p_queue: float = 0.45  # cars ahead start as a standing queue (vs flowing traffic)
    insert_rate: float = 1.0 / 25.0  # cross traffic turning into the next segment [1/s]
    noise: float = 1.0  # scale of tracker noise
    shield: bool = True  # cap the policy's acceleration with the classical safety envelope (`shield_bound`)
    episode_steps: int = EP_STEPS


class Seg(NamedTuple):
    L: jnp.ndarray
    vmax: jnp.ndarray
    cycle: jnp.ndarray
    green: jnp.ndarray  # length of this approach's phase (green + yellow + all-red) [s]
    offset: jnp.ndarray
    signal: jnp.ndarray  # bool
    arrives: jnp.ndarray  # bool
    cap: jnp.ndarray  # hand-over speed cap (9 straight, 5 turn)


class Drivers(NamedTuple):
    v0f: jnp.ndarray
    thw: jnp.ndarray
    amax: jnp.ndarray
    b: jnp.ndarray
    s0: jnp.ndarray
    yrun: jnp.ndarray


class State(NamedTuple):
    key: jnp.ndarray
    t: jnp.ndarray  # signal clock
    seg_t: jnp.ndarray
    step: jnp.ndarray
    seg: Seg
    pos: jnp.ndarray  # [N] front bumpers
    v: jnp.ndarray
    length: jnp.ndarray
    alive: jnp.ndarray
    stalled: jnp.ndarray
    drv: Drivers
    ego_s: jnp.ndarray
    ego_v: jnp.ndarray
    ego_a: jnp.ndarray
    ego_cmd: jnp.ndarray
    fol_pos: jnp.ndarray  # follower front bumper (always present; reacts to the ego)
    fol_v: jnp.ndarray
    fol_drv: Drivers
    tail_pos: jnp.ndarray  # rear of the last vehicle on the next segment, from its entry
    tail_v: jnp.ndarray
    jam_pos: jnp.ndarray
    jam_rate: jnp.ndarray
    # tracker model
    lead_idx: jnp.ndarray
    lead_age: jnp.ndarray  # steps since this leader's track appeared
    vl_hat: jnp.ndarray
    gap_prev: jnp.ndarray
    dgap: jnp.ndarray
    track_alpha: jnp.ndarray
    blind: jnp.ndarray  # steps until tracks exist after entering a segment
    noise: jnp.ndarray
    red_done: jnp.ndarray
    prev_light: jnp.ndarray
    snap_pos: jnp.ndarray  # [N] other cars as of the last world update (what the AV's sensors see, extrapolated)
    snap_v: jnp.ndarray
    snap_age: jnp.ndarray
    yellow_stop: jnp.ndarray  # the yellow-light decision, latched when the light turns yellow (no dilemma-zone dithering)
    # episode metrics
    m_dist: jnp.ndarray
    m_hard: jnp.ndarray
    m_coll: jnp.ndarray
    m_red: jnp.ndarray
    m_blocked: jnp.ndarray
    m_segs: jnp.ndarray
    m_jerk: jnp.ndarray
    m_stall_time: jnp.ndarray


def idm(v, v0, gap, vl, amax, b, thw, s0):
    s_star = s0 + jnp.maximum(0.0, v * thw + v * (v - vl) / (2.0 * jnp.sqrt(amax * b)))
    return amax * (1.0 - (v / v0) ** 4 - (s_star / jnp.maximum(gap, 0.1)) ** 2)


def idm_free(v, v0, amax):
    return amax * (1.0 - (v / v0) ** 4)


def light_state(seg: Seg, t):
    """0 red, 1 yellow, 2 green, with the world's timing (network.rs: 3 s yellow, 1 s all-red)."""
    phi = jnp.mod(t + seg.offset, seg.cycle)
    left = seg.green - phi
    on = jnp.where(left > 4.0, 2, jnp.where(left > 1.0, 1, 0))
    return jnp.where(seg.signal, jnp.where(phi < seg.green, on, 0), 2)


ROOM_MARGIN = 5.0  # extra room the driver wants before committing to a junction (don't block the box)
YELLOW_GO_MAX_T = 2.3  # go on yellow only if the line is reached within this time (3 s yellow - 0.5 s hand-over latency - margin)


def entry_room(tail_pos, v, margin=0.0):
    """sim.rs `entry_room_for`: body + 1 m + a 3 m/s^2 stopping distance behind the next segment's tail (+ margin)."""
    return tail_pos >= 4.5 + 1.0 + v * v / 6.0 + margin


# All randomness of a step is drawn in ONE uniform call and sliced: per-call PRNG overhead dominated the step otherwise.
U_DRV = 12  # uniforms per driver
U_SEG = 8
U_TRAFFIC = 3 * N + 5 + U_DRV * N
U_EXIT = 5
U_NEWSEG = U_SEG + U_TRAFFIC + U_EXIT + 1
U_RESET = 5 + U_DRV + U_NEWSEG
U_STEP = 8


def _lerp(u, lo, hi):
    return lo + (hi - lo) * u


def sample_drivers(u):
    """u: [U_DRV, ...] uniforms -> (Drivers, length) with the world's distributions (sim.rs make_veh)."""
    truck = u[0] < 0.08
    mixed = u[1] < 0.5  # half the drivers from the --drivers mix profiles
    cautious, aggressive = u[2] < 0.25, u[2] > 0.75
    pick = lambda c, n, a: jnp.where(cautious, c, jnp.where(aggressive, a, n))
    w = u[3:9]
    v0f = jnp.where(mixed, pick(_lerp(w[0], 0.82, 0.95), _lerp(w[0], 0.92, 1.08), _lerp(w[0], 1.05, 1.22)), _lerp(w[0], 0.88, 1.12))
    thw = jnp.where(mixed, pick(_lerp(w[1], 1.5, 2.2), _lerp(w[1], 1.1, 1.6), _lerp(w[1], 0.7, 1.1)), _lerp(w[1], 1.0, 1.8))
    amax = jnp.where(mixed, pick(_lerp(w[2], 1.0, 1.5), _lerp(w[2], 1.4, 2.0), _lerp(w[2], 2.0, 2.8)), _lerp(w[2], 1.2, 2.0))
    b = jnp.where(mixed, pick(_lerp(w[3], 1.4, 2.0), _lerp(w[3], 1.8, 2.4), _lerp(w[3], 2.3, 3.2)), _lerp(w[3], 1.6, 2.4))
    s0 = jnp.where(mixed, pick(_lerp(w[4], 2.3, 3.0), _lerp(w[4], 1.8, 2.4), _lerp(w[4], 1.2, 1.7)), 2.0)
    yrun = w[5] < jnp.where(mixed, pick(0.0, 0.10, 0.45), 0.0)
    v0f = jnp.where(truck, _lerp(u[9], 0.78, 0.9), v0f)
    amax = jnp.where(truck, _lerp(u[10], 0.6, 1.0), amax)
    length = jnp.where(truck, 11.0, _lerp(u[11], 4.1, 5.2))
    return Drivers(v0f, thw, amax, b, s0, yrun & ~truck), length


def sample_segment(u, cfg: EnvConfig):
    arterial = u[0] < 0.4
    vmax = jnp.where(arterial, 16.7, 11.1)
    L = jnp.where(u[1] < 0.85, _lerp(u[2], 120.0, 260.0), _lerp(u[2], 45.0, 120.0))
    cycle = jnp.where(u[3] < 0.8, 60.0, 70.0)
    green = cycle * jnp.where(arterial, 0.6, 0.4)
    return Seg(L=L, vmax=vmax, cycle=cycle, green=green, offset=70.0 * u[4], signal=u[5] >= cfg.p_unsignalised,
               arrives=u[6] < cfg.p_arrive, cap=jnp.where(u[7] < 0.6, 9.0, 5.0))


def sample_traffic(u, seg: Seg, cfg: EnvConfig):
    """Vehicles ahead on a fresh segment: a standing queue at the line or flowing traffic, never violating the ego's entry room."""
    drv, length = sample_drivers(u[3 * N + 5:].reshape(U_DRV, N))
    queue = u[0] < cfg.p_queue
    n_cars = jnp.floor(u[1] * (N + 1))
    v_flow = seg.vmax * drv.v0f * _lerp(u[5:5 + N], 0.5, 1.0)
    v = jnp.where(queue, 1.5 * u[5:5 + N], v_flow)
    gaps = jnp.where(queue, drv.s0 + 2.5 * u[5 + N:5 + 2 * N], drv.s0 + v * drv.thw + 40.0 * u[5 + N:5 + 2 * N])
    first = jnp.where(queue, seg.L - 0.3 - 3.0 * u[2], seg.L - 0.6 * u[2] * seg.L)
    # front of car i = first - sum_{j<i} (len_j + gap_j)
    step_back = jnp.concatenate([jnp.zeros(1), jnp.cumsum(length + gaps)[:-1]])
    pos = first - step_back
    alive = jnp.arange(N) < n_cars
    # incident: a stalled vehicle in the last slot; cars it overlaps are removed
    stall = u[3] < cfg.p_stall
    sp = _lerp(u[4], 40.0, jnp.maximum(seg.L - 5.0, 41.0))
    overlap = (pos > sp - 4.6 - 3.0) & (pos - length < sp + 3.0)
    alive = jnp.where(stall, alive & ~overlap, alive)
    stalled = jnp.zeros(N, bool).at[N - 1].set(stall)
    pos = pos.at[N - 1].set(jnp.where(stall, sp, pos[N - 1]))
    v = v.at[N - 1].set(jnp.where(stall, 0.0, v[N - 1]))
    length = length.at[N - 1].set(jnp.where(stall, 4.6, length[N - 1]))
    alive = alive.at[N - 1].set(jnp.where(stall, True, alive[N - 1]))
    return pos, v, length, alive, stalled, drv


def sample_exit(u, cfg: EnvConfig):
    cong = u[0] < cfg.p_congested_exit
    jam = jnp.where(cong, _lerp(u[1], -5.0, 40.0), 1e4)
    tail = jnp.where(cong, jnp.maximum(jam, 0.0) + 15.0 * u[2], _lerp(u[2], 15.0, 200.0))
    tv = jnp.where(cong, 1.5 * u[3], _lerp(u[3], 4.0, 13.0))
    return tail, tv, jam, _lerp(u[4], -0.6, 1.5)


class NewSeg(NamedTuple):
    seg: Seg
    pos: jnp.ndarray
    v: jnp.ndarray
    length: jnp.ndarray
    alive: jnp.ndarray
    stalled: jnp.ndarray
    drv: Drivers
    tail: jnp.ndarray
    tv: jnp.ndarray
    jam: jnp.ndarray
    rate: jnp.ndarray
    blind: jnp.ndarray


def sample_new_segment(u, cfg: EnvConfig) -> NewSeg:
    a, b, c = U_SEG, U_SEG + U_TRAFFIC, U_SEG + U_TRAFFIC + U_EXIT
    seg = sample_segment(u[:a], cfg)
    pos, v, length, alive, stalled, drv = sample_traffic(u[a:b], seg, cfg)
    tail, tv, jam, rate = sample_exit(u[b:c], cfg)
    blind = 2 + jnp.floor(4.0 * u[c]).astype(jnp.int32)
    return NewSeg(seg, pos, v, length, alive, stalled, drv, tail, tv, jam, rate, blind)


def admit(ns: NewSeg, ego_v):
    """Drop sampled vehicles that would violate the entry-room rule that admitted the ego at speed `ego_v`."""
    need = 4.5 + 1.0 + jnp.minimum(ego_v, 9.0) ** 2 / 6.0
    return ns.alive & (ns.pos - ns.length >= need + ENTRY_S - EGO_LEN)


class Spare(NamedTuple):
    """Pre-sampled randomness consumed by `step`: the next fresh segment and the next episode start.

    Sampling these costs ~40x more than a physics step, so the trainer refreshes them once per rollout (`make_spare`);
    an env that needs two fresh segments within one rollout reuses the spare (rare and harmless)."""

    newseg: NewSeg
    start: object


def reset(key, cfg: EnvConfig = EnvConfig(), u=None) -> State:
    if u is None:
        key, sub = jax.random.split(key)
        u = jax.random.uniform(sub, (U_RESET,))
    ego_v = 9.0 * u[0]
    ns = sample_new_segment(u[5 + U_DRV:], cfg)
    seg, pos, v, length, stalled, drv, tail, tv, jam, rate, blind = ns.seg, ns.pos, ns.v, ns.length, ns.stalled, ns.drv, ns.tail, ns.tv, ns.jam, ns.rate, ns.blind
    alive = admit(ns, ego_v)
    fol_drv, _ = sample_drivers(u[5:5 + U_DRV])
    z = jnp.zeros(())
    return State(
        key=key, t=1000.0 * u[1], seg_t=z, step=jnp.zeros((), jnp.int32), seg=seg,
        pos=pos, v=v, length=length, alive=alive, stalled=stalled, drv=drv,
        ego_s=jnp.asarray(ENTRY_S), ego_v=ego_v, ego_a=z, ego_cmd=z,
        fol_pos=ENTRY_S - EGO_LEN - _lerp(u[2], 3.0, 40.0), fol_v=ego_v, fol_drv=fol_drv,
        tail_pos=tail, tail_v=tv, jam_pos=jam, jam_rate=rate,
        lead_idx=jnp.asarray(-1), lead_age=jnp.zeros((), jnp.int32), vl_hat=z, gap_prev=jnp.asarray(SENSOR_RANGE), dgap=z,
        track_alpha=_lerp(u[3], 0.25, 0.6), blind=blind,
        noise=cfg.noise * _lerp(u[4], 0.5, 1.5), red_done=jnp.asarray(False),
        prev_light=jnp.asarray(2), yellow_stop=jnp.asarray(True),
        snap_pos=pos, snap_v=v, snap_age=jnp.floor(SNAP_EVERY * u[0]).astype(jnp.int32) % SNAP_EVERY,
        m_dist=z, m_hard=z, m_coll=z, m_red=z, m_blocked=z, m_segs=z, m_jerk=z, m_stall_time=z,
    )


def _leader_of(pos_query, pos, length, alive):
    """(gap, index) of the nearest alive vehicle whose front is ahead of `pos_query` (gap = its rear - pos_query)."""
    ahead = alive & (pos > pos_query + 1e-3)
    gap = jnp.where(ahead, pos - length - pos_query, jnp.inf)
    i = jnp.argmin(gap)
    return gap[i], i


def yellow_decision(s: State, light):
    """Stop for a yellow only if we could stop comfortably (3 m/s^2) when it turned yellow; the decision is then kept."""
    d = s.seg.L - s.ego_s
    onset = (light == 1) & (s.prev_light != 1)
    stop = (s.ego_v ** 2 / 6.0 < d + 1.0) | (d > YELLOW_GO_MAX_T * jnp.maximum(s.ego_v, 0.1))
    return jnp.where(onset, stop, s.yellow_stop)


def seen_pos(s: State):
    """Other cars as the AV perceives them: the last 2 Hz world snapshot moved on at constant velocity."""
    return s.snap_pos + s.snap_v * s.snap_age * DT


def stop_required(s: State, light):
    """Red, a yellow we decided to stop for, or no room (with margin) at the next segment."""
    room = entry_room(s.tail_pos, jnp.minimum(s.ego_v, 9.0), ROOM_MARGIN)
    must = (light == 0) | ((light == 1) & yellow_decision(s, light)) | ~room
    return must & ~s.seg.arrives, room


def observe(s: State, key=None):
    """The features the Rust agent can compute from its own state, the world snapshot and its confirmed tracks."""
    light = light_state(s.seg, s.t)
    must, room = stop_required(s, light)
    d = s.seg.L - s.ego_s
    sp = seen_pos(s)
    gap1, i1 = _leader_of(s.ego_s, sp, s.length, s.alive)
    seen1 = (gap1 < SENSOR_RANGE) & (s.blind <= 0) & (s.lead_age >= 3)
    gap1o = jnp.where(seen1, s.gap_prev, SENSOR_RANGE)
    vl = jnp.where(seen1, s.vl_hat, s.ego_v)
    # second vehicle ahead (the leader of the leader)
    pos2 = jnp.where(jnp.arange(N) == i1, -jnp.inf, sp)
    gap2, _ = _leader_of(s.ego_s, pos2, s.length, s.alive)
    seen2 = seen1 & (gap2 < SENSOR_RANGE)
    req_line = jnp.where(must, s.ego_v ** 2 / (2.0 * jnp.maximum(d - 1.0, 0.5)), 0.0)
    req_lead = jnp.where(seen1, jnp.maximum(s.ego_v - vl, 0.0) ** 2 / (2.0 * jnp.maximum(gap1o - 2.0, 0.5)), 0.0)
    o = jnp.stack([
        s.ego_v / V_SCALE, s.ego_a / 3.0, s.seg.vmax / V_SCALE, jnp.clip(d / 100.0, -0.1, 3.0),
        (light == 0).astype(jnp.float32), (light == 1).astype(jnp.float32), (light == 2).astype(jnp.float32),
        room.astype(jnp.float32), s.seg.arrives.astype(jnp.float32), must.astype(jnp.float32),
        seen1.astype(jnp.float32), jnp.clip(gap1o / 50.0, -0.1, 1.2), vl / V_SCALE, (s.ego_v - vl) / 10.0,
        jnp.clip(s.dgap / 10.0, -2.0, 2.0) * seen1,
        seen2.astype(jnp.float32), jnp.clip(jnp.where(seen2, gap2, SENSOR_RANGE) / 60.0, 0.0, 1.0),
        s.ego_cmd / 3.0, jnp.clip(req_line / 8.0, 0.0, 2.0), jnp.clip(req_lead / 8.0, 0.0, 2.0),
        jnp.clip(jnp.where(seen1, gap1o / jnp.maximum(s.ego_v, 1.0), 6.0) / 3.0, 0.0, 2.0),
        jnp.minimum(s.seg_t / 60.0, 2.0) * (s.ego_v < 0.5),
    ]).astype(jnp.float32)
    return o


def shield_bound(o):
    """Upper bound on the acceleration from the observation alone (so Rust computes the identical number):
    an IDM stop towards a point 1 m before the line when a stop is required, an IDM bound behind the leader with tight
    parameters (0.6 s, 2 m: it only binds when genuinely close), and the speed limit. Mirrors `learned.rs::shield_bound`."""
    v, vmax, d = o[0] * V_SCALE, o[2] * V_SCALE, o[3] * 100.0
    must, seen = o[9] > 0.5, o[10] > 0.5
    gap, vl = o[11] * 50.0, o[12] * V_SCALE
    v0 = jnp.maximum(vmax, 1.0)
    a_stop = jnp.where(must, idm(v, v0, d - 1.0, 0.0, 2.0, 3.0, 0.5, 1.0), 99.0)
    a_lead = jnp.where(seen, idm(v, v0, gap, jnp.maximum(vl, 0.0), 2.0, 3.0, 0.6, 2.0), 99.0)
    a_speed = 1.5 * (vmax - v)
    return jnp.minimum(jnp.minimum(a_stop, a_lead), jnp.maximum(a_speed, -2.0))


def action_to_accel(u):
    u = jnp.clip(u, -1.0, 1.0)
    return jnp.where(u >= 0.0, MAX_ACCEL * u, MAX_DECEL * u)


def make_spare(key, cfg: EnvConfig = EnvConfig()) -> Spare:
    k1, k2 = jax.random.split(key)
    u = jax.random.uniform(k1, (U_NEWSEG,))
    return Spare(sample_new_segment(u, cfg), reset(k2, cfg))


def step(s: State, u, cfg: EnvConfig = EnvConfig(), spare: Spare = None):
    """One 0.1 s step. `u` in [-1, 1] maps to [-8, 3] m/s^2. Returns (state, obs, reward, done, info).

    `spare` supplies the next fresh segment / episode start; without it they are sampled here (slow, fine for eval)."""
    key, sub = jax.random.split(s.key)
    if spare is None:
        key, ks = jax.random.split(key)
        spare = make_spare(ks, cfg)
    u_step = jax.random.uniform(sub, (U_STEP,), minval=1e-7, maxval=1.0)
    light = light_state(s.seg, s.t)
    a_pol = action_to_accel(u)
    a_cmd = jnp.maximum(jnp.minimum(a_pol, shield_bound(observe(s))), -MAX_DECEL) if cfg.shield else a_pol
    shield_cut = a_pol - a_cmd

    # ---- ego (vehicle.rs: first-order lag, no reversing)
    alpha = DT / (TAU + DT)
    ego_a = s.ego_a + alpha * (a_cmd - s.ego_a)
    ego_v = jnp.maximum(s.ego_v + ego_a * DT, 0.0)
    ego_a = jnp.where((ego_v == 0.0) & (ego_a < 0.0), 0.0, ego_a)
    ego_s = s.ego_s + 0.5 * (s.ego_v + ego_v) * DT

    # ---- world vehicles ahead (sim.rs compute_accels: IDM, stop line for the head vehicle, entry blocking)
    gaps_l = jnp.where(s.alive[None, :] & (s.pos[None, :] > s.pos[:, None] + 1e-3), s.pos[None, :] - s.length[None, :] - s.pos[:, None], jnp.inf)
    j = jnp.argmin(gaps_l, axis=1)
    has_lead = jnp.isfinite(gaps_l[jnp.arange(N), j])
    lead_gap, lead_v = gaps_l[jnp.arange(N), j], s.v[j]
    dist_end = s.seg.L - s.pos
    can_stop = s.v ** 2 / (2.0 * 4.5) <= dist_end - 0.5
    blocked = s.tail_pos < s.length + 1.0 + 1.0
    stop_line = (((light == 0) | ((light == 1) & ~s.drv.yrun)) & can_stop) | (blocked & can_stop)
    v0 = jnp.maximum(s.seg.vmax * s.drv.v0f, 1.0)
    head_gap = jnp.where(stop_line, dist_end - 0.3, dist_end + s.tail_pos)
    head_v = jnp.where(stop_line, 0.0, s.tail_v)
    g = jnp.where(has_lead, lead_gap, jnp.where(s.seg.arrives, jnp.inf, head_gap))
    gv = jnp.where(has_lead, lead_v, head_v)
    acc = jnp.where(jnp.isfinite(g), idm(s.v, v0, g, gv, s.drv.amax, s.drv.b, s.drv.thw, s.drv.s0), idm_free(s.v, v0, s.drv.amax))
    acc = jnp.where(s.stalled, 0.0, jnp.clip(acc, -9.0, s.drv.amax))
    nv = jnp.maximum(s.v + acc * DT, 0.0)
    npos = s.pos + 0.5 * (s.v + nv) * DT
    # no overlap among world vehicles (sim.rs integrate)
    rear_lead = jnp.where(has_lead, s.pos[j] - s.length[j], jnp.inf)
    clamp = npos > rear_lead - 0.1
    npos = jnp.where(clamp, jnp.maximum(rear_lead - 0.1, s.pos), npos)
    nv = jnp.where(clamp, jnp.minimum(nv, lead_v), nv)
    nv = jnp.where(s.stalled, 0.0, nv)
    npos = jnp.where(s.stalled, s.pos, npos)
    # vehicles crossing the line join the next segment (and become its tail)
    crossed = s.alive & (npos > s.seg.L) & ~s.stalled
    alive = s.alive & ~crossed
    new_tail = jnp.where(crossed, npos - s.seg.L - s.length, jnp.inf).min()
    tail_from_cross = jnp.isfinite(new_tail)

    # ---- the next segment's tail vehicle (the junction exit), plus random cross traffic turning in
    tail_acc = idm(s.tail_v, 12.0, s.jam_pos - s.tail_pos, 0.0, 1.5, 2.0, 1.4, 2.0)
    tail_v = jnp.maximum(s.tail_v + jnp.clip(tail_acc, -9.0, 1.5) * DT, 0.0)
    tail_pos = s.tail_pos + tail_v * DT
    jam_pos = s.jam_pos + s.jam_rate * DT
    # cross traffic never fills the exit once the ego is within its comfortable stopping distance (it could not react)
    ins = (u_step[0] < cfg.insert_rate * DT) & (s.seg.L - s.ego_s > s.ego_v ** 2 / 6.0 + 3.0)
    ins_pos = 12.0 * u_step[1]
    ins_v = _lerp(u_step[2], 1.0, 8.0)
    take_cross = tail_from_cross & (new_tail < tail_pos)
    tail_v = jnp.where(take_cross, s.v[jnp.argmax(crossed)], tail_v)
    tail_pos = jnp.where(take_cross, new_tail, tail_pos)
    take_ins = ins & (ins_pos < tail_pos)
    tail_v = jnp.where(take_ins, ins_v, tail_v)
    tail_pos = jnp.where(take_ins, ins_pos, tail_pos)
    far = tail_pos > 220.0  # the old tail drove away: a fresh one somewhere behind it
    tail_pos = jnp.where(far, 60.0 + 100.0 * u_step[3], tail_pos)

    # ---- follower (an IDM driver behind the ego)
    fg = ego_s - EGO_LEN - s.fol_pos
    fa = idm(s.fol_v, jnp.maximum(s.seg.vmax * s.fol_drv.v0f, 1.0), fg, ego_v, s.fol_drv.amax, s.fol_drv.b, s.fol_drv.thw, s.fol_drv.s0)
    fol_v = jnp.maximum(s.fol_v + jnp.clip(fa, -9.0, s.fol_drv.amax) * DT, 0.0)
    fol_pos = s.fol_pos + 0.5 * (s.fol_v + fol_v) * DT

    # ---- safety events
    gap_now, li = _leader_of(ego_s, npos, s.length, alive)
    coll_front = gap_now < 0.0
    coll_rear = (ego_s - EGO_LEN - fol_pos) < 0.0
    fol_pos = jnp.where(coll_rear, ego_s - EGO_LEN - 0.1, fol_pos)  # world integrate never lets it overlap for long
    collision = coll_front
    d_line = s.seg.L - ego_s
    room = entry_room(tail_pos, jnp.minimum(ego_v, 9.0))
    at_line = d_line <= 0.2
    allowed = (light != 0) & room
    # the world only hands the AV over with permission; otherwise it stays (possibly past the line) until it may go
    crossing = (s.seg.L - s.ego_s >= 0.0) & (d_line < 0.0)
    red_run = (d_line < 0.0) & (light == 0) & ~s.seg.arrives & ~s.red_done
    blocked_run = crossing & ~room & ~s.seg.arrives & (light != 0)
    past = (d_line < 0.0) & ~allowed & ~s.seg.arrives
    deep = (d_line < -DEEP) & ~s.seg.arrives
    collision = coll_front | deep
    handoff = at_line & allowed & ~s.seg.arrives
    finish = at_line & s.seg.arrives
    forced = s.seg_t > SEG_TIMEOUT
    new_seg = handoff | finish | forced

    # ---- reward
    hard = (ego_a < -4.0) & (s.ego_a >= -4.0) & (ego_v > 3.0)
    jerk = (ego_a - s.ego_a) / DT
    thw = gap_now / jnp.maximum(ego_v, 1.0)
    r = (W_PROGRESS * (ego_s - s.ego_s)
         - W_COMFORT * ego_a ** 2
         - W_HARD * jnp.maximum(-ego_a - 3.0, 0.0) * (ego_v > 2.0)
         - W_JERK * jerk ** 2
         - W_HEADWAY * jnp.maximum(0.8 - thw, 0.0) * (ego_v > 1.0)
         - W_GAP * jnp.maximum(2.5 - gap_now, 0.0)
         - W_SPEED * jnp.maximum(ego_v - s.seg.vmax, 0.0)
         - W_PAST * past
         - W_SHIELD * shield_cut
         - W_IDLE * (ego_v < 0.5) * ~stop_required(s, light)[0] * (gap_now > 8.0)
         + R_COLLISION * collision + R_RED * red_run + R_BLOCKED * blocked_run
         + 0.5 * R_COLLISION * coll_rear)

    # ---- 2 Hz world snapshot, then the tracker model for the leader seen next step
    take = (s.step + 1) % SNAP_EVERY == 0
    snap_pos = jnp.where(take, npos, s.snap_pos)
    snap_v = jnp.where(take, nv, s.snap_v)
    snap_age = jnp.where(take, 0, s.snap_age + 1)
    gap_obs, li_o = _leader_of(ego_s, snap_pos + snap_v * snap_age * DT, s.length, alive)
    new_track = (li_o != s.lead_idx) | ~jnp.isfinite(gap_obs)
    r_ = jnp.sqrt(-2.0 * jnp.log(u_step[4:6]))  # Box-Muller
    nz = r_ * jnp.cos(2.0 * jnp.pi * u_step[6:8])
    gap_meas = gap_obs + s.noise * (0.15 + 0.01 * jnp.minimum(gap_obs, 100.0)) * nz[0]
    vl_meas = snap_v[li_o] + 0.3 * s.noise * nz[1]
    vl_hat = jnp.where(new_track, 0.0, s.vl_hat + s.track_alpha * (vl_meas - s.vl_hat))  # tracks start at vx = 0
    dgap = jnp.where(new_track, 0.0, 0.5 * s.dgap + 0.5 * (gap_meas - s.gap_prev) / DT)
    lead_age = jnp.where(new_track, 0, s.lead_age + 1)

    m_dist = s.m_dist + (ego_s - s.ego_s)
    st = s._replace(
        key=key, t=s.t + DT, seg_t=s.seg_t + DT, step=s.step + 1, pos=npos, v=nv, alive=alive,
        ego_s=ego_s, ego_v=ego_v, ego_a=ego_a, ego_cmd=a_cmd, fol_pos=fol_pos, fol_v=fol_v,
        tail_pos=tail_pos, tail_v=tail_v, jam_pos=jam_pos,
        lead_idx=jnp.where(jnp.isfinite(gap_obs), li_o, -1), lead_age=lead_age, vl_hat=vl_hat,
        gap_prev=jnp.where(jnp.isfinite(gap_obs), gap_meas, SENSOR_RANGE), snap_pos=snap_pos, snap_v=snap_v, snap_age=snap_age, dgap=dgap, blind=s.blind - 1,
        red_done=s.red_done | red_run, prev_light=light, yellow_stop=yellow_decision(s, light),
        m_dist=m_dist, m_hard=s.m_hard + hard, m_coll=s.m_coll + collision + coll_rear, m_red=s.m_red + red_run,
        m_blocked=s.m_blocked + blocked_run, m_segs=s.m_segs + new_seg, m_jerk=s.m_jerk + jerk ** 2 * DT,
        m_stall_time=s.m_stall_time + DT * (ego_v < 0.3),
    )

    # ---- hand-over to a fresh segment (world: speed capped, tracks restart)
    ego_v_new = jnp.where(handoff, jnp.minimum(ego_v, s.seg.cap), ego_v)
    ns = spare.newseg
    seg, pos2, v2, len2, stalled2, drv2, tail2, tv2, jam2, rate2, blind2 = ns.seg, ns.pos, ns.v, ns.length, ns.stalled, ns.drv, ns.tail, ns.tv, ns.jam, ns.rate, ns.blind
    alive2 = admit(ns, ego_v_new)
    fol_shift = jnp.maximum(fol_pos - s.seg.L, ENTRY_S - EGO_LEN - 60.0)
    st_new = st._replace(
        seg=seg, seg_t=jnp.zeros(()), pos=pos2, v=v2, length=len2, alive=alive2, stalled=stalled2, drv=drv2,
        ego_s=jnp.asarray(ENTRY_S), ego_v=ego_v_new, fol_pos=jnp.minimum(fol_shift, ENTRY_S - EGO_LEN - 2.0),
        tail_pos=tail2, tail_v=tv2, jam_pos=jam2, jam_rate=rate2, lead_idx=jnp.asarray(-1), lead_age=jnp.zeros((), jnp.int32),
        vl_hat=jnp.zeros(()), gap_prev=jnp.asarray(SENSOR_RANGE), dgap=jnp.zeros(()), blind=blind2, red_done=jnp.asarray(False), prev_light=jnp.asarray(2),
        snap_pos=pos2, snap_v=v2, snap_age=jnp.zeros((), jnp.int32),
    )
    st = jax.tree.map(lambda a, b: jnp.where(new_seg, a, b), st_new, st)

    done = collision | (st.step >= cfg.episode_steps)
    info = dict(dist=st.m_dist, hard=st.m_hard, coll=st.m_coll, red=st.m_red, blocked=st.m_blocked, segs=st.m_segs,
                jerk=st.m_jerk, stopped=st.m_stall_time, steps=st.step.astype(jnp.float32), done=done,
                # per-step increments (for logging without waiting for episodes to end)
                s_dist=ego_s - s.ego_s, s_hard=hard.astype(jnp.float32), s_coll=(collision | coll_rear).astype(jnp.float32),
                s_red=red_run.astype(jnp.float32), s_blocked=blocked_run.astype(jnp.float32), s_segs=new_seg.astype(jnp.float32),
                s_shield=(shield_cut > 0.5).astype(jnp.float32))
    st_reset = spare.start._replace(key=key)
    st = jax.tree.map(lambda a, b: jnp.where(done, a, b), st_reset, st)
    return st, observe(st), r, done, info
