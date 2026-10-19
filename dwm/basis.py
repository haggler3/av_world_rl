"""Fixed linear decoders between a small latent the diffusion model generates and the 30 x 6 future.

All bases work on the physical residual (future minus the kinematic prediction, in m, m/s, ...) and are folded into the
model's normalised space, so "smooth" means smooth in the real world:

    steps        identity (30 values per channel)
    knots        `nb` control points per channel, linear interpolation
    dct          `nb` cosine coefficients per channel (smooth everywhere, no kinks)
    leader_phys  the leader's acceleration as `nb` control points, integrated once into the leader-speed residual and
                 twice into the gap residual (gap also loses whatever the AV's own speed residual adds). Small, down-weighted
                 free terms absorb tracking noise and leader switches; because they are down-weighted, the least-squares
                 encoding prefers the physical explanation.
"""
import numpy as np

from .data import DT, F, NC



def knots_matrix(n):
    pos = np.linspace(0, F - 1, n)
    U = np.zeros((F, n))
    for f in range(F):
        j = min(np.searchsorted(pos, f, side="right") - 1, n - 2)
        w = (f - pos[j]) / (pos[j + 1] - pos[j])
        U[f, j], U[f, j + 1] = 1 - w, w
    return U


def dct_matrix(n):
    f = np.arange(F)[:, None] + 0.5
    k = np.arange(n)[None]
    B = np.cos(np.pi * f * k / F)
    return B / np.linalg.norm(B, axis=0, keepdims=True) * np.sqrt(F / n)


def physical_decoder(basis, nb, free_weight=0.2):
    """[F*C, Z] matrix from latent to the physical residual (row index = f * C + c)."""
    if basis == "steps":
        return np.eye(F * NC)
    per = knots_matrix(nb) if basis in ("knots", "leader_phys") else dct_matrix(nb)
    if basis in ("knots", "dct"):
        D = np.zeros((F * NC, nb * NC))
        for c in range(NC):
            D[c::NC, c * nb:(c + 1) * nb] = per
        return D
    # leader_phys: latent blocks [speed, d_line, seen, must, leader accel, gap free, lead_v free]
    I1 = np.tril(np.ones((F, F))) * DT  # integrate once (residual is zero before the first step)
    blocks = 7
    D = np.zeros((F * NC, nb * blocks))
    col = lambda b: slice(b * nb, (b + 1) * nb)  # noqa: E731
    for b, c in enumerate((0, 1, 2, 5)):
        D[c::NC, col(b)] = per
    D[4::NC, col(4)] = I1 @ per  # leader speed residual = integral of leader acceleration
    D[3::NC, col(4)] = I1 @ I1 @ per  # gap residual: leader displacement ...
    D[3::NC, col(0)] -= I1 @ per  # ... minus the AV's extra displacement
    D[3::NC, col(5)] = free_weight * per
    D[4::NC, col(6)] = free_weight * per
    return D


def codec(basis, nb, sd, scale, ridge=None):
    """(Dec, Enc) in the model's normalised space: r_norm = Dec @ z, z = Enc @ r_norm.

    The encoder is ridge-regularised least squares (relative strength `ridge`): the integrating leader decoder is badly
    conditioned (condition number ~3e6), and a plain pseudo-inverse explains the data with huge cancelling coefficients
    that turn small sampling errors into tens of metres of gap error."""
    D = physical_decoder(basis, nb)
    to_norm = 1.0 / (sd * scale[None]).reshape(-1)  # physical -> normalised residual, per (f, c)
    Dn = (D * to_norm[:, None]).astype(np.float64)
    if ridge is None:
        ridge = 1e-4 if basis == "leader_phys" else 0.0  # floor 0.30 m, 5x less sensitive than knots (explore results)
    G = Dn.T @ Dn
    Enc = np.linalg.solve(G + ridge * np.trace(G) / len(G) * np.eye(len(G)), Dn.T) if ridge > 0 else np.linalg.pinv(Dn)
    return Dn.astype(np.float32), Enc.astype(np.float32)
