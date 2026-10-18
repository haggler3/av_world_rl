| variant | steps | latent | skill_vs_kinematic | crps_gap_1s | crps_gap_3s | crps_lead_v_3s | coverage90_gap_3s | roughness_vs_real | projection_floor_gap_3s_m |
|---|---|---|---|---|---|---|---|---|---|
| control points (10/channel) | 4300 | 60 | 0.562 | 0.208 | 0.536 | 0.268 | 0.805 | 4.89 | 0.051 |
| DCT (8 coefficients/channel) | 4600 | 48 | 0.5 | 0.215 | 0.453 | 0.24 | 0.715 | 2.25 | 0.3 |
| leader physics decoder (accel knots, integrated) | 4300 | 70 | 0.528 | 0.186 | 0.524 | 0.282 | 0.523 | 0.21 | 0.336 |
| leader physics + classifier-free guidance, guidance 1.0 | 4200 | 70 | 0.543 | 0.191 | 0.561 | 0.302 | 0.541 | 0.21 | 0.336 |
| leader physics + classifier-free guidance, guidance 1.5 | 4200 | 70 | 0.545 | 0.191 | 0.521 | 0.296 | 0.559 | 0.21 | 0.336 |
| DCT + observation noise | 4700 | 48 | 0.496 | 0.215 | 0.45 | 0.237 | 0.713 | 2.76 | 0.3 |
| leader physics + observation noise | 4000 | 70 | 0.536 | 0.185 | 0.535 | 0.291 | 0.513 | 5.59 | 0.336 |
