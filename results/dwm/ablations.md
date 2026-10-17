| variant | skill_vs_kinematic | crps_speed_3s | crps_gap_1s | crps_gap_3s | crps_lead_v_3s | coverage90_gap_3s | acc_lead_seen_3s | acc_must_stop_3s | steps | params |
|---|---|---|---|---|---|---|---|---|---|---|
| constant velocity | 27.918 | 1.181 | 0.307 | 1.551 | 0.873 |  | 0.939 | 0.825 | 0 | 0 |
| kinematic | 1.0 | 0.039 | 0.338 | 1.624 | 0.873 |  | 0.939 | 0.825 | 0 | 0 |
| base: residual target, MLP 256x3, v-pred, 10 DDIM steps | 0.546 | 0.015 | 0.269 | 0.56 | 0.259 | 0.967 | 0.989 | 0.912 | 5500 | 1110964 |
| target = change from current value (no physics prior) | 5.277 | 0.221 | 0.608 | 1.152 | 0.342 | 0.988 | 0.986 | 0.893 | 2500 | 1110964 |
| deterministic regressor, same network | 0.613 | 0.018 | 0.324 | 0.756 | 0.351 |  | 0.985 | 0.907 | 2500 | 1110964 |
| predict eps | 9.541 | 0.161 | 6.62 | 13.773 | 3.561 | 1.0 | 0.914 | 0.733 | 3000 | 1110964 |
| predict x0 | 0.55 | 0.015 | 0.279 | 0.584 | 0.265 | 0.848 | 0.987 | 0.897 | 3800 | 1110964 |
| 1-D temporal conv denoiser | 3.49 | 0.076 | 2.125 | 3.349 | 0.835 | 0.999 | 0.939 | 0.825 | 800 | 454726 |
| bigger MLP 512x4 | 1.243 | 0.039 | 0.664 | 1.332 | 0.513 | 0.988 | 0.959 | 0.855 | 1200 | 5107380 |
| no future actions in the condition | 1.033 | 0.04 | 0.46 | 1.2 | 0.436 | 0.942 | 0.968 | 0.843 | 3200 | 1103284 |
| history = current step only | 0.729 | 0.021 | 0.364 | 0.783 | 0.285 | 0.983 | 0.986 | 0.895 | 3700 | 1060276 |
| base, 2 DDIM steps at sampling | 0.56 | 0.015 | 0.274 | 0.528 | 0.256 | 0.912 | 0.989 | 0.911 | 5500 | 1110964 |
| base, 5 DDIM steps at sampling | 0.539 | 0.015 | 0.264 | 0.539 | 0.253 | 0.949 | 0.989 | 0.909 | 5500 | 1110964 |
| base, 20 DDIM steps at sampling | 0.557 | 0.015 | 0.277 | 0.58 | 0.263 | 0.971 | 0.989 | 0.911 | 5500 | 1110964 |
