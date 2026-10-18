| benchmark driver | model | skill_vs_kinematic | crps_speed_3s | crps_gap_1s | crps_gap_3s | crps_lead_v_3s | roughness_vs_real |
|---|---|---|---|---|---|---|---|
| planner | constant velocity | 15.923 | 2.17 | 0.458 | 2.306 | 0.877 | 0.0 |
| planner | kinematic | 1.0 | 0.08 | 0.355 | 1.697 | 0.877 | 0.332 |
| planner | diffusion · 30 steps (full data, 7.5 min) | 2.83 | 0.124 | 1.98 | 5.408 | 1.037 | 79.806 |
| planner | diffusion · 10 control points | 1.882 | 0.108 | 0.999 | 3.737 | 0.877 | 9.256 |
| planner | diffusion · DCT basis | 1.854 | 0.108 | 1.143 | 3.083 | 0.864 | 6.868 |
| planner | diffusion · leader physics decoder | 1.06 | 0.099 | 0.303 | 1.453 | 0.75 | 0.404 |
| planner | flow matching · DCT basis | 2.136 | 0.124 | 1.243 | 3.347 | 0.839 | 9.453 |
| learned | constant velocity | 26.59 | 1.245 | 0.323 | 1.691 | 1.048 | 0.0 |
| learned | kinematic | 1.0 | 0.035 | 0.391 | 1.976 | 1.048 | 0.217 |
| learned | diffusion · 30 steps (full data, 7.5 min) | 0.316 | 0.01 | 0.129 | 0.341 | 0.197 | 29.429 |
| learned | diffusion · 10 control points | 0.419 | 0.013 | 0.196 | 0.504 | 0.248 | 4.968 |
| learned | diffusion · DCT basis | 0.434 | 0.011 | 0.233 | 0.467 | 0.236 | 2.411 |
| learned | diffusion · leader physics decoder | 0.442 | 0.014 | 0.201 | 0.58 | 0.277 | 0.221 |
| learned | flow matching · DCT basis | 0.375 | 0.009 | 0.228 | 0.438 | 0.221 | 3.305 |
