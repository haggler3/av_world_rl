# Learned driver: version history


Each version was trained, exported and driven through the city; every fix came from the city's collision / red-light logs.

| version | change | city result that drove the next change |
|---|---|---|
| v1 | first reward | 64 trips 1.6 min faster than ordinary drivers, 0.1 hard brakes/trip, but 1 collision, 5 red runs: it **sped** (19.6 m/s on 16.7 m/s roads, outside the LQR's gain table) and dithered on yellow |
| v2 | speed-limit penalty, yellow decision latched at onset, blocked junction entry −15 | 64 trips: 0 collisions, 0 red runs; 30 % fleet: 18 collisions, 9 red runs: AVs crept past the line into full junctions (the env teleported them through; the world does not) |
| v3 | no hand-over without permission (red / full exit), penalty for waiting past the line, 2 Hz snapshots in the env | most AV-side contacts were now at walking pace: it parked within 0.2 m of the car ahead |
| v4 | standstill-gap penalty; hand-over checked only every 0.5 s like the world | worse: many AVs stuck past the line (reverted the 0.5 s check) |
| v5 | 5 m "don't block the box" margin, go on yellow only if the line is < 2.3 s away | in the env still worse than the rule driver at stop lines: PPO plateaued with residual errors |
| **v6** | **classical safety envelope around the policy, in training and deployment** | results above; one 64-trip run had a trip stuck > 30 min in spill-back → deploy margin 2 m |

Still wrong or missing:
* The envelope is part of the system: the learned policy alone is not safe enough (table above).
* The shipped policy was trained with a 5 m junction-room margin and is deployed with 2 m (5 m lost its place in
  spill-back queues: 30+ min vs 26.6 for the planner on one trip); 0 m was faster still, but blocks junctions more.
* 1.2 collisions per 1000 km remain in the env (the rule driver: 0), and a few red-light entries per city run at 30 % share.
* Trained only against IDM drivers: other AVs (which brake differently and are seen through 2 Hz snapshots) are the hardest case at high shares.
* Longitudinal only, one lane, no pedestrians: it covers exactly what the city asks of the AV, not `small_av`'s pedestrian or cut-in scenarios (the planner still handles those there).
* One training seed per version; the city comparisons are paired but 64 trips cannot resolve differences much below 1 min.

