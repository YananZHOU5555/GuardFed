# AD2 vs AD2+ Additive Sweep

10% clean server data, 70 rounds. Fairness ranking uses valid-ACC rule as main table.

| Method | fairness metric | risk weight | violation weight | Fair first/second | Fair first | ACC first/second | Avg ACC | Avg AEOD | Avg ASPD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| GuardFed-AD2 | aeod_aspd | 0.75 | 0.2 | 13/16 | 7/16 | 0/8 | 73.12 | 0.075 | 0.050 |
| GuardFed-AD2+ | aeod_aspd | 0.6 | 0.1 | 9/16 | 2/16 | 0/8 | 73.13 | 0.075 | 0.046 |
| GuardFed-AD2+ | aeod_aspd | 0.9 | 0.25 | 9/16 | 3/16 | 0/8 | 73.15 | 0.075 | 0.059 |
| GuardFed-AD2+ | max | 0.8 | 0.2 | 7/16 | 2/16 | 0/8 | 73.37 | 0.080 | 0.067 |

Conclusion: fixed additive AD2+ did not beat fairness-first AD2 in this sweep. The current best main-table variant remains GuardFed-AD2 with risk=0.75, violation=0.2, root-norm scaling, original calibration objective.