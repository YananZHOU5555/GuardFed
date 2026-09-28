# CelebA: non-IID

COMPLETED VALIDATION COHORT - 10 shared seeds (91001-91010).

Snapshot: 2026-09-28T10:20:44.788275+00:00; 700/700 accepted records.

| Category | Method | Metric | Benign | F-Flip | FedSA | S-DFA | Sp-DFA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla FL | FedAvg | ACC (%) ↑ | 86.85 ± 5.45 | 89.51 ± 1.82 | 74.76 ± 13.38 | 74.51 ± 13.37 | 85.48 ± 2.58 |
|  |  | AEOD ↓ | 0.0428 ± 0.0137 | 0.0492 ± 0.0045 | 0.0644 ± 0.1097 | 0.0522 ± 0.0913 | 0.0267 ± 0.0157 |
|  |  | ASPD ↓ | 0.0958 ± 0.0353 | 0.1161 ± 0.0143 | 0.0614 ± 0.0893 | 0.0591 ± 0.0756 | 0.0741 ± 0.0247 |
| Fairness-aware | FairFed* | ACC (%) ↑ | 89.59 ± 0.97 | 89.83 ± 0.92 | 71.45 ± 11.85 | 67.06 ± 11.91 | 86.18 ± 2.41 |
|  |  | AEOD ↓ | 0.0388 ± 0.0057 | 0.0462 ± 0.0081 | 0.0759 ± 0.0860 | 0.1307 ± 0.1184 | 0.0328 ± 0.0133 |
|  |  | ASPD ↓ | 0.1092 ± 0.0071 | 0.1163 ± 0.0071 | 0.0730 ± 0.0885 | 0.1171 ± 0.1067 | 0.0829 ± 0.0204 |
| Robust FL | Median | ACC (%) ↑ | 86.22 ± 1.31 | 86.51 ± 1.24 | 84.47 ± 1.30 | 84.48 ± 1.35 | 85.47 ± 1.26 |
|  |  | AEOD ↓ | 0.0518 ± 0.0068 | 0.0594 ± 0.0084 | 0.0425 ± 0.0118 | 0.0461 ± 0.0142 | 0.0502 ± 0.0111 |
|  |  | ASPD ↓ | 0.0963 ± 0.0096 | 0.1067 ± 0.0097 | 0.0799 ± 0.0113 | 0.0839 ± 0.0124 | 0.0929 ± 0.0086 |
| Robust FL | FLTrust | ACC (%) ↑ | 89.68 ± 0.60 | 89.83 ± 0.69 | 89.48 ± 0.70 | 89.48 ± 0.70 | 89.64 ± 0.71 |
|  |  | AEOD ↓ | 0.0445 ± 0.0049 | 0.0493 ± 0.0054 | 0.0441 ± 0.0063 | 0.0441 ± 0.0063 | 0.0475 ± 0.0052 |
|  |  | ASPD ↓ | 0.1132 ± 0.0060 | 0.1197 ± 0.0044 | 0.1120 ± 0.0041 | 0.1120 ± 0.0041 | 0.1153 ± 0.0053 |
| Robust + fair | FairGuard* | ACC (%) ↑ | 89.39 ± 1.11 | 89.84 ± 0.76 | 59.95 ± 16.51 | 61.29 ± 15.59 | 78.37 ± 14.94 |
|  |  | AEOD ↓ | 0.0314 ± 0.0054 | 0.0394 ± 0.0103 | 0.0066 ± 0.0094 | 0.0108 ± 0.0161 | 0.0209 ± 0.0104 |
|  |  | ASPD ↓ | 0.1026 ± 0.0086 | 0.1109 ± 0.0090 | 0.0169 ± 0.0251 | 0.0206 ± 0.0317 | 0.0552 ± 0.0238 |
| Robust + fair | FLTrust+FairGuard* | ACC (%) ↑ | 89.69 ± 1.68 | 90.07 ± 1.34 | 81.90 ± 11.66 | 80.44 ± 12.06 | 79.69 ± 13.94 |
|  |  | AEOD ↓ | 0.0357 ± 0.0055 | 0.0426 ± 0.0062 | 0.0400 ± 0.0214 | 0.0359 ± 0.0262 | 0.0408 ± 0.0124 |
|  |  | ASPD ↓ | 0.1085 ± 0.0102 | 0.1144 ± 0.0088 | 0.0851 ± 0.0350 | 0.0763 ± 0.0174 | 0.0892 ± 0.0276 |
| Ours | GuardFed-AD2+ | ACC (%) ↑ | 88.59 ± 1.17 | 88.44 ± 0.74 | 88.41 ± 0.92 | 88.22 ± 0.78 | 88.34 ± 0.96 |
|  |  | AEOD ↓ | 0.0070 ± 0.0042 | 0.0100 ± 0.0070 | 0.0078 ± 0.0056 | 0.0126 ± 0.0093 | 0.0087 ± 0.0066 |
|  |  | ASPD ↓ | 0.0651 ± 0.0090 | 0.0589 ± 0.0092 | 0.0616 ± 0.0141 | 0.0584 ± 0.0189 | 0.0596 ± 0.0146 |

All 70 method/distribution/scenario cells use the same 10 seeds. Seed subsets follow the frozen protocol; all cells within each table use identical seeds.  
Round 70; validation only (19,867 images). Mean ± sample SD (ddof = 1); ACC in %, gaps on [0, 1].  
AEOD denotes the implemented absolute TPR gap. No significance claim follows from mean ranks alone.  
* Project adaptations. GuardFed includes training-root group calibration; baseline outputs are uncalibrated.  
Recipes were selected on non-IID Benign/S-DFA, including seed 91001; this is coverage transfer, not IID-specific tuning.  
This table uses 336 cu128 and 14 cu130 records; the source cohort contains 14 cu130 records. No 70-round equivalence claim.  
Stage A covers seven implementations; ten further manuscript baselines and mechanism controls remain outstanding.  
