# CelebA [raw]: IID

COMPLETED VALIDATION COHORT - 10 shared seeds (91001-91010).

Snapshot: 2026-10-09T15:46:28.401491+00:00; 900/900 accepted records.

| Category | Method | Metric | Benign | F-Flip | FedSA | S-DFA | Sp-DFA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla FL | FedAvg | ACC (%) ↑ | 84.72 ± 11.74 | 89.57 ± 0.78 | 68.19 ± 5.77 | 66.01 ± 3.57 | 84.37 ± 2.38 |
|  |  | AEOD ↓ | 0.0384 ± 0.0176 | 0.0519 ± 0.0096 | 0.1004 ± 0.0410 | 0.0950 ± 0.0477 | 0.0280 ± 0.0076 |
|  |  | ASPD ↓ | 0.0937 ± 0.0366 | 0.1212 ± 0.0064 | 0.0993 ± 0.0380 | 0.0807 ± 0.0413 | 0.0693 ± 0.0161 |
| Fairness-aware | FairFed* | ACC (%) ↑ | 89.47 ± 0.85 | 89.79 ± 0.76 | 69.47 ± 4.65 | 62.59 ± 5.11 | 85.55 ± 2.31 |
|  |  | AEOD ↓ | 0.0414 ± 0.0076 | 0.0489 ± 0.0042 | 0.1158 ± 0.0635 | 0.1594 ± 0.0822 | 0.0288 ± 0.0081 |
|  |  | ASPD ↓ | 0.1102 ± 0.0077 | 0.1188 ± 0.0043 | 0.1120 ± 0.0554 | 0.1461 ± 0.0668 | 0.0781 ± 0.0116 |
| Robust FL | Median | ACC (%) ↑ | 86.28 ± 1.31 | 86.52 ± 1.35 | 84.66 ± 1.47 | 85.00 ± 1.38 | 85.54 ± 1.50 |
|  |  | AEOD ↓ | 0.0510 ± 0.0081 | 0.0610 ± 0.0087 | 0.0438 ± 0.0135 | 0.0484 ± 0.0090 | 0.0534 ± 0.0096 |
|  |  | ASPD ↓ | 0.0980 ± 0.0117 | 0.1062 ± 0.0111 | 0.0825 ± 0.0142 | 0.0852 ± 0.0127 | 0.0965 ± 0.0124 |
| Robust FL | FLTrust | ACC (%) ↑ | 89.43 ± 0.75 | 89.67 ± 0.82 | 89.68 ± 0.74 | 89.68 ± 0.74 | 89.72 ± 0.64 |
|  |  | AEOD ↓ | 0.0432 ± 0.0075 | 0.0518 ± 0.0072 | 0.0422 ± 0.0069 | 0.0422 ± 0.0069 | 0.0496 ± 0.0061 |
|  |  | ASPD ↓ | 0.1111 ± 0.0062 | 0.1207 ± 0.0064 | 0.1119 ± 0.0071 | 0.1119 ± 0.0071 | 0.1177 ± 0.0057 |
| Adaptive FL | FedAA-DDPG* | ACC (%) ↑ | 88.35 ± 1.33 | 89.06 ± 0.92 | 88.80 ± 1.06 | 88.82 ± 1.21 | 88.60 ± 1.12 |
|  |  | AEOD ↓ | 0.0455 ± 0.0116 | 0.0442 ± 0.0137 | 0.0423 ± 0.0116 | 0.0385 ± 0.0131 | 0.0493 ± 0.0089 |
|  |  | ASPD ↓ | 0.1083 ± 0.0100 | 0.1097 ± 0.0143 | 0.1071 ± 0.0115 | 0.1042 ± 0.0152 | 0.1136 ± 0.0095 |
| Robust FL | LASA* | ACC (%) ↑ | 89.39 ± 1.42 | 89.46 ± 0.79 | 85.75 ± 1.71 | 85.96 ± 1.62 | 88.20 ± 1.40 |
|  |  | AEOD ↓ | 0.0377 ± 0.0049 | 0.0493 ± 0.0066 | 0.0330 ± 0.0109 | 0.0366 ± 0.0105 | 0.0450 ± 0.0073 |
|  |  | ASPD ↓ | 0.1074 ± 0.0088 | 0.1176 ± 0.0059 | 0.0811 ± 0.0138 | 0.0839 ± 0.0152 | 0.1050 ± 0.0113 |
| Robust + fair | FairGuard* | ACC (%) ↑ | 89.18 ± 0.83 | 89.57 ± 1.24 | 49.32 ± 1.59 | 50.33 ± 1.72 | 83.20 ± 1.76 |
|  |  | AEOD ↓ | 0.0317 ± 0.0026 | 0.0369 ± 0.0071 | 0.0042 ± 0.0132 | 0.0000 ± 0.0000 | 0.0205 ± 0.0098 |
|  |  | ASPD ↓ | 0.0998 ± 0.0062 | 0.1078 ± 0.0092 | 0.0049 ± 0.0156 | 0.0000 ± 0.0000 | 0.0491 ± 0.0199 |
| Robust + fair | FLTrust+FairGuard* | ACC (%) ↑ | 89.93 ± 1.10 | 90.37 ± 0.60 | 77.83 ± 12.15 | 75.99 ± 16.52 | 86.44 ± 5.76 |
|  |  | AEOD ↓ | 0.0365 ± 0.0075 | 0.0363 ± 0.0086 | 0.0298 ± 0.0299 | 0.0293 ± 0.0146 | 0.0368 ± 0.0144 |
|  |  | ASPD ↓ | 0.1074 ± 0.0113 | 0.1118 ± 0.0100 | 0.0607 ± 0.0354 | 0.0652 ± 0.0419 | 0.0903 ± 0.0325 |
| Ours | GuardFed-AD2+ | ACC (%) ↑ | 88.55 ± 1.04 | 88.70 ± 0.73 | 88.83 ± 0.90 | 89.06 ± 0.84 | 88.77 ± 0.92 |
|  |  | AEOD ↓ | 0.0403 ± 0.0103 | 0.0368 ± 0.0134 | 0.0372 ± 0.0070 | 0.0367 ± 0.0038 | 0.0347 ± 0.0139 |
|  |  | ASPD ↓ | 0.1031 ± 0.0101 | 0.1008 ± 0.0103 | 0.1026 ± 0.0116 | 0.1019 ± 0.0066 | 0.0998 ± 0.0106 |

All displayed cells use the same 10 seeds. Seed subsets follow the frozen protocol; all cells within each table use identical seeds.  
Round 70; validation only (19,867 images). Mean +/- sample SD (ddof = 1); ACC in %, gaps on [0, 1].  
AEOD is the implemented absolute TPR gap. Descriptive values do not establish statistical significance.  
* Adaptations: FairFed, FairGuard, hybrid; FedAA-DDPG round/policy adapter; LASA with local-Adam update differences.  
Recipes fixed before coverage. Seven methods used non-IID Benign/S-DFA search; FedAA/LASA used both distributions, Benign/S-DFA, seed 91001.  
All records in this table use torch 2.11/cu128. The source cohort contains 14 cu130 records, excluded from this table.  
Nine methods only; eight additional manuscript baselines and final frozen evaluation remain. Native/shared main endpoint is pending.  
Raw view: all nine methods use uncalibrated argmax (margin > 0); same saved checkpoint per ID.  
All outcomes, including constant predictions, are retained. Inference: 434 CPU / 466 GPU in the 900-ID source; no uniform-device claim.  
