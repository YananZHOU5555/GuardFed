# CelebA [shared calibration]: IID

COMPLETED VALIDATION COHORT - 10 shared seeds (91001-91010).

Snapshot: 2026-10-09T15:46:28.401491+00:00; 900/900 accepted records.

| Category | Method | Metric | Benign | F-Flip | FedSA | S-DFA | Sp-DFA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla FL | FedAvg | ACC (%) ↑ | 84.38 ± 11.63 | 89.17 ± 0.91 | 68.98 ± 5.78 | 66.69 ± 3.47 | 84.04 ± 2.31 |
|  |  | AEOD ↓ | 0.0061 ± 0.0058 | 0.0104 ± 0.0075 | 0.0114 ± 0.0079 | 0.0099 ± 0.0083 | 0.0115 ± 0.0061 |
|  |  | ASPD ↓ | 0.0546 ± 0.0251 | 0.0689 ± 0.0156 | 0.0085 ± 0.0070 | 0.0141 ± 0.0121 | 0.0299 ± 0.0173 |
| Fairness-aware | FairFed* | ACC (%) ↑ | 89.11 ± 0.83 | 89.36 ± 0.77 | 69.73 ± 4.94 | 63.30 ± 4.77 | 85.17 ± 2.24 |
|  |  | AEOD ↓ | 0.0095 ± 0.0067 | 0.0076 ± 0.0074 | 0.0091 ± 0.0068 | 0.0110 ± 0.0099 | 0.0088 ± 0.0080 |
|  |  | ASPD ↓ | 0.0704 ± 0.0089 | 0.0752 ± 0.0054 | 0.0071 ± 0.0061 | 0.0124 ± 0.0066 | 0.0417 ± 0.0112 |
| Robust FL | Median | ACC (%) ↑ | 85.87 ± 1.33 | 86.20 ± 1.40 | 84.30 ± 1.57 | 84.56 ± 1.36 | 85.20 ± 1.52 |
|  |  | AEOD ↓ | 0.0118 ± 0.0067 | 0.0132 ± 0.0078 | 0.0120 ± 0.0125 | 0.0082 ± 0.0080 | 0.0130 ± 0.0074 |
|  |  | ASPD ↓ | 0.0577 ± 0.0113 | 0.0598 ± 0.0107 | 0.0398 ± 0.0179 | 0.0431 ± 0.0132 | 0.0532 ± 0.0138 |
| Robust FL | FLTrust | ACC (%) ↑ | 89.13 ± 0.84 | 89.35 ± 0.92 | 89.27 ± 0.85 | 89.27 ± 0.85 | 89.29 ± 0.78 |
|  |  | AEOD ↓ | 0.0086 ± 0.0065 | 0.0076 ± 0.0067 | 0.0074 ± 0.0061 | 0.0074 ± 0.0061 | 0.0127 ± 0.0112 |
|  |  | ASPD ↓ | 0.0750 ± 0.0076 | 0.0743 ± 0.0116 | 0.0650 ± 0.0099 | 0.0650 ± 0.0099 | 0.0769 ± 0.0181 |
| Adaptive FL | FedAA-DDPG* | ACC (%) ↑ | 88.05 ± 1.44 | 88.62 ± 1.12 | 88.49 ± 1.22 | 88.46 ± 1.22 | 88.18 ± 1.02 |
|  |  | AEOD ↓ | 0.0140 ± 0.0105 | 0.0124 ± 0.0103 | 0.0130 ± 0.0131 | 0.0113 ± 0.0116 | 0.0144 ± 0.0093 |
|  |  | ASPD ↓ | 0.0572 ± 0.0224 | 0.0566 ± 0.0222 | 0.0506 ± 0.0211 | 0.0574 ± 0.0187 | 0.0547 ± 0.0203 |
| Robust FL | LASA* | ACC (%) ↑ | 88.97 ± 1.40 | 89.00 ± 0.83 | 85.45 ± 1.71 | 85.64 ± 1.62 | 87.80 ± 1.39 |
|  |  | AEOD ↓ | 0.0095 ± 0.0074 | 0.0086 ± 0.0070 | 0.0098 ± 0.0070 | 0.0049 ± 0.0030 | 0.0078 ± 0.0044 |
|  |  | ASPD ↓ | 0.0662 ± 0.0126 | 0.0725 ± 0.0066 | 0.0483 ± 0.0090 | 0.0455 ± 0.0086 | 0.0590 ± 0.0099 |
| Robust + fair | FairGuard* | ACC (%) ↑ | 88.92 ± 0.91 | 89.18 ± 1.24 | 51.67 ± 0.01 | 51.67 ± 0.00 | 83.15 ± 1.26 |
|  |  | AEOD ↓ | 0.0086 ± 0.0049 | 0.0079 ± 0.0054 | 0.0002 ± 0.0003 | 0.0000 ± 0.0001 | 0.0101 ± 0.0059 |
|  |  | ASPD ↓ | 0.0593 ± 0.0118 | 0.0676 ± 0.0158 | 0.0001 ± 0.0002 | 0.0000 ± 0.0001 | 0.0221 ± 0.0078 |
| Robust + fair | FLTrust+FairGuard* | ACC (%) ↑ | 89.60 ± 1.11 | 89.98 ± 0.66 | 80.70 ± 6.93 | 76.87 ± 14.90 | 86.81 ± 3.77 |
|  |  | AEOD ↓ | 0.0093 ± 0.0071 | 0.0085 ± 0.0066 | 0.0104 ± 0.0111 | 0.0107 ± 0.0082 | 0.0079 ± 0.0045 |
|  |  | ASPD ↓ | 0.0689 ± 0.0168 | 0.0707 ± 0.0145 | 0.0196 ± 0.0242 | 0.0255 ± 0.0216 | 0.0508 ± 0.0205 |
| Ours | GuardFed-AD2+ | ACC (%) ↑ | 88.26 ± 1.04 | 88.39 ± 0.63 | 88.47 ± 0.91 | 88.69 ± 0.81 | 88.38 ± 0.89 |
|  |  | AEOD ↓ | 0.0097 ± 0.0075 | 0.0107 ± 0.0096 | 0.0061 ± 0.0042 | 0.0077 ± 0.0046 | 0.0165 ± 0.0072 |
|  |  | ASPD ↓ | 0.0625 ± 0.0138 | 0.0607 ± 0.0103 | 0.0640 ± 0.0142 | 0.0638 ± 0.0070 | 0.0559 ± 0.0156 |

All displayed cells use the same 10 seeds. Seed subsets follow the frozen protocol; all cells within each table use identical seeds.  
Round 70; validation only (19,867 images). Mean +/- sample SD (ddof = 1); ACC in %, gaps on [0, 1].  
AEOD is the implemented absolute TPR gap. Descriptive values do not establish statistical significance.  
* Adaptations: FairFed, FairGuard, hybrid; FedAA-DDPG round/policy adapter; LASA with local-Adam update differences.  
Recipes fixed before coverage. Seven methods used non-IID Benign/S-DFA search; FedAA/LASA used both distributions, Benign/S-DFA, seed 91001.  
All records in this table use torch 2.11/cu128. The source cohort contains 14 cu130 records, excluded from this table.  
Nine methods only; eight additional manuscript baselines and final frozen evaluation remain. Native/shared main endpoint is pending.  
Shared-calibration view: all nine methods use saved clean-training-root group thresholds; no validation-label fitting.  
All outcomes, including constant predictions, are retained. Inference: 434 CPU / 466 GPU in the 900-ID source; no uniform-device claim.  
