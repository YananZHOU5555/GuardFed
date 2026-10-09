# CelebA [shared calibration]: non-IID

COMPLETED VALIDATION COHORT - 10 shared seeds (91001-91010).

Snapshot: 2026-10-09T15:46:28.401491+00:00; 900/900 accepted records.

| Category | Method | Metric | Benign | F-Flip | FedSA | S-DFA | Sp-DFA |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Vanilla FL | FedAvg | ACC (%) ↑ | 86.50 ± 5.47 | 89.07 ± 1.89 | 75.03 ± 12.51 | 74.59 ± 12.78 | 85.01 ± 2.51 |
|  |  | AEOD ↓ | 0.0080 ± 0.0052 | 0.0080 ± 0.0066 | 0.0101 ± 0.0068 | 0.0110 ± 0.0073 | 0.0161 ± 0.0112 |
|  |  | ASPD ↓ | 0.0586 ± 0.0248 | 0.0689 ± 0.0175 | 0.0127 ± 0.0109 | 0.0168 ± 0.0122 | 0.0333 ± 0.0219 |
| Fairness-aware | FairFed* | ACC (%) ↑ | 89.22 ± 1.04 | 89.55 ± 0.96 | 71.78 ± 11.30 | 67.66 ± 11.30 | 85.84 ± 2.32 |
|  |  | AEOD ↓ | 0.0088 ± 0.0049 | 0.0087 ± 0.0065 | 0.0090 ± 0.0073 | 0.0085 ± 0.0066 | 0.0102 ± 0.0035 |
|  |  | ASPD ↓ | 0.0712 ± 0.0083 | 0.0760 ± 0.0084 | 0.0054 ± 0.0049 | 0.0087 ± 0.0071 | 0.0534 ± 0.0169 |
| Robust FL | Median | ACC (%) ↑ | 85.84 ± 1.25 | 86.11 ± 1.25 | 84.12 ± 1.25 | 84.15 ± 1.35 | 85.08 ± 1.28 |
|  |  | AEOD ↓ | 0.0108 ± 0.0071 | 0.0163 ± 0.0085 | 0.0083 ± 0.0036 | 0.0117 ± 0.0101 | 0.0092 ± 0.0090 |
|  |  | ASPD ↓ | 0.0561 ± 0.0108 | 0.0642 ± 0.0094 | 0.0430 ± 0.0106 | 0.0472 ± 0.0108 | 0.0505 ± 0.0089 |
| Robust FL | FLTrust | ACC (%) ↑ | 89.36 ± 0.63 | 89.55 ± 0.76 | 89.09 ± 0.72 | 89.09 ± 0.72 | 89.24 ± 0.74 |
|  |  | AEOD ↓ | 0.0075 ± 0.0056 | 0.0122 ± 0.0072 | 0.0097 ± 0.0070 | 0.0097 ± 0.0070 | 0.0096 ± 0.0066 |
|  |  | ASPD ↓ | 0.0738 ± 0.0076 | 0.0771 ± 0.0118 | 0.0681 ± 0.0116 | 0.0681 ± 0.0116 | 0.0691 ± 0.0101 |
| Adaptive FL | FedAA-DDPG* | ACC (%) ↑ | 88.24 ± 1.16 | 88.12 ± 1.30 | 88.28 ± 1.27 | 88.16 ± 1.54 | 88.07 ± 1.26 |
|  |  | AEOD ↓ | 0.0090 ± 0.0047 | 0.0133 ± 0.0078 | 0.0095 ± 0.0076 | 0.0123 ± 0.0066 | 0.0117 ± 0.0084 |
|  |  | ASPD ↓ | 0.0595 ± 0.0137 | 0.0572 ± 0.0219 | 0.0537 ± 0.0158 | 0.0568 ± 0.0165 | 0.0535 ± 0.0225 |
| Robust FL | LASA* | ACC (%) ↑ | 88.92 ± 0.88 | 89.14 ± 0.86 | 85.50 ± 1.65 | 85.59 ± 1.57 | 87.77 ± 1.19 |
|  |  | AEOD ↓ | 0.0101 ± 0.0050 | 0.0076 ± 0.0057 | 0.0060 ± 0.0047 | 0.0086 ± 0.0067 | 0.0047 ± 0.0056 |
|  |  | ASPD ↓ | 0.0691 ± 0.0132 | 0.0738 ± 0.0046 | 0.0437 ± 0.0075 | 0.0453 ± 0.0106 | 0.0646 ± 0.0080 |
| Robust + fair | FairGuard* | ACC (%) ↑ | 89.09 ± 1.05 | 89.55 ± 0.66 | 61.22 ± 15.39 | 62.27 ± 14.60 | 78.33 ± 14.24 |
|  |  | AEOD ↓ | 0.0105 ± 0.0085 | 0.0153 ± 0.0095 | 0.0051 ± 0.0099 | 0.0066 ± 0.0086 | 0.0127 ± 0.0102 |
|  |  | ASPD ↓ | 0.0597 ± 0.0141 | 0.0725 ± 0.0175 | 0.0044 ± 0.0133 | 0.0048 ± 0.0099 | 0.0191 ± 0.0168 |
| Robust + fair | FLTrust+FairGuard* | ACC (%) ↑ | 89.31 ± 1.62 | 89.73 ± 1.31 | 81.54 ± 11.50 | 80.46 ± 10.78 | 81.23 ± 11.08 |
|  |  | AEOD ↓ | 0.0111 ± 0.0083 | 0.0066 ± 0.0053 | 0.0132 ± 0.0102 | 0.0125 ± 0.0099 | 0.0136 ± 0.0135 |
|  |  | ASPD ↓ | 0.0675 ± 0.0122 | 0.0697 ± 0.0112 | 0.0369 ± 0.0299 | 0.0273 ± 0.0203 | 0.0414 ± 0.0212 |
| Ours | GuardFed-AD2+ | ACC (%) ↑ | 88.59 ± 1.17 | 88.44 ± 0.74 | 88.41 ± 0.92 | 88.22 ± 0.78 | 88.34 ± 0.96 |
|  |  | AEOD ↓ | 0.0070 ± 0.0042 | 0.0100 ± 0.0070 | 0.0078 ± 0.0056 | 0.0126 ± 0.0093 | 0.0087 ± 0.0066 |
|  |  | ASPD ↓ | 0.0651 ± 0.0090 | 0.0589 ± 0.0092 | 0.0616 ± 0.0141 | 0.0584 ± 0.0189 | 0.0596 ± 0.0146 |

All displayed cells use the same 10 seeds. Seed subsets follow the frozen protocol; all cells within each table use identical seeds.  
Round 70; validation only (19,867 images). Mean +/- sample SD (ddof = 1); ACC in %, gaps on [0, 1].  
AEOD is the implemented absolute TPR gap. Descriptive values do not establish statistical significance.  
* Adaptations: FairFed, FairGuard, hybrid; FedAA-DDPG round/policy adapter; LASA with local-Adam update differences.  
Recipes fixed before coverage. Seven methods used non-IID Benign/S-DFA search; FedAA/LASA used both distributions, Benign/S-DFA, seed 91001.  
This table uses 436 cu128 and 14 cu130 records; the source cohort contains 14 cu130 records. No 70-round equivalence claim.  
Nine methods only; eight additional manuscript baselines and final frozen evaluation remain. Native/shared main endpoint is pending.  
Shared-calibration view: all nine methods use saved clean-training-root group thresholds; no validation-label fitting.  
All outcomes, including constant predictions, are retained. Inference: 434 CPU / 466 GPU in the 900-ID source; no uniform-device claim.  
