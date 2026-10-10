# FLGMM: six complete CelebA validation scenes

Frozen Dirichlet partitions: IID alpha = 5000; non-IID alpha = 5. ACC is reported in percent; AEOD and ASPD are fractions. AEOD means absolute TPR difference, not full equalized odds. See [caption and evidence boundaries](CAPTION.md) for fixed seed panels, validation selection history and the already adopted 61-record source.

## raw · 10 fixed seeds

| Metric | IID: Benign | IID: F-Flip | IID: FedSA | IID: S-DFA | IID: Sp-DFA | non-IID: Benign |
| --- | --- | --- | --- | --- | --- | --- |
| ACC (%) ↑ | 89.33 ± 0.99 | 89.72 ± 0.83 | 89.34 ± 1.01 | 89.34 ± 1.01 | 89.49 ± 0.93 | 89.02 ± 0.91 |
| AEOD ↓ | 0.0404 ± 0.0065 | 0.0458 ± 0.0078 | 0.0424 ± 0.0070 | 0.0423 ± 0.0075 | 0.0460 ± 0.0061 | 0.0396 ± 0.0066 |
| ASPD ↓ | 0.1074 ± 0.0073 | 0.1171 ± 0.0049 | 0.1113 ± 0.0081 | 0.1113 ± 0.0080 | 0.1143 ± 0.0063 | 0.1071 ± 0.0078 |

## raw · 9 fixed seeds

| Metric | IID: Benign | IID: F-Flip | IID: FedSA | IID: S-DFA | IID: Sp-DFA | non-IID: Benign |
| --- | --- | --- | --- | --- | --- | --- |
| ACC (%) ↑ | 89.40 ± 1.02 | 89.81 ± 0.84 | 89.47 ± 0.98 | 89.48 ± 0.97 | 89.63 ± 0.87 | 89.18 ± 0.81 |
| AEOD ↓ | 0.0401 ± 0.0069 | 0.0454 ± 0.0082 | 0.0424 ± 0.0074 | 0.0423 ± 0.0080 | 0.0465 ± 0.0062 | 0.0392 ± 0.0068 |
| ASPD ↓ | 0.1081 ± 0.0074 | 0.1175 ± 0.0050 | 0.1120 ± 0.0083 | 0.1120 ± 0.0082 | 0.1153 ± 0.0058 | 0.1080 ± 0.0077 |

## raw · 6 fixed seeds

| Metric | IID: Benign | IID: F-Flip | IID: FedSA | IID: S-DFA | IID: Sp-DFA | non-IID: Benign |
| --- | --- | --- | --- | --- | --- | --- |
| ACC (%) ↑ | 89.12 ± 1.17 | 89.66 ± 1.00 | 89.10 ± 1.01 | 89.12 ± 1.01 | 89.32 ± 0.92 | 89.05 ± 0.99 |
| AEOD ↓ | 0.0411 ± 0.0082 | 0.0436 ± 0.0093 | 0.0431 ± 0.0092 | 0.0429 ± 0.0098 | 0.0460 ± 0.0073 | 0.0398 ± 0.0039 |
| ASPD ↓ | 0.1064 ± 0.0083 | 0.1162 ± 0.0044 | 0.1091 ± 0.0089 | 0.1091 ± 0.0088 | 0.1121 ± 0.0044 | 0.1067 ± 0.0079 |

## native · 10 fixed seeds

| Metric | IID: Benign | IID: F-Flip | IID: FedSA | IID: S-DFA | IID: Sp-DFA | non-IID: Benign |
| --- | --- | --- | --- | --- | --- | --- |
| ACC (%) ↑ | 89.33 ± 0.99 | 89.72 ± 0.83 | 89.34 ± 1.01 | 89.34 ± 1.01 | 89.49 ± 0.93 | 89.02 ± 0.91 |
| AEOD ↓ | 0.0404 ± 0.0065 | 0.0458 ± 0.0078 | 0.0424 ± 0.0070 | 0.0423 ± 0.0075 | 0.0460 ± 0.0061 | 0.0396 ± 0.0066 |
| ASPD ↓ | 0.1074 ± 0.0073 | 0.1171 ± 0.0049 | 0.1113 ± 0.0081 | 0.1113 ± 0.0080 | 0.1143 ± 0.0063 | 0.1071 ± 0.0078 |

## native · 9 fixed seeds

| Metric | IID: Benign | IID: F-Flip | IID: FedSA | IID: S-DFA | IID: Sp-DFA | non-IID: Benign |
| --- | --- | --- | --- | --- | --- | --- |
| ACC (%) ↑ | 89.40 ± 1.02 | 89.81 ± 0.84 | 89.47 ± 0.98 | 89.48 ± 0.97 | 89.63 ± 0.87 | 89.18 ± 0.81 |
| AEOD ↓ | 0.0401 ± 0.0069 | 0.0454 ± 0.0082 | 0.0424 ± 0.0074 | 0.0423 ± 0.0080 | 0.0465 ± 0.0062 | 0.0392 ± 0.0068 |
| ASPD ↓ | 0.1081 ± 0.0074 | 0.1175 ± 0.0050 | 0.1120 ± 0.0083 | 0.1120 ± 0.0082 | 0.1153 ± 0.0058 | 0.1080 ± 0.0077 |

## native · 6 fixed seeds

| Metric | IID: Benign | IID: F-Flip | IID: FedSA | IID: S-DFA | IID: Sp-DFA | non-IID: Benign |
| --- | --- | --- | --- | --- | --- | --- |
| ACC (%) ↑ | 89.12 ± 1.17 | 89.66 ± 1.00 | 89.10 ± 1.01 | 89.12 ± 1.01 | 89.32 ± 0.92 | 89.05 ± 0.99 |
| AEOD ↓ | 0.0411 ± 0.0082 | 0.0436 ± 0.0093 | 0.0431 ± 0.0092 | 0.0429 ± 0.0098 | 0.0460 ± 0.0073 | 0.0398 ± 0.0039 |
| ASPD ↓ | 0.1064 ± 0.0083 | 0.1162 ± 0.0044 | 0.1091 ± 0.0089 | 0.1091 ± 0.0088 | 0.1121 ± 0.0044 | 0.1067 ± 0.0079 |

## shared_calibration · 10 fixed seeds

| Metric | IID: Benign | IID: F-Flip | IID: FedSA | IID: S-DFA | IID: Sp-DFA | non-IID: Benign |
| --- | --- | --- | --- | --- | --- | --- |
| ACC (%) ↑ | 88.95 ± 1.04 | 89.36 ± 0.82 | 88.99 ± 1.00 | 88.99 ± 1.00 | 89.16 ± 0.93 | 88.61 ± 0.92 |
| AEOD ↓ | 0.0079 ± 0.0032 | 0.0071 ± 0.0053 | 0.0086 ± 0.0049 | 0.0092 ± 0.0049 | 0.0093 ± 0.0080 | 0.0065 ± 0.0049 |
| ASPD ↓ | 0.0657 ± 0.0075 | 0.0712 ± 0.0094 | 0.0659 ± 0.0085 | 0.0661 ± 0.0080 | 0.0749 ± 0.0086 | 0.0684 ± 0.0067 |

## shared_calibration · 9 fixed seeds

| Metric | IID: Benign | IID: F-Flip | IID: FedSA | IID: S-DFA | IID: Sp-DFA | non-IID: Benign |
| --- | --- | --- | --- | --- | --- | --- |
| ACC (%) ↑ | 89.02 ± 1.08 | 89.43 ± 0.83 | 89.12 ± 0.97 | 89.12 ± 0.97 | 89.30 ± 0.86 | 88.78 ± 0.79 |
| AEOD ↓ | 0.0080 ± 0.0034 | 0.0077 ± 0.0053 | 0.0090 ± 0.0050 | 0.0097 ± 0.0050 | 0.0090 ± 0.0084 | 0.0061 ± 0.0051 |
| ASPD ↓ | 0.0653 ± 0.0078 | 0.0718 ± 0.0098 | 0.0656 ± 0.0089 | 0.0657 ± 0.0084 | 0.0749 ± 0.0092 | 0.0694 ± 0.0062 |

## shared_calibration · 6 fixed seeds

| Metric | IID: Benign | IID: F-Flip | IID: FedSA | IID: S-DFA | IID: Sp-DFA | non-IID: Benign |
| --- | --- | --- | --- | --- | --- | --- |
| ACC (%) ↑ | 88.70 ± 1.22 | 89.35 ± 1.02 | 88.79 ± 1.05 | 88.78 ± 1.04 | 89.02 ± 0.96 | 88.67 ± 0.95 |
| AEOD ↓ | 0.0076 ± 0.0041 | 0.0089 ± 0.0053 | 0.0078 ± 0.0043 | 0.0087 ± 0.0043 | 0.0080 ± 0.0099 | 0.0067 ± 0.0061 |
| ASPD ↓ | 0.0649 ± 0.0098 | 0.0756 ± 0.0097 | 0.0657 ± 0.0109 | 0.0659 ± 0.0102 | 0.0730 ± 0.0070 | 0.0689 ± 0.0048 |

## Scope and interpretation

- Terminal round70; validation only,19,867 images; all three metrics and views use the same checkpoint per ID. Mean ± sample SD (ddof=1); ACC in percent, AEOD/ASPD on [0,1]. AEOD is the absolute TPR gap, not full equalized odds.
- Panels are fixed:10 seeds91001–91010;9 seeds91002–91010;6 seeds91005–91010. No per-cell best-seed selection. The9-seed panel excludes selection seed91001; it does not undo repeated validation exposure. The6-seed panel preserves the earlier common-seed comparison subset; all61 FLGMM records themselves use the same declared cu128 version.
- Selected recipe: warmup20, control width2.0, local learning rate0.001. Recipe selection used seed91001 validation scores across IID/non-IID Benign/S-DFA; three of the four reused screen checkpoints enter these complete tables. The fourth, non-IID S-DFA seed91001, is retained in records61.json but excluded from scene statistics.
- FLGMM follows the frozen author-code adaptation, including its largest-cluster choice, upstream bounds behavior and declared zero-standard-deviation extension. Native and raw are identical uncalibrated margin>0 outputs, not independent replications. Shared calibration applies the existing common clean train-root group thresholds (margin≥threshold), not FLGMM-native calibration.
- All61 accepted records were trained on CUDA with torch2.11.0+cu128 and evaluated on CPU with torch2.11.0+cu128; this is not a CPU/GPU training-equivalence claim. Windows saved-output audit runtime is separately retained in the root proofs.
- Original Linux whole checks supply root-refit evidence. The original Windows47 exact-refit/whole failure remains preserved; new13 Windows checking used saved fits with zero refits. Root61 retains five group-KL micro-difference records among its60 main records plus the prior interface difference; these do not change saved predictions/metrics/counts. No universal bitwise recalibration claim.
- This is six complete FLGMM scenes (60 records) plus one retained partial record, not FLGMM100, a17-method comparison or final test. Values are descriptive; all outcomes are retained, without a superiority or significance claim.
