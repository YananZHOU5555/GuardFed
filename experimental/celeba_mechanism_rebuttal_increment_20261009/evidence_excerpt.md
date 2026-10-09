# Accepted three-view evidence excerpt

**DO_NOT_SUBMIT_BEFORE_FULL_COHORT — interim author-review material.**

This is a direct extraction of existing summary cells, with no recalculation of scientific metrics or statistics. Full-precision numbers and source JSON pointers are in [numeric_references.json](numeric_references.json).

Six scenes; 60 minus_U checkpoints and 60 matched historical Full checkpoints. n=10 shared seeds 91001–91010 in every row. Values are mean ± sample SD (ddof=1); Δ is minus_U−Full. ACC is %, ΔACC is percentage points, and both gaps are on [0,1]. AEOD is the absolute TPR gap, not full equalized odds.

The complete equal-rule n=10/9/6 panels remain in the [accepted table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim_20261009T145900Z/TABLES.md). The nine-seed panel excludes 91001; the six-seed panel retains 91005–91010. Neither is an untouched confirmation set. Directions discussed in the replies refer only to the n=10 panel.

## native

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) / ΔACC (pp) | AEOD / ΔAEOD | ASPD / ΔASPD |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID | Benign | minus_U | 10 | 87.152 ± 1.796 | 0.01138 ± 0.00831 | 0.03982 ± 0.01737 |
| IID | Benign | minus_U minus Full | 10 | -1.107 ± 2.044 | 0.00164 ± 0.00571 | -0.02272 ± 0.01963 |
| IID | F Flip | Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID | F Flip | minus_U | 10 | 88.082 ± 1.551 | 0.01276 ± 0.00997 | 0.05153 ± 0.02025 |
| IID | F Flip | minus_U minus Full | 10 | -0.309 ± 1.337 | 0.00208 ± 0.01164 | -0.00916 ± 0.02366 |
| IID | FedSA | Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID | FedSA | minus_U | 10 | 88.143 ± 1.359 | 0.01061 ± 0.00888 | 0.05184 ± 0.01671 |
| IID | FedSA | minus_U minus Full | 10 | -0.327 ± 1.294 | 0.00453 ± 0.01009 | -0.01217 ± 0.01722 |
| IID | S-DFA | Full | 10 | 88.689 ± 0.809 | 0.00769 ± 0.00463 | 0.06377 ± 0.00697 |
| IID | S-DFA | minus_U | 10 | 87.759 ± 1.228 | 0.01191 ± 0.01163 | 0.05051 ± 0.02281 |
| IID | S-DFA | minus_U minus Full | 10 | -0.931 ± 1.140 | 0.00421 ± 0.01321 | -0.01325 ± 0.02639 |
| IID | Sp-DFA | Full | 10 | 88.382 ± 0.889 | 0.01648 ± 0.00719 | 0.05588 ± 0.01561 |
| IID | Sp-DFA | minus_U | 10 | 86.998 ± 4.345 | 0.01311 ± 0.01074 | 0.05267 ± 0.02474 |
| IID | Sp-DFA | minus_U minus Full | 10 | -1.384 ± 3.938 | -0.00337 ± 0.01161 | -0.00321 ± 0.03629 |
| non-IID | Benign | Full | 10 | 88.594 ± 1.168 | 0.00696 ± 0.00423 | 0.06511 ± 0.00902 |
| non-IID | Benign | minus_U | 10 | 87.290 ± 1.870 | 0.01300 ± 0.00732 | 0.05140 ± 0.01598 |
| non-IID | Benign | minus_U minus Full | 10 | -1.303 ± 0.892 | 0.00604 ± 0.00899 | -0.01371 ± 0.01219 |

## raw

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) / ΔACC (pp) | AEOD / ΔAEOD | ASPD / ΔASPD |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 10 | 88.549 ± 1.038 | 0.04027 ± 0.01035 | 0.10308 ± 0.01007 |
| IID | Benign | minus_U | 10 | 87.391 ± 2.237 | 0.03930 ± 0.01011 | 0.09930 ± 0.01160 |
| IID | Benign | minus_U minus Full | 10 | -1.158 ± 2.504 | -0.00097 ± 0.01606 | -0.00378 ± 0.01447 |
| IID | F Flip | Full | 10 | 88.703 ± 0.733 | 0.03684 ± 0.01343 | 0.10080 ± 0.01032 |
| IID | F Flip | minus_U | 10 | 88.401 ± 1.511 | 0.03210 ± 0.01597 | 0.09698 ± 0.01028 |
| IID | F Flip | minus_U minus Full | 10 | -0.302 ± 1.330 | -0.00474 ± 0.01405 | -0.00382 ± 0.00806 |
| IID | FedSA | Full | 10 | 88.832 ± 0.899 | 0.03716 ± 0.00699 | 0.10257 ± 0.01155 |
| IID | FedSA | minus_U | 10 | 88.456 ± 1.346 | 0.02835 ± 0.00976 | 0.09138 ± 0.01511 |
| IID | FedSA | minus_U minus Full | 10 | -0.376 ± 1.336 | -0.00880 ± 0.01183 | -0.01120 ± 0.01949 |
| IID | S-DFA | Full | 10 | 89.059 ± 0.840 | 0.03671 ± 0.00377 | 0.10194 ± 0.00664 |
| IID | S-DFA | minus_U | 10 | 88.005 ± 1.297 | 0.03623 ± 0.01343 | 0.09576 ± 0.01605 |
| IID | S-DFA | minus_U minus Full | 10 | -1.054 ± 1.111 | -0.00048 ± 0.01566 | -0.00618 ± 0.01346 |
| IID | Sp-DFA | Full | 10 | 88.766 ± 0.924 | 0.03465 ± 0.01394 | 0.09978 ± 0.01059 |
| IID | Sp-DFA | minus_U | 10 | 87.264 ± 4.711 | 0.03612 ± 0.01022 | 0.09364 ± 0.02496 |
| IID | Sp-DFA | minus_U minus Full | 10 | -1.502 ± 4.227 | 0.00147 ± 0.02085 | -0.00614 ± 0.03121 |
| non-IID | Benign | Full | 10 | 88.875 ± 1.209 | 0.03157 ± 0.00643 | 0.09978 ± 0.00770 |
| non-IID | Benign | minus_U | 10 | 87.723 ± 1.779 | 0.03036 ± 0.01038 | 0.09083 ± 0.01732 |
| non-IID | Benign | minus_U minus Full | 10 | -1.152 ± 0.775 | -0.00122 ± 0.01165 | -0.00896 ± 0.01156 |

## shared_calibration

| Distribution | Scenario | Procedure / paired difference | n | ACC (%) / ΔACC (pp) | AEOD / ΔAEOD | ASPD / ΔASPD |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID | Benign | minus_U | 10 | 87.152 ± 1.796 | 0.01138 ± 0.00831 | 0.03982 ± 0.01737 |
| IID | Benign | minus_U minus Full | 10 | -1.107 ± 2.044 | 0.00164 ± 0.00571 | -0.02272 ± 0.01963 |
| IID | F Flip | Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID | F Flip | minus_U | 10 | 88.082 ± 1.551 | 0.01276 ± 0.00997 | 0.05153 ± 0.02025 |
| IID | F Flip | minus_U minus Full | 10 | -0.309 ± 1.337 | 0.00208 ± 0.01164 | -0.00916 ± 0.02366 |
| IID | FedSA | Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID | FedSA | minus_U | 10 | 88.143 ± 1.359 | 0.01061 ± 0.00888 | 0.05184 ± 0.01671 |
| IID | FedSA | minus_U minus Full | 10 | -0.327 ± 1.294 | 0.00453 ± 0.01009 | -0.01217 ± 0.01722 |
| IID | S-DFA | Full | 10 | 88.689 ± 0.809 | 0.00769 ± 0.00463 | 0.06377 ± 0.00697 |
| IID | S-DFA | minus_U | 10 | 87.759 ± 1.228 | 0.01191 ± 0.01163 | 0.05051 ± 0.02281 |
| IID | S-DFA | minus_U minus Full | 10 | -0.931 ± 1.140 | 0.00421 ± 0.01321 | -0.01325 ± 0.02639 |
| IID | Sp-DFA | Full | 10 | 88.382 ± 0.889 | 0.01648 ± 0.00719 | 0.05588 ± 0.01561 |
| IID | Sp-DFA | minus_U | 10 | 86.998 ± 4.345 | 0.01311 ± 0.01074 | 0.05267 ± 0.02474 |
| IID | Sp-DFA | minus_U minus Full | 10 | -1.384 ± 3.938 | -0.00337 ± 0.01161 | -0.00321 ± 0.03629 |
| non-IID | Benign | Full | 10 | 88.594 ± 1.168 | 0.00696 ± 0.00423 | 0.06511 ± 0.00902 |
| non-IID | Benign | minus_U | 10 | 87.290 ± 1.870 | 0.01300 ± 0.00732 | 0.05140 ± 0.01598 |
| non-IID | Benign | minus_U minus Full | 10 | -1.303 ± 0.892 | 0.00604 ± 0.00899 | -0.01371 ± 0.01219 |

Native and shared-calibration saved metrics and group confusion-count dictionaries are identical for all 120 displayed checkpoints. This is not a newly verified prediction-vector equality claim, and gives no independent calibration-gain evidence.

Displayed Full inference uses 5 CPU and 55 GPU checkpoints; all 60 minus_U checkpoints use CPU. Historical Full training comprises 59 cu128 and one cu130 checkpoint (non-IID Benign, seed91001); minus_U training uses cu128 on the current driver595 environment. Driver equality was not established. This is not a uniform-device final fairness comparison.

The [separate native-only seven-scene table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/interim_tables_20261009T144558Z/TABLES.md) also has non-IID F Flip. That seventh scene is not inserted into these three-view tables or their claims.
