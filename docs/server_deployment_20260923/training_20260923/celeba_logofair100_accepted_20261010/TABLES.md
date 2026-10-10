# LoGoFair fixed recipe07: validation100

ACC%; AEOD absolute TPR gap; ASPD absolute positive-rate gap. Mean ± sampleSD(ddof1). Model seeds vary; fitseed1719 is fixed. No new recipe ranking.

## ten_seed

| Distribution | Scene | n | ACC% | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID | Benign | 10 | 83.637 ± 11.342 | 0.04531 ± 0.01886 | 0.00708 ± 0.00592 |
| IID | F Flip | 10 | 88.032 ± 0.794 | 0.05644 ± 0.00773 | 0.00794 ± 0.00397 |
| IID | FedSA | 10 | 69.184 ± 5.690 | 0.01608 ± 0.01154 | 0.01028 ± 0.00714 |
| IID | S-DFA | 10 | 66.762 ± 3.389 | 0.01289 ± 0.01063 | 0.00997 ± 0.00549 |
| IID | Sp-DFA | 10 | 83.606 ± 2.262 | 0.02836 ± 0.01152 | 0.00773 ± 0.00606 |
| non-IID | Benign | 10 | 85.643 ± 4.921 | 0.04452 ± 0.02319 | 0.00555 ± 0.00509 |
| non-IID | F Flip | 10 | 87.962 ± 1.663 | 0.05353 ± 0.01469 | 0.00597 ± 0.00360 |
| non-IID | FedSA | 10 | 74.940 ± 12.898 | 0.01917 ± 0.01472 | 0.00842 ± 0.00543 |
| non-IID | S-DFA | 10 | 74.528 ± 12.819 | 0.01962 ± 0.01781 | 0.00899 ± 0.00610 |
| non-IID | Sp-DFA | 10 | 84.530 ± 2.270 | 0.03577 ± 0.01224 | 0.00513 ± 0.00453 |
| Seed-first | All10 scenes | 10 | 79.882 ± 3.795 | 0.03317 ± 0.00715 | 0.00771 ± 0.00349 |

## exclude_selection

| Distribution | Scene | n | ACC% | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID | Benign | 9 | 87.189 ± 1.666 | 0.05035 ± 0.01073 | 0.00787 ± 0.00570 |
| IID | F Flip | 9 | 88.204 ± 0.614 | 0.05627 ± 0.00818 | 0.00767 ± 0.00411 |
| IID | FedSA | 9 | 70.055 ± 5.281 | 0.01512 ± 0.01181 | 0.01091 ± 0.00726 |
| IID | S-DFA | 9 | 67.325 ± 3.058 | 0.01221 ± 0.01104 | 0.01062 ± 0.00539 |
| IID | Sp-DFA | 9 | 84.123 ± 1.657 | 0.02781 ± 0.01208 | 0.00787 ± 0.00641 |
| non-IID | Benign | 9 | 86.756 ± 3.644 | 0.04880 ± 0.01997 | 0.00534 ± 0.00535 |
| non-IID | F Flip | 9 | 88.322 ± 1.284 | 0.05675 ± 0.01126 | 0.00603 ± 0.00382 |
| non-IID | FedSA | 9 | 75.648 ± 13.472 | 0.02021 ± 0.01522 | 0.00768 ± 0.00521 |
| non-IID | S-DFA | 9 | 75.264 ± 13.371 | 0.02180 ± 0.01743 | 0.00818 ± 0.00586 |
| non-IID | Sp-DFA | 9 | 84.711 ± 2.331 | 0.03556 ± 0.01296 | 0.00440 ± 0.00413 |
| Seed-first | All10 scenes | 9 | 80.760 ± 2.746 | 0.03449 ± 0.00616 | 0.00766 ± 0.00369 |

## matching_six

| Distribution | Scene | n | ACC% | AEOD | ASPD |
|---|---|---:|---:|---:|---:|
| IID | Benign | 6 | 86.821 ± 1.923 | 0.04811 ± 0.01202 | 0.00761 ± 0.00597 |
| IID | F Flip | 6 | 88.223 ± 0.748 | 0.05407 ± 0.00856 | 0.00987 ± 0.00298 |
| IID | FedSA | 6 | 70.210 ± 6.012 | 0.01521 ± 0.01209 | 0.00994 ± 0.00814 |
| IID | S-DFA | 6 | 68.054 ± 3.013 | 0.01327 ± 0.01357 | 0.01009 ± 0.00414 |
| IID | Sp-DFA | 6 | 84.876 ± 1.230 | 0.02812 ± 0.01199 | 0.00962 ± 0.00680 |
| non-IID | Benign | 6 | 85.676 ± 4.119 | 0.04112 ± 0.02005 | 0.00606 ± 0.00626 |
| non-IID | F Flip | 6 | 88.022 ± 1.500 | 0.05472 ± 0.01287 | 0.00662 ± 0.00462 |
| non-IID | FedSA | 6 | 78.309 ± 11.314 | 0.02633 ± 0.01406 | 0.00772 ± 0.00606 |
| non-IID | S-DFA | 6 | 78.541 ± 11.019 | 0.03064 ± 0.01413 | 0.00846 ± 0.00655 |
| non-IID | Sp-DFA | 6 | 85.072 ± 2.690 | 0.03437 ± 0.01591 | 0.00564 ± 0.00449 |
| Seed-first | All10 scenes | 6 | 81.381 ± 2.139 | 0.03459 ± 0.00696 | 0.00816 ± 0.00437 |

All negative/constant predictions are retained. Twenty image-ID virtual cohorts are not true training clients. Calibrated DP adaptation does not establish EO or aggregation-only effects.

Seed91001 participated in validation selection; other validation seeds were historically observed. Accepted FedAvg sources have mixed runtime/device histories; sigmoid of accepted float32 margins may differ from fresh softmax. Historical test attribute/split metadata exposure remains; this is valid-only, not final test. Primary manuscript endpoint and final claims remain author decisions.
