# CelebA mechanism ablation: interim accepted scenes

Validation-only terminal-round results; this is an interim extraction, not the completed 900-record mechanism comparison.

Each displayed scene contains all ten declared paired seeds (91001–91010). The two additional panels apply the same rule to every method: omit selection seed 91001, or retain seeds 91005–91010. These previously observed validation seeds are not an untouched confirmation set.

## All 10 seeds

| Distribution | Scenario | Variant | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 10 | 88.258 ± 1.039 | 0.00973 ± 0.00751 | 0.06254 ± 0.01377 |
| IID | Benign | minus_U | 10 | 87.152 ± 1.796 | 0.01138 ± 0.00831 | 0.03982 ± 0.01737 |
| IID | F Flip | Full | 10 | 88.391 ± 0.626 | 0.01068 ± 0.00958 | 0.06068 ± 0.01026 |
| IID | F Flip | minus_U | 10 | 88.082 ± 1.551 | 0.01276 ± 0.00997 | 0.05153 ± 0.02025 |
| IID | FedSA | Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| IID | FedSA | minus_U | 10 | 88.143 ± 1.359 | 0.01061 ± 0.00888 | 0.05184 ± 0.01671 |
| IID | S-DFA | Full | 10 | 88.689 ± 0.809 | 0.00769 ± 0.00463 | 0.06377 ± 0.00697 |
| IID | S-DFA | minus_U | 10 | 87.759 ± 1.228 | 0.01191 ± 0.01163 | 0.05051 ± 0.02281 |
| IID | Sp-DFA | Full | 10 | 88.382 ± 0.889 | 0.01648 ± 0.00719 | 0.05588 ± 0.01561 |
| IID | Sp-DFA | minus_U | 10 | 86.998 ± 4.345 | 0.01311 ± 0.01074 | 0.05267 ± 0.02474 |
| non-IID | Benign | Full | 10 | 88.594 ± 1.168 | 0.00696 ± 0.00423 | 0.06511 ± 0.00902 |
| non-IID | Benign | minus_U | 10 | 87.290 ± 1.870 | 0.01300 ± 0.00732 | 0.05140 ± 0.01598 |
| non-IID | F Flip | Full | 10 | 88.443 ± 0.741 | 0.01002 ± 0.00696 | 0.05893 ± 0.00916 |
| non-IID | F Flip | minus_U | 10 | 87.447 ± 1.643 | 0.01229 ± 0.00975 | 0.04775 ± 0.01623 |
| non-IID | FedSA | Full | 10 | 88.413 ± 0.920 | 0.00779 ± 0.00559 | 0.06157 ± 0.01408 |
| non-IID | FedSA | minus_U | 10 | 87.812 ± 1.015 | 0.01113 ± 0.00694 | 0.05410 ± 0.01484 |
| non-IID | S-DFA | Full | 10 | 88.215 ± 0.781 | 0.01255 ± 0.00926 | 0.05844 ± 0.01889 |
| non-IID | S-DFA | minus_U | 10 | 87.762 ± 1.080 | 0.00881 ± 0.00777 | 0.05598 ± 0.01109 |

Values are mean ± sample SD (ddof=1). Paired differences are retained in the accompanying JSON; ACC differences use percentage points. No hypothesis test or superiority claim is made.

## Exclude selection seed: 9 seeds

| Distribution | Scenario | Variant | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 9 | 88.352 ± 1.056 | 0.00975 ± 0.00797 | 0.06232 ± 0.01459 |
| IID | Benign | minus_U | 9 | 87.107 ± 1.899 | 0.01125 ± 0.00881 | 0.03948 ± 0.01838 |
| IID | F Flip | Full | 9 | 88.412 ± 0.660 | 0.01147 ± 0.00981 | 0.06043 ± 0.01085 |
| IID | F Flip | minus_U | 9 | 88.073 ± 1.645 | 0.01266 ± 0.01056 | 0.04963 ± 0.02052 |
| IID | FedSA | Full | 9 | 88.622 ± 0.819 | 0.00544 ± 0.00388 | 0.06675 ± 0.01202 |
| IID | FedSA | minus_U | 9 | 88.230 ± 1.412 | 0.01108 ± 0.00929 | 0.05251 ± 0.01758 |
| IID | S-DFA | Full | 9 | 88.845 ± 0.681 | 0.00675 ± 0.00376 | 0.06329 ± 0.00721 |
| IID | S-DFA | minus_U | 9 | 87.925 ± 1.177 | 0.01199 ± 0.01234 | 0.05347 ± 0.02207 |
| IID | Sp-DFA | Full | 9 | 88.546 ± 0.765 | 0.01706 ± 0.00737 | 0.05548 ± 0.01650 |
| IID | Sp-DFA | minus_U | 9 | 86.979 ± 4.608 | 0.01314 ± 0.01140 | 0.05060 ± 0.02530 |
| non-IID | Benign | Full | 9 | 88.591 ± 1.239 | 0.00632 ± 0.00394 | 0.06384 ± 0.00856 |
| non-IID | Benign | minus_U | 9 | 87.245 ± 1.978 | 0.01230 ± 0.00740 | 0.04929 ± 0.01540 |
| non-IID | F Flip | Full | 9 | 88.504 ± 0.759 | 0.01039 ± 0.00727 | 0.05787 ± 0.00904 |
| non-IID | F Flip | minus_U | 9 | 87.595 ± 1.670 | 0.01307 ± 0.01000 | 0.04772 ± 0.01722 |
| non-IID | FedSA | Full | 9 | 88.394 ± 0.974 | 0.00826 ± 0.00571 | 0.06088 ± 0.01476 |
| non-IID | FedSA | minus_U | 9 | 87.846 ± 1.071 | 0.01204 ± 0.00669 | 0.05449 ± 0.01569 |
| non-IID | S-DFA | Full | 9 | 88.241 ± 0.824 | 0.01187 ± 0.00954 | 0.06046 ± 0.01885 |
| non-IID | S-DFA | minus_U | 9 | 87.854 ± 1.104 | 0.00913 ± 0.00817 | 0.05679 ± 0.01144 |

Values are mean ± sample SD (ddof=1). Paired differences are retained in the accompanying JSON; ACC differences use percentage points. No hypothesis test or superiority claim is made.

## Seeds 91005–91010: 6 seeds

| Distribution | Scenario | Variant | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---|---|---:|---:|---:|---:|
| IID | Benign | Full | 6 | 88.558 ± 0.680 | 0.01138 ± 0.00734 | 0.06649 ± 0.01593 |
| IID | Benign | minus_U | 6 | 87.467 ± 1.093 | 0.01129 ± 0.00914 | 0.04017 ± 0.01468 |
| IID | F Flip | Full | 6 | 88.288 ± 0.746 | 0.01484 ± 0.01041 | 0.05808 ± 0.01290 |
| IID | F Flip | minus_U | 6 | 87.676 ± 1.877 | 0.01099 ± 0.01064 | 0.04632 ± 0.02276 |
| IID | FedSA | Full | 6 | 88.569 ± 0.867 | 0.00550 ± 0.00361 | 0.06965 ± 0.01099 |
| IID | FedSA | minus_U | 6 | 88.576 ± 0.521 | 0.00728 ± 0.00729 | 0.05668 ± 0.00947 |
| IID | S-DFA | Full | 6 | 88.745 ± 0.786 | 0.00703 ± 0.00401 | 0.06171 ± 0.00850 |
| IID | S-DFA | minus_U | 6 | 87.961 ± 0.921 | 0.01546 ± 0.01384 | 0.04828 ± 0.02248 |
| IID | Sp-DFA | Full | 6 | 88.354 ± 0.873 | 0.01710 ± 0.00828 | 0.05595 ± 0.02031 |
| IID | Sp-DFA | minus_U | 6 | 86.259 ± 5.627 | 0.01084 ± 0.00958 | 0.04996 ± 0.02617 |
| non-IID | Benign | Full | 6 | 88.585 ± 1.423 | 0.00623 ± 0.00389 | 0.06411 ± 0.00673 |
| non-IID | Benign | minus_U | 6 | 87.497 ± 2.201 | 0.01282 ± 0.00747 | 0.04494 ± 0.01300 |
| non-IID | F Flip | Full | 6 | 88.431 ± 0.874 | 0.01102 ± 0.00861 | 0.05696 ± 0.00900 |
| non-IID | F Flip | minus_U | 6 | 88.011 ± 1.798 | 0.01009 ± 0.00944 | 0.05325 ± 0.01677 |
| non-IID | FedSA | Full | 6 | 88.488 ± 0.881 | 0.00831 ± 0.00443 | 0.05660 ± 0.00808 |
| non-IID | FedSA | minus_U | 6 | 87.957 ± 1.309 | 0.01229 ± 0.00545 | 0.05699 ± 0.01589 |
| non-IID | S-DFA | Full | 6 | 88.423 ± 0.829 | 0.00670 ± 0.00677 | 0.06111 ± 0.01143 |
| non-IID | S-DFA | minus_U | 6 | 88.078 ± 1.232 | 0.00663 ± 0.00760 | 0.06188 ± 0.00682 |

Values are mean ± sample SD (ddof=1). Paired differences are retained in the accompanying JSON; ACC differences use percentage points. No hypothesis test or superiority claim is made.

AEOD is the absolute TPR gap, not full equalized odds. Full reuses historical terminal checkpoints; controls were trained in the current cu128 environment. Driver differences remain a limitation even where the PyTorch build matches. Reported native metrics include each procedure’s original calibration; these tables do not isolate aggregation from calibration.

Completed paired scenes are included by coverage, regardless of which variant wins. Incomplete scenes and other variants remain pending. Lower disparity after a deletion is retained; these interim results do not establish that every component is indispensable.

Source inspection SHA256: `cd357e05eca1ae6a7d0c9bca170484af4e31d208a96c493681d70206b34e163a`.
Original evidence tool SHA256: `3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef`.
