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

Values are mean ± sample SD (ddof=1). Paired differences are retained in the accompanying JSON; ACC differences use percentage points. No hypothesis test or superiority claim is made.

AEOD is the absolute TPR gap, not full equalized odds. Full reuses historical terminal checkpoints; controls were trained in the current cu128 environment. Driver differences remain a limitation even where the PyTorch build matches. Reported native metrics include each procedure’s original calibration; these tables do not isolate aggregation from calibration.

Completed paired scenes are included by coverage, regardless of which variant wins. Incomplete scenes and other variants remain pending. Lower disparity after a deletion is retained; these interim results do not establish that every component is indispensable.

Source inspection SHA256: `57bb5297cb4e5f418c726439fddeef60da30a1230c56e9385eeced5709c89ca4`.
Original evidence tool SHA256: `3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef`.
