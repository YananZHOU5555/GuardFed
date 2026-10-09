# Strict score separation: offline diagnostics

Snapshot: 2026-09-23T08:44:57.970136+00:00
Accepted completed runs: 140; available round records: 8440; rejected files: 0.
Direct-weight formula checks: 8400 rounds; maximum absolute difference: 0.0; failures: 0.

| Source | Distribution | Component | Runs | Rounds available / expected | Positive / zero / negative margin | Selected with no malicious | Max malicious weight |
|---|---|---|---:|---:|---:|---:|---:|
| historical_full | IID | none | 10 | 20 / 700 | 17 / 0 / 3 | 20 | 0.0 |
| historical_full | non-IID | none | 10 | 20 / 700 | 14 / 0 / 6 | 20 | 0.0 |
| new_completed | IID | A | 10 | 700 / 700 | 568 / 0 / 132 | 689 | 0.9999999998137692 |
| new_completed | IID | C | 10 | 700 / 700 | 435 / 0 / 265 | 632 | 1.0 |
| new_completed | IID | F | 10 | 700 / 700 | 620 / 0 / 80 | 700 | 0.0 |
| new_completed | IID | N | 10 | 700 / 700 | 566 / 0 / 134 | 698 | 3.9125453000257624e-24 |
| new_completed | IID | U | 10 | 700 / 700 | 434 / 0 / 266 | 685 | 1.0 |
| new_completed | IID | V | 10 | 700 / 700 | 631 / 0 / 69 | 694 | 1.0 |
| new_completed | non-IID | A | 10 | 700 / 700 | 460 / 0 / 240 | 699 | 0.9992333413793926 |
| new_completed | non-IID | C | 10 | 700 / 700 | 375 / 0 / 325 | 687 | 0.21203728054433987 |
| new_completed | non-IID | F | 10 | 700 / 700 | 579 / 0 / 121 | 699 | 0.0 |
| new_completed | non-IID | N | 10 | 700 / 700 | 417 / 0 / 283 | 700 | 0.0 |
| new_completed | non-IID | U | 10 | 700 / 700 | 355 / 0 / 345 | 696 | 0.42688950648449925 |
| new_completed | non-IID | V | 10 | 700 / 700 | 625 / 0 / 75 | 699 | 0.0 |

- Strict margin=min(benign trust_scores)-max(malicious trust_scores); every negative and zero margin is retained.
- trust_scores is the additive score from normalized components; logit margin additionally divides by the chosen candidate temperature. No extra score normalization is invented.
- All-pre-gate, hard-gate and final selected sets are reported separately. Empty-class margins are undefined, not positive separation.
- The implementation can select gate-excluded clients when the top-k count exceeds gate size, then applies softmax to original scores. Therefore selected need not be a subset of hard_gate; selected_outside_hard_gate_n explicitly records this existing behavior.
- Historical Full contains only rounds 1 and 70. Its missing rounds are neither imputed nor treated as failed/successful separation.
- Historical weights are reconstructed only from stored scores, selected indices and chosen-candidate temperature, using the unchanged frozen softmax and normalization formula.
- Malicious weight is on norm-scaled updates; it is not total adversarial influence, nor a model accuracy/fairness guarantee.
- These are conditional softmax diagnostics, not a complete robustness guarantee. Round observations are correlated and are not independent statistical replicates.
- Coverage is completed files at script start, not all planned experiments. Re-run after more runs complete.
