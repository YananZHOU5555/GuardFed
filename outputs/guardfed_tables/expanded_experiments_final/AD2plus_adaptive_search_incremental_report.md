# AD2+ Adaptive Search Incremental Report

This file records the additional 5090 experiments run after the first expanded workbook. Values are derived from `raw_results.jsonl` with the same joint-last10 rule: choose one actual round from the last 10 rounds by maximizing `ACC - 0.5 * (AEOD + ASPD)`.

## Additional Raw Experiment Counts

- `ad2plus_adaptive_smoke`: 4 runs.
- `ad2plus_adaptive_fedsa_grid`: 48 runs.
- `ad2plus_adaptive_calibration_grid`: 24 runs.
- `ad2plus_adaptive_middle_grid`: 16 runs.
- `ad2plus_profile_bank_fedsa`: 4 runs.

Total added AD2+ runs: 96.

## Main Finding

The previous FedSA table used `--ad2-plus-mode fixed`, so `GuardFed-AD2+` was effectively the same aggregation rule as `GuardFed-AD2`. I reran AD2+ in adaptive mode and searched root-calibration/fairness-budget variants.

The strongest true 3-seed candidate so far is not a single universal winner across all four FedSA settings:

| Candidate | Adult IID | Adult non-IID | COMPAS IID | COMPAS non-IID |
| --- | ---: | ---: | ---: | ---: |
| Original fixed AD2+ | score rank 1 | score rank 2 | score rank 5 | score rank 4 |
| `b006_d0020_n25` | score rank 4 | score rank 1 | score rank 1 | score rank 3 |
| `b008_cal003_q81` | score rank 3 | score rank 4 | score rank 3 | score rank 1 |
| `b008_d0005_n25` | score rank 5 | score rank 5 | score rank 4 | score rank 6 |

Detailed values are in:

- `ad2plus_candidate_combined_fedsa_joint_summary.csv`
- `ad2plus_adaptive_fedsa_grid_joint_summary.csv`

## Best Observed True Values

`b006_d0020_n25`:

| Dataset | Distribution | ACC | AEOD | ASPD | Score | Score rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Adult | IID | 81.47 | 0.0054 | 0.0622 | 0.7809 | 4 |
| Adult | non-IID | 82.79 | 0.0089 | 0.0817 | 0.7826 | 1 |
| COMPAS | IID | 66.27 | 0.0351 | 0.0210 | 0.6347 | 1 |
| COMPAS | non-IID | 65.26 | 0.0635 | 0.0254 | 0.6082 | 3 |

`b008_cal003_q81`:

| Dataset | Distribution | ACC | AEOD | ASPD | Score | Score rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Adult | IID | 81.08 | 0.0029 | 0.0553 | 0.7817 | 3 |
| Adult | non-IID | 81.54 | 0.0116 | 0.0609 | 0.7791 | 4 |
| COMPAS | IID | 64.67 | 0.0565 | 0.0314 | 0.6027 | 3 |
| COMPAS | non-IID | 66.13 | 0.0244 | 0.0150 | 0.6416 | 1 |

## Interpretation

- Adaptive mode is useful, but not automatically superior for every dataset/distribution.
- A single static AD2+ parameterization still leaves one or two FedSA cells outside rank 1.
- The clean-root profile-bank code path was added and tested, but the first profile-bank seed123 run did not outperform the best static adaptive candidates.
- The next productive direction is to make the profile-bank selection itself more principled, likely by using a root-validated multi-objective selector across both aggregation profiles and calibration profiles, then rerun 3 seeds for the final single AD2+ configuration.

## Current Status

The broader objective is not fully complete yet because the evidence does not prove that one final AD2+ configuration is best across all FedSA settings, and the synthetic/root-data COMPAS `10% real clean` setting still was not top 2 in the earlier joint summary.
