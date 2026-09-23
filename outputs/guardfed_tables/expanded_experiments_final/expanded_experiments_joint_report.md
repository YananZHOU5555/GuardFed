# GuardFed-AD2+ expanded advisor experiments

This report is generated from the raw experiment log without manual value edits. The joint-last10 rule selects one actual round from the last 10 rounds for each run, then reports ACC, AEOD, and ASPD from that same round.

## Completed runs
- expanded_synthetic_ratios: 840 rows
- expanded_server_dist30_labelpreserve: 720 rows
- fedsa_all_methods: 264 rows

## Key rule
- Joint score: `ACC - 0.5 * (AEOD + ASPD)`.
- Fairness ranking gate: Adult ACC >= 80%, COMPAS ACC >= 60%; below the gate, AEOD/ASPD may be reported but should not win fairness ranking.
- Display floor: values below 0.0001 are shown as 0.0001 in paper-style tables; exact raw values remain in CSV.

## Synthetic/root-data result
### adult
| Rank | Setting | ACC | AEOD | ASPD | Score |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 9% real + 1% tvae | 82.03 | 0.0050 | 0.0737 | 0.7809 |
| 2 | 10% real clean | 82.29 | 0.0091 | 0.0750 | 0.7809 |
| 3 | 7% real + 3% forest_diffusion | 82.54 | 0.0102 | 0.0802 | 0.7802 |
| 4 | 9% real + 1% forest_diffusion | 81.97 | 0.0065 | 0.0737 | 0.7796 |
| 5 | 2% real + 8% forest_diffusion | 81.89 | 0.0077 | 0.0712 | 0.7795 |
- 10% real clean rank: score 2, ACC 10, AEOD 9, ASPD 22.

### compas
| Rank | Setting | ACC | AEOD | ASPD | Score |
| --- | --- | ---: | ---: | ---: | ---: |
| 1 | 7% real + 3% gaussian_copula | 66.21 | 0.0374 | 0.0149 | 0.6359 |
| 2 | 7% real + 3% forest_diffusion | 66.13 | 0.0349 | 0.0227 | 0.6324 |
| 3 | 3% real + 7% gaussian_copula | 65.50 | 0.0159 | 0.0302 | 0.6320 |
| 4 | 2% real + 8% forest_diffusion | 65.76 | 0.0140 | 0.0397 | 0.6307 |
| 5 | 5% real + 5% smote | 64.50 | 0.0169 | 0.0123 | 0.6304 |
- 10% real clean rank: score 25, ACC 21, AEOD 30, ASPD 21.

## FedSA all-method AD2+ ranks
- adult IID: ACC=82.09 (rank 14), AEOD=0.0024 (rank 1), ASPD=0.0746 (rank 5), score rank=1.
- adult non-IID: ACC=82.37 (rank 19), AEOD=0.0102 (rank 4), ASPD=0.0745 (rank 3), score rank=1.
- compas IID: ACC=65.06 (rank 19), AEOD=0.0835 (rank 2), ASPD=0.0446 (rank 1), score rank=2.
- compas non-IID: ACC=65.37 (rank 19), AEOD=0.0764 (rank 2), ASPD=0.0396 (rank 2), score rank=2.

## Files
- `joint_selected_all_raw.csv`
- `synthetic_joint_summary.csv`
- `synthetic_joint_by_dataset_summary.csv`
- `fedsa_joint_summary.csv`
- `fedsa_paper_style_joint.md`
- `server_dist30_joint_summary.csv`