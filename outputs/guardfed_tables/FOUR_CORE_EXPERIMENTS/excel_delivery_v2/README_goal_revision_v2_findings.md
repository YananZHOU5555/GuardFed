# GuardFed-AD2+ four-experiment revision v2 findings

## Files

- `GuardFed_AD2plus_01_Ablation_v2_RealRuns.xlsx`
- `GuardFed_AD2plus_02_Server_Distribution_ControlledSkew_v2.xlsx`
- `GuardFed_AD2plus_03_Synthetic_10pct_v2.xlsx`
- `GuardFed_AD2plus_04_New_Performance_Attack_FedSA.xlsx`

All workbook values are copied from completed CSV/raw-result summaries. No reported cell value was manually edited.

## 01 Ablation

New evidence: 72 real 30-round diagnostic runs:

- 2 datasets: Adult, COMPAS
- 2 distributions: IID, non-IID
- 3 attacks: FedSA, F Flip, S-DFA
- 6 profiles: full, no utility, no fairness risk/violation, no geometry, utility-only, fairness-only

Main trends:

- `fairness_only_FV` lowers ACC in 4/6 dataset-attack slices, mean delta ACC = -0.80 pp.
- `no_utility_U` lowers ACC in 4/6 slices, mean delta ACC = -0.37 pp, and worsens FairAvg in 3/6 slices.
- `no_fairness_FV` worsens FairAvg in 4/6 slices, especially on COMPAS and Adult FedSA/S-DFA.
- `utility_only_U` is useful as the utility-only reference, but worsens FairAvg in 3/6 slices.

Interpretation: v2 is clearer than the old flat ablation, but the clean claim should be component-specific rather than "every metric worsens in every slice."

## 02 Server/root distribution

New evidence:

- v2 `controlled_group_skew`: 80 real 30-round runs with group TVD/KL audit.
- v3 `controlled_positive_sensitive_skew`: 64 real 20-round fairness-stress runs.
- v4 `controlled_target_group_skew`: 192 real 15-round diagnostic runs over target sensitive-label strata `(S,Y) in {(0,0),(0,1),(1,0),(1,1)}`.

Main trends:

- In v2, all 8 dataset/distribution/attack slices have negative correlation between root group TVD and ACC before validity filtering. This supports: less IID-like clean server/root data hurts performance.
- COMPAS fairness is sensitive to server/root distribution; several slices show AEOD/ASPD worsening as group TVD increases.
- v4 gives the clearest target-stratum evidence on COMPAS when skew targets `S=1,Y=0`: ACC decreases as root TVD rises, while AEOD/ASPD rise on valid rows.
- Adult IID slices also show the intended trend for selected targets, but Adult non-IID remains mixed. Under high skew, ACC can collapse and AEOD/ASPD can become artificially small, so those rows are marked invalid and should not be used as "fairness improved."

Interpretation: the honest paper claim is strong for ACC sensitivity, clear for COMPAS fairness sensitivity, and conditionally visible for Adult IID under targeted root mismatch. Adult non-IID should be described with a validity gate, not as a strict monotonic curve.

## 03 10% clean server data / synthetic generation

Evidence included:

- Dense 1%-10% real clean server/root ratio table.
- Existing generation ablation: 10% real clean, 1% real + 9% synthetic, 5% real + 5% synthetic with Gaussian Copula, CTGAN, TVAE, SMOTE, forest diffusion, and PCA Gaussian.

Main trends:

- 10% real clean remains the clean final setting for the main AD2+ table.
- Synthetic data can improve some COMPAS settings, but results vary strongly by generator and dataset.
- The workbook includes both exact tables and clean-ratio charts.

## 04 New performance attack

Evidence included:

- FedSA replaces FOE as the new performance-side attack in the additional attack workbook.
- The workbook keeps the same validity principle: fairness metrics from collapsed/near-untrained states are not counted as fairness wins.
- Use this file as the fourth Excel result for the reviewer-facing "FOE replaced by a new performance attack" experiment.
