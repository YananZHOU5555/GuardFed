# Accepted evidence addendum: F-deletion and the hybrid control

Author-review text for R3.2, R3.7 and R3.8. This addendum extends the accepted F20 draft; it does not replace the reviewers' 24 original comments or claim that the submitted manuscript has been edited. All results below are validation results, not a frozen final-test evaluation.

## R3.2 and R3.7 — Individual components and conditional effects

The F-deletion comparison now covers three complete IID scenarios—Benign, F Flip and FedSA—with ten matched Full/deletion seeds per scenario. We retain raw, native and shared-calibration predictions from the same round-70 checkpoint for each model. The [complete table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_three_scenes30_20261011/TABLES.md) reports mean ± sample SD (ddof=1), paired differences and the fixed 10/9/6-seed sensitivity panels. The previous Benign and F Flip records and statistics remain unchanged.

For IID FedSA, the ten-seed native results are:

| Variant | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---:|---:|---:|---:|
| Full | 10 | 88.470 ± 0.909 | 0.00607 ± 0.00416 | 0.06402 ± 0.01425 |
| Remove F | 10 | 88.606 ± 1.323 | 0.00929 ± 0.00785 | 0.06794 ± 0.01401 |
| Paired difference: remove F − Full | 10 | +0.136 ± 1.125 pp | +0.00322 ± 0.00911 | +0.00393 ± 0.01621 |

Removing F slightly increases mean accuracy while increasing both measured disparities in the ten-seed panel. This direction also occurs in the nine-seed native/shared panel and all three raw panels. In the six-seed native/shared panel, however, removal increases ACC and AEOD but decreases ASPD (paired mean −0.00560). We preserve that reversal. Together with the earlier Benign and F Flip results, the evidence supports conditional utility–disparity trade-offs, rather than universal necessity of F. These are descriptive paired means; they do not establish significance or a consistent effect in every seed.

Deleting F removes its scoring contribution while retaining prediction calibration. Native and shared-calibration metrics and group counts coincide for these records, so they are not independent confirmations. Full predictions comprise 2 CPU and 28 GPU evaluations; all 30 deletion predictions use CPU. The selected models retain their recorded Torch 2.11.0+cu128 training provenance. Exposed validation, the selection history of seed91001, historical test exposure and mixed replay devices limit causal and confirmatory interpretations. AEOD here is the absolute TPR gap, not full equalized odds.

The accepted three-view mechanism coverage is now U100 + C100 + A100 + F30 = 330 deletion checkpoints, with Full controls explicitly reused. Seven F scenarios and four other controls remain incomplete: 470 planned mechanism-control checkpoints still lack complete accepted three-view evidence. Native training acceptance is tracked separately at 333/800; those three additional native checkpoints are not counted as completed three-view comparisons.

## R3.8 — Postprocessing and baseline adaptations

The [hybrid control's IID Benign table](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_hybrid_three_view_IID_Benign10_20261011/TABLES.md) now includes ten checkpoints and the same three prediction views. Native/raw ten-seed means are ACC 88.027 ± 1.386%, AEOD 0.02920 ± 0.00903 and ASPD 0.09251 ± 0.01410. Shared calibration yields 87.708 ± 1.270%, 0.02321 ± 0.01652 and 0.03383 ± 0.01892. All three fixed seed panels exhibit lower mean ACC together with lower mean disparities after shared calibration. We report this accuracy cost and do not attribute the entire disparity change to aggregation. Seed91001 remains the original selected screen checkpoint; the other nine are coverage checkpoints. The hybrid is a project control, not a faithful implementation of an external named method.

Huber uses the author-approved identity projection for the CNN and is labelled an empirical CNN project adaptation, without inheriting the original constrained-domain theoretical guarantee. LoGoFair uses 20 deterministic image-ID virtual cohorts with root-only fitting. These evaluate population-group behavior, not fairness among real training clients. Neither adaptation is presented as evidence for a stronger original-theory claim.

The remaining target-method coverage, other image controls, final evaluation boundary and submitted-manuscript integration are still pending. Existing negative and constant-prediction results remain in the evidence record.
