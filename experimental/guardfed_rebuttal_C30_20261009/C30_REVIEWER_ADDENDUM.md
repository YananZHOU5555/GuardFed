# Author-review addendum: CelebA C deletion in three IID scenes

**AUTHOR_REVIEW / DO_NOT_SUBMIT. Separate evidence addendum; not applied to the submitted manuscript.**

This addendum supplements R3.2 and R3.7 in the [frozen complete response](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C20_20261009/rebuttal_integrated_20261009.md). Its original reviewer comments, other experimental findings and pending author decisions remain unchanged. The accepted extension adds IID FedSA to the previously accepted IID Benign and IID F Flip scenes. It contains 30 minus_C checkpoints paired by scene and seed with 30 historical Full checkpoints: 60 unique records across exactly three complete IID scenes, not a C100 study.

The actual [three-scene table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_three_scenes_20261009/snapshot/TABLES.md) is accepted in the [root verification](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_three_scenes_20261009/ROOT_VERIFICATION.json); the [independent arithmetic review](E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_C30_root_arithmetic_review_20261009/ROOT_ARITHMETIC_REVIEW.json) checked 486 mean/SD scalars, 243 display cells and 540 record-view metric values. The original 40 two-scene records, 324 statistics and 162 cells are preserved. These are evidence-verification counts, not new independent experiments.

## R3.2 — Individual components

**Original comment (verbatim, unchanged).**

> 2. The ablation study better isolates the contribution of each individual component of GuardFed. The current ablation study groups several components together. For example, C and A, as well as F and V are removed together, it is difficult to determine the individual contribution of each component. So, I suggest evaluating each component separately, or providing further justification for why these components are evaluated jointly.

**Additional response.** We extend the image C-deletion evidence to a third complete scene, IID FedSA. The C mask removes its score contribution after each candidate override while retaining geometric hard filtering and the other mechanisms. It does not remove prediction calibration. This extends the existing individual-component evidence; it does not establish that C, or every remaining component, is necessary.

Each scene uses ten matched training seeds and one round-70 checkpoint per model. All three metrics and raw/native/shared-calibration views use that same checkpoint. The native view retains each procedure's original prediction rule; raw uses uncalibrated predictions and shared uses the frozen common root-only rule. No threshold refitting or model inference was performed by the table builder or this writing update. ACC is percent and ΔACC is in percentage points. AEOD denotes the absolute TPR gap, not full equalized odds; AEOD and ASPD are on [0,1], with smaller gaps indicating lower measured disparity.

The following IID FedSA paired differences are minus_C−Full, mean ± sample SD (ddof=1). All outcomes are retained.

| View | Matched seeds | ΔACC (pp) | ΔAEOD | ΔASPD |
|---|---:|---:|---:|---:|
| Native | 10 | +0.279 ± 0.498 | -0.00024 ± 0.00472 | +0.00099 ± 0.01311 |
| Native | 9 | +0.219 ± 0.489 | +0.00026 ± 0.00472 | -0.00233 ± 0.00832 |
| Native | 6 | +0.148 ± 0.569 | +0.00016 ± 0.00415 | -0.00593 ± 0.00634 |
| Raw | 10 | +0.235 ± 0.535 | -0.00132 ± 0.01034 | +0.00142 ± 0.01245 |
| Raw | 9 | +0.158 ± 0.505 | -0.00277 ± 0.00983 | -0.00072 ± 0.01109 |
| Raw | 6 | +0.206 ± 0.556 | +0.00033 ± 0.00842 | +0.00241 ± 0.01063 |
| Shared calibration | 10 | +0.279 ± 0.498 | -0.00024 ± 0.00472 | +0.00099 ± 0.01311 |
| Shared calibration | 9 | +0.219 ± 0.489 | +0.00026 ± 0.00472 | -0.00233 ± 0.00832 |
| Shared calibration | 6 | +0.148 ± 0.569 | +0.00016 ± 0.00415 | -0.00593 ± 0.00634 |

In the ten-seed FedSA native/shared panel, deleting C raises mean ACC, lowers AEOD slightly and raises ASPD. The native/shared nine- and six-seed panels instead have positive AEOD differences and negative ASPD differences. In the raw nine-seed panel, deletion improves all three means: ACC rises while both gaps fall. In the raw six-seed panel, ACC and both gaps rise. Thus the comparison exposes prediction-rule and subset dependence rather than a uniformly favorable component effect. Native and shared-calibration metrics and saved group counts coincide for all 60 records; these views are parallel presentations, not independent repetitions or an additional calibration gain.

## R3.7 — Dataset-dependent behavior and retained counterexamples

**Original comment (verbatim, unchanged).**

> 7. The ablation results require further analysis. For example, when the reward terms U, C, and A are removed, the Adult non-IID accuracy decreases, whereas the corresponding COMPAS accuracy is slightly higher than that of Full GuardFed. The manuscript currently mainly emphasizes the Adult result. The authors should explain this dataset-dependent behavior.

**Additional response.** The new FedSA scene reinforces the need to preserve trade-offs, alongside the existing COMPAS, U-deletion and C20 counterexamples. It does not isolate candidate compensation, score redundancy or root-estimation variability as causal explanations. The previous two-scene findings remain unchanged. For the ten-seed panel, paired ACC/AEOD/ASPD differences are:

| Scene and view | ΔACC (pp) | ΔAEOD | ΔASPD |
|---|---:|---:|---:|
| IID Benign, native/shared | -0.083 ± 1.378 | +0.00292 ± 0.01315 | -0.00142 ± 0.01953 |
| IID F Flip, native/shared | +0.518 ± 0.814 | +0.00139 ± 0.01487 | +0.01098 ± 0.02176 |
| IID F Flip, raw | +0.530 ± 0.875 | +0.00465 ± 0.01053 | +0.00703 ± 0.01157 |

For F Flip, deletion improves the ten-seed accuracy mean while worsening both disparity means. Its native/shared AEOD difference is +0.00025 ± 0.01530 for nine seeds and -0.00291 ± 0.01773 for six seeds. This previously reported mean-direction reversal is retained. The FedSA extension supplies another scene-specific trade-off, not evidence of universal component benefit. No significance, component-necessity or causal claim follows from these descriptive means.

## Comparison boundary and remaining scope

The ten-seed panel includes recipe-selection seed 91001. The nine-seed panel excludes it, and the six-seed panel retains 91005–91010; the same seed sets apply to Full and minus_C. Prior validation and official-test exposure remain disclosed. These valid-only results use 19,867 evaluation images and are not an untouched test or a final fairness comparison. Subsetting exposed data does not create an independent confirmation set. The native/shared primary endpoint and final evaluation remain pending author decisions.

For this C30 comparison, Full replay comprises two CPU and 28 GPU records; all 30 minus_C replays use CPU. Both training cohorts contain 30 checkpoints from PyTorch 2.11.0+cu128. These counts do not replace the historical U100 or nine-method CUDA/driver and cu130 disclosures in the frozen response; historical/current driver equality was not established. Mixed replay devices and the selection history limit attribution.

Only IID Benign, IID F Flip and IID FedSA are complete here. Six accepted IID S-DFA C records are retained in [the excluded-source file](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_three_scenes_20261009/snapshot/excluded_partial_C_records.json) and are excluded from every scene mean. The other seven C distribution–scenario cells and six other image-control variants remain outside these completed C30 conclusions. C and those six controls remain incomplete. The separately accepted complete U100 study, other experiments, all negative findings and the remaining benchmark/final-evaluation/manuscript-integration items in the frozen response remain unchanged. This addendum neither certifies whole-mechanism completion nor records an applied manuscript revision.

Every quoted mean/SD and scope fact is mapped in [SOURCE_POINTERS.json](E:/OneDrive/文档/GuardFed/tmp/guardfed_rebuttal_C30_20261009/SOURCE_POINTERS.json); the local [writing-source checker](E:/OneDrive/文档/GuardFed/tmp/guardfed_rebuttal_C30_20261009/check_sources.py) verifies those pointers and displayed values against the accepted files. It does not rerun training, inference or the complete statistical audit.
