# F20 clear response: bounded editorial update

Author-review draft only; no manuscript application or final-test completion. The original 24 comments and their order are unchanged. No new statistics, arrays, fits, inference or literature searches were used.

Sources:
- `E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_clear_A100_20261011/rebuttal_clear_20261011.md`
  SHA256: `e3ed0f76c452e6f6a0ac03922802f46ccb7ab6aae336631f48f0e0400f9ec452`
- `E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/ROOT_VERIFICATION.json`
  SHA256: `9d222cea9e55a00442ff278eb47fa1b622938611049c44d0565644047a8894ac`
- `E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/TABLES.md`
  SHA256: `afce33a5ae59f5814c7b2164ce91638813acd82aa8bf8b475351ac41f46e668c`

## Edit 1: AE: add only the two accepted F scenes

Before:

Individual-component evidence includes complete U- and C-deletion comparisons and ten complete A-deletion scenes.

After:

Individual-component evidence includes complete U-, C- and A-deletion comparisons, plus F-deletion results for IID Benign and F Flip, each with ten paired seeds.

## Edit 2: AE: distinguish partial F coverage from all-control completion

Before:

The other five image controls, broader method comparability, synthetic provenance, and frozen final evaluation remain unresolved.

After:

Completion of the other five image controls, including the remaining F-deletion scenes, broader method comparability, synthetic provenance, and frozen final evaluation remain unresolved.

## Edit 3: R3.2: exact F20 scope, same-checkpoint pairing and retained calibration

Before:

The image study separately checks U, C and A using matched model seeds and round-70 checkpoints. U and C each cover all ten IID/non-IID–scenario cells with 100 deletion/Full pairs. A now also covers all ten cells with 100 pairs. The [A table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/TABLES.md) includes every raw/native/shared view and fixed 10/9/6-seed panel. Non-IID A Sp-DFA now has all ten seeds. The other five image controls remain unfinished; this response is not an all-component completion claim.

After:

The image study separately checks U, C and A using matched model seeds and round-70 checkpoints. U and C each cover all ten IID/non-IID–scenario cells with 100 deletion/Full pairs. A now also covers all ten cells with 100 pairs. The [A table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/TABLES.md) includes every raw/native/shared view and fixed 10/9/6-seed panel. Non-IID A Sp-DFA now has all ten seeds. The added [F table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/TABLES.md) covers IID Benign and F Flip (frozen Dirichlet α=5000), with ten matched Full/deletion pairs per scene and the same raw/native/shared views and fixed 10/9/6-seed panels. All three metrics for a model come from one round-70 checkpoint. F deletion retains prediction calibration; native/shared outputs coincide here and are not independent confirmations. The other eight F scenes and the other four image controls remain unfinished; this is not F100 or an all-component completion claim.

## Edit 4: R3.2: actual F20 evaluation-device boundary, without treating it as an isolated mechanism intervention

Before:

These experiments do not support calling every term indispensable. All 12 COMPAS deletion conditions have higher mean accuracy than Full, and six improve all three means. R3.7 discusses these counterexamples and the prediction-rule dependence. Mixed historical environments and replay devices limit causal attribution; deleting a score term is not an isolated removal of every related pipeline operation.

After:

These experiments do not support calling every term indispensable. All 12 COMPAS deletion conditions have higher mean accuracy than Full, and six improve all three means. R3.7 discusses these counterexamples and the prediction-rule dependence. For these two F scenes, the reused Full predictions comprise 2 CPU and 18 GPU evaluations, whereas all 20 F-deletion evaluations use CPU; the selected models all retain their recorded Torch 2.11.0+cu128 training provenance. Mixed historical environments and replay devices limit causal attribution; deleting a score term is not an isolated removal of every related pipeline operation.

## Edit 5: R3.7: F Flip ten-seed benefit and nine/six accuracy reversal; Benign negative evidence and selection history

Before:

Correlated scores, compensation by candidate selection and root-estimation variability are plausible explanations, not isolated causes. The supplied discussion presents both datasets and the raw/calibrated comparisons, and narrows the claim to conditional trade-offs. No best Full seed is compared with deletion means.

After:

The new IID F-deletion comparison reinforces this conditional interpretation. Under F Flip, Full has higher mean ACC and lower AEOD and ASPD in the ten-seed panel for all three prediction views. Native/shared paired differences (minus_F−Full, mean±sample SD) are ACC −0.066±1.239 percentage points, AEOD +0.00372±0.00942 and ASPD +0.00532±0.02066. In the fixed nine- and six-seed panels, deletion instead has higher mean ACC in both raw and calibrated views, while Full retains lower mean disparities.

In IID Benign, native/shared ten-seed deletion improves mean ACC and AEOD but worsens ASPD; raw deletion improves ACC while worsening both disparities. The six-seed raw panel favors deletion on all three means. We retain these reversals, not just the favorable F Flip panel. AEOD is the absolute TPR gap, not full equalized odds. Seed91001 was used in recipe selection; removing it for the nine-seed panel or retaining seeds91005–91010 for the six-seed panel does not create an untouched confirmation set. These scene-specific descriptive results support neither universal necessity nor statistical significance; no cross-scene F aggregate is reported.

Correlated scores, compensation by candidate selection and root-estimation variability are plausible explanations, not isolated causes. The supplied discussion presents both datasets and the raw/calibrated comparisons, and narrows the claim to conditional trade-offs. No best Full seed is compared with deletion means.

## Edit 6: R3.9: describe current draft scope without claiming the historical Git snapshot contains F20

Before:

The current draft adds complete A100 evidence while retaining the A90 findings;

After:

The current draft adds complete A100 and two-scene F-deletion evidence while retaining the A90 findings;

## Edit 7: P2: accepted mechanism cutoff 320 and remaining 480; distinguish completion of three controls from partial F20

Before:

| P2 — Image mechanisms | Finish the eight-control CelebA study: 800 new runs with 100 explicitly reused Full controls. At this draft's accepted cutoff, 300 new models have native and three-view evidence. U100, C100 and A100 each cover all ten cells. Finish the other five controls, then produce their complete matched-seed tables. |

After:

| P2 — Image mechanisms | Finish the eight-control CelebA study: 800 new runs with 100 explicitly reused Full controls. At this draft's accepted cutoff, 320 new models have native and three-view evidence: U100, C100 and A100 each cover all ten cells, and F20 covers IID Benign and F Flip with ten paired seeds per scene. The remaining 480 control runs comprise 80 F-deletion runs and 100 runs for each of four other controls. Complete their accepted evidence and matched-seed tables; the mechanism study is not finished. |

## Edit 8: Supporting material: direct adopted F20 table link

Before:

- [Complete U-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md), [complete C-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_full100_20261010/snapshot/TABLES.md) and [complete ten-scene A-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/TABLES.md).

After:

- [Complete U-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md), [complete C-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_full100_20261010/snapshot/TABLES.md) and [complete ten-scene A-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/TABLES.md).
- [F-deletion: IID Benign and F Flip, matched raw/native/shared tables](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_two_scenes20_20261011/TABLES.md).
