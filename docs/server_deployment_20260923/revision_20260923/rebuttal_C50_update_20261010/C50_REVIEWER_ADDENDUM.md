# Author-review addendum: C deletion across five IID CelebA scenes

**AUTHOR_REVIEW / DO_NOT_SUBMIT_BEFORE_FULL_COHORT. Separate candidate; not applied to the submitted manuscript or the complete 24-comment response.**

This update uses the [adopted C50 snapshot](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010/snapshot/TABLES.md), accepted on 9 October 2026 UTC ([root verification](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010/ROOT_VERIFICATION.json)). It pairs 50 minus_C checkpoints with 50 historical Full checkpoints in IID Benign, F Flip, FedSA, S-DFA and Sp-DFA. The previous four-scene records and tables are unchanged. The [complete response](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C20_20261009/rebuttal_integrated_20261009.md), including the tabular and COMPAS counterexamples, remains unchanged.

## R3.2 — Individual components

**Original comment (verbatim).**

> 2. The ablation study better isolates the contribution of each individual component of GuardFed. The current ablation study groups several components together. For example, C and A, as well as F and V are removed together, it is difficult to determine the individual contribution of each component. So, I suggest evaluating each component separately, or providing further justification for why these components are evaluated jointly.

**Additional response.** The fifth complete C-deletion scene, IID Sp-DFA, shows a trade-off. The intervention removes C's score contribution after each candidate override while retaining geometric hard filtering and prediction calibration. All metrics and views for each model use its same round-70 checkpoint. Native retains the original prediction rule; raw is uncalibrated; shared calibration uses the frozen common root-only rule. Deletion is therefore not a removal of calibration or an isolated intervention on every aggregation operation.

The Sp-DFA entries below are paired minus_C−Full differences for ten matched seeds, mean ± sample SD (ddof=1). ACC is percent and ΔACC is in percentage points (pp); AEOD/ASPD are on [0,1], with smaller gaps better. AEOD is the absolute TPR gap, not full equalized odds.

| View | ΔACC (pp) | ΔAEOD | ΔASPD |
|---|---:|---:|---:|
| Native / shared calibration | +0.042 ± 0.796 | -0.00668 ± 0.00848 | +0.00810 ± 0.02432 |
| Raw | +0.097 ± 0.909 | +0.00523 ± 0.01289 | +0.00392 ± 0.01422 |

Deleting C raises mean ACC in both views, but only native/shared lowers AEOD; ASPD increases in each view. The native/shared ACC difference is +0.134 ± 0.786 for nine seeds and -0.066 ± 0.881 for six seeds. This mean-direction reversal is retained, alongside lower native/shared AEOD and higher ASPD in both subsets. Raw retains ACC-up/AEOD-up/ASPD-up in both subsets. These descriptive differences do not establish necessity, significance or a pure aggregation causal effect.

## R3.7 — Counterexamples and scope

**Original comment (verbatim).**

> 7. The ablation results require further analysis. For example, when the reward terms U, C, and A are removed, the Adult non-IID accuracy decreases, whereas the corresponding COMPAS accuracy is slightly higher than that of Full GuardFed. The manuscript currently mainly emphasizes the Adult result. The authors should explain this dataset-dependent behavior.

**Additional response.** The new scene reinforces the need to report conditional trade-offs. The earlier F Flip ten-seed native/shared accuracy gain with larger gaps, its nine-to-six-seed AEOD reversal, and the Benign counterexample remain. FedSA retains its native/shared subset reversals, the raw-nine-seed all-three-means improvement after deletion, and the raw-six-seed increase in both gaps. S-DFA retains its native/shared six-seed ACC reversal. None of these unfavorable or reversed outcomes is removed from the linked tables. Native/shared metrics and saved group counts coincide for all 100 displayed records; they are parallel views, not independent repetitions or evidence of an additional calibration gain.

The [cross-scene summary](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010/snapshot/cross_scene_seed_first.json) first averages the five IID scenes within each seed, then summarizes across seeds. Its ten-seed native/shared mean also trades lower AEOD for higher ASPD after deletion; raw has larger means for both gaps. This does not imply a uniform component benefit across scenes, datasets or metrics.

## Comparison boundary

The fixed ten-seed panel includes selection seed 91001; the nine-seed panel excludes it; the six-seed panel retains 91005–91010, identically for Full and minus_C. Prior validation and official-test exposure remain disclosed. Evaluation uses 19,867 validation images, not an untouched test or confirmation set; the final test under a frozen final protocol has not been run. The native/shared primary endpoint remains pending author decision.

Full replay uses three CPU and 47 GPU records; all 50 minus_C replays use CPU. Both training cohorts use PyTorch 2.11.0+cu128, but historical/current driver equality was not established. Historical U100 and nine-method cu130/CUDA disclosures remain in force. Mixed replay devices, validation selection and prior exposure limit attribution; this is not a uniform-device final comparison.

Only the five IID C scenes are complete in this snapshot; the five non-IID C scenes and six other image-control variants remain incomplete. The accepted U100 study and earlier negative results remain unchanged. This addendum neither completes the whole mechanism study nor selects a final endpoint. No training, inference, threshold refitting or new statistical analysis was performed for this writing update. [Exact quoted sources and JSON pointers](E:/OneDrive/文档/GuardFed/tmp/rebuttal_C50_update_prepared_20261010/SOURCE_POINTERS.json) and the [local source/link checker](E:/OneDrive/文档/GuardFed/tmp/rebuttal_C50_update_prepared_20261010/check_sources.py) accompany the candidate [manuscript insertions](E:/OneDrive/文档/GuardFed/tmp/rebuttal_C50_update_prepared_20261010/C50_MANUSCRIPT_INSERTIONS.md).
