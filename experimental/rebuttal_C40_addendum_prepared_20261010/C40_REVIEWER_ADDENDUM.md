# Author-review addendum: C deletion in four IID CelebA scenes

**AUTHOR_REVIEW / DO_NOT_SUBMIT. Separate evidence addendum; not applied to the submitted manuscript.**

This short extension supplements R3.2/R3.7 and the [accepted C30 addendum](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_C30_addendum_20261009/C30_REVIEWER_ADDENDUM.md). The [complete 24-comment response](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C20_20261009/rebuttal_integrated_20261009.md) and its original comments, other experiments and pending author decisions remain unchanged. The [adopted four-scene table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_four_scenes_20261010/snapshot/TABLES.md) pairs 40 minus_C checkpoints with 40 historical Full checkpoints: 80 unique records in IID Benign, F Flip, FedSA and S-DFA. It is C40, not C100.

The [root verification](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_four_scenes_20261010/ROOT_VERIFICATION.json) and [independent arithmetic review](E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_C40_root_arithmetic_review_20261010/ROOT_ARITHMETIC_REVIEW.json) bind 648 mean/SD scalars, 324 display cells and 720 record-view metric values. The original 60 C30 records, 486 statistics and 243 cells are unchanged; the six previously partial S-DFA records are retained and joined with four newly accepted records. These verification counts are not new independent experiments.

## R3.2 — Individual components

**Original comment (verbatim, unchanged).**

> 2. The ablation study better isolates the contribution of each individual component of GuardFed. The current ablation study groups several components together. For example, C and A, as well as F and V are removed together, it is difficult to determine the individual contribution of each component. So, I suggest evaluating each component separately, or providing further justification for why these components are evaluated jointly.

**Additional response.** IID S-DFA now supplies a fourth complete C-deletion scene. The mask removes C's score contribution after each candidate override while retaining geometric hard filtering and the other mechanisms; it does not remove prediction calibration. Each model's three metrics and raw/native/shared-calibration views use its same round-70 checkpoint. Native retains the original prediction rule, raw is uncalibrated, and shared uses the frozen common root-only rule. No model inference or threshold refitting was performed by the table builder or this writing update.

The S-DFA entries below are paired minus_C−Full differences, mean ± sample SD (ddof=1), for the same matched seeds in both procedures. ACC is percent and ΔACC is in percentage points (pp). AEOD is the absolute TPR gap, not full equalized odds; AEOD/ASPD are on [0,1], with smaller gaps indicating lower measured disparity.

| View | Matched seeds | ΔACC (pp) | ΔAEOD | ΔASPD |
|---|---:|---:|---:|---:|
| Native | 10 | -0.077 ± 0.715 | +0.00050 ± 0.00760 | +0.00022 ± 0.01411 |
| Native | 9 | -0.154 ± 0.714 | +0.00146 ± 0.00739 | +0.00026 ± 0.01496 |
| Native | 6 | +0.022 ± 0.770 | +0.00051 ± 0.00843 | +0.00646 ± 0.01382 |
| Raw | 10 | -0.035 ± 0.715 | +0.00200 ± 0.00936 | +0.00144 ± 0.01139 |
| Raw | 9 | -0.137 ± 0.677 | +0.00315 ± 0.00915 | +0.00067 ± 0.01180 |
| Raw | 6 | -0.070 ± 0.660 | +0.00259 ± 0.00937 | +0.00095 ± 0.01126 |
| Shared calibration | 10 | -0.077 ± 0.715 | +0.00050 ± 0.00760 | +0.00022 ± 0.01411 |
| Shared calibration | 9 | -0.154 ± 0.714 | +0.00146 ± 0.00739 | +0.00026 ± 0.01496 |
| Shared calibration | 6 | +0.022 ± 0.770 | +0.00051 ± 0.00843 | +0.00646 ± 0.01382 |

Deleting C lowers S-DFA ACC and raises both disparity means in every ten- and nine-seed view. In the six-seed native/shared panel its ACC difference becomes positive while both gap differences remain positive; the six-seed raw panel still has lower ACC and larger gaps. This ACC mean-direction reversal is retained. It does not establish C necessity or a causal mechanism. Native and shared metrics and saved group counts coincide for all 80 records; they are parallel views, not independent repetitions or an additional calibration gain.

## R3.7 — Retained counterexamples and scope

**Original comment (verbatim, unchanged).**

> 7. The ablation results require further analysis. For example, when the reward terms U, C, and A are removed, the Adult non-IID accuracy decreases, whereas the corresponding COMPAS accuracy is slightly higher than that of Full GuardFed. The manuscript currently mainly emphasizes the Adult result. The authors should explain this dataset-dependent behavior.

**Additional response.** The S-DFA pattern complements rather than removes the earlier counterexamples. For F Flip, the ten-seed native/shared paired ACC/AEOD/ASPD differences remain +0.518 ± 0.814 / +0.00139 ± 0.01487 / +0.01098 ± 0.02176: deletion raises accuracy while worsening both gaps. Its native/shared AEOD difference is +0.00025 ± 0.01530 for nine seeds and -0.00291 ± 0.01773 for six seeds. FedSA retains its ten-seed native/shared ACC-up/AEOD-down/ASPD-up trade-off, the nine-/six-seed AEOD-up/ASPD-down directions, and the raw-nine-seed counterexample in which deleting C improves all three means. FedSA raw-six-seed deletion raises ACC and both gaps. Benign's ten-seed native/shared ACC-down/AEOD-up/ASPD-down trade-off and all other panel results remain in the linked tables. No significance, universal component benefit, necessity or causal explanation is claimed.

## Comparison boundary

The ten-seed panel includes selection seed 91001; the nine-seed panel excludes it, and the six-seed panel retains 91005–91010, identically for Full and minus_C. Prior validation and official-test exposure remain disclosed. These results use 19,867 validation images, not an untouched test or confirmation set; the final test under a frozen final protocol has not been run. The native/shared primary endpoint and final evaluation remain pending author decisions.

Full replay uses three CPU and 37 GPU records; all 40 minus_C replays use CPU. Both training cohorts contain 40 checkpoints from PyTorch 2.11.0+cu128. Historical U100 and nine-method cu130/CUDA/driver disclosures remain in force; historical/current driver equality was not established. Mixed replay devices and selection history limit attribution.

Only these four IID C scenes are complete here; six other C scenes remain incomplete. There are eight planned image-control variants: U is complete, C is partial, and six other image-control variants remain incomplete. The accepted complete U100 study and the prior tabular/benchmark negative findings remain unchanged. This extension does not certify whole-mechanism or whole-rebuttal completion.

[Every quoted scalar and scope fact](E:/OneDrive/文档/GuardFed/tmp/rebuttal_C40_addendum_prepared_20261010/SOURCE_POINTERS.json) is bound to the actual files; the [local writing checker](E:/OneDrive/文档/GuardFed/tmp/rebuttal_C40_addendum_prepared_20261010/check_sources.py) checks pointers, display values, comments, seed/device/scope facts and links. It references the accepted arithmetic review without rerunning its statistical audit. The companion [candidate manuscript paragraphs](E:/OneDrive/文档/GuardFed/tmp/rebuttal_C40_addendum_prepared_20261010/C40_MANUSCRIPT_INSERTIONS.md) remain an unapplied author-review copy.
