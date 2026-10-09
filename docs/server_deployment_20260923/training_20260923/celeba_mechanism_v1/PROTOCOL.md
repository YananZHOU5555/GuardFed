# CelebA full mechanism coverage — prepared 2026-10-09

Status: PREPARED, NOT DISPATCHED. The user explicitly requires completing all rebuttal experiments. Existing accepted cohorts remain immutable. The current server's SSH endpoint is refusing connections; this document and its prepared manifest are not evidence of a running queue or of scientific results. Freeze dispatch identities only after live source/data/reused-model checks and image gates pass.

## Question and full paired design

Determine the contribution of the actual AD2+ scoring, norm treatment, filtering and candidate selection on the image task. The strongest competing explanation is that prediction calibration or a subset of mechanisms explains the apparent advantage. Preserve negative effects and dataset dependence; no assertion that every component is necessary.

Nine variants × IID(alpha5000)/non-IID(alpha5) × Benign/F Flip/FedSA/S-DFA/Sp-DFA × shared seeds91001..91010 =900 model records. Reuse all100 accepted StageA Full models. Train only eight missing controls,800 jobs,70rounds. Full is not retrained for the formal table. Pipeline-only3round Full gates use the same horizon as their unchanged-worker regression references and do not count as formal runs.

Keep selected recipe lr0.0005_drop0.005, official train162770/valid19867, RGB64 CNN,20clients/four nominal malicious clients,root10%,batch64,one local Adam epoch and deterministic FP32. All numerical/data settings inherit the accepted Full recipe. Apart from variant/seed/distribution/attack and bookkeeping, no tuning is performed on the ablations. Every formal condition uses the same ten seeds; no selected Full seed versus ablation means.

## Exact controls

| Variant | Intervention |
|---|---|
| Full | Unchanged frozen core; reuse100 accepted records |
| minus_U/C/A/F/V | Existing component mask after every candidate override, applied to all ten candidates; verify the deleted contribution is zero on every call |
| minus_N | Existing N mask: cancel both root-norm rescaling and adaptive clipping by setting all selected-update scales to1; soft weights remain |
| no_hard_screen | Remove the intermediate distance/alignment gate AND top-k/min-keep truncation; retain all clients for soft weights and existing norm processing in every candidate |
| fixed_balanced | Use the predeclared original balanced lens; remove the adaptive candidate search and its candidate root evaluations. Balanced coefficients are fixed before collecting results, not selected using ablation performance |

The intermediate gate is not an absolute exclusion guarantee. The original implementation puts a negative sentinel at gate-excluded positions, then selects top-k over all clients. If k exceeds the gate size, excluded positions can re-enter; the final softmax uses original scores. Full and the score/norm controls preserve this existing behavior. The no_hard_screen intervention removes the complete selection block, rather than silently repairing the original method. See the independently restored Adult score audit for observed re-entry; it is not a CelebA result.

The core source bytes are not edited. A worker-local adapter makes the two missing controls explicit. No-hard-screen compiles the original function after replacing the one verified selection block; source/adapter hashes prevent silent upstream drift. Fixed-balanced also changes the number of inference-loader iterations and therefore potentially the later RNG stream; this is part of removing the actual selector, not proof of an isolated coefficient-only causal effect. Disclose it and compare paired full procedures rather than claiming identical local-update trajectories across variants.

## Gates, execution and recovery

Local CPU gates cover unchanged-Full output/diagnostics/RNG/function identity, all six masks across all ten candidates, all-client inclusion, fixed-balanced equivalence/no root candidate evaluation, and rejection/restoration after invalid controls. These are component evidence, not full-image pipeline or CUDA equivalence.

Before GPU dispatch: read current server guide; verify frozen core/data/config identities; verify100 reused Full model/result identities and numeric recipe equivalence; ensure no duplicate worker or partial attempt. Run18 separate three-round real-image gates (all nine variants × both distributions, S-DFA,seed91001). Compare Full against unchanged-worker references at the same three-round horizon, including model tensors, all metrics/diagnostics and RNG when available. Inspect attack audit and finite model/metrics for each intervention. Candidate-level ledger must contain30 checked aggregates per3round mask/no-screen gate and3 for fixed-balanced; Full remains unwrapped. Store gate/source/job/checkpoint identities before freezing a dispatch receipt.

Reuse the existing bounded fail-stop queue pattern, eight processes across the two GPUs with one CPU thread each. Preserve existing completed results; stop new dispatch on the first error while active jobs finish. Never overwrite partial directories or failure evidence. No automatic midround recovery guarantee is added; external recovery needs the original source/data/config/job identities and validated accepted-result skips. No driver/instance changes or resource purchases.

## Acceptance, calibration and reporting

For each job verify exact requested variant/config/source/data/adapter/job/protocol identity; all trajectory and diagnostic rounds1..70; the terminal checkpoint SHA; finite tensors and all three metrics from that same terminal checkpoint; valid19867/train162770/disjointness and prediction/group support. Verify every candidate mask/admission call, not only the selected candidate. Preserve failures, constant predictions, empty-group limitations and unfavorable results.

Report each distribution/scene/variant with ten-seed mean and sampleSD(ddof1), plus paired per-seed differences to Full. Cross-scene averages are first formed within seed. ACC is percent; AEOD is the implemented absolute TPR gap, not full equalized odds; ASPD is absolute positive-rate disparity. No synthetic score substitutes for raw metrics or determines which seeds enter the table.

After acceptance, reuse all900 terminal models for raw/native and a common train-root-only calibration control. Cache margins and sample/group identities once; afterprocessing adds no training. The accepted100 Full shared-calibration results can be reused where identities and common calibration match. This is required to distinguish scoring/filtering effects from group-threshold effects. Test labels must not tune recipes or thresholds.

Incrementally back up only newly accepted IDs, including weights/raw results/config/logs/candidate audit and acceptance, with archive SHA/member hashes and an off-server restore chain. Keep old100 model chains unchanged. This stage's800 jobs do not include the separate missing-baseline800-record comparison or frozen final evaluation; completing one does not complete the whole rebuttal.
