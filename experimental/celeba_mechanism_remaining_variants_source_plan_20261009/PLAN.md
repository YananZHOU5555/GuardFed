# Remaining mechanism variants: source-only extension proposal

Status: **PREPARED_NOT_APPROVED / UNEXECUTED**. No SSH, training, CNN, test access, dispatch, new inventory, acceptance, resource allocation or Full inference. The bound input inventory has 82 native terminals, all minus_U; this is an input snapshot, not a fresh live progress claim. No other variant's actual terminal is supplied or accepted here. The in-flight after71 replay and its files are untouched.

## Frozen intervention mapping

| Actual variant | config.ablation_component | Actual source behavior | Original 70-round acceptance |
|---|---|---|---|
| minus_U | U | Zero standardized clean-root accuracy utility contribution after each candidate override | Every candidate mask verified; 700 audit calls, selected-round U=0 |
| minus_C | C | Zero standardized centrality contribution | 700 calls; selected-round C=0 |
| minus_A | A | Zero standardized update-to-root alignment contribution | 700 calls; selected-round A=0 |
| minus_F | F | Zero act_risk_weight × standardized fairness-risk contribution | 700 calls; selected-round F=0 |
| minus_V | V | Zero act_violation_weight × dual_lambda × standardized violation contribution | 700 calls; selected-round V=0 |
| minus_N | N | Set selected update rescaling to exactly1, bypassing both root norm scaling and adaptive clipping; retain soft weights | 700 calls; every stored norm scale1 |
| no_hard_screen | none | Replace whole distance/alignment gate and top-k/min_keep block with all-client gate/selection, every candidate; retain scores/soft weights/norm behavior | 700 calls; every candidate selected_count20; round gate/selected IDs exactly0..19 |
| fixed_balanced | none | Replace adaptive candidate selection with the original predeclared balanced override; one aggregation per round; no candidate root evaluation loop | 70 calls; each round mode fixed_balanced_without_candidate_selection and candidate balanced |

Full is reference-only, never one of the800 new jobs. Its mask is none; it remains the original adaptive method. Both structural controls also have mask none: a config flag alone cannot distinguish them from Full. Their job.variant, raw frozen job SHA, adapter source SHA, result.revision_job.variant, per-candidate ledger, per-round diagnostics and mechanism_acceptance identities remain required through the ORIGINAL acceptor.

Exact source pins and local paths are in INPUTS.json. Relevant source locations:
- Frozen deployment_snapshot/prepare.py: generated config update and exact800 manifest construction.
- Frozen adapter.py: mechanism, without_hard_screen, fixed, verify_ledger.
- Frozen worker.py: checked, original.checked_result, model finite tensors and audit/model/result/job SHA binding.
- Frozen core reproduce_paper_tables.py: lines1093–1145 component masks/norms/softweights,1188–1220 candidate overrides.
- evidence_v4.py: validate_design104–124; terminal_checks132–164; partition_identity167–182; accept_new185–204.

The balanced override is source-defined: aeod_aspd, risk.75, violation.20, keep.80, temperature.35, utility1.00, centrality.35, alignment.35; base score_clip0, norm_mode root, calibration_objective original, norm_clip_scale and max_acc_drop inherited from frozen config. No candidate is selected from new outcomes. Fixed balanced removes loader traversals for candidate evaluation and may therefore change later RNG trajectories; this is a paired procedural intervention, not an identical-update trajectory claim. Existing sentinel-gate/top-k re-entry behavior is preserved in every original method; no_hard_screen removes that entire selection block rather than repairing it.

## Exact recipe and paired control

All800 planned job files were read and matched to their manifest SHA. Each variant has100 jobs, two distributions × five attacks × ten seeds. All compare exactly to their own paired accepted Full config except the original six-field IGNORE_RECIPE set:
`seed, client_alpha, ablation_component, experiment_suite, experiment_tag, full_round_diagnostics`.
This exclusion set is UNCHANGED. Seed/actual alpha must match the record and paired cell; component must match the explicit eight-name mapping; new helper requires suite celeba_mechanism_v1, tag exact ID, diagnostics true. Across the actual800 paired configs, only suite/tag differ for structural controls; the six masks additionally change ablation_component. No actual learning-rate, clipping, fairness/calibration weight, network, optimizer, batch-size or round difference was found. Exact differing-field lists and example job SHA are in VARIANT_CONFIG_MAPPING.json.

Unchanged recipe is lr0.0005_drop0.005,70round,20clients/four nominal malicious,RGB64CNN,batch64,one local Adam epoch,root fraction.1,strictFP32. Distribution is actual alpha5000/5; attacks preserve exact spelling including `F Flip`. Each record must keep fulltrain162770, valid19867, root16277, client total146493, disjoint train/valid and root/client, clean unsynthesized root and native calibration enabled. Root/train/valid ordered IDs/cache/partition contract must equal the same distribution/attack/seed Full record. Full100 checkpoint/result/raw-job SHA and baseline record canonical SHA remain references only; use actual accepted three-view Full bindings, never substitute native for raw/shared or infer Full again.

## Minimal code change and batch preparation

BRIDGE_SEMANTICS_DRAFT.diff is a review artifact, NOT an executable widened cohort. It inserts the small variant_metadata_draft.py body, replaces the minus_U-only predicate by the exact eight-name map and replaces suffix parsing by the explicit component map. Existing82 accepted IDs, excluded71, selected11, inspection SHA and approval constants are deliberately unchanged. All11 other top-level functions, including bind_runtime with its nested replay/accept, are byte/AST identical. No strict/scoring/root calibration/native tolerance code is changed. Applying only this diff cannot authorize any pending/new record.

For each future actual terminal batch, reuse the existing prepare.py constructor and archive/member verifier. Change only the new package's scope, source/count/inspection pins, actual cumulative accepted IDs, prior adopted replay IDs and exact selected difference. Replace its hardcoded job.variant == minus_U assertion by the explicit map after SHA-bound frozen-job validation. Do not discover pending models dynamically at replay time. Required input chain:
1. Original strict native inspection for terminal70, original adapter audit and real job/result/checkpoint identities; original evidence.accept_new must pass.
2. Offserver archive/member SHA proof plus root adoption reference for those actual terminal IDs. Original previous accepted records remain JSON-identical; no pending record gains fabricated SHA or runtime.
3. Exact cumulative-native minus adopted-three-view difference, explicitly frozen before execution, excluding all prior closed records and Full100. Partial/failed/active records remain outside the replay input. Pending list remains IDs-only and cannot become a numerator.
4. New source/inventory/scope/parent-proof seals and separately reviewed exact execution approval, namespace and real resources. No automatic reuse of old after71 approval. Keep fresh-child sequence and failure-stop semantics; no automatic retry.

The current record constructor has no minus_U-specific numerical transformation: it copies original result/config/provenance, accepted_v4 row, archive model/result/job member identities and paired Full. Only its explicit identity assertion and bridge metadata predicate need variant expansion. Do not change bind_runtime: its original evidence.accept_new already validates all eight semantics using frozen worker/adapter; its original replay_one derives raw/native/shared from the same loaded terminal model and root-only fit. Existing Full references/actual900 bindings, source/data checks, same checkpoint and native1e-12 checks remain exact. A future mismatch is a preserved failure, never permission to relax1e-12 or switch evaluation runtime silently.

## Smallest next empirical gate, not performed

Once the first previously unseen variant has an actual strictly accepted/offserver terminal, prepare one exact already-trained checkpoint for the unchanged three-view valid replay. Require original native match1e-12, unchanged model/source/data before/after, all three views' prediction/metric/confusion and root-only fit receipts, then strict offserver closure. This single checkpoint validates that variant's representation/acceptance path only; it is not scientific performance evidence or coverage of the other six variants. Use one such existing terminal for each of the seven previously untested variant branches as they become available. Reuse that result in the later explicit cohort rather than replaying it twice. No new training/three-round simulation is required for this replay interface. Existing training gates and minus_U replays do not prove these new replay branches. Distribution/attack identities remain checked for every eventual record.

## Local verification and limits

Run `python -B tmp/celeba_mechanism_remaining_variants_source_plan_20261009/selfcheck.py` from project root. It checks pinned source bytes, all800 real planned job SHAs/configs, byte/AST preservation of11 old functions, the original82 inventory's metadata acceptance, and53 metadata/identity rejection cases. It never calls bind_runtime or imports torch/scientific modules. Generated proof files are immutable-on-rerun. These are source/config checks, not image gates, completed remaining-variant experiments, or replay acceptance. No new runtime installer, service, output namespace, current resource measurement or execution approval was created.
