# Rebuttal completion: active work, 2026-10-09

User objective: complete all experiments needed for the rebuttal and produce a sufficiently clear, complete response. This authorizes useful implementation and missing experiments; it does not authorize invented results, selective reporting or changing a method merely to force a win.

## Evidence that remains complete

The accepted nine-method validation comparison contains900 records, both distributions, five scenes and ten paired seeds. Paper tables and the10/9/6-seed statistical audits are in `../../../outputs/guardfed_tables/celeba_nine_method_final_20261004/`. Old TableII has480 traced source values with their actual replication counts; unsupported SD are not invented. The2454 previously accepted new trainings and their backup chains remain unchanged. A completed validation comparison does not establish the entire rebuttal or an untouched-test result.

## New implementation and writing

- Complete CelebA mechanism design:800 missing controls plus100 existing Full records; prepared manifest, isolated controls, strict result/ledger checks, bounded fail-stop launcher,18 real-image gates and two unchanged-worker references. Six CPU/synthetic component checks and the full prepared grid pass. No new CelebA/GPU execution has occurred. Entry: `celeba_mechanism_v1/EXECUTION.md`.
- Four additional method components are implemented with CPU checks: true same-point gradients/Fed-NGA, fixed-threshold sample-weighted Huber, the original cosine/fairness control, and official LoGoFair optimization/calibration. These are not completed GPU baselines. LoGoFair still needs an explicit client-ID evaluation adaptation; gradient methods need a declared attack-message/training mapping.
- FLGMM author-code aggregation is integrated into the unchanged CNN/local-Adam/attack path, with a checked full-state-to-delta bridge and a three-round synthetic-image pipeline test. Thirty-two screen jobs are prepared, not frozen or run. FedWA/SmartFL/FedDNA source recovery remains unresolved.
- Updated English rebuttal covers24 original comment blocks with source alignment, manuscript insertions and eight checked recommended references. Entry: `../revision_20260923/rebuttal_20261009/README.md`. The missing Adult score archive is restored and fully checked:140 original files,8440 visible rounds and388240 fields; observed gate re-entry and the conditional theorem's limited scope are retained. See `score_analysis/restoration_and_score_report_20261009.md`. Pending experimental evidence is labeled explicitly.
- CelebA realized-partition audit is complete:20 original Sp-DFA records,400 client counts and40 original sensitive-group counts match exactly. IID Male ratios range40.495%–42.969%, non-IID9.832%–74.303%; root support is counted once per shared seed. See `celeba_partition_audit_20261009/README.md`. Client label-joint counts remain unavailable without metadata.

## Additional accepted local work

- The stronger-alpha Adult/COMPAS audit is now complete:60 partitions,120 original paired jobs,1200 client totals,2400 sensitive-group margins and240 S-DFA flip counts match. Restored archived client label-joint counts are distinguished from independently replayed sensitive counts. Alpha0.1 includes empty and single-group clients, and nominal malicious IDs can hold over65percent of client samples. See `tabular_partition_audit_20261009/README.md`.
- Historical synthetic numerical lineage is checked for840 records. All840 joint-checkpoint rows match; an older independent-column export has797 rows without a common checkpoint. Seventy settings now also have three-seed summaries formed after averaging the four scenarios within each seed. ForestDiffusion implementation/fit/cache identity, the submitted Fig.3 input chain and the separate PCA suite remain unresolved. See `synthetic_lineage_audit_20261009/README.md`; the900 CelebA validation records are unaffected.
- Fed-NGA/Huber have an isolated same-point-gradient worker and64 search proposals. Eighteen synthetic CNN rounds,108 attack sign-oracle cases and22 acceptance checks pass. No client Adam difference is presented as a gradient. Five explicit protocol choices and real-image gates remain unresolved; nothing is frozen or dispatched. Local entry: `../../../tmp/celeba_baselines/gradient_bridge_20261009/REPORT.md`; Git snapshot entry: `experimental/celeba_baselines/gradient_bridge_20261009/REPORT.md`.
- All900 terminal model/result/raw-job identities have been located in25 existing SHA-verified backups. The900 final-evaluation job proposals are PREPARED_NOT_FROZEN; this is model preservation and planning, not test inference. Live source/data/environment, target sample identity, valid replay and final reporting rules still require verification. See `final_evaluation_prepared_20261009/README.md`.

The numerical superiority target is a research objective. Current accepted results have accuracy/fairness tradeoffs; no rule change, deleted negative result or selected-seed-versus-mean comparison is justified by that target.

## Live blocker

The user confirms instance52514165 is running and the endpoint is unchanged. Current SSH and TCP attempts to213.224.31.105:26712 are refused before authentication; the instance's internal SSH listener and GPU state cannot be observed. This does not prove that the instance is stopped or that authentication failed. The most recent successful full server observation is2026-10-07 and is historical. A request for the web-console output of `ss -lntp` is pending. No further connection retry loop or completed-queue restart is warranted without new evidence.

## Remaining actual completion criteria

1. Restore observable SSH connectivity and verify the guide/source/data/current processes.
2. Pass real-image gates and complete the missing faithful baselines, validation selection and full paired coverage; disclose any unresolved unavailable original method instead of substituting its old simplified branch.
3. Complete the800 new mechanism jobs and900-model native/raw/shared-calibration comparison, strict acceptance and off-server backups.
4. Freeze the final evaluation identities and rules, evaluate accepted models without test tuning, and disclose prior test exposure. This may reuse terminal models and is not automatically another full training campaign.
5. Replace the rebuttal's pending-evidence fields only with accepted measurements; finalize manuscript tables/text, response and all linked artifacts.

The active goal remains incomplete. Prepared files, working toy adapters and a finished response draft are concrete progress, not evidence that the missing scientific experiments have finished.
