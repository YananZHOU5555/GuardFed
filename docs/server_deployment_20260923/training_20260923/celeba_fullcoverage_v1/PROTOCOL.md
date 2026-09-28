# CelebA full coverage v1 — 2026-09-26

## Authorization and scope

The user's explicit instruction is to perform sufficiently comprehensive experiments, following their correction that both IID and non-IID coverage is missing. This authorizes supplementing missing conditions, faithful baseline integration, and controlled mechanism comparisons on the already rented two-GPU instance. Do not purchase resources or rerun completed historical experiments. Stage completion is not completion of the entire revision campaign.

Stage A freezes a complete coverage matrix for the seven already verified implementations: FedAvg, FairFed (project adaptation), Median, FLTrust, FairGuard (project adaptation), FLTrust+FairGuard (project adaptation), GuardFed-AD2+. Use both IID (alpha=5000) and non-IID (alpha=5), each with Benign, F-Flip, FedSA, S-DFA, Sp-DFA. The canonical attack strings must follow the frozen worker implementation. Ten shared seeds are 91001 through 91010. Total: 7 x 2 x 5 x 10 = 700 conditions. Reuse the exact 56 accepted non-IID Benign/S-DFA seed91001–91004 records from seedcheck and its explicit reused_jobs; run only 644 missing conditions. No old formal240 or other1390 results are overwritten or mixed into this matrix.

## Hypothesis and frozen recipes

Test whether the already selected recipes maintain useful accuracy and group fairness across distributions and attacks. The competing explanation is that an advantage was specific to the non-IID selection conditions or arose from group-threshold postprocessing. Stage A tests transfer/coverage, not attribution and not IID-specific hyperparameter optimality. Separate mechanism controls and faithful baselines remain required.

Use exactly the seven recipes chosen by the preceding validation screen: GuardFed LR0.0005/max_acc_drop0.005; FLTrust0.0005; Median0.0005; FedAvg0.002; FairFed0.001; FairGuard0.001; FLTrust+FairGuard0.001. All remaining model, attack, calibration, and root settings are inherited unchanged from the validated seedcheck source jobs. Change only distribution, effective client_alpha, attack, seed, experiment bookkeeping/output paths. Record exact differences. Freeze protocol, preparer, selected recipes, training source/data, and telemetry launcher hashes before dispatch. Do not modify this protocol after dispatch; amendments need a separate version and preserve this cohort.

Full official train162770 and validation19867, Smiling label/Male sensitive attribute, RGB64 CNN,70rounds,20clients/4nominal malicious,root10%,Adam,batch64,deterministicFP32. Server root calibration uses training-root data, never the evaluation labels. Every reported metric comes from the same final-round checkpoint for that condition. No test evaluation or test-guided selection in Stage A. Test confirmation is a distinct frozen evaluation after implementations and comparison settings are settled; previous test exposure must be disclosed.

## Scientific reporting

Primary table has method/category/metric rows and ten scenario columns grouped under IID/non-IID, matching the manuscript table structure. Show ACC in percent and AEOD/ASPD on [0,1], mean and sampleSD(ddof1) over the same ten seeds in each cell. AEOD here is the implemented absolute TPR gap, not full equalized odds. Retain raw values, prediction rates, every negative/degenerate outcome and failed-attempt history; no N/E suppression of low values. Report paired seed differences and uncertainty with sample size. Do not count scenarios or repeated checkpoints as independent seeds.

Also report seed91002–91010 separately (nine seeds excluding the configuration selection seed) and seed91005–91010 separately (six previously unobserved seeds fixed before collection). The original seed91001 was involved in recipe choice; other first-four-seed outcomes were previously viewed. Do not label the ten-seed validation matrix as an untouched-test result or as fully prospective seed evidence. No cherry-picked bestseed table is the primary table. No automatic hyperparameter changes or seed additions based on interim results.

The former 14 reused seed91001 runs used torch2.11+cu130; all new runs and the prior42 seedcheck runs use torch2.11+cu128. Preserve per-run runtime provenance and report same-environment subsets. Four first-round migration canaries matched exactly, which does not prove70round environment equivalence.

If reporting the pre-existing composite score, preserve ACC-.35*(.45*AEOD+.45*ASPD+.10*max(AEOD,ASPD))-.10*max(0,max(AEOD,ASPD)-.06). Compute it per condition; average any predeclared set of conditions within each seed before cross-seed statistics. Primary conclusions rely on all three original metrics, not solely this scalar. No promise of superiority.

## Execution and acceptance

Server213.224.31.105:26712, instance52514165; repository/workers /workspace/GuardFed-celeba-expanded; new service guardfed_celeba_fullcoverage. New manifest results/revision_20260926/celeba_fullcoverage_v1/manifest.json. Use the existing cgroup-v1 telemetry launcher and frozen worker,8concurrency; retain prior throughput-tested setting. Do not chase transient GPU utilization or change numerical settings. Old89.22.197.55 is archival and must not run queues.

Prelaunch: source/data/protocol hashes match; no competing workers; all56 reused outputs pass existing checked_result plus70round/valid/same-checkpoint/recipe identity; all700 effective condition keys unique and complete. Verify IID/alpha and attack control flow in source, rather than trusting labels. Fullsource/data identity and unchanged worker permit reuse of prior image and deterministic migration canaries; additionally inspect initial new-task progress and finite diagnostics for newly added scenarios. On first numerical/logical failure, preserve evidence and stop dispatch; do not loop-retry. Only external interruption may be resumed after identity/no-duplicate/skip-complete checks.

Incrementally accept new outputs through the existing checked_result and verify final70round, seed, distribution, attack, recipe, evaluation split/counts, source/data and model SHA. Existing runner summaries cover only644 new runs: the final table must explicitly join manifest.reused_jobs56 to yield700 records. Do not claim700 on service exit alone. Off-server incremental archives must include new checkpoints, raw results/configs/diagnostics/logs and checksum inventory, with local SHA/member validation. No repeated backup of unchanged models.

## Remaining campaign stages

1. Reconcile every method in manuscript Tables I/II against implemented/verified CelebA baselines. Complete faithful adapters and real-image canaries, giving each method appropriate bounded validation tuning. Existing approximations must retain adaptation names. Separate work specification: BASELINE_AND_MECHANISM_PLAN.md. Seven-method Stage A alone does not close baseline expansion.
2. Evaluate raw and shared-root-calibrated predictions on the same available checkpoints for GuardFed and strong baselines. Match root/sensitive-attribute access, fitting rules and calibration budgets. Cache scores and reuse models where possible. Preserve native end-to-end results separately.
3. Version a CelebA mechanism matrix that separates scoring contributions from hard filtering and prediction calibration; paired seeds and common reporting rules. Reuse completed Adult/COMPAS single-component, strongheterogeneity,rootnoise/rootshare,ACS and originalCelebA experiments. Do not rerun them merely to seek a preferred result.
4. After implementation/tuning gates, freeze final comparative evaluation, independently record prior test exposure, and evaluate final checkpoints without selecting on test metrics. New methods/configurations may require new training; unchanged checkpoints are reused. No blanket campaign is launched for undefined adapter parameters.

This protocol makes Stage A executable now and records the remaining work explicitly. Monitor Stage A without autonomously changing its methods/seeds/configurations or treating its finish as entire-project completion.
