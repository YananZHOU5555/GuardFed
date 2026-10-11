# NGA96 incremental acceptance — source only

No SSH, F-drive access, result verification, Torch import, inference, fitting, training, Git or STATE changes have been performed by this package. The supplied 03:08 observation of one terminal run is **not** adopted by this package. All actual terminal IDs, checker/source bindings, F-volume observations and archive proofs remain null in templates.

## Reused implementation

`collect_delta.py`, `pinned_collect_one.py` and `run_once.py` derive from the already used `gradient_native_after42_20261011` incremental collector and transport. `verify_delta.py` derives from `gradient64_final_acceptance_prepared_20261011`, including its original full tensor-state key/order/shape/dtype/finiteness check. The original server and off-server per-ID scientific loops and archive loops remain byte-identical. `pinned_collect_one.py` is an AST helper source, **not a standalone entrypoint**.

The actual frozen coverage `accept_result.py` is imported from the source-approved NGA package without changes. Its already frozen extension from screen64 is recorded in FROZEN_CHECKER_VS_SCREEN64.patch: coverage labels, job-bound seeds and five-scene attack checks. No mathematical rule or tolerance is altered here.

Scope is exactly 96 new NGA jobs plus the four explicit selected screen records in SCOPE.json. Reuse their existing acceptance identity; do not rearchive their models. Acceptance counters describe **new coverage jobs only**. Neither a terminal file nor a successful one-item delta means the method100 table is complete.

## Root binding and commands

1. Inspect SOURCE_DIFF.patch, SOURCE_ADAPTATIONS.json, SOURCE_CHECK.json and source seals. Read the current server guide and resource constraints before future execution. Collect a fresh source-bound terminal/failure/worker observation; freeze a nonempty ordered, nonoverlapping subset of the 96 new IDs. No implicit all-current-results selection exists.
2. Fill `AUTHORIZED_SNAPSHOT.json` from its template with actual observation path/SHA, actual source approval/package/checker/manifest pins, actual gate14 path/SHA and selected IDs. `status` must become `ROOT_BOUND_EXACT_NGA96_DELTA_NOT_ACCEPTED`. The expected frozen hashes are in SCOPE.json; they are not live observations. For the first batch, `accepted_prior_ids=[]`, `accepted_prior_count=0` and every previous-proof field remains null. For later batches, use actual root-adopted prior coverage IDs and copy matching prior proofs as `PREVIOUS_ROOT_ADOPTION.json` / `PREVIOUS_OFFSERVER_ACCEPTANCE.json`.
3. A prior coverage root must record `root_adopted=true`, `test=false`, `scientific_package_sha256`, `accepted_ids`, `accepted_total`, and agree with the prior off-server proof. Do not use the old screen64 cohort as the coverage96 prior counter.
4. Create `ROOT_REVIEW.json` from its template only after root review: status `PARENT_AUTHORIZED_FIXED_NGA96_INCREMENTAL_DELTA`, `root_adoption_performed=false`, current source-seal SHA, actual authorization SHA and exact IDs. Its filename may be external; pass its actual path and SHA on every command.
5. Using the existing Python environment with Torch for the off-server phase, run each phase once:
   - `python -B run_once.py collect --review <actual review> --review-sha256 <actual SHA>`
   - `python -B run_once.py download --review <actual review> --review-sha256 <actual SHA>`
   - `python -B run_once.py verify --review <actual review> --review-sha256 <actual SHA>`
   - `python -B run_once.py close --review <actual review> --review-sha256 <actual SHA>`
6. Review the measured command logs, member proof, original strict proof and fullstate proof before creating any root adoption. `OFFSERVER_ACCEPTANCE.json` deliberately has `root_adopted=false` and `coverage100_completed_claim=false`. This package does not write canonical STATE or claim a complete table. Reuse the same source in a separately named, fresh batch namespace for the next explicit delta; never overwrite successful or failed batch evidence.

Transport preserves the previously used CPU110/thread1/nice10/idleIO/no-CUDA collector and checks it is free. It does not change the CPU105/GPU1 training allocation. Running service is allowed; completed EXITED service is allowed only with the exact96 completion list and no workers. Selected producer PID must be gone, progress70 present, and original strict checks must pass. Failures stop the batch, with no automatic retry.

## Evidence and minimum backup

Strict acceptance requires all70 ordered rounds, final metrics from that same checkpoint, exact job/recipe/source/data/component identities, correct seed/distribution/alpha/attack, valid split19867, train162770, root/client separation, original gradient diagnostics and actual uploaded attack assignment. Off-server verification additionally checks the complete saved CNN tensor state without forward/fit/data loading. Original negative and collapsed predictions remain in raw results and counters.

Each archive contains only the explicitly selected new terminal run directories, the small frozen source package, per-batch identity/resource proofs and a labelled shared-log prefix. Old models, raw image caches, unselected results and historic archives are not repacked. Full shared logs remain on the server. Download and restore go only to guarded `F:/YananResearchStorage/GuardFed/<batch namespace>/batch`; `Yanan 2TB` label/capacity checks run before writes, with no internal-drive fallback.

The eventual method100 table must explicitly merge strict/root-adopted96 with manifest.reused_jobs4, preserve all outcomes, and use per-condition means/sampleSD(ddof1) over the same10 seeds. Fixed9 and fixed6 descriptive panels remain separate; no scenario is treated as an independent seed, no best-seed replacement or test selection is allowed. Partial batches provide partial evidence only.

All necessary frozen code is already available locally; no additional server source fetch is required for this preparation. Future live result/proof files are intentionally not supplied or guessed.
