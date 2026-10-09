# FLGMM fullcoverage incremental backup — source prepared, no actual backup

Ownership is restricted to this new directory. The tools do not SSH, start/restart services, run CNNs, refit thresholds, evaluate test, select recipes or edit the sealed training stage/canonical state/Git. At preparation, the authorized stage had **0/96 new completed results**; this is context, not a fresh server observation or a manufactured backup.

## Minimal reuse

`collect_delta.py` is adapted from the already used `celeba_flgmm_screen_20261009_v2_dispatch/accepted_delta_after6_20261009/collect_delta.py`. Its complete per-terminal acceptance/model/raw/log inventory loop is AST-identical (`SOURCE_CHECK.json`). Hardcoded old6/32/screen paths become CLI-pinned previous offserver chain /96-new/fullcoverage stage. The original fullcoverage `screen_common.accepted` and `source/accept_result.checked_result` are imported unchanged. `verify_delta_offserver.py` retains the existing archive/member-SHA plus original-checker approach, with parameterized local stage and Python3.10-compatible streaming safe restoration. There is no new scientific acceptance framework.

The fixed stage is `/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage`, package SHA `6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230`. Original fullcoverage source, 96 jobs and four reuse references are verified by `local_identity`. For each candidate new ID the original producer PID must be gone, it must be absent from both the sampled and current active queue, and all terminal files must exist. Original terminal70/seed/config/source/adapter/data/valid19867/train162770/root16277, true alpha, attack, same-checkpoint metrics, deterministic identity, full controller and model hashes are checked. Sources/data are rehashed before and after the bounded batch. Failures and negative values are retained; incomplete output never triggers a restart or becomes accepted evidence.

## Next actual inspection and one bounded collection

Root first confirms a genuinely free single CPU and whether any terminal job has left the active queue. Copy these two helper scripts into an isolated server helper directory, then invoke the collector once with a **new output path** and the actual collector source SHA. The parent of that path must already exist and resolve literally. CPU106 below is an example requiring an actual free-owner check; CPUs102/103,104,112–119 are disallowed to protect the established queues. The collector checks actual affinity/nice/idleIO, one-thread environment, restricted Python CPU ownership and current declared thread budget against real cgroup quota.

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 106 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B HELPER_DIRECTORY/collect_delta.py --out NEW_ABSOLUTE_BATCH_DIRECTORY --cpu 106 --source-sha256 ACTUAL_COLLECTOR_SHA --initial-empty
```

`--initial-empty` is only for the first batch when root has confirmed **no new fullcoverage ID has previously been accepted offserver**. It records that empty input set explicitly as `EXPLICIT_INITIAL_EMPTY_NEW_RESULT_SET_NOT_BACKUP`; it does not claim a prior scientific backup. On all subsequent batches replace it with:

```text
--previous EXACT_PREVIOUS_OFFSERVER_ACCEPTANCE.json --previous-sha256 ACTUAL_PREVIOUS_PROOF_SHA
```

The previous proof must be `PARTIAL_ACCEPTED_OFFSERVER_VERIFIED` from this same package, carrying the cumulative new-result `accepted_job_ids`. Root supplies the latest adopted proof; the collector never edits a canonical ledger or chooses one implicitly. This single invocation freezes a snapshot, selects only completed IDs minus that exact prior set and stops at that snapshot. No continuous monitoring or chasing later completions occurs. If the difference is empty, only `live_snapshot.json`, input bindings and `NO_NEW_TERMINAL.json` are retained; there is no archive or accepted-count claim.

## Transfer and local verification

For a successful nonempty batch, complete actual SCP of `accepted_delta.tar.gz`, `BACKUP_SHA256.json`, `MEMBERS.json` and `PARTIAL_ACCEPTANCE.json` into a fresh local batch directory. Root pins the actual server receipt SHA, then invokes:

```powershell
python -B tmp/celeba_flgmm_fullcoverage_incremental_20261009/verify_delta_offserver.py --batch NEW_LOCAL_BATCH_DIRECTORY --release tmp/celeba_flgmm_fullcoverage_root_operations_20261009/attempt_20261009T200518912319Z/verified_manual_v2/stage --receipt-sha256 ACTUAL_SERVER_BACKUP_RECEIPT_SHA
```

The local release is the already accepted bound metadata/source package. Its original archive SHA is `9d6ea7126224343dc08bb08113181de6edbc19111d4a15b335b81fc72e5be24c`; that recovery reference is recorded in the new acceptance. The four reused seventy-round models, 32-search models and prior three-round canary outputs are **references only**, not repackaged. Each archive contains only new IDs' entire final output directories (models/controller/raw/result/config/provenance), closed logs, snapshot/prior proof, input bindings and the two small helper sources. No scientific source/model bundle is duplicated per batch.

The server verifies every archive member and stable pre/post file SHA; the local verifier checks archive/receipt/member hashes, safe file paths and exact sizes, replays the original checker on restored new results, compares the original same-checkpoint metrics/identity against the server acceptance, and emits `OFFSERVER_ACCEPTANCE.json`. That proof carries cumulative accepted new IDs for the next root-reviewed batch. It records local verification runtime separately and never presents local CPU Torch as the training cu128/GPU environment. Source/data files are physically rehashed on the server; checkpoint/records and original packaged science are verified locally. Source or dependency failures stop with evidence, without relaxing scientific checks.

Every batch keeps failures in its own directory. Existing output/restoration paths are rejected. The collector does not overwrite trained outputs, logs, past proofs, canonical LATEST files or summary/selection rules. Accepted-new counts range0–96; the separate four reused results do not enter the difference. Even accepted_new=96 does not make this incremental backup a substitute for the original full100 final summary/adoption.

## Prepared validation

`SOURCE_DIFF.patch` gives exact changes from the prior helper. `SOURCE_CHECK.json` pins both parent helpers and the actual bound scientific entry points, proves the original per-result loop AST unchanged, and records CLI/optimized-Python refusal checks. No remote collection, restored model or new scientific sample exists from this preparation. There is no extra approval-template hierarchy or automatic dispatch.
