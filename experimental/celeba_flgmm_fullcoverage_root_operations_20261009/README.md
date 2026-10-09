# Root operations: prepared, not executed

Three small helpers only. Source and operation seals stay separate. No SSH, binding, old-result check, canary, supervisor modification or new training has run during this preparation. The currently reviewed v2 source remains byte-exact. A missing remote stage is a legitimate NOT_BOUND/NOT_STARTED observation, not a failure to be repaired by starting work.

`deploy_bind.py` requires explicit SHA-bound v2 independent review and root binding approval. It verifies source28/summary/final32/adoption inputs locally, creates one fresh local attempt, checks the guide/CPU vacancy and exact fresh remote namespace, uploads a safe member-hashed archive, calls only `bind_stage.py`, then performs original4 strict acceptance through the already reviewed isolated legacy subprocess path. It verifies all actual21 source/data hashes before/after, including the six exact resolved data targets. It does NOT write EXECUTION_AUTHORIZATION, run_canaries, run_fullcoverage or supervisor config. It retrieves the complete new metadata package,96 jobs/4 original references/5 canary jobs, protocol, approval and source review; every returned member is SHA/size checked and every package/job hash is checked locally. Original checkpoint files are never repackaged. Offserver checks are metadata/member identity checks; the four model acceptance calls happen on the original server and do not claim local scientific re-execution.

Fixed remote paths:

- Original `/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2`
- Source `/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/source_prepared`
- Bound stage `/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage`

The root binding approval defaults to the separately root-owned `tmp/celeba_flgmm_fullcoverage_binding_20261009/BIND_APPROVAL.json`. It must satisfy the v2 binder schema, plus actual `independent_source_review_sha256=a22118cfc7ba4f7cae592fa9131a596731c38c4d5a97f4efb2466f830b681e3b`, `summary_adoption_sha256=e602761016e199da157862da3f24c9f9d0f191cfde10fad49074540c672b4a7f`, reviewed helper_seal_sha256 and actual chosen helper_cpu. Selected recipe is read from the actual adopted summary, not a hardcoded favorable choice. Root chooses a currently free allowed CPU; there is no default assignment that could collide with CPU106/main8/Hybrid. Execution is CPU1/nice10/idleIO, CUDA hidden. It invokes only local config/hash checks and the old original strict result checker, not any CNN or training call.

Future root-reviewed command:

```text
python -B deploy_bind.py --source-review ACTUAL_ROOT_READY_REVIEW.json --source-review-sha256 a22118cfc7ba4f7cae592fa9131a596731c38c4d5a97f4efb2466f830b681e3b --binding-approval-sha256 ACTUAL_ROOT_BINDING_APPROVAL_SHA --helper-seal-sha256 REVIEWED_THIS_SEAL_SHA --cpu ACTUAL_FREE_CPU
```

Both the fixed remote namespace and new stage must be absent at first execution. Any failed transfer, timeout, check or binding leaves the exact attempt and original evidence intact; no automatic retries, deletion, moving or overwrite exist. A timeout does not prove the remote helper stopped. Inspect that namespace before any separate recovery. `OPERATION_FAILURE.json`/per-command logs/timeouts are local; remote late failures also create BIND_FAILURE files. A failed fresh parent creation/transfer is not permission to delete it and retry.

`observe.py` is a single read-only SSH snapshot, saving local stdout/stderr/returncode/SNAPSHOT. It reads exact package/job SHA,96 new/5 canary/2 reference progress, source changes, result-file presence, failures/log tail, relevant service statuses, actual process argv/affinity, cgroup-v2 CPU/memory, disk and GPU. Missing services are recorded with their real returncodes; file presence never implies scientific acceptance. It does not install a monitor, repair state, check model numerics, select a recipe or mutate remote data. Run only when root requests an actual observation; it has not run here.

The optional launch_canaries helper is deliberately deferred to keep this first deliverable metadata-only and promptly reviewable. Root must inspect the actual returned bound package first. Any later launcher needs a fresh120sec resource receipt, package-bound exact7 scope, CPU102/103/main8/Hybrid104 conflict checks, oldFL EXITED, a separate normal supervisor and its own source review. None of those launch operations is smuggled into the binding helper.
