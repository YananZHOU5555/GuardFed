# FLGMM 96-new + 4-reused validation launch — PREPARED ONLY

This independent envelope reuses the actually successful seven-canary launch transport/resource checks. It launches the unchanged bound `run_fullcoverage.py run` only after actual seven-canary strict, offserver and root closure. No SSH, inference, training or service mutation was performed while preparing it. Existing scientific source, recipe, seed grid, alpha, attack definitions, seventy-round budget and selection/statistical rules are unchanged.

## Required real evidence

The bound stage/package remains `/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage` / SHA `6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230`. The exact seven-canary offserver receipt must have `status=PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON`, with local saved verification accepted_new=5, same_horizon_pairs=2, rounds=3 and formal_table_samples=0. Root's separately written adoption must have:

```json
{
  "status": "ROOT_SEVEN_CANARY_CLOSURE_ADOPTED",
  "package_sha256": "ACTUAL_BOUND_PACKAGE_SHA",
  "gate_sha256": "ACTUAL_GATE_ACCEPTANCE_SHA",
  "offserver_sha256": "ACTUAL_OFFSERVER_VERIFICATION_JSON_SHA",
  "receipt_sha256": "ACTUAL_REMOTE_BACKUP_RECEIPT_SHA",
  "archive_sha256": "ACTUAL_CANARY_ARCHIVE_SHA"
}
```

These six fields bind the actual gate/archive/offserver chain. The launcher rejects absent or pending proof. `APPROVAL_TEMPLATE.json` is deliberately unapproved; root supplies its actual approved external copy, exact helper seal, both proof SHA values, old canary authorization SHA and a current root-live baseline SHA. No actual closure hash or approval has been invented in this prepared package.

```powershell
python -B tmp/celeba_flgmm_fullcoverage_launch_operations_20261009/launch.py --approval ACTUAL_APPROVAL_PATH --approval-sha256 ACTUAL_APPROVAL_SHA --gate-offserver ACTUAL_OFFSERVER_VERIFICATION_PATH --gate-offserver-sha256 ACTUAL_OFFSERVER_SHA --root-closure ACTUAL_ROOT_CLOSURE_PATH --root-closure-sha256 ACTUAL_ROOT_CLOSURE_SHA --baseline ACTUAL_ROOT_LIVE_PATH --baseline-sha256 ACTUAL_ROOT_LIVE_SHA --package-sha256 6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230 --helper-seal-sha256 ACTUAL_HELPER_SEAL
```

## Exact scope transition

The stage already contains the canary `EXECUTION_AUTHORIZATION.json`. After checking its actual SHA, status/scope/package, the new helper preserves its **exact bytes** as `PREVIOUS_CANARY_AUTHORIZATION.json`. After real gate artifact hashes and source/data checks, it prepares a new external authorization and atomically replaces only the stage authorization via a unique pending file. The new authorization follows actual `screen_common.authorized` fields: scope `96_new_70round_valid_only`, package SHA, resource path/SHA/time, protected-resource metadata, max_workers=2, cpu_threads_per_worker=1, final_test=false, and `gate_acceptance_sha256`. It cannot authorize test or a new scientific scope.

Existing complete canary `preflight` evidence is allowed and rehashed exactly against the adopted gate. Any fullcoverage `runs`, `logs`, queue progress, summary, coordinator lock or queue failure blocks this initial-launch envelope. Prior partial outputs are not resumed. All four legacy references are rechecked through the original `reused_records` function. The original runner's strict accepted-skip and fail-stop/drain semantics remain unchanged; this initial launcher does not create a new recovery policy.

## Resource and service boundary

The new service is exactly `guardfed_celeba_flgmm_fullcoverage`, autostart/autorestart=false, startretries=0. The old screen and completed canary services must both be EXITED, sglang STOPPED, and no stage producer present. Configuration existence is checked for the exact new service, allowing the distinct completed canary configuration to remain untouched.

Only this new service is updated/started. Taskset limits the manager and its two children to allowed CPUs102/103; each child's computational libraries use one thread. This is the unchanged runner's shared two-core affinity, **not** a new per-worker exclusive core scheduler. Original runner maps one child to each GPU. Nice10/idleIO and CPU1 library environment propagate. Main8, HybridCPU104 and every other service remain untouched.

The original `main_health.py` is byte-identical. Main protection accepts actual1–8 workers, current queue/log/round evidence, and measured round/completion growth relative to the SHA-bound recent baseline, permitting normal terminal handover. All observed Python thread ownership/budgets are preserved; restricted affinities cannot overlap102/103. New planned budget adds four conservative slots: two children, coordinator and transient launcher. Actual cgroup v2 quota/memory/events, GPU free memory/Recovery, disk and allowed affinities are checked after source/data/legacy acceptance. A resource receipt must be ≤120 seconds old at supervisor start. No utilization-based restarts occur.

## Evidence and limitations

`SOURCE_DIFF.patch` is the exact change against the previously executed canary envelope; `LINEAGE.json` pins parent helpers and scientific entry points. Eighteen local checks exercise the real local approval prefix through the transport boundary, fourteen refusals, unchanged main-health bytes and optimized-Python rejection. No Linux resource or future training success is claimed by these fixtures.

The base64/stdin transport avoids Python3.10 tar extraction APIs. New operations directory, authorization backups, command receipts and failures are retained. Timeout or post-authorization failure requires inspection of the actual namespace/service, never blind rerun. Startup receipts explicitly report only start-command success, not accepted results. Root performs actual closure review and deployment; this package never starts test evaluation or alters the frozen science. Three-round gate success does not establish universal seventy-round equivalence.
