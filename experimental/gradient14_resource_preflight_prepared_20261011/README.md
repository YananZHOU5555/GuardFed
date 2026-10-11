# Prepared only: one-job gradient14 resource receipt

This is source-only. No SSH, Linux sampling, deployment, approval, recipe selection, model/array read, fit, forward pass, or training was performed. The original adapter/metadata/compare_saved package remains unchanged.

Parent resource implementation: `tmp/celeba_gradient_screen64_v2_root_operations_20261010/remote_preflight_attempt2.py`. `sha`, `read`, `command`, and original gate `formal_growth` are source-exact. The original source/data hash checks, nominal compute budget, GPU recovery/free-memory, cgroup RAM, disk and cgroup usage sampling remain. Changes add root-pinned exact14 metadata, next-one-job fail-stop, old64 exit/failure checks, and real main800 progress. CPU105/one GPU compute thread becomes CPU88–95/eight CPU threads, CUDA hidden. All live /proc task affinities are checked, including non-project owners; restricted masks <=32 intersecting either target or SMT sibling reject. Broad masks are not exclusive reservations. Permission/read failures fail closed; only disappeared processes are skipped. Targets must be eight unique physical cores on one socket, each with its eligible online SMT sibling. No automatic CPU relocation.

The root changed the requested pool from96–103 to88–95 because the former overlaps the running FL service at102/103. This is a source binding, not an actual finding about current CPU88–95 availability. Nice10/idleIO/environment are launch requirements; preflight does not pretend it changed a future process. Existing `exclusive_cpu_ids` is a legacy adapter field meaning requested affinity only, not guaranteed CPU exclusivity or idleness.

## Late binding

Fill `AUTHORIZATION_TEMPLATE.json` only after actual64 strict/offserver acceptance, actual source selection/frozen192 approval, and original14 metadata.bind. It is an actual root source/operational authorization, not a scientific result. All future paths/hashes and ordered jobs are null now. Pin the actual main800 manifest. Pin the actual original adapter.py and metadata.py. The preflight checks the actual root64 and frozen source statuses and full source/data hashes on each invocation. It does not derive winners or freeze a protocol.

Run preflight with its natural broad eligible mask (do not taskset it to eight CPUs, since it must check sibling eligibility). Pass a prior SHA-pinned original gate snapshot, previous resource receipt containing main_snapshot, or original five-queue raw observation. The latter is normalized directly from its main queue and worker rounds, retaining original file SHA; it is not a new observation.

```sh
python -B PREPARED/preflight.py --stage ACTUAL14_STAGE --repo /workspace/GuardFed-celeba-expanded --package ORIGINAL14_PACKAGE --authorization ROOT_AUTHORIZATION.json --authorization-sha256 ACTUAL_SHA --job 'ONE_EXACT_ID' --previous-main PRIOR_ACTUAL.json --previous-main-sha256 PRIOR_SHA --out NEW_UNIQUE_RESOURCE.json
```

Save command/stdout/stderr/exit and any refusal. Do not retry automatically. A failed preflight creates no approval receipt. No queue/runner/monitor is included.

## Required root serial-call check

The unchanged adapter.runtime_receipt checks all14 job hashes but does **not** consume the new per-job field. Before each original adapter invocation, root must check the receipt's SHA and assert:

```python
assert receipt['job_id'] == cli_job == receipt['one_job_runtime_scope']['job_id']
assert receipt['job_sha256'] == scope_entry['job_sha256']
assert receipt['one_job_runtime_scope']['sequential_index'] == scope['jobs'].index(scope_entry)
assert receipt['one_job_runtime_scope']['fresh_receipt_required_for_next_job'] is True
assert receipt['scope_sha256'] == actual_scope_sha256
assert receipt['jobs'] == {e['id']: e['job_sha256'] for e in scope['jobs']}
assert 0 <= time.time() - receipt['measured_unix'] <= 120
```

Then pass exactly that receipt/SHA and same cli_job to the original adapter; each subsequent job requires a new actual preflight receipt. Preserve original serial order and failure/partial-output refusal. The root's intended command uses `env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 taskset -c 88-95 nice -n 10 ionice -c 3 python -B ORIGINAL14_PACKAGE/adapter.py --stage ACTUAL14_STAGE --repo /workspace/GuardFed-celeba-expanded --job ONE_EXACT_ID --approval NEW_UNIQUE_RESOURCE.json --approval-sha256 ACTUAL_RECEIPT_SHA`.

The stdlib preflight never imports Torch/scientific modules. It executes only the two SHA-pinned original pure metadata functions for exact job/config order. Scientific adapter/whole comparison and downstream192 remain unexecuted and separately gated. Health metadata that all800 are complete is not scientific/offserver acceptance. The true800 ID set, empty failed/active/pending sets and correct service status are required for the completion alternative.

## Superseding draft V2

Author second read-only review found a predecessor receipt identity gap. V2 adds only acceptance.status=PASS, acceptance.job_sha256=previous job and acceptance.scope_sha256=current scope checks. The original V1 source/seal are preserved under the review namespace and superseded before any Linux execution. The existing 30 pure metadata tests concern unchanged helper functions and retain their original source pin; only the new exact predecessor-loop positive/wrong-job regression was run for this change. This is author review, not an independent reviewer or actual Linux acceptance.
