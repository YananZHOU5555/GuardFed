# Root-operated future execution

These are instructions only. Root independently reviews the fixed package and actual Linux readiness before invoking them. No approval is supplied by this file.

Fresh remote prepared directory: `/workspace/guardfed_checks/celeba_mechanism_valid_C_after20_20261009`. Runtime: its `execution_candidate` child. Service: `guardfed_celeba_mechanism_valid_C_after20`. Prior service required EXITED/no worker: `guardfed_celeba_mechanism_valid_C_after12`.

1. Verify `PACKAGE_SHA256.json`, `PACKAGE_RECEIPT.json`, science/execution seals and the actual new independent root source review. Read `/etc/vast-agents-guide.md`; require its pinned SHA. Confirm sglang STOPPED, previous service EXITED/no worker and fresh namespace. Reuse the C-after12 root deployment's existing safe extract/upload logic; upload the new source tar only, without old models, outputs or approvals.
2. Root creates runtime `ROOT_APPROVED.json` from `ROOT_REVIEW_TEMPLATE.json`: set `status=ROOT_REVIEW_PASS_BOUNDED_C_AFTER20_VALID_REPLAY`, `execution_authorized_within_existing_user_request=true`, and `execution_seal_sha256` to the actual execution seal. Preserve exact source/scope/inventory/bridge pins, selected5, `closed120_must_not_replay`, budgets and `source_only_root_review_sha256=b1ff1fad6f5fe08cebec3008b6df861396c2e11c289b8f05815debb78dc3ed14`. Record the actual new independent source-review path/SHA as additional provenance; the inherited lineage field must remain distinct. No template itself counts as review or approval.
3. Root creates `EXECUTION_DRAFT.json` from `APPROVED_TEMPLATE.json`: set `status=APPROVED_C_AFTER20_MECHANISM_VALID_REPLAY_ONLY`, `root_approval_sha256` to the actual bytes of the new ROOT_APPROVED, and `execution_seal_sha256` to the fixed seal. Preserve selected5, exact output/dependency maps, CPU112–119/eight threads/max1, valid-only, `1e-12`, Full inference0 and retry=false. Hash the actual draft bytes externally.
4. Invoke the unchanged installer once with that actual external SHA:

```sh
cd /workspace/guardfed_checks/celeba_mechanism_valid_C_after20_20261009/execution_candidate
taskset -c 112-119 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B install_once.py --draft-sha256 ACTUAL_EXECUTION_DRAFT_SHA256
```

Installer preflight, rather than this document, measures Linux CPU ownership, conservative nominal budget plus three reservations, quota, GPU recovery, disk and all original source/data/artifact hashes. It registers and starts the normal supervisor service with autostart/autorestart=false. Supervisor starts the worker from its own priority; worker nice10 is not accumulated from the installer.

All Windows-to-SSH Python transport must use short argv and send the full Python program on stdin from the first attempt:

```python
subprocess.run(["ssh", "-p", "60350", "root@89.22.197.55", "python -B -"],
               input=code.encode("utf-8"), capture_output=True, check=True)
```

Do not interpolate a batch-complete dictionary into argv. Do not adopt prior transport-failure prerequisites; this is a fresh namespace.

After actual terminal completion, root freezes the exact progress/approval/seal SHAs and calls the original new candidate archiver once. A smaller strictly producer-closed delta is supported; preserve its receipt chain and never retry failed inference:

```sh
cd /workspace/guardfed_checks/celeba_mechanism_valid_C_after20_20261009/execution_candidate
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B backup_completed.py
```

Copy the actual returned archive and receipt into a new local `execution_candidate/backups/<actualtag>`; then run `python -B execution_candidate/verify_backup.py <actual_local_delta>`. The verifier uses safe manual extraction and the original saved-array rules, not a CNN. Root independently binds terminal progress/receipt/member identities before adoption. No old model or Full weight is archived. If all five close in one first archive, the expected total is 68 members (67 content), 45 metric checks, 120 counts and 15 prediction rules; these are expectations, not measured future results. A failure stops the service and remains preserved; there is no automatic restart or acceptance.
