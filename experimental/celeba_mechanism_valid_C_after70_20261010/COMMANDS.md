# Root commands — prepared only, not executed

Fresh namespace `/workspace/guardfed_checks/celeba_mechanism_valid_C_after70_20261010`, service `guardfed_celeba_mechanism_valid_C_after70`. Root independently reviews the source and confirms guide SHA, sglang STOPPED, prior C_after60 EXITED/no worker, CPU112–119 availability and empty namespace. Do not resume prior outputs or replay old170/Full.

Root creates `ROOT_APPROVED.json` from the root review template with status `ROOT_REVIEW_PASS_BOUNDED_C_AFTER70_VALID_REPLAY`, actual execution seal, explicit authorization, exact10 IDs and `closed170_must_not_replay`. Inherited `source_only_root_review_sha256=b1ff1fad6f5fe08cebec3008b6df861396c2e11c289b8f05815debb78dc3ed14` identifies historical approved science, not a new after70 source review. Bind the new independent review separately. Templates grant no authority.

Root creates external `EXECUTION_DRAFT.json` from APPROVED_TEMPLATE with status `APPROVED_C_AFTER70_MECHANISM_VALID_REPLAY_ONLY`, actual ROOT_APPROVED SHA and execution seal. Preserve exact10, CPU112–119/8threads/max1, valid/native1e-12, original dependency/output/source identities, Full inference0 and retry=false. Hash the actual draft before invoking:

```sh
cd /workspace/guardfed_checks/celeba_mechanism_valid_C_after70_20261010/execution_candidate
taskset -c 112-119 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B install_once.py --draft-sha256 ACTUAL_EXECUTION_DRAFT_SHA256
```

Transport uses fixed short argv `ssh -p 60350 root@89.22.197.55 'python -B -'` with code supplied through stdin (`input=code.encode('utf-8')`). Do not embed whole proof dictionaries in argv. No historical transport recovery prerequisite is reused.

Only after genuinely closed strict results, unchanged source/approval and no matching producer:

```sh
cd /workspace/guardfed_checks/celeba_mechanism_valid_C_after70_20261010/execution_candidate
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B backup_completed.py
```

Copy actual archive/receipt to fresh local `execution_candidate/backups/<actualtag>` and run `python -B execution_candidate/verify_backup.py <actual_local_delta>`. If the first backup closes all10, the unchanged seven-artifacts/ID +science10/execution11 source and fixed extras expects103 total members/102content,90 metrics/240counts/30rules. These are contracts, not produced results. Root separately checks/adopts actual evidence; no automatic retry, old model repack or Full inference.
