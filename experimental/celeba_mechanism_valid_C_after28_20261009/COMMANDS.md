# Future root invocation; not executed

Root verifies final `PACKAGE_SHA256.json`, `PACKAGE_RECEIPT.json`, source tar and science/execution seals against its independent source review. The fixed new prepared namespace is `/workspace/guardfed_checks/celeba_mechanism_valid_C_after28_20261009`; runtime is its `execution_candidate`. Service is `guardfed_celeba_mechanism_valid_C_after28`. The previous service `guardfed_celeba_mechanism_valid_C_after25` must be EXITED with no worker. Read/hash the guide, confirm sglang STOPPED and the new namespace empty; use the existing safe source-only deployment process.

Root fills `ROOT_APPROVED.json` from `ROOT_REVIEW_TEMPLATE.json`: status `ROOT_REVIEW_PASS_BOUNDED_C_AFTER28_VALID_REPLAY`, execution authorization true, and actual fixed execution seal. Preserve science/scope/inventory/bridge pins, exact8 and `closed128_must_not_replay`. Keep inherited scientific lineage `source_only_root_review_sha256=b1ff1fad6f5fe08cebec3008b6df861396c2e11c289b8f05815debb78dc3ed14` distinct from the actual new independent review path/SHA.

Root fills `EXECUTION_DRAFT.json` from `APPROVED_TEMPLATE.json`: status `APPROVED_C_AFTER28_MECHANISM_VALID_REPLAY_ONLY`, `root_approval_sha256` equal to the actual new ROOT_APPROVED bytes, and actual execution seal. Preserve exact8, CPU112–119/eight threads/max1, output/dependency maps, valid split, native `1e-12`, Full inference0 and retry=false. Independently hash the actual draft; invoke the original installer once:

```sh
cd /workspace/guardfed_checks/celeba_mechanism_valid_C_after28_20261009/execution_candidate
taskset -c 112-119 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B install_once.py --draft-sha256 ACTUAL_EXECUTION_DRAFT_SHA256
```

Installer measures original source/data/model/guide/resource readiness and installs the normal supervisor service with autostart/autorestart=false. No current Linux health is claimed here. For SSH Python transport use short argv `['ssh','-p','60350','root@89.22.197.55','python -B -']` and `input=code.encode('utf-8')`; never interpolate a full terminal dictionary into argv or inherit unrelated failure prerequisites.

After actual producer-closed strict output, root binds terminal/start/approval/source identities and calls the original archiver once for the new completed difference:

```sh
cd /workspace/guardfed_checks/celeba_mechanism_valid_C_after28_20261009/execution_candidate
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B backup_completed.py
```

Copy the actual archive/receipt into a fresh local `execution_candidate/backups/<actualtag>`; run `python -B execution_candidate/verify_backup.py <actual_local_delta>`. This safely verifies saved arrays without a CNN. If all eight close in one first archive, expected totals are 89 members/88 content, 72 metric checks, 192 confusion counts and 24 prediction rules. These are expectations only; actual receipt/proof SHAs must come from execution. Root independently adopts offserver results. Preserved failed/partial state is never automatically restarted or counted accepted.
