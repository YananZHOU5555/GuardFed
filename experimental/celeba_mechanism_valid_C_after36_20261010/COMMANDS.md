# Future root commands — not executed

Root verifies PACKAGE/source tar/science/execution against its independent review, checks guide SHA, sglang STOPPED, prior `guardfed_celeba_mechanism_valid_C_after28` EXITED/no worker and empty new namespace. New service is `guardfed_celeba_mechanism_valid_C_after36`; namespace `/workspace/guardfed_checks/celeba_mechanism_valid_C_after36_20261010`.

Fill `ROOT_APPROVED.json` from ROOT_REVIEW_TEMPLATE: status `ROOT_REVIEW_PASS_BOUNDED_C_AFTER36_VALID_REPLAY`, actual execution seal and explicit execution authorization. Preserve exact4, `closed136_must_not_replay` and all source/scope/inventory pins. Historical `source_only_root_review_sha256=b1ff1fad6f5fe08cebec3008b6df861396c2e11c289b8f05815debb78dc3ed14` is inherited scientific lineage; bind the actual new independent review separately, never invent it.

Fill EXECUTION_DRAFT from APPROVED_TEMPLATE: status `APPROVED_C_AFTER36_MECHANISM_VALID_REPLAY_ONLY`, actual root approval SHA and execution seal. Preserve CPU112–119/8threads/max1, valid/native1e-12, exact4, original outputs/dependencies, Full inference0 and retry=false. Hash draft bytes externally, then one-shot installer:

```sh
cd /workspace/guardfed_checks/celeba_mechanism_valid_C_after36_20261010/execution_candidate
taskset -c 112-119 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B install_once.py --draft-sha256 ACTUAL_EXECUTION_DRAFT_SHA256
```

Use short SSH argv `ssh -p 60350 root@89.22.197.55 'python -B -'` and Python code via stdin (`input=code.encode('utf-8')`). Do not embed full terminal dictionaries into argv or reuse previous failure preconditions.

After actual producer-closed strict results and terminal/source/approval binding, invoke the original incremental archiver once:

```sh
cd /workspace/guardfed_checks/celeba_mechanism_valid_C_after36_20261010/execution_candidate
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B backup_completed.py
```

Download actual archive/receipt into fresh `execution_candidate/backups/<actualtag>` and run local `python -B execution_candidate/verify_backup.py <actual_local_delta>`. For one first archive closing all4, expected61 members/60content, 36metric checks/96count checks/12rules; these are expectations only. No future archive/proof SHA is asserted here. Root independently adopts actual offserver evidence; partial/failure state is preserved and cannot trigger automatic retry.
