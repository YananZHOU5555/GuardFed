# Seven-canary closure — PREPARED, NOT EXECUTED

This separate package accepts and backs up only existing terminal outputs from the current seven-run, three-round canary scope. It performs no SSH during preparation, no inference or training, no service mutation, no 96-run dispatch, no recipe selection, and no claim of seventy-round equivalence. Original bound package/source files are not changed.

## Existing scientific acceptance reused

Bound package SHA is `6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230`; original `run_canaries.py` SHA is `5052d4f31fe2fb63ac16706fb287b8ce64efe61b8974f83d5d091b475216cee7`. There is no separate `verify_canaries.py` in that package. `verify_saved.py` therefore selects the exact original final read-only comparison block by AST (import torch, initialize pairs, iterate preflight jobs). `SOURCE_CHECK.json` proves the selected block's AST equality. It imports the original `equal`, original strict five-result checker and package checks; it does not copy or relax scientific comparisons.

The original block checks all five terminal results and the two Benign/S-DFA reference pairs: metrics, trajectory, round summaries, evaluation stats/data contract, state/control diagnostics, all model tensors, round1/2/3 and final Python/NumPy/Torch CPU/CUDA RNG state, training generator states and advanced-import indices. Imported RNG evidence is retained, with the original comparison boundary unchanged. The seven-run preflight artifact set must exactly equal the original GATE_ACCEPTANCE hash inventory, contain no failure record and have five accepted runs plus two references. Model/data/job/source/checkpoint identity stays under the original strict checker. Source/data hashes and six exact resolved targets are rechecked on the server before and after acceptance.

The three-round gate retains selected Tg=20 and covers the early path, not later UCL/monitor behavior. It creates zero formal table samples. No already accepted seventy-round screen models or Full controls are repackaged.

## Root-reviewed future execution

Only after the supervisor service is EXITED and **no stage producer remains**, root may deploy all sealed helper files to:
`/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/seven_canary_closure/`.

Source must be pinned to the actual FILES_SHA256 SHA before invoking the collector once:

```sh
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 106 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B /workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/seven_canary_closure/collect_once.py --out /workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/seven_canary_closure/output --source-seal-sha256 ACTUAL_HELPER_SEAL
```

The collector verifies the guide SHA, all helper files, actual one-core affinity/nice/environment, CPU106 thread ownership, total declared CPU budget versus real cgroup quota, and at least 8 GiB remaining cgroup RAM. CPU106 must really be free; root must not assume an earlier reservation is current. It reads the target supervisor status without treating other stopped services or a target EXITED return code as failure. It never restarts the canary service. The output path must be new and exact; any failure is preserved there and prevents blind repetition.

Transfer only new `seven_canaries.tar.gz`, `BACKUP_RECEIPT.json` and `MEMBERS.json` with an actual completed SCP. The archive contains the original bound source/jobs/package, seven new canary models and complete output/RNG/log files, original gate, authorization/startup evidence, collector source and remote strict acceptance. Every file is hashed before/after archiving; inventory and archive SHA are recorded separately. Root should pin the actually fetched receipt SHA and then run on the local copies:

```powershell
python -B tmp/celeba_flgmm_seven_canary_closure_20261009/restore_verify.py --archive DOWNLOADED_ARCHIVE --receipt DOWNLOADED_RECEIPT --receipt-sha256 ACTUAL_REMOTE_RECEIPT_SHA --out NEW_LOCAL_RECOVERY_DIRECTORY
```

Restoration works on Python3.10: manual exclusive file writes, no `extractall(filter=...)`, no links, duplicate names, absolute paths, traversal or overwritten output. Every archive member is checked for exact size/SHA. It reexecutes the original saved-output comparison locally, including tensor equality and strict metadata/controller checks. Different local Torch/NumPy/Python versions are explicitly recorded; this is saved-record verification, not a claim that the local runtime is the original cu128/GPU training environment. If a dependency or genuine comparison fails, the failure is preserved and no passing proof is invented. No model evaluation occurs.

## Prepared checks and remaining boundary

`SOURCE_CHECK.json` records original comparison AST equivalence. `LOCAL_CHECKS.json` records syntax, CLI and optimization refusals plus safe-recovery metadata failures, with no remote/scientific execution. Actual canary outputs were unavailable to this preparation, so no gate acceptance, archive SHA or terminal result is claimed here. Root must review the helper and run acceptance after real completion; successful startup alone is insufficient.
