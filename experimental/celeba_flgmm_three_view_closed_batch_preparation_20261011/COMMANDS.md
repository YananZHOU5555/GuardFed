# Future root-controlled execution; not executed

First independently review `SOURCE_DIFF.patch`, `SOURCE_CHECK.json`, the exact47 list, all external proof links, and the screen4 byte adapter. Deploy only this sealed compact source/proof package to a new Linux namespace. Do not copy local E/F absolute paths into Linux file operations: the manifest's `path_map` translates each compact origin to a package member or an explicitly pinned original server artifact.

Root must supply actual, hash-bound files:

- source review: `source_adoptable=true`, `package_sha256=<this FILES_SHA256 SHA>`;
- fresh preflight: `status=ROOT_LINUX_FLGMM_EXACT47_PREFLIGHT_PASS`, actual UTC no older than 300 seconds, CPU120–127 eligible and free across restricted threads, no duplicate batch, nominal reservations within actual quota, selected producers quiescent, source/model/data hashes, healthy services/GPU/cgroup/RAM/storage;
- authorization: `status=ROOT_AUTHORIZED_FLGMM_CLOSED_EXACT47_THREE_VIEW`, exact47 IDs in manifest order, matching package/manifest/source-review/preflight SHAs, CPU120–127, `device=cpu`, `max_processes=1`, `test=false`.

The following is a command template only. Each `<...>` must be replaced by actual root-bound values. Output must not already exist.

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 120-127 nice -n 10 ionice -c 3 \
  /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B \
  /workspace/guardfed_checks/celeba_flgmm_three_view_closed_batch_20261011/source/candidate.py \
  --package-sha256 <actual_FILES_SHA256_sha256> \
  --source-review <actual_source_review.json> --source-review-sha256 <actual_sha256> \
  --preflight <actual_Linux_preflight.json> --preflight-sha256 <actual_sha256> \
  --authorization <actual_authorization.json> --authorization-sha256 <actual_sha256> \
  --output /workspace/guardfed_checks/celeba_flgmm_three_view_closed_batch_20261011/outputs/<fresh_attempt>
```

No deployment, authorization, preflight, queue/service edit, SSH, CNN, root fit, download, saved-array check, or scientific adoption has been performed by this preparation. The entry is fail-stop and preserves partial outputs; do not retry a scientific failure automatically. Its output remains root-adopted count zero until original strict whole saved checks, F offserver member/array verification, and root adoption actually succeed. Use the unchanged original `check_saved` pinned in `SOURCE_REUSE.json`; do not replace whole-receipt equality with the prior diagnostic-only Windows array-block result.
