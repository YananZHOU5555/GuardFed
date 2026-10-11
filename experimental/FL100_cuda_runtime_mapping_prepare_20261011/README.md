# FL100 historical Linux mapping proposal

This is an actual-path proposal for fresh root binding, not a current server verification or authorization. The frozen API requires **11 runtime dependency roles**; its source package contains12 files. No source adapter changes are proposed.

| Identity source | Records | Historically deployed candidate directory |
|---|---:|---|
| fl_identity0 | 47 | `/workspace/guardfed_checks/celeba_flgmm_three_view_closed_batch_20261011/source` |
| fl_identity1 | 13 | `/workspace/guardfed_checks/fl_three_view_after48_20261011/source` |
| fl_identity2 | 10 | `/workspace/guardfed_checks/fl_three_view_FFlip10_20261011/source` |
| fl_identity3 | 10 | `/workspace/guardfed_checks/fl_FedSA10_three_view_prepared_20261011/source` |
| fl_identity4 | 9 | `/workspace/guardfed_checks/fl_SDFA9_three_view_source_prepare_20261011/source` |
| fl_identity5 | 10 | `/workspace/guardfed_checks/fl_SpDFA10_final100_isolated_v3_20261011/source` |
| fl_identity6 | 1 | `/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010/source` |

Each role in `RUNTIME_DEPENDENCIES.json` is already the exact `{path, sha256}` shape expected by `bind_fl`. Paths come from seven real `SOURCE_DEPLOYMENT.json` receipts. Core/CNN use the original repository locations, with manifest pins and an earlier actual source hash observation. All seven saved_science roles deliberately use the verified final-v3 `source/originals/saved_science.py`; the earliest package does not contain that member. It is identical d512e5… science, explicitly allowed by the current adapter layout checks.

**FFlip boundary:** fl_identity2 must use its API-frozen earlier deployed3f1c… candidate/26cf… seal. The separate capacity_pool32 package that completed CPU evaluation has971edf… candidate/699e… seal. Do not substitute the latter path or claim the earlier candidate produced those CPU results. This affects package provenance, not checkpoint selection.

`RECORD_INPUT_REFS.json` points to each exact deployed manifest row. Root resolves model/result/job/provenance/fullstate and other artifacts from that row’s `runtime_artifacts`: use **server_path**, never the historical `path` which may refer to E/F storage. It covers100 rows and704 unique artifact paths without duplicating the original large manifests or registries. Preserve the four explicit screen records and the original bridge’s CRLF/LF original-job binding. Source/proof path translation stays in each unchanged candidate.bound_bridge; copying a Windows string into a Linux Path is not a replacement.

Root execution shape after actual approval:

```python
row = exact100["records"][index]  # SHA-bound deployed cohort
approval["runtime_dependencies"] = mapping[row["identity_source"]]
# Freshly verify actual paths/SHA, complete candidate package and original source/data/artifacts.
# Root bootstraps exactly one torch/pandas runtime with approved device/threads/affinity.
args = SimpleNamespace(repo=Path("/workspace/GuardFed-celeba-expanded"),
                       torch_module=torch, pandas_module=pandas)
api = adapter.bind_fl(candidate, args, exactcohortpin, actualapprovalpin, resource_gate)
api["replay_one"](wall_seconds=1800)
accepted = api["accept_saved_predictions"]()
```

`original_inputs` in the template is a preparation index for the root caller, not a new adapter argument. The adapter reads original inputs from its unchanged manifest/bridge closure. Actual approval must match that API’s template, including row ID/native index/identity SHA, ordered100, author decision, frozen source seal, fresh preflight, one-worker resources and new output namespace. Supply both SIGALRM and SIGTERM interruption handling in the root executable and avoid stdlib-shadowing filenames such as queue.py.

Unbound: present server source/proof/artifact/data hashes, current availability, new formal adapter/cohort paths, candidate module instance, CUDA UUID/CPU affinity/resource callback/preflight/deadline, actual authorization, outputs and root acceptance. All remain null or explicitly pending; no CNN, cached fit, model access, SSH, F-drive access, STATE or Git operations were performed.

Validation:77 dependency slots match frozen hashes; seven actual deployment receipts;11 shared source/data paths match the historical actual observation;100 unique IDs with four screens; local metadata only. The first generator attempt assumed saved_science existed in every deployed package, which the earliest package disproved. No output was published from that failed attempt; the final mapping uses the actual shared-v3 member and records that fact explicitly.
