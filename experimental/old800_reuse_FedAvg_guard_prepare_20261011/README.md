# Strict reuse of the one accepted attempt002 FedAvg CUDA record

`reuse_fedavg.py` is stdlib-only. `verify_reused(auth, scope)` reads only the one fixed observed FedAvg acceptance chain and returns `{FedAvg_ID: original_ACCEPTED_object}`. It does not run CNN/fit, parse arrays, infer success from partial outputs, or mark root/offserver adoption.

```python
from reuse_fedavg import verify_reused
reused = verify_reused(auth, scope)
# Root queue must remove exactly these IDs from its planned work.
# attempt003 canaries: only original FedAA and LASA; remaining797 remains root-gated.
```

`AUTH_REUSE.fragment.json` contains the exact requested auth.reused_cuda schema. Task ACCEPTED a050…, old deployment c411…, queue auth c568…, cohort d39… and every receipt/binding/array/acceptance/approval hash are frozen from the actual local observation and INSTALL receipt. The exact FedAvg inventory canonical hash is frozen too. No substitute checkpoint or CPU result is eligible.

The check follows original structured full_hashes identity: `{sha256,resolved_path,bytes,stat}`, where stat is the original v3 `stat_token` body (identical apart from line endings). Before/after equality and current content/stat equality are all required. It rehashes original source/data/artifact files on the server when root calls it, which may include the full shared image cache; it does not read any such file during this source preparation. Existing accepted symlink resolution is preserved in structured identities.

The old sealed deployment, its external source pins, old approval/cohort, task links, original Linux saved pass, native-zero comparison, same-checkpoint metrics/model bytes, round70/full valid19867, original training environment, and exact valid-ID digest are checked. The native inventory target_image_ids_sha256 field is null; the known original replay VALID_IDS_SHA is used instead, without manufacturing an inventory value.

FedAvg-specific failure evidence refuses reuse. The preserved global canary queue failure for the different FedAA task is not treated as a failure of the already accepted FedAvg. Historical preflight files remain hash-bound through original source closure; their age is not required to be fresh now. New tasks still require their own actual fresh resource approval.

13 synthetic stdlib fixtures exercised the actual helper with explicitly substituted in-memory test constants, not the production evidence. They check a valid chain and refusal of changed arrays/links/source bytes/source stat/artifact stat/own failure/old auth/cohort/extra reuse/CPU receipt/native discrepancy/binding. No NumPy/model/fit was imported. Fresh server verification has not run: receipt/binding/acceptance body semantics follow the original adapter contract and fail closed if the actual files differ. All original evidence stays untouched.
