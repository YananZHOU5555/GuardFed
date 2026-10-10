# Hybrid native after9: fixed three-record source preparation

SOURCE_ONLY / NOT_EXECUTED / NOT_ADOPTED. This package freezes the actual 18:01:36.558278Z observation, not future completions. Original strict collection requires a separate root source review and execution authorization.

The difference between the observed 12 terminal records and the nine root-accepted records is exactly `CosineFairness_lam20.0_tau0.1_lr0.001_IID_F Flip_seed91001_fullcoverage`, seed91002, and seed91003. The existing nine are IID Benign seed91002–91010. Their original root and cumulative offserver objects are copied byte-for-byte, in original order. Four selected-screen reuses and seven three-round canaries stay separate. No F Flip ten-seed table is generated.

## Exact reuse and changes

`collect_once.py`, `verify_saved.py`, `restore_verify.py`, and `storage.py` are byte-exact copies of the already executed `tmp/hybrid_native_after1_20261011` versions. The original checker still validates saved round70/full-train/valid records and CPU-restored tensors; its seed/host identity bridge is unchanged. No CNN, calibration fit, training, final test, method, recipe, seed, or concurrency changes.

`execute_transport.py` changes only the E/F/remote namespace, actual parent9 SHA/count, exact3 IDs, and use of the parent's cumulative `accepted_ids` instead of only its last eight `accepted_new_ids`. The fixed source seal remains an explicit required argument. `SOURCE_DIFF.patch` lists every change. The seven in-memory scope fixtures pass, including rejection of a future seed, reordered prior IDs, wrong prior count, source changes, and wrong seals.

`finalize_saved.py` retains the original per-record full-state/tensor/config/checkpoint loop byte-for-byte, with the original `tensor_sha` source digest. Its metadata counts become 3 models/24 tensors and prior9→12, archive member counts must agree with the actual receipt, and the former ten-seed statistics block is removed. `FINALIZER_DIFF.patch` lists these changes. It uses the original CPU108 release query once after transport; this is a future action, not already observed. The finalizer has only been compiled.

The previous root adopter is `tmp/hybrid_after1_root_adopter_20261011/adopt_exact8.py`. Its original archive/member rehash, native-replay/final-trajectory/config/checkpoint joins and canary/reuse separation are the basis for a later minimal root adopter. Do not run that old adopter for this batch: it hard-codes parent1/exact8/Benign and actual old receipts. A new root adopter needs this batch's real receipt/archive/delivery pins after execution, and is intentionally not supplied with invented expected outcomes.

## Runtime gates retained

CPU108 is only a candidate lane. At execution, all process threads are checked for a narrow affinity reservation on CPU108; any owner blocks the operation. The collector rechecks selected producer absence, original guide SHA, original package/source/data identity, supervisor identity, actual cgroup CPU quota and RAM, and CUDA-hidden one-thread/nice10/idle-I/O settings. No assumption that the historical CPU is still free. A failure is retained and stops this batch; no automatic retry or substitution of a different CPU.

Bulk goes only to fresh `F:/YananResearchStorage/GuardFed/hybrid_native_after9_20261011`, after the original `Yanan 2TB`/Healthy/capacity guard. Existing prior models and screen/canary models are not repacked. No bulk F read/write or SSH has occurred during preparation.

## Commands after root review and authorization

From the repository root, set `SOURCE_SHA` to the delivered `FILES_SHA256.json` SHA, then execute each step once, stopping on any failure:

```text
python -B tmp/hybrid_native_after9_20261011/execute_transport.py collect SOURCE_SHA
python -B tmp/hybrid_native_after9_20261011/execute_transport.py download SOURCE_SHA
python -B tmp/hybrid_native_after9_20261011/execute_transport.py verify SOURCE_SHA
python -B tmp/hybrid_native_after9_20261011/finalize_saved.py SOURCE_SHA
```

Retain outer command/exit/stdout/stderr for the finalizer as well as the original transport logs. The finalizer emits root-ready metadata only; parent root adoption and canonical/STATE/LATEST/Git changes remain outside this package. Successful future acceptance would make 12 of 96 new formal records, plus four separately accepted reused screen records; it would not establish complete100, numerical cross-runtime equivalence, three-view replay, or a ten-seed F Flip result.
