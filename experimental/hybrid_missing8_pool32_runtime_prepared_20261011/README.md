# Hybrid IID Benign exact8 — prepared source only

Only the eight **already native-root-adopted** formal checkpoints with seeds91003–91010 are selected. Their native parent is `ROOT_HYBRID_EXACT8_ORIGINAL_STRICT_OFFSERVER_CHAIN_ADOPTED`, SHA `1434b40de5116bf3d53bc5a6ae2bd3b90f4e54222f23ad099ff72b1fd3ef1775`. The selected recipe remains `CosineFairness_lam20.0_tau0.1_lr0.001`. Original 70-round CUDA training provenance is retained, distinct from proposed FP32 CPU replay.

The accepted seed91002 three-view checkpoint is skipped by its actual original exact3 adoption (`631d7ee2acf1cbe3453d849523e262f29ff5b354b5ddb56c60e5779e94364456`), receipt, checkpoint and saved-array bindings. The screen seed91001 has a separate identity gap and is excluded. This is neither a future100 inventory nor a new native acceptance.

`candidate.py` loads the unchanged original FL closed-batch helper definitions and original replay AST. It binds the existing **Hybrid private identity** function, not FL strict or `checked_result`. All17 evaluator/root/metric functions and the frozen shared-calibration recipe are reused from their original pinned source/code objects. The original Hybrid GPU strict body is not executed by this bridge. Only the external identity/path hooks, scope text, fixed eight records and resource binding differ. `SOURCE_DIFF.patch` shows the candidate changes; `SOURCE_CHECK_FINAL.json` checks the actual eight metadata joins and scientific source/AST reuse without Torch or scientific calls.

The first real replay must be seed91003. The unchanged replay requires native metric difference≤`1e-12`, preserves the receipt/arrays on mismatch and raises before any next record. Only after success does the runner write `CANARY_PASS.json` and continue the remaining seven in order. Native/raw use strict positive margins; shared uses the original group `>=` thresholds. Root-only calibration uses original clean-train labels; valid labels enter scoring after fit/predictions are fixed. Official train162770/valid19867 and all root/client IDs/config/checkpoint pins remain bound. No test inference or test-label load is authorized.

Eight **compute threads** are fixed; affinity is separately late-bound by root to either eight logical CPUs or a measured same-socket32-CPU pool. No resource is declared available. FL32–63, old11–18 and102–119 are excluded from replay affinity. Root preflight must verify all-thread conflicts, eligible CPU mask, measured quota/memory/storage/GPU health, unchanged data/source/model hashes, no duplicate replay and selected producers quiescent. The unchanged original resource gate has one engineering adaptation: affinity size equals the externally bound pool size; its8-thread/1-interop and thread-containment checks remain. A fresh root approval and SHA-bound preflight (≤300 seconds old) are mandatory.

## Linux replay command for root, not executed here

Install these compact source files under the distinct server namespace `/workspace/guardfed_checks/celeba_hybrid_three_view_missing8_20261011/source`. Native checkpoints, RGB cache and their existing results remain at their original server paths. `MANIFEST.path_map` explicitly translates Windows/F origins to either package-relative sources/proofs or absolute Linux artifacts; no Windows absolute path is passed to Linux.

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c "$ROOT_SELECTED_CPU_MASK" nice -n 10 ionice -c 3 \
  /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B \
  /workspace/guardfed_checks/celeba_hybrid_three_view_missing8_20261011/source/candidate.py \
  --package-sha256 <FILES_SHA256_SHA> --cpus "$ROOT_SELECTED_CPU_MASK" \
  --source-review <actual_review.json> --source-review-sha256 <actual_SHA> \
  --preflight <fresh_actual_preflight.json> --preflight-sha256 <actual_SHA> \
  --authorization <actual_root_authorization.json> --authorization-sha256 <actual_SHA> \
  --output /workspace/guardfed_checks/celeba_hybrid_three_view_missing8_20261011/outputs/attempt001
```

`--cpus` requires a comma-separated **expanded** sorted list, not range notation. The taskset mask can use that same expanded list. Expected service name for later whole validation is `guardfed_hybrid_missing8_valid`. Root supplies actual review/preflight/authorization files; none is fabricated here. The authorization status must be `ROOT_AUTHORIZED_HYBRID_MISSING8_CANARY_FIRST_VALID`; the exact required fields are in `authorize`. Preflight status must be `ROOT_LINUX_HYBRID_MISSING8_PREFLIGHT_PASS`; a32-CPU pool additionally needs measured `same_socket_cpu_pool=true`.

## Saved validation, separately bound roles

After all8 actually succeed and the replay service is `EXITED` (supervisor rc3, zero worker), root may run the unchanged original Linux **whole** `check_saved` once. This rebuilds the original root and refits the saved root margins, with no CNN. CPU110 must first be free at all threads and the FL evaluation service must be EXITED; `--post-replay-preflight` explicitly binds these actual facts. Do not run this concurrently with FL's whole checker.

```bash
env CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  taskset -c 110 nice -n 10 ionice -c 3 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B \
  /workspace/guardfed_checks/celeba_hybrid_three_view_missing8_20261011/source/check_saved.py \
  --mode linux-whole --cpu 110 --allow-original-cached-root-refit \
  --package-sha256 <FILES_SHA256_SHA> --gate-result-sha256 <actual_GATE_SHA> \
  --post-replay-preflight <fresh_actual_post_replay_preflight.json> --post-replay-preflight-sha256 <actual_SHA> \
  --gate-dir /workspace/guardfed_checks/celeba_hybrid_three_view_missing8_20261011/outputs/attempt001 \
  --output /workspace/guardfed_checks/celeba_hybrid_three_view_missing8_20261011/LINUX_SAVED_CHECK.json
```

Only saved predictions and necessary receipts need a fresh F transport. Original model/data caches need not be downloaded again. The verifier requires exactly19 members: `bundle/GATE_RESULT.json`, `bundle/metadata_receipt.json`, `LINUX_SAVED_CHECK.json`, and each of8 `bundle/<ID>/{receipt.json,validation_predictions.npz}`. Root's SHA-bound transport proof must contain `status=HYBRID_MISSING8_F_SAVED_MEMBERS_SHA_PASS`, `verified_extract`, `members`, actual package/gate/Linux SHA fields. Each member is rehashed; original F native checkpoint/job/result bytes are rehashed without loading models.

The Windows consumer is **zero-fit saved-output consistency**: it calls the accepted original saved-output payload validator and seven unchanged original array/prediction/metric/native statements. Only its fit method label is rebound to Hybrid. The original two refit statements are excluded, explicitly; this is not an original whole-check PASS and does not establish Windows bit-identical recalibration. Original root-ID/partition reconstruction is still checked; root-audit differences are retained. No `fit_views`, threshold fitter, CNN or inference is called on the Windows branch.

```powershell
python -B tmp/celeba_hybrid_three_view_missing8_prepared_20261011/check_saved.py `
  --mode windows-saved-output --allow-saved-output-zero-fit `
  --package-sha256 <FILES_SHA256_SHA> --gate-result-sha256 <actual_GATE_SHA> `
  --transport-proof <actual_F_transport_proof.json> --transport-proof-sha256 <actual_SHA> `
  --linux-proof-sha256 <actual_LINUX_WHOLE_SHA> `
  --metadata-npz <existing_F_metadata.npz> --gate-dir <actual_F_verified_extract/bundle> `
  --output F:/YananResearchStorage/GuardFed/hybrid_missing8_20261011/attempt001/WINDOWS_SAVED_OUTPUT_CHECK.json
```

Fresh actual `Yanan 2TB`/Healthy/capacity guards precede F array consumption/output. No E fallback. Existing output, started or failure files reject repeated execution before scientific imports. Failures are retained, no automatic retry. Original exact3 Windows FL group-KL failure and later FL47 Windows refit failure are not relabelled or attributed to Hybrid. Linux whole and Windows zero-fit reports need separate scientific root judgment; both remain unexecuted in this package.

Current measurements are only source compilation and eight small-record identity joins. New forward/fit/training/transport/root-adoption counts are all0. Native1e-12 canary success, all8 actual receipts, runtime feasibility and saved-consumer execution remain pending. No full100, final-primary, superiority, significance or final-test claim is made.
