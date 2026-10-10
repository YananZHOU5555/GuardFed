# F IID S-DFA ten-record saved-output increment — source preparation only

This package has not bound a future native proof, contacted the server, collected an archive, evaluated a model, fitted a calibration rule, or accepted any new result. `PREPARED.json` deliberately keeps `native_binding`, `native_target_total`, future root path/hash and authorized snapshot path/hash null.

The fixed replay scope is `minus_F_IID_S-DFA_seed91001` through `seed91010`. Parent replay330 is the actually adopted FedSA batch, root `62475bb89325d0b229977629bb879d99994c4d313cb7924db9a5efbce6c9ec25`, index `d580cbe700a139cd073b79e7d2defd7e99fcd0d016a53347d459713a5023525b`. If this exact batch later passes, replay acceptance becomes 340/800 and the remaining620 transported/accepted subset becomes 160, from 150. Native and replay totals remain distinct.

## Minimal changes from the executed FedSA package

- Namespace, exact scene IDs, parent replay/receipt pointers, counts 330→340 and 150→160.
- Native source validation explicitly joins two archives: already accepted native333 supplies seeds 91002/91003/91004; a future root delta must supply the remaining seven. It may include other native IDs, but only these ten enter this replay increment. The new native proof must be the next direct delta after native333, retain all 433 old record objects/order and the 47-entry ledger prefix, and have total ≥340. If native acceptance is split over multiple future deltas, this package rejects it; do not silently loosen that chain.
- The new binder requires an independently SHA-bound full-five snapshot containing all ten remaining620 remote-closed IDs and no recorded remaining620 failure or source change. This is an additional metadata readiness gate, not a substitute for the original fresh server preflight.
- The join's full per-record scientific/identity body is byte-identical. The adopter's saved-record loop is byte-identical; its native archive loop differs only in the fixed attack literal. `execute_once.py`, `remote_cpu111_export.py` and `cpu_release_readonly.py` are byte-identical. The remote preflight body after its fixed candidate/previous-receipt header is byte-identical, including guide/source/data identity, CPU111 ownership/quota, producer/duplicate/failure checks and exact-ten closure.

No statistical definitions, native tolerance (1e-12), source/parameter/seed/affinity protocol or underlying original transport/saved-array checker was changed. Evaluation remains from the existing frozen remaining620 lane (CPU112–119, eight threads); transport is CPU111/single-thread/nice10/idle IO/CUDA hidden. Fresh F volume `Yanan 2TB`/Healthy/capacity checks remain mandatory. The archive is only the ten new saved-output records (92 members); no Full weights, previous models, images or old archives are repackaged.

## Actual source checks

`SOURCE_CHECK_V2.stdout.json` is the final local metadata/source check: 13 sources compile, original loops/guards compare as described, one synthetic positive and twelve independent rejection cases pass. Future proof/snapshot fixture objects exist only in memory and are never persisted as actual evidence. The initial fixture run is retained separately: its mutable synthetic new-ID list could contaminate later negative cases. V2 copies that list and rechecks the positive base before every rejection; runtime sources did not change. Neither run executes scientific code.

## Later execution, only after root source review and actual dependencies

From the repository root, root supplies real existing arguments; the placeholders below are not authorizations or invented future hashes:

```powershell
python -B tmp/celeba_remaining_F_SDFA10_prepared_20261011/bind_native.py --native-root ACTUAL_NEXT_ROOT --native-root-sha256 ACTUAL_NEXT_ROOT_SHA --snapshot ACTUAL_FULL_FIVE_SNAPSHOT --snapshot-sha256 ACTUAL_SNAPSHOT_SHA --prepared-seal-sha256 SOURCE_FILES_SHA256_JSON_SHA
python -B tmp/celeba_remaining_F_SDFA10_prepared_20261011/execute_once.py
python -B tmp/celeba_remaining_F_SDFA10_prepared_20261011/join_saved.py
```

Capture exact outer argv, source SHA, UTC, stdout/stderr and return code for each actual command. The existing finalizer expects `ACTUAL_EXECUTE_EXIT.json`, `ACTUAL_JOIN_EXIT.json`, and original read-only `CPU_RELEASE.json`/`CPU_RELEASE_EXIT.json`. Reuse the original CPU-release read-only invocation after export has stopped; do not run the remote-only helper locally or claim release from a historical snapshot. Failures stop the sequence; retain evidence and do not automatically retry. The original `execute_once` invokes the fresh preflight, original export and guarded download/original offserver verifier once.

After actual saved verification and native join, `finalize_actual_handoff.py --archive-sha256 ACTUAL --offserver-sha256 ACTUAL --index-sha256 ACTUAL --join-sha256 ACTUAL` creates only the compact actual handoff. Root then closes the actual delivery seal and `ROOT_ACTUAL_PINS.json` using the unchanged predecessor contract (delivery SHA/member count, handoff SHA, proposed index SHA, native SHA/count, replay_total=340, new_replays=10). Only root may execute `adopt_by_root.py --actual-pins-sha256 ACTUAL`. None of these actual receipts or root pins is created by this source preparation.

No new table, manuscript claim, final test or overall experiment-completion claim follows from this source package.
