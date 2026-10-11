# Source-only: minus_F non-IID FedSA10 → saved380 / F80

This package has **not** executed transport, inference, fitting, scientific verification, table generation or root adoption. Future native380, saved380 and F80 evidence pins remain null. Only source compilation, byte-exact inverse checks, unchanged science-loop checks, and synthetic JSON identity tests have run.

Fixed new IDs are `minus_F_non-IID_FedSA_seed91001` through `seed91010`. Preserve accepted saved370 and F70. The old native370 raw inspection contains an additional observed FedSA seed91001: it is retained as raw evidence, while its already adopted `ACCEPTED_NATIVE_SCOPE.json` supplies the exact old370 + Full100 = 470 ordered record tokens. The new accepted scope adds only these ten IDs; it never promotes arbitrary observed records.

## Required root review and execution order

The root operator must inspect SOURCE_DIFF.patch and SOURCE_ADAPTATIONS.json, verify both source seals, and use actual external evidence. Do not run with optimized Python or retry failed science. None of the commands below was executed during preparation.

1. Obtain the **actually root-adopted** native380 proof and its SHA, and a source-bound live snapshot SHA. Bind once:
   `python -B bind_native.py --native-root <actual ROOT_DELTA_VERIFICATION.json> --native-root-sha256 <actual SHA> --prepared-seal-sha256 <SOURCE_FILES_SHA256.json SHA> --snapshot <actual snapshot> --snapshot-sha256 <actual SHA>`.
2. After checking the full server guide and the existing resource controls, run `python -B execute_once.py` exactly once. Retain command/stdout/stderr/exit records; write `ACTUAL_EXECUTE_EXIT.json` with the measured `returncode`. This invokes the unchanged closed-ID / no-active-worker / CPU111 / source / receipt-chain gates, then the original exporter and original off-server saved-array checker. It neither fits nor replays models. If the remote saved arrays are not ready, do not claim acceptance.
3. Run `python -B join_saved.py` once, retaining measured `ACTUAL_JOIN_EXIT.json`. It validates unchanged native/model/config/data/checkpoint and Full-reference joins, 90 metrics / 240 confusion counts / 30 prediction rules, with native tolerance 1e-12 and exact observed difference 0 required.
4. Execute the existing `cpu_release_readonly.py` on the same server under the inherited CPU111 command, retaining `CPU_RELEASE.json` stdout and measured `CPU_RELEASE_EXIT.json`. No resource settings change.
5. Finalize: `python -B finalize_actual_handoff.py --archive-sha256 <actual> --offserver-sha256 <actual> --index-sha256 <MECHANISM_INDEX.json SHA> --join-sha256 <NATIVE_IDENTITY_JOIN.json SHA>`. Seal the actual compact delivery as `ACTUAL_DELIVERY_FILES_SHA256.json`, with no large artifacts copied into this namespace.
6. Root fills `ROOT_ACTUAL_PINS.json` from its template using only actual native/transport/join evidence and independently recorded source review. Run `python -B adopt_root.py --actual-pins-sha256 <actual SHA>` once. The output is a new saved380 root, not canonical STATE and not a statistical table.
7. Only after that adoption, fill `ROOT_BINDING.json` from its template with the actual saved380 adoption/index/native-root paths and SHAs, root status, and `FILES_SHA256.json` SHA. Preserve a root source review as `ROOT_SOURCE_REVIEW.json`. Run `python -B build.py --binding-sha256 <actual SHA>`, then `python -B verify_saved.py --binding-sha256 <actual SHA>` once; retain verifier stdout as `ACTUAL_VERIFY.stdout.json`.
8. Root adopts the checked table: `python -B root_adopt_actual.py --binding-sha256 <actual SHA> --verifier-sha256 <ACTUAL_VERIFY.stdout.json SHA> --root-source-review-sha256 <actual SHA>`.

F80 requires 160 Full/minus_F records, 8 complete scenes, 1296 mean/SD scalars and 648 display cells. Original F70's 140 record objects/JSON tokens, 1134 scalars and 567 cells must remain exact. Fixed 10/9/6 seed panels, paired differences, metric definitions, calibration settings, source functions and tolerance are unchanged. No cross-scene aggregate or necessity/significance claim is introduced.

All future bulk output remains on the server or under the existing guarded `F:/YananResearchStorage/GuardFed/` destination. The original exporter/download/verifier checks require the `Yanan 2TB` volume and capacity; there is no internal-drive fallback. This source package performs no F-volume access itself.
