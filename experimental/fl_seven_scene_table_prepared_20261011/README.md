# FLGMM seven-scene source preparation — not built

The input scope is the accepted61-record FLGMM table plus exactly ten new non-IID F Flip records, seeds91001–91010. The future71-record adoption is required; successful replay or Linux checking alone cannot open the table builder. The non-IID S-DFA screen singleton remains stored and excluded. The existing six-scene records, order,324 statistical scalars and162 display cells are preserved, with no old-scene numeric recalculation. Five old table artifacts, including the adaptation caption, are checked against the accepted six-scene root and copied with exact original bytes.

This is a small adaptation of the already prepared seven-scene source, which in turn reuses `tmp/flgmm_six_scene_table_20261011/{build,verify}.py`. Original per-record identity, new-scene mean/sample-SD, count reconstruction and fsum/display loop bodies are AST-identical. `check_source.py` compiled the four Python sources and passed nine invalid-metadata rejection fixtures. It did not execute the builder or scientific verifier. The second source check followed the small fixed-root-path and old-caption preservation changes; no scientific calculation occurred in either check.

## Required late binding

Create `ROOT_BINDING.json` from the unbound template only after actual root adoption:

- `root_adopted=true`; `root71=tmp/fl_FFlip10_capacity_pool32_20261011/ROOT_SCIENTIFIC_ADOPTION.json` with its realSHA. Root schema is the previous61 schema: `records` contains the old60 objects in their original order followed by the exact new10; `prior_interface_explicitly_reused` retains the original one object; total71/new10. Linux original whole/root-fit and Windows zero-fit saved-output evidence have distinct roles. Windows whole remains false and the original failure is not relabelled.
- `candidate=tmp/fl_FFlip10_capacity_pool32_20261011`; `candidate_seal_sha256=699e24a9421e684f446346d0eb46252020806ee3250a3d677a3d90c7c821de39`.
- `transport` points to the actual `TRANSPORT_VERIFICATION.json` with its realSHA. The proof must provide `verified_extract` under `F:/YananResearchStorage/GuardFed` and `members`, including `bundle/<exact-ID>/receipt.json` and `validation_predictions.npz` hashes/sizes. The builder reads only the ten small JSON receipts, never the arrays. Receipt/array/checkpoint identities must equal the adopted ten records and frozen manifest. F label/Healthy checks remain mandatory even for receipt reads. Root reported transportSHA `fe2494601b79dffd6245de1a437593b7cf9ce1a286f3fbdb86621bb4bd57d2e7`; it is intentionally not made an executable binding before root71 is available.

One authorized build, with exact actual bindingSHA:

```powershell
python -B tmp/fl_seven_scene_table_prepared_20261011/build.py --binding-sha256 ACTUAL_ROOT_BINDING_SHA256
```

Then root runs the limited independent verification once:

```powershell
python -B tmp/fl_seven_scene_table_prepared_20261011/verify.py
```

The fresh `candidate/` directory gate prevents an unnoticed repeated build. A failure leaves evidence; no automatic retry is provided. Root adoption of the table is separate and is not fabricated here.

## Expected scope, not a measured result

Seven complete scenes/70 records plus one excluded partial record; raw/native/shared calibration, fixed10/9/6 panels. The expanded table has378 statistical scalars/189 display cells. Only the new scene's54 scalars/27 display cells are computed; independent checking reconstructs90 metrics from240 base counts for the new ten receipts. Original324 scalars/162 cells are compared directly with the accepted prior objects. Original six-scene tables and caption remain byte-exact copies. Native=raw is an identity, not independent evidence.

The existing adaptation/validation-selection caveats, alpha5000/5, ACC percent and AEOD absolute TPR-gap definition remain. Original Windows47 refit failure and later root-audit micro-differences remain in the adopted proofs; new10 diagnostics must remain explicit there. Device/environment claims are checked against records. This is not FLGMM100,17 methods, final test, a superiority/significance claim, or cross-platform bitwise-refit evidence. No SSH, inference, fit, model/array reads, STATE, Git or canonical mutation is part of this source task.
