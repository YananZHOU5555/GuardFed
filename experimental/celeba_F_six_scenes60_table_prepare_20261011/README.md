# F60 table source preparation — not executed

This is the smallest scoped derivation of the executed F50 table implementation. The original builder and verifier have hard-coded scene cardinalities, so merely supplying a different binding cannot produce a six-scene table. Six source files are derived; the original statistic/summarize implementation is referenced unchanged.

Scope: the five accepted IID scenes plus non-IID Benign seeds91001–91010; Full/minus_F paired by distribution, attack, seed, data contract and checkpoint. IID remains alpha5000; the added scene is non-IID alpha5. Prior views350 and the exact ten additional IDs are mandatory. A larger native accepted set cannot expand this table.

Expected only, not measured:120 records/60 pairs/six scenes; three views and fixed10/9/6 panels;972 mean/SD scalars,486 display cells,1080 count-derived metrics and2880 base-count checks. Old F50's100 record fragments/order,810 statistics and405 display cells must survive exact checks. Old row selection uses distribution plus attack, so the added non-IID Benign cannot be confused with old IID Benign. The source keeps all signed paired differences and original sampleSD(ddof1); no cross-scene aggregation or significance claim.

## Actual dependencies

Parent F50 ROOT_VERIFICATION SHA cf89d438b80f48df617c994084a69d8d913cf7dc35c65c762d8800517b89590b; tables.json c7a6e30ef60485ee4955052ec26abfa2b709e9bcde92ff6e4a18fa871f9da44f. SOURCE_INPUTS retains prior source/data-metadata pins and adds the current parent and six source hashes. No model or prediction array was read during preparation.

Future adopted views360 root/index/native pins are null in ROOT_BINDING.template.json. There is deliberately no ROOT_BINDING.json. Source compile and reversible-diff checks are not table or scientific acceptance. Parent original artifacts were not modified.

## Root execution after actual adoption and source review

1. Save actual root source review as ROOT_SOURCE_REVIEW.json. Copy the template to ROOT_BINDING.json only after genuine views360 adoption. Set status to ACTUAL_REPLAY360_BOUND_FOR_F_SIX_SCENES60 and root_adopted true; fill actual adoption/index/native_root paths and SHA, actual root_status, and this FILES_SHA256.json SHA. The original loader verifies the exact350→360 chain and root/native/index inspection join before any table arithmetic.
2. Execute once, with actual argv/start/end/stdout/stderr/exit saved externally:
   `python -B tmp/celeba_F_six_scenes60_table_prepare_20261011/build.py --binding-sha256 ACTUAL_BINDING_SHA`
3. Independently run the original saved checker once and save its stdout JSON as ACTUAL_VERIFY.stdout.json:
   `python -B tmp/celeba_F_six_scenes60_table_prepare_20261011/verify_saved.py --binding-sha256 ACTUAL_BINDING_SHA`
4. Root only, after reviewing actual outputs and verifier:
   `python -B tmp/celeba_F_six_scenes60_table_prepare_20261011/root_adopt_actual.py --binding-sha256 ACTUAL_BINDING_SHA --verifier-sha256 ACTUAL_VERIFIER_SHA --root-source-review-sha256 ACTUAL_ROOT_REVIEW_SHA`

Canonical destination: docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_six_scenes60_20261011. The adopter fails if it exists and uses actual output device counts; no future mean, sign, device count or SHA is fabricated. Remaining scope after actual acceptance would be four non-IID F scenes plus four other controls, not complete F100.

## Preparation checks and limitations

SOURCE_DIFF.patch contains the full six-file source diff; INVERSE_EDITS.json records exact forward substitutions. SOURCE_CHECK documents in-memory compilation, declared edit replay and reversible ndiff restoration to the original bytes; original build helpers and inner mean/SD and confusion-count/metric loops are byte-identical. The per-record identity loop differs only in new distribution/attack/alpha.

The first source-only inverse check failed because reverse global replacements collide; it did not run statistics. Subsequent source reading caught a destination-string match that also changed the historical parent path; both unexecuted drafts and their findings are preserved. The final parent/destination identities are now separately asserted through AST. These are preparation errors, not numerical failures. Do not run prepare_source.py again: it is the provenance of this completed source preparation, not the table runner.

No SSH, F writes, model/array reads, fit/CNN/train/test, table generation, statistical checker, canonical adoption, STATE or Git operation was performed. This package does not claim that future runtime gates have passed.
