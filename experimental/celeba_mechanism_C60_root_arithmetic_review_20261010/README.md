# Prepared independent C60 arithmetic and provenance review

This source reuses the accepted C50 independent standard-library checker. It has not read an actual C60 handoff or snapshot and creates no arithmetic PASS before root supplies the actual handoff, delivery seal and after56 adoption. `raw_records`, `verify_files` and `mean_sd` remain source-identical to C50. No builder, adopter, evaluator, checkpoint inference or threshold fitting is called.

After the real inputs arrive, root calls once:

```powershell
python -B tmp/celeba_mechanism_C60_root_arithmetic_review_20261010/review.py --handoff tmp/celeba_mechanism_C_six_scenes_prepared_20261010/ACTUAL_HANDOFF.json --handoff-sha256 ACTUAL_HANDOFF_SHA256 --seal-sha256 ACTUAL_DELIVERY_SHA256
```

The output is exclusively `ROOT_ARITHMETIC_REVIEW.json`; errors are retained in `ROOT_ARITHMETIC_FAILURE.json`, with no automatic retry. The result follows `tmp/adopt_C60_table_root_20261010.py:review_guard` and does not adopt or write canonical data.

Expected actual scope is120 unique records /60 matched pairs: five IID scenes plus non-IID Benign, each with10 matched seeds. The original10/9/6 panels remain parallel for native/raw/shared_calibration. The independent reader verifies972 mean/sampleSD scalars,486 display cells,1080 confusion-derived metrics and2880 count consistency checks, each seed-paired difference, and the unchanged100 old record JSON spans/order,810 old scalars,405 old cells and162 IID aggregate scalars. The five-IID aggregate file must be byte-identical; no six-scene aggregate is created. N denotes seeds, not models or scenes.

Provenance reuses the adopted C50 proof for old100 records and connects only the new non-IID Benign6+4 strict/offserver archives/receipts and ten additional actual accepted Full900 records. Archive/member inventories, source/config/checkpoint/data/root/valid/70-round identity, unchanged weights and native1e-12 remain required. The source-seal reader handles the actual list-of-members schema. Actual replay device and training Torch summaries come from saved records; no common-device claim is inferred.

Report all subset directions and negative outcomes. AEOD remains absoluteTPRgap, not full equalized odds. The other four non-IID C scenes are incomplete. Selection seed91001, validation selection/history and historical test exposure remain disclosed; the primary endpoint stays pending author choice. No significance, necessity, pure-aggregation causality, final-test or whole-rebuttal-completion claim follows from these six scenes.

Preparation is local only. Source contracts/AST are checked without future statistics. A failed Windows `rg` wildcard probe is retained as `SOURCE_READ_PROBE_FAILURE.json`; the subsequent directory/glob read succeeded and no scientific check was bypassed.
