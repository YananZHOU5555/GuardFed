# A60 current delivery — saved-evidence checks passed

The authorized A60 generation is now complete. `build.py`, `finish.py` and the original adapted `verify_saved.py` each ran once and exited 0. The current entry points are `REPORT.md`, `HANDOFF.json` and `SAVED_VERIFICATION.json`. Root table adoption remains pending.

`README.md`, `SOURCE_PREPARATION.json` and `PREPARATION_FILES_SHA256.json` are preserved records of the earlier source-only preparation phase, before the real root260/index binding arrived. Their original bytes remain unchanged; their “not yet generated” statements describe that earlier phase only.

Current scope is five complete IID scenes plus non-IID Benign, 120 records / 60 matched pairs. All fixed 10/9/6-seed panels and three prediction views remain. The adopted A50 five-IID aggregate JSON is byte-identical, with no mixed-distribution aggregate. Other four non-IID scenes and native-only F Flip records are excluded.

Actual checks cover 1,134 statistic scalars, 567 mean/SD display cells, 1,080 metrics from saved group counts and 2,880 base-count checks. Current Full replay is 5 CPU / 55 GPU, with 59 cu128 / 1 cu130 training records; minus_A is 60 CPU and 60 cu128. Native/shared equality is not independent confirmation. Validation selection, prior official-test exposure, final-endpoint/final-test and author-review limits remain.

No new model inference, threshold fit, training, SSH, bulk download, canonical, STATE or Git operation occurred. TeX is an uncompiled fragment. Root may reproduce the saved verifier via `python -B tmp/celeba_mechanism_A60_candidate_20261010/verify_saved.py`; do not rerun the fresh-output builder or finisher.
