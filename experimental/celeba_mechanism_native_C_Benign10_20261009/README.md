# Native minus_C: one completed CelebA scene

The [table](TABLES.md) contains the accepted IID Benign Full–minus_C ten-seed comparison at round 70. [tables.json](tables.json) retains unrounded means and sample SDs; ACC uses percent and paired differences use percentage points. The 10/9/6 panels use the same seed sets for both procedures. Differences are computed within seed before the original statistics function summarizes them.

This is a fixed extraction of native112: Full100 reused, minus_U100 and minus_C12 accepted. The complete C scene contributes twenty original records. The other C scenes remain incomplete: IID F Flip has two accepted seeds and eight scenes have none. Those partial IDs remain visible in [coverage.json](coverage.json); they are excluded from the single-scene means. This is not a completed C100 or mechanism900 comparison, and it is not a three-view table.

[INPUTS.json](INPUTS.json) pins the inspection, actual strict/offserver root proofs, frozen ledger, two accepted source archives, original manifest, Full inventory and statistics/rendering sources. [records.json](records.json) preserves all twenty original inspection rows. [identity_bindings.json](identity_bindings.json) joins each C job/config/result/acceptance to its archive member SHA and same-seed Full reference. It binds the terminal checkpoint, seventy rounds, original recipe, source/adapter hashes, valid split, train/root/evaluation IDs and client partition. No model is unpacked or copied. The full212 original records remain in the pinned source inspection; all previous204 records are checked unchanged.

`build.py` imports unchanged `evidence_v4.statistic` and `summarize`. It adapts the prior native100 extraction to exactly one C scene; the renderer’s existing mean/SD formatting is retained and the paired row is additionally displayed. No scientific metric, calibration fitting, training or inference is implemented. [MINIMAL_SOURCE_REUSE.md](MINIMAL_SOURCE_REUSE.md) specifies the changes. Run once from the repository root:

```powershell
python -B tmp/celeba_mechanism_native_C_Benign10_20261009/build.py
python -B tmp/celeba_mechanism_native_C_Benign10_20261009/verify.py
```

All generated artifacts use exclusive creation; an existing snapshot is not overwritten. The verifier independently computes the 54 mean/SD scalars with `math.fsum` and sample variance (ddof=1), checks 27 rendered cells and 30 individual paired differences, and verifies the unchanged prior Full panel values. Six scope/identity mutations must be refused. [verification.json](verification.json) records measured roundoff and input-source SHA preservation. Rerunning these commands in this sealed directory intentionally refuses existing outputs.

Native AEOD is the absolute TPR gap, not full equalized odds. Native metrics retain the original root-fitted calibration for each procedure. This does not isolate aggregation from calibration. Both shown ten-seed groups use 2.11.0+cu128; their historical/current drivers differ, and the current control driver is 595.84. The broader Full100 has 98 cu128 and 2 cu130 records; neither cu130 record appears in this scene. These are original training-native metrics, distinct from later CPU/GPU three-view replays.

Seed 91001 participated in validation-based recipe selection. The 9-seed panel removes it and the 6-seed panel keeps 91005–91010, but all have validation exposure. Historical test exposure remains disclosed; this extraction performs no test access. Native/shared primary endpoint selection remains with the author. No significance test, seed search, claim of C necessity/causality, or final-test claim is made. Lower disparity or an accuracy improvement after deletion is retained without filtering.

This package is a local table candidate for root review. It does not update canonical paper text, scientific acceptance ledgers, STATE, RUNNING or Git.
