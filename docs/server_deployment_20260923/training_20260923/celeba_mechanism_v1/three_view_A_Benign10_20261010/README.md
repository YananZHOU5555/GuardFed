# Actual Full–minus_A IID Benign table candidate

This is one complete validation scene (10 paired seeds), with raw/native/shared_calibration views and the predefined 10/9/6 seed panels. It is pending root table review. `TABLES.md` is the paper-readable table; `tables.json` retains full floating-point precision. ACC is percent; paired ΔACC is percentage points; every paired difference is minus_A minus Full. AEOD is the absolute TPR gap, not full equalized odds. SD is sample SD (ddof=1).

`records.json` contains 24 accepted records: 10 Full + 10 minus_A for the table, and the two Full–minus_A IID F Flip pairs retained only for coverage and individual values. F Flip2 never enters a mean. This is not A100, all controls, final test, or completion of the rebuttal.

`build.py` ran once successfully from the repository root:

```powershell
python -B docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_Benign10_20261010/build.py
```

It refuses existing result output. The command and exit record are preserved. Original `evidence_v4.statistic/summarize`, C-single-scene panels and its independent arithmetic checker are reused. Only exact variant string constants change C→A in the latter two. The original `receipt_identity/normalized` and `full_record` functions join the accepted A12 saved receipts and the same Full900/Full100 references. No model, prediction array or archive is copied here; no CNN or calibration fit runs. Source SHA pins and per-record receipt/binding paths are in `INPUTS.json`, `SOURCE_BINDINGS.json` and record provenance.

Checks: 162 mean/SD scalars; 81 literal Markdown cells; 216 metrics from group counts and 576 structural count checks across all24 records. The independent arithmetic maximum difference is 1.4210854715202004e-14 (within original1e-12); native replay differences remain exactly0 in the root-adopted A12 chain. Native/shared metrics and counts are identical for all24 retained records, so they are not independent evidence of calibration gain.

The n10 native/shared paired means are ACC −0.4298585594201455 pp, AEOD +0.0023127203911768924, ASPD −0.003141867501773782: removing A lowers accuracy and raises the TPR gap while lowering statistical parity disparity. Raw n10 differences are −0.33522927467660396 pp, +0.0006437691249272404 and −0.0018385339232523002. Subsets preserve reversals: raw n9 ΔACC and ΔASPD become positive; n6 ΔAEOD is negative in all views. These are descriptive paired results, without significance or necessity/causal-isolation claims.

For the complete scene, Full replay is2CPU+8GPU and minus_A is10CPU; both training subsets are10cu128. The broader Full100 history includes98cu128+2cu130, and original/current driver/runtime provenance remains in the referenced receipts. This is not a uniform-device final comparison. Native retains the original root-only recipe; shared uses the frozen common root-only rule; the formal primary endpoint remains pending. Seed91001 selection, prior validation exposure and historical test exposure remain limitations; the9/6 panels are not untouched confirmation cohorts. No test inference is performed here.

`SCHEMA_PROBE_FAILURE.log` preserves the initial read-only index-schema probe error; it occurred before the successful builder. Root's later independent proof is intentionally absent from and excluded from the author seal.
