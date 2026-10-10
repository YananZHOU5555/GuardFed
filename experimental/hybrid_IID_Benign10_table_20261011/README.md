# Hybrid IID Benign10: actual three-view table candidate

[TABLES.md](TABLES.md) presents raw/native/shared calibration for the fixed 10, 9 and 6 seed panels. [records.json](records.json) retains all ten accepted checkpoint receipts, three saved metrics and group counts per view, root reconstruction, original fit rules, training/replay environment and source hashes. [SOURCE_BINDINGS.json](SOURCE_BINDINGS.json) binds the actual root adoption and each F receipt. [VERIFICATION.json](VERIFICATION.json) is the actual local arithmetic result.

Both build and verify were executed exactly once and exited 0; the command/source hashes, stdout, stderr and exit records are retained. The builder extracts the original evidence_v4 `statistic` unchanged. The checker separately uses math.fsum/sample SD (ddof=1), and the original F20 per-record count loop unchanged: 54 scalars, 27 display cells, 90 metrics from group counts and 240 structural count checks passed; maximum scalar difference 2.220446049250313e-16. It reads compact accepted receipt JSON, never arrays, models or ZIPs. This agent wrote both entries; this is an independent arithmetic implementation, not a second-person review or a new replay acceptance.

All-10 native/raw: ACC 88.027 ± 1.386%, AEOD 0.02920 ± 0.00903, ASPD 0.09251 ± 0.01410. Shared: ACC 87.708 ± 1.270%, AEOD 0.02321 ± 0.01652, ASPD 0.03383 ± 0.01892. Raw and native are exactly equal per record. In all three fixed panels, shared has lower mean ACC and lower mean AEOD/ASPD; the trade-off is retained without choosing a preferred view. The individual negative outcomes and diagnostics remain in records.json. AEOD denotes the absolute TPR gap.

Seed91001 remains the selected screen checkpoint; seeds91002–91010 are nine formal-coverage checkpoints. The n9 panel excludes only91001; n6 is fixed91005–91010. Training was CUDA with 2.11.0+cu128 for all ten; replay was CPU with the same torch version, eight Torch threads. Original root-only shared thresholds and `>0` versus `>=` rules are preserved. Validation was used in development after initial official-test results had already been viewed. These panels are descriptive; no untouched confirmation, primary-endpoint decision, new final-test evaluation, full100, significance, causal or necessity claim is made.

Linux whole supplied original root-refit validation; Windows supplied zero-fit saved-output checks. The original Windows8 resource-import failure remains preserved (before first record, completed0). Seed91005 retains the exact group-KL diagnostic discrepancy -2.168404344971009e-19; this table does not relabel a combined Windows whole/refit as PASS. Other seven formal records and the screen record have empty exact diagnostic maps. No fit/inference/training was performed by this table task.

The older native-only `outputs/guardfed_tables/celeba_hybrid_IID_Benign10_20261011` already existed and remains untouched; all eight sealed member hashes match. The new candidate remains only in this owned tmp directory. Root may independently review and copy it to the separate `outputs/guardfed_tables/celeba_hybrid_three_view_IID_Benign10_20261011` path. No STATE, entry or Git changes were made.

Execution already completed; root should review the saved outputs rather than rerun the fresh-output-only commands:

```text
python -B tmp/hybrid_IID_Benign10_table_20261011/build.py
python -B tmp/hybrid_IID_Benign10_table_20261011/verify_numeric.py
```
