# One-record Windows fit diagnostic — source only

Fixed record: `FLGMM_Tg20_L2.0_lr0.001_IID_Benign_seed91006_fullcoverage`, index 4 of the sealed exact47. The original failed whole-batch array command and its failure remain unchanged. This is a distinct diagnostic, not recovery, acceptance, or a retry of 47 records.

The helper extracts the unchanged setup prefix from sealed `attempt_v2/verify_arrays.py`, stopping before its result loop. It reuses the original `verify_one` identity preamble, root reconstruction, `fit_views`, `predict_views`, scoring and recursive field-diff helper. It calls `fit_views` once, only on this record's original saved root margins. Original source/member/volume guards remain active. The original 1e-12 native comparator is only reported diagnostically; differences are not converted into acceptance.

Root may execute once from the repository, after reviewing the source seal:

```powershell
python -B tmp/flgmm47_windows_fit_diagnostic_source_20261011/diagnose_one.py --source-seal-sha256 <FILES_SHA256_SHA> --allow-single-record-cached-root-diagnostic
```

Output is the separate `saved_acceptance_actual001/OFFSERVER_SINGLE_RECORD_DIAGNOSTIC.json`; `.started.json` prevents blind repeats, and diagnostic failure is preserved separately. No original failure/output is overwritten. Output includes complete actual/saved fits, recursive fit/root differences with float hex values, per-view prediction mismatch counts, complete actual/saved metrics and group counts, metric/count differences, original checkpoint/array identities and actual runtime versions. No platform-cause or harmless-difference conclusion is preset.

Local preparation performed compilation and AST-only checks; no Torch import, fit, CNN, SSH, transport or scientific acceptance. The original setup hashes all already transported members before the single-record diagnostic; it does not refit other records. Large arrays remain on the guarded F volume. Final scientific adoption remains blocked by the preserved Windows array-fit failure until root explicitly determines the evidence boundary.
