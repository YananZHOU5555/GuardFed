# Single-record diagnostic: bounded engineering recovery

Prepared only; no fit or scientific execution occurred here. The first diagnostic completed one cached-root fit but its recursive comparator failed when it sorted integer group keys together with the saved JSON string keys. Its source, started marker and failure are pinned and unchanged. That comparator error does not explain the original scientific fit mismatch.

V2 fixes only JSON representation and evidence retention. Before any diff, it saves complete actual/saved fits, root receipts, view metrics/counts, prediction mismatch counts and runtime to `OFFSERVER_SINGLE_RECORD_DIAGNOSTIC_V2.MEASURED.json`. Actual fits/root/scored objects then pass through standard JSON dump/load; dictionary keys use the receipt representation while floating-point values remain unchanged. Duplicate normalized keys are refused. Original recursive diff still reports differing float hex values.

The same exact fifth record is fixed: `FLGMM_Tg20_L2.0_lr0.001_IID_Benign_seed91006_fullcoverage`. The original setup and identity guards remain reused, and original root reconstruction, fit, prediction and scoring call ASTs are unchanged. At most one call to original cached-root `fit_views` occurs. Original 1e-12 comparator and all failed evidence remain unchanged. No 47-record retry, CNN, test, transport or scientific adoption.

Root may execute this separate attempt once after reviewing its seal:

```powershell
python -B tmp/flgmm47_windows_fit_diagnostic_source_20261011/attempt_v2/diagnose_one.py --source-seal-sha256 <FILES_SHA256_SHA> --allow-single-record-cached-root-diagnostic
```

Output, started marker, failure and measured objects all have separate `OFFSERVER_SINGLE_RECORD_DIAGNOSTIC_V2` names under the existing root execution's `saved_acceptance_actual001` directory. Any existing V2 artifact prevents another attempt. The helper pins prior source and failure bytes both before and after measurement. Its output is diagnostic only, regardless of whether predictions/metrics match.

Preparation checks: source compilation; equal integer/string key fixture; preservation of float hex and signed zero; an actual one-ULP difference remains visible; duplicate-key rejection; MEASURED save precedes differences; unchanged scientific call ASTs. All checks use stdlib synthetic objects and execute no fit.
