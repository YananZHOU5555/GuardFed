# A50 IID table candidate — author/root review pending

Owned directory only; no canonical table, STATE, Git, training, model, predictions or thresholds were changed. `TABLES.md` and `TABLES.tex` include every raw/native/shared 10/9/6 panel, paired difference, and a separate five-IID seed-first summary. `records.json` contains 50 Full/50 minus_A records; `SOURCE_BINDINGS.json` contains all 50 exact Full joins. The non-IID singleton is preserved through the adopted 251-record source index but excluded from the table.

## Read-only verification

```powershell
python -B tmp/celeba_mechanism_A50_IID_candidate_20261010/verify_saved.py
```

The builder and finisher refuse existing output. Their already executed commands were:

```powershell
python -B tmp/celeba_mechanism_A50_IID_candidate_20261010/build.py --adoption tmp/celeba_mechanism_remaining620_after240_root_adoption_20261010/ROOT_ADOPTION.json --adoption-sha256 edd16e71d6fef9f6e2fc7fd15a7b824d2dfb2a7f809290990878b23477a1e838 --index tmp/celeba_mechanism_remaining620_after240_root_adoption_20261010/MECHANISM251_INDEX.json --index-sha256 fd1531be09ccaa90190e174dc46296632023db38fdfc11ed249c1e188310c5e6
python -B tmp/celeba_mechanism_A50_IID_candidate_20261010/finish.py
```

`SOURCE_DIFF.patch` is against the sealed canonical A40 builder. `SOURCE_REUSE.json` binds byte-identical scientific functions and the original per-record identity loop. `AGGREGATE_SOURCE_PINS.json` binds the accepted C100 seed-first code, adapted only by the already reviewed exact variant-string AST mapper. No new statistical rule or tolerance is introduced. `verify_saved.py` reuses the independent fsum/sample-SD/count checker and checks old A40 records, numeric rows, and all rendered cells.

The LaTeX artifact is an include fragment requiring booktabs, not a standalone manuscript. No LaTeX/PDF compilation or visual rendering was performed. Candidate claims and limits are in REPORT.md; final endpoint and paper adoption remain pending.
