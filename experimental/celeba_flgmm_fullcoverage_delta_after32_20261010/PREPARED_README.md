# FL96 exact6: source prepared, not collected

Fixed IDs: FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005..91010_fullcoverage, in manifest order. Actual observed terminal38 and root-adopted32 are inputs; this package accepts zero new records. Proposed successful cumulative result is38/96 plus four separately reused references, subject to the original strict/offserver checks and later root adoption.

After root source review, execute once from E:/OneDrive/文档/GuardFed:

```powershell
python -B tmp/celeba_flgmm_fullcoverage_delta_after32_20261010/run_once.py collect --review <ACTUAL_ROOT_REVIEW_PATH> --review-sha256 <ACTUAL_ROOT_REVIEW_SHA256>
python -B tmp/celeba_flgmm_fullcoverage_delta_after32_20261010/run_once.py verify
python -B tmp/celeba_flgmm_fullcoverage_delta_after32_20261010/run_once.py finalize
```

Run phases sequentially only after the preceding phase truly succeeded. collect requires an actual external root review containing `source_adoptable:true`, `exact_selected_ids` equal to the exact six, and `prepared_seal_sha256` equal to this actual PREPARED_FILES_SHA256.json. Missing/bad review or changed source/latest/previous binding refuses before SSH. No future review SHA is invented. If CPU110 is occupied, original owner preflight records it and does not dispatch; do not proceed without SERVER_COLLECTOR_RECEIPT.json. On any transport/scientific/partial-output failure preserve it and stop, without automatic retry or rerunning successful strict.

All new archive/models/raw/logprefix/restored evidence will go directly to F:/YananResearchStorage/GuardFed/celeba_flgmm_fullcoverage_delta_after32_20261010/batch after fresh Yanan 2TB Healthy/capacity guards. E holds source/config/compact receipts/index/report only. Original archive extraction/member, record and saved full-state tensor checks are reused, with zero forward/data/optimizer/training calls; no prediction-array recomputation is claimed. Shared LATEST/STATE/Git/service/recipe remain untouched. Root independently adopts the actual final handoff.
