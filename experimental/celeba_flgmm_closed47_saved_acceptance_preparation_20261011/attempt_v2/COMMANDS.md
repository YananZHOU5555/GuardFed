# Later root commands (not executed)

Run from `E:/OneDrive/文档/GuardFed`. Read the current server guide first. Obtain the actual full47 `GATE_RESULT.json` SHA only after service `guardfed_flgmm_closed47_valid` is EXITED with rc3, no live candidate worker and no failures. No repeated polling or automatic retry is supplied. Check CPU110 across all threads before the one-shot Linux command; the remote source also refuses restricted-thread overlap.

```powershell
python -B tmp/celeba_flgmm_closed47_saved_acceptance_preparation_20261011/attempt_v2/check_linux.py --gate-result-sha256 <ACTUAL_GATE_SHA256> --report-dir tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001 --allow-original-cached-root-refit
```

The returned `linux_proof_sha256` hashes the exact remote proof bytes copied through stdout. Stop on any error. A timeout leaves remote state unknown: inspect the existing process/receipt before any recovery; never resubmit blindly. Linux whole validation re-fits the unchanged cached-root calibration; it performs no CNN inference.

After actual whole PASS, use its actual SHA:

```powershell
python -B tmp/celeba_flgmm_closed47_saved_acceptance_preparation_20261011/attempt_v2/transport.py --gate-result-sha256 <SAME_ACTUAL_GATE_SHA256> --linux-proof-sha256 <ACTUAL_LINUX_PROOF_SHA256> --destination F:/YananResearchStorage/GuardFed/flgmm_closed47_saved_20261011/attempt001 --report-dir tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001
```

Fresh F guard requires >1 GiB plus 256 MB for archive/extraction. Destination must be new; failure keeps partial bytes and reports. If transport completed but only local verification needs explicit recovery, `--verify-existing` reads that same archive/stderr without SSH or downloading. It does not overwrite any existing extraction.

After successful F transport, bind its compact proof SHA:

```powershell
python -B tmp/celeba_flgmm_closed47_saved_acceptance_preparation_20261011/attempt_v2/verify_arrays.py --transport-proof tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001/TRANSPORT_VERIFICATION.json --transport-proof-sha256 <ACTUAL_TRANSPORT_PROOF_SHA256> --linux-proof-sha256 <SAME_ACTUAL_LINUX_PROOF_SHA256> --metadata-npz F:/YananResearchStorage/GuardFed/added_cnn_exact3_valid_20261010/attempt001/verified_extract/metadata.npz --output tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001/OFFSERVER_ARRAY_REFIT_CHECK.json --allow-original-cached-root-refit
```

Windows reports original array-block validation and actual root-receipt differences separately. It never reports whole-check PASS. Root must review the actual47 Linux proof, F transport/member proof, Windows array report/differences and original accepted FL identities before adopting any additional three-view records. The prior adopted FL interface stays excluded. These commands do not update shared acceptance or dispatch further work.
