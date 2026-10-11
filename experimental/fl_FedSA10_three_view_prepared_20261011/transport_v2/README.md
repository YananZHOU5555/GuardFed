# FL FedSA exact10 transport V2 (SOURCE ONLY)

The original candidate exact10 and Linux whole check already passed. The first transport failed at the old fixed F Flip ID list before any NPZ was read or ZIP created. Its exit1, command, stderr, bound source and F attempt001 zero-byte ZIP remain unchanged. The original source/delivery seals remain valid for their original bytes; they are not relabelled V2.

Only `transport_remote.py` changes its single `ids` literal to the existing FedSA manifest. `transport.py`, `contract.py` and `FIXED_IDS.json` are raw copies of their originals. The original remote scientific/archive/resource body is unchanged. `SOURCE_CHECK.json` uses saved actual GATE fields from `runtime/TERMINAL_GATE.json` and the actual Linux proof, not a fabricated GATE or a new download. The first source-fixture line-selection failure is preserved separately; it ran no transport or science.

Root review must precede this ONE transport command (fresh F volume/capacity and CPU110/thread owner guards remain in the original implementation):

```powershell
python -B tmp/fl_FedSA10_three_view_prepared_20261011/transport_v2/transport.py --gate-result-sha256 6bdf450e37432f8071b4ca0fc574edc22441ebb26c6ff7d29453868913953fc6 --linux-proof-sha256 30423f89729798d4726490d05a8c4ed187c2aa826094ea0b15228c919c1894c1 --destination F:/YananResearchStorage/GuardFed/fl_FedSA10_three_view_prepared_20261011/attempt002 --report-dir tmp/fl_FedSA10_three_view_prepared_20261011/actual/saved002
```

Before binding, root may copy the three existing Linux proof/command/exit files from saved001 to saved002, checking each SHA in `LINUX_REUSE_AND_FAILURE_PINS.json`. This is byte reuse, not another Linux execution. No such copy or transport has occurred in this source preparation. First error stops; no automatic retry.

The downstream changes are only identity/namespace binding:

* `bind_audit.py` retains the original per-record/member checks, uses saved002 and the corrected V2 remote source, and requires extra `--transport-v2-seal-sha256` and `--transport-v2-review-sha256`. Root's V2 review must contain `source_adoptable: true` and `transport_v2_seal_sha256` equal to this package seal. All existing source/runtime/actual proof CLI pins remain mandatory. Its future source-pins include the V2 seal/review and four original failed transport files. Original saved_v2 STATIC_SOURCE remains unchanged. Only the original already planned, not-yet-created saved_v2 SOURCE_PINS/FILES are written when root later invokes the binder.
* `audit_saved.py` hash-loads the original zero-fit consumer; only ACTUAL, OUTPUT, STARTED and FAILED globals point to saved002. Original saved_fits/predict/counts/fit-exclusion logic remains untouched. Invoke it only after binding: `python -B tmp/fl_FedSA10_three_view_prepared_20261011/transport_v2/audit_saved.py --source-seal-sha256 ACTUAL_AUDIT_SEAL_SHA --allow-saved-output-audit`. Original verify_arrays.main is never called by this wrapper.
* `adopt_root.py` keeps all original 19 proof roles and scientific/member loops, changing only saved002 and ROOT_INPUTS_v2.json. Root fills the template from actual successful files and exact diagnostic maps before invoking `--inputs-sha256 ACTUAL_INPUTS_SHA --allow-complementary-root-adoption`. This preparation does not authorize or execute root adoption.

For binder, retain the original arguments and add the two V2 SHA arguments. Actual originals: gate 6bdf450e..., Linux 30423f89..., root source-review 8d638726..., preflight 9aa1ebd9..., deployment 2104559a..., authorization 42782e71..., start 1ac50789.... The new transport proof/V2 review SHA remain REQUIRED late bindings.

Native77 plus explicit screen4 supplies 81 checkpoints; prior accepted three-view71 plus this exact10 could become81 only after actual transport, zero-fit audit and root adoption. Current accepted views remain71. Linux whole PASS10 is a separate actual proof; Windows47 original refit failure (completed4) remains failed. No claim of Windows whole/refit PASS, test, full100, or an eight-scene table is made.
