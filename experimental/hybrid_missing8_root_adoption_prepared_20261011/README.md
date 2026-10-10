# Fixed Hybrid8 + screen1 root adopter — prepared, not executed

`adopt_root.py` derives from the actual FL adopter `tmp/fl_FFlip10_capacity_pool32_20261011/adopt_root.py` (SHA `2f62a89d9532fd591f2063b963e3633769940c712abbbc52d255a8e84a077166`). It closes only the two fixed, actually measured batches: eight formal IID Benign seeds 91003–91010 and the originally selected screen seed 91001. Prior Hybrid91002 is reused unchanged from the accepted exact3 root. An executed PASS would therefore cover ten unique Benign checkpoints, comprising nine formal endpoints and one selected screen endpoint; it would not close full100 or create a performance table.

All 33 roles in `ROOT_INPUTS.template.json` point to real files and actual byte hashes. No future values remain. Root must read the source/diff, then copy the template to `ROOT_INPUTS.json`, hash that exact file and invoke once:

```powershell
Copy-Item -LiteralPath tmp/hybrid_missing8_root_adoption_prepared_20261011/ROOT_INPUTS.template.json -Destination tmp/hybrid_missing8_root_adoption_prepared_20261011/ROOT_INPUTS.json -ErrorAction Stop
$hybridRootInputsSha = (Get-FileHash -LiteralPath tmp/hybrid_missing8_root_adoption_prepared_20261011/ROOT_INPUTS.json -Algorithm SHA256).Hash.ToLowerInvariant()
python -B tmp/hybrid_missing8_root_adoption_prepared_20261011/adopt_root.py --inputs-sha256 $hybridRootInputsSha --allow-complementary-root-adoption
```

The adopter hashes both source seals, the two original Linux whole proofs, Windows zero-fit proofs, command/exit/source-review records, F ZIPs and saved members. It reads no images, loads no model or NPZ, and invokes no fit, inference, Torch, SSH or original scientific checker. F archive/member byte hashing runs only when root later executes adoption. The checks retain the original checkpoint/weights/receipt/array/native guard and original `1e-12`; the Hybrid field name is `saved_predictions_metrics_counts_exact`, unlike the FL-specific schema.

Linux whole supplies eight plus one exact original cached-root-refit checks. Windows supplies eight plus one zero-fit saved-output audits. These are complementary roles, not a combined Windows whole PASS or cross-platform exact recalibration. Seed91005 has the exact root-reviewed group-KL diagnostic difference −2.168404344971009e−19; the other seven and screen91001 have none. The adopter requires exact map equality and hexadecimal/value agreement, without a tolerance substitution.

The failed original Windows8 attempt completed zero records because an unused Linux `resource` import occurred before the first record. Its command, exit1, `.started` and `.failure` remain pinned. Windows002 uses the explicitly reviewed canonical-only wrapper, fresh output and exit0. Historical exact3 combined Windows failure belonged to FLGMM; the Hybrid91002 record itself had exact saved evidence and is not relabeled as failed.

`SOURCE_CHECK.json` records compile, positive/three refusal metadata fixtures, original utility and archive-member-loop raw-byte equality, six unchanged original per-record guard AST statements in each fixed batch, and complete source inverse restoration. This is source preparation only. The first static check rejected a stale parent SHA before any fixtures or archive/science work; its original checker and failure are preserved. The corrected check binds the actual parent file. Source-only schema correction for the old exact3 failure uses its actual document hash plus the original `SCIENCE_STDERR.txt` pin; no old evidence was edited.

No STATE, shared LATEST, Git, running package, canonical adoption or scientific output was modified. `ROOT_SCIENTIFIC_ADOPTION.json` does not exist yet and can be written only by root's authorized invocation. If any guard fails, preserve the failure and do not blindly retry.

Final source adds one exact outer single-consumer SHA9408 guard. Inner saved_science remains d512, as both actual proof fields record. The unexecuted nine-member first seal/source and its actually executed metadata fixtures are preserved under source_v1_unexecuted. The final SOURCE_CHECK records compile and one-span inverse equivalence; it does not claim those fixtures were repeated. SOURCE_DIFF remains the original full parent diff; SOURCE_GUARD_ADDITION is the exact final one-line supplement.
