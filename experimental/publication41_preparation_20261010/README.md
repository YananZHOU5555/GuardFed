# Increment41 — exact closed evidence, frozen02:30:59 state

Parent40: `59ec6455c1402ff3bfdac454cbf8de9daf0d216d`. Actual scope is native180 / views170 / FLnew18 plus four separately reused runs / Hybrid23 / baseline900. C table stays70 pairs and complete English author-review prose staysC60. C80 source, future replays and all-complete/test claims are excluded.

ACTUAL_CLOSED_INPUTS is the explicit file/destination manifest. It contains the native exact10 FedSA original archive, receipt,27-entry ledger and root/independent identity proofs; FL exact2; Hybrid exact1. Exactly three new result archives are allowed, with no old model duplication. Native180 archive intentionally remains inside its original root_delta subdirectory. Old26-entry recovery chain and originalstrict source remain referenced in the parent commit.

All FL/Hybrid delivery-seal members are verified even when two nonessential copies are omitted: the identical original checked_record_body is recovered from the explicit parent-commit path/SHA; the historical upload-only collector_transport.tar is not republished, because its seven metadata files are already retained separately and in the result archive. Regeneration is not claimed byte-identical to that historical envelope. No original third-party scientific body is newly published.

Fifteen mutable entry/helper files were frozen into owned frozen_inputs at the root-provided exactSHA version; publication writes those bytes to their original canonical destinations. Later live changes do not silently enter this snapshot. Recovery can recreate owned frozen_inputs from each `frozen_snapshot_destinations` commit path and must match the recordedSHA. This is a publication snapshot, not a replacement of current shared runtime state.

The small publisher uses the exact source-pinned40 AST try-block for copy/-f/-text/index/failure handling instead of copying another large implementation. The verifier executes source-pinned40 code with only fixed parent/count/receipt-name/table-not-republished substitutions; its committed-blob loop is byte-identical. Only two positive/six rejection checks were added. A local preparation KeyError for FLseal files-versus-members schema is retained; no scientific or Git execution occurred, and already frozen bytes were reused unchanged.

Root stage command, only after review:

    python -B publish_increment41.py --closed-inputs ACTUAL_CLOSED_INPUTS.json --closed-inputs-sha256 <actual SHA> --output <this directory>/stage --execute-stage

After a separately controlled commit, verify_increment41.py accepts --receipt/--receipt-sha256/--output and optional --remote, exactly like40. This preparation only ran source checks and read-only plan, with no SSH, inference, training, statistics, stage, commit or push.
