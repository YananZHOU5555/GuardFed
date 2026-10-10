# Reuse and changes

- Queue V2 (`a3461e20…`) is read-only. Its original `source_identity`, inventory projection and approval metadata guards are reused. PLAN and its future checkpoint nulls are unchanged.
- New transport-only code selects explicit remaining620 IDs, verifies actual closed binding/output/native file hashes, freezes a batch snapshot, and maintains its own previous-receipt/ID chain. It excludes original180, Full and already transported IDs; it never creates scientific acceptance.
- Original archive writer: the `with tarfile.open(archive, 'w:gz')` block and immediately following original input-hash loop are extracted verbatim from the pinned after70 `backup_completed.py`. Only caller-supplied file paths and inventory metadata differ. `evidence_v4.verify_archive` is imported unchanged.
- Original saved-array verifier: the function `verify` is reused with exactly two metadata substitutions: the old literal inventory SHA is replaced by `EXPECTED_SNAPSHOT_SHA`; the exact10 descriptive label becomes an explicit remaining620-subset label. All metric/count/prediction/native tolerance/runtime checks remain original source/AST. The original cache SHA is unchanged.
- First export freezes source/review dependencies; later deltas do not repeat them or old models. Both remote and local proofs remain pending independent root adoption with `accepted_offserver=0`.

The executable tests cover metadata/refusal and original-body source compatibility only. No real archive or arrays have been generated or verified by this new adapter yet.
