# Live mechanism evidence v3

For new inspections use v4; `README_v4.md` records the independently found fail-closed diagnostic defect and its correction. The five-model v3 archive remains valid.

Original `evidence.py`, `evidence_v2.py`, their checks and seals remain unchanged. v3 changes acceptance-tool scheduling and diagnostic preservation; the frozen training worker, methods, data, seeds, metrics and manifest remain unchanged.

The first v2 inspection accepted one real terminal result and 100 reused Full records, but also misclassified eight healthy active output directories as invalid. Backup preflight then correctly refused a growing active log (`minus_U_IID_Benign_seed91002.log`) whose hash differed from the inspection. No archive was created; the empty initialized ledger remains the same cohort. This was a tool failure, not a failed training result. The original inspection and the observed failure record are preserved under the project evidence directory.

v3 identifies a live worker by exact adapter/repository/job argv, non-zombie PID and process start ticks. A matching active job is pending, including finalization; it cannot be counted as terminal evidence. Existing failure files remain invalid even with a live PID. Orphan partial outputs remain invalid. Complete outputs still pass all original checked-result, paired Full, source/data/config/checkpoint, 70-round and metric checks. Completed-artifact SHA guards remain unchanged.

Invalid evidence is copied into immutable inspection snapshots, preserving original paths, inspected and snapshot hashes and any change between observations. Backup uses snapshots and deduplicates failures by original-path/content identity. Active logs are not backup inputs. Accepted new model IDs remain an exact difference against the verified restore ledger; reused Full weights are not repacked.

`selfcheck_v3.json` verifies the original four acceptance/backup groups, including checkpoint mixing, corrupt archive/member rejection, and duplicate-ledger rejection. `live_boundaries_v3.json` checks exact-owner versus wrong-job PID, live partials, failure/progress identity rejection, orphan partials and immutable diagnostic snapshots. These are software checks, not new scientific observations.

Actual v3 inspection accepted five new terminal models plus 100 reused Full, with zero invalid and 795 pending (eight verified live owners). The first incremental archive contains only those five new IDs; its 70 members and SHA256 `de867e84fe6185af62368e5ddacbf618d23f432112bb5cc3238dd49d25d08cd8` passed independent off-server verification on BlueBook. This does not complete the 800-job queue or its separate raw/native/shared evaluation.
