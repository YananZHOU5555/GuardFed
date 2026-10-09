# Source reuse and scope changes

Baseline sources are pinned in `INPUTS.json`: `native100/build.py`, the existing interim renderer and accepted `evidence_v4.py` (SHA3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef).

- The native104 scope check becomes an exact native112 check: Full100+U100+C12. All204 earlier records must remain identical. Filtering selects only Full/minus_C, IID Benign, seeds91001–91010; the C F Flip partial pair is retained in coverage.
- The build imports the existing `statistic` and `summarize` functions unchanged. Their mean/sample-SD(ddof=1), completeness and seed-pairing rules are the scientific calculation. `summarize` supplies the ten individual C−Full deltas; each10/9/6 panel calls `statistic` on the corresponding individual records or within-seed deltas.
- The existing renderer’s3-decimal ACC/5-decimal gap formatting and10/9/6 seed panels are retained. One row for paired C−Full is shown beside the two procedure rows; the previous renderer stored that row only in JSON. No metric or calibration implementation is added.
- Original accepted archive JSON metadata is read for the ten C jobs. Its config/source/data/checkpoint/terminal identities are joined to the accepted same-seed Full inventory. No model bytes are extracted; original archives and proofs remain unchanged.
- The independent verifier is the native100 verifier reduced from ten scenes/540 scalars to one scene/54 scalars, with27 displayed mean/SD cells and30 per-seed delta checks. It also checks all three prior Full rows exactly and six finite identity/coverage refusal cases.

The scope is one accepted native scene. Other C scenes, other seven controls, formal endpoint selection and final test remain outside this package. There is no new inference, calibration fit, scientific acceptance framework, service or remote command.
