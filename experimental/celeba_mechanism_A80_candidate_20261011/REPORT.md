# A80 saved-evidence table — built and finished, root verification pending

The actual accepted root280/index were SHA-bound before generation. `build.py` and `finish.py` each executed once and returned exit0, with empty stderr. Command arguments, timestamps, stdout and SHA receipts are retained. `verify_saved.py` has **not** been executed by this agent: root will run it independently once before any table adoption. Neither the candidate nor this report claims root table acceptance.

The table contains80 Full–minus_A pairs /160 records: the original five IID scenes plus non-IID Benign, F Flip and FedSA. Each scene has ten matched seeds and the same terminal round70 checkpoint across raw/native/shared calibration. All10/9/6 seed panels and sample SD(ddof1) remain. Saved group counts supply1440 recomputed metrics and3840 base-count checks. The scene statistics check covers1296 scalars; the unchanged IID seed-first aggregate adds162, for1458 scalars and729 mean/SD display cells. Maximum observed numerical verification difference was1.4210854715202004e-14 under the unchanged original checker.

The original A60 projection retains all120 serialized record objects and their order, all972 per-scene statistics and486 cells. The five-IID aggregate remains byte-identical (`ce6088b23dd0e2181e83393510075a919dc52719d4b654f70a29205d398b5baf`); it averages the five IID scenes within each seed before calculating across-seed mean/sample SD. Non-IID records do not enter that aggregate, and no mixed-distribution eight-scene summary is introduced.

For the new ten-seed non-IID scenes, differences below are minus_A − Full, ACC in percentage points and gaps in their original scale:

| Scene | View | ΔACC | ΔAEOD | ΔASPD |
|---|---|---:|---:|---:|
| F Flip | native/shared | −0.40318 | −0.00341363 | +0.00489673 |
| F Flip | raw | −0.38758 | +0.01095060 | +0.00544140 |
| FedSA | native/shared | +0.05335 | +0.00371859 | +0.00206705 |
| FedSA | raw | +0.07802 | −0.00508330 | −0.00170270 |

These are descriptive paired means, not significance or all-seed claims. Native/shared have exactly identical metric/count objects and are not independent confirmation. In raw FedSA, deletion improves the three reported means, so an unqualified necessity claim would conflict with these results. All other panels, directions, constant values and negative outcomes remain in the artifacts.

Actual replay environments are Full5CPU/75GPU and minus_A80CPU. Full training builds are79cu128+1cu130; minus_A80cu128. Configuration, checkpoint, source and runtime provenance remain per record. Recipe-selection seed91001, validation development exposure and historical test exposure remain disclosed. Non-IID S-DFA/Sp-DFA and the other mechanism controls are incomplete; this is not A100, final-test evidence or completion of the reply. No CNN, fit, training or test ran during table generation. TeX is an uncompiled fragment.

`README.md`, `SOURCE_CHECK.json`, `FILES_SHA256.json` and `HANDOFF.json` retain the original source-preparation phase and bytes. `README_CURRENT.md`, this report, `SUMMARY.json` and `ACTUAL_HANDOFF.json` describe the current built state. The original preparation source seal remains valid; its missing-root-binding source check must not be rerun as a current scientific check.

Root next command: `python -B tmp/celeba_mechanism_A80_candidate_20261011/verify_saved.py`. Do not rerun the fresh-output builder or finisher.
