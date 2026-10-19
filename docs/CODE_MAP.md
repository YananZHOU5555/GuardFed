# Implementation map

## Shared engine

`scripts/reproduce_paper_tables.py` is an importable engine. Run experiments through `scripts/run_experiment.py`.

| Operation | Definition |
|---|---|
| Configuration | `ExperimentConfig` |
| Dataset / root / client construction | `load_bundle`, `src/data_loader.py`, `src/celeba_data.py` |
| AD2 aggregation | `guardfed_act_aggregate` |
| AD2+ candidates and aggregation | `ad2plus_candidate_overrides`, `guardfed_ad2plus_adaptive_aggregate` |
| Training | `run_experiment` |
| Aggregation dispatch | `aggregate_round` |
| Metrics | `compute_metrics` |
| Root-fitted thresholds | `fit_group_thresholds` |

The release runner supports the shared-engine implementations of FedAvg, FairFed, Median, FLTrust, FairGuard, GuardFed, GuardFed-AD2 and GuardFed-AD2+. Compatibility branches in the engine are not a substitute for the author-based operators below.

## Additional baseline operators

These are aggregation / postprocessing components. The release does not provide a new unified training queue for every external baseline.

| Component | Source | Interface / boundary |
|---|---|---|
| Fed-NGA | `baselines/fednga/fednga.py` | `normalized_gradient_step`: actual gradients at a common parameter point |
| Huber / gradient collection / hybrid control | `baselines/aggregation/group_a/adapters.py` | `empirical_gradient`, `huber_center`, `huber_thresholds`, `cosine_fairness_hybrid`; no Adam-delta substitution |
| FLGMM author-code adaptation | `baselines/flgmm/flgmm_adapter.py` | Stateful `FLGMMAdapter`; pinned source and MIT notice under `sources/` |
| FedAA policy and round integration | `baselines/fedaa/fedaa_official_adapter.py`, `baselines/aggregation/fedaa/fedaa_round_adapter.py` | `OfficialFedAA`, `FedAARounds`; pinned DDPG source under `baselines/fedaa/official/` |
| LASA | `baselines/lasa/adapter.py` | `lasa_aggregate`: additive parameter deltas, explicit numerical edge handling |
| LoGoFair | `baselines/logofair/adapter.py` | `OfficialLoGoFairDP`: requires the official client source and optional `netcal` calibration |

LoGoFair's expected source is [FedFairPostClient.py at commit 7044815](https://github.com/liizhang/LoGofair/blob/7044815cf813cdad53fca7bab426c4d2194ab505/fedlearn/models/FedFairPostClient.py), placed under `baselines/logofair/upstream/fedlearn/models/`. Its LF SHA256 is `244e8bd3f3a6f5b4945cd8cf93035d5996954acc7f5b61e41e89651d9f88fba2`; the adapter verifies it. That source is not bundled. DP here means demographic parity.

FLGMM preserves the author's maximum-component selection rule, including the documented difference from the paper description. Huber projection, server step size and group/client mapping remain experimental-protocol choices; component availability does not certify every adaptation as an exact original-paper reproduction. Virtual image-ID groups do not establish fairness across real training clients.

## Provenance and historical outputs

Scientific engine definitions are checked against the server source. Removed legacy reporting and scheduling utilities are not dependencies of the release runner. Source hashes changed where entry points or documentation changed; old frozen jobs must not be relabelled with new hashes or treated as newly accepted runs.

Earlier reporting utilities allowed metric-wise run/round selection. Historical tables produced by those utilities require a separate audit and are not automatically equivalent to this release's terminal-checkpoint, full-cohort statistics. The original research branch and server artifacts retain the evidence needed to audit those outputs.
