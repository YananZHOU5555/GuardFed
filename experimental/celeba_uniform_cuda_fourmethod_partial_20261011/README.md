# partial four-method uniformly-CUDA validation — source delivery

The four actual adopted CUDA100 roots are bound in `INPUT_PINS.json` and `ROOT_BINDING.template.json`: FedAvg, GuardFed-AD2+ Full, FLGMM and LoGoFair-DP. All400 IDs/checkpoints, per-method100-cell grids and the corresponding compact audit/transport identities have been checked. Statistics have **not** been executed. The template explicitly withholds statistical execution authority until root source review.

Only new root-adopted **native** metrics enter the table. For FedAvg/Full/FLGMM this is `records[id].views.native`; for the unchanged LoGo composite it is `records[id].metrics`, its sole frozen-postprocessing view. The old adopted1100 record JSON is read solely to identify each method/distribution/scenario/seed/checkpoint and provide a metadata pointer. None of its old metrics enters a mean or SD. All paths opened by the builder must resolve to compact workspace JSON; F, raw arrays, models, pickle, SSH, fitting and inference are outside this script.

`statistics_original.py` reuses the original11-method builder's mean/stdev and seed-first loops. Only its old-ten-method equality guard is removed because this is an explicitly different four-method CUDA stage. The formatter still loads the original `values_for` function by pinned source AST. `STATS_SOURCE.json` retains the entire parent summaries function and each minimal change. Function text uses normalized LF; no unsupported raw-byte equivalence is claimed. Original statistics remain `statistics.mean` and `statistics.stdev`, sample SD with ddof1.

Outputs after root authorization:

- `TABLES.md`: IID and non-IID blocks; columns Benign/F-Flip/FedSA/S-DFA/Sp-DFA, rows method × ACC/AEOD/ASPD. ACC uses percentage with2 decimals; fairness gaps4 decimals, each mean±sample SD. No best-cell highlighting, composite score, seed selection or rankings.
- `summary_statistics.json/.csv`: the10-seed panel, excluding91001 (9 seeds), and91005–91010 (6 seeds).
- `seed_first_aggregates.json/.csv`: average scenes within each seed first, then across-seed mean/SD; separate IID/non-IID and balanced-all10 scopes. Scenes never serve as independent seeds.
- `records_native400.json`: compact new native metrics with adopted root pointer and old metadata-only pointer, retaining all negative outcomes.
- `constant_predictions.json` and an explicit Markdown count table: every panel/method/scenario reports constant-count/n, including zero counts. The constant class derives from the adopted confusion counts and must agree with original root flags.
- `BUILD_RECEIPT.json`: source/input/output hashes and explicit partial-scope limitations.

The header must remain **“partial four-method uniformly-CUDA validation”**. This is not final17-method coverage or a newly chosen manuscript primary endpoint. Validation selection exposure is disclosed; n9/n6 are sensitivity views, not a pristine test set. Native methods include their own postprocessing and do not isolate aggregation causality. AEOD remains absolute TPR gap. LoGo uses the original recipe07, fitseed1719, netcal1.3.6 CPU postprocessing and20 image-ID virtual cohorts, not actual training clients; DP means demographic parity. Original cu128/cu130 training histories remain unchanged; CUDA forward evaluation does not assert training-runtime equivalence.

## Root execution after review

Do not rerun the exclusive preparation/check scripts after sealing. Create a **new** actual binding file with the same pinned four roots, set `status` to `ROOT_AUTHORIZED_PARTIAL_FOUR_METHOD_UNIFORM_CUDA_NATIVE_TABLE`, `root_execution_authorized:true`, and bind actual root source-review path/SHA. The source review must declare `source_adoptable:true` and this package seal. Preserve native/no-test/no-fit/no-inference fields.

From the repository root:

`python -B tmp/celeba_uniform_cuda_fourmethod_partial_20261011/build.py --inputs <actual_binding.json> --inputs-sha256 <actual_SHA> --source-seal-sha256 <FILES_SHA256.json_SHA> --output tmp/celeba_uniform_cuda_fourmethod_partial_20261011/actual_partial400`

The output must not exist. This source delivery does not execute that command, edit old tables, make a standalone TeX/PDF, alter STATE/Git or adopt the resulting table.
