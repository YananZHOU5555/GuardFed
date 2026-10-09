# Internal evidence delta for the rebuttal package

This is a proposed evidence update, not a claim that the manuscript/figure has already been corrected. The original submission, prior audits and current rebuttal package were not edited by this recovery task.

## Newly closed numerical subitem

The separate historical `server_generation_ablation` suite has been recovered from the pinned 5090 archive. Its 260 original records cover Adult and COMPAS, two client distributions, five attacks, and 13 root-construction settings, all at seed123. This includes all 40 PCA-Gaussian records. Three historical raw exports agree exactly with these records, and all 26 setting summaries have been checked to numerical precision. Each setting's n=10 refers to ten distribution–attack scenarios at one seed, not ten independent seeds.

The historical export independently maximized accuracy and minimized AEOD and ASPD over the last ten test-evaluated rounds. For 250 of 260 records, including all 40 PCA records, no common round attains the exported triplet. These exports therefore cannot be presented as the utility and fairness of one checkpoint. The recovery preserves each metric's attaining rounds, the terminal triplet, and a same-round historical joint-selection triplet without replacing the original results or removing their test-selection limitation.

## Figure discrepancy to resolve before a final response

The submitted Fig.3 is a raster on page 11 of the supplied PDF. A candidate using the ten-scenario historical setting averages and `1−0.5*(AEOD+ASPD)` is strongly compatible with all 26 marker positions/colours: 24 candidate centres are within 2 pixels of a same-colour pixel, and the two occluded centres remain within the visible marker extent. The manuscript instead describes S-DFA-only results and defines `FairScore=1−(AEOD+ASPD)`. The original plotting script and point-to-input receipt have not been recovered, so this compatibility must not be written as proof of the exact original plotting source. The corrected source/selection/aggregation/axis definition needs an explicit author-approved figure decision.

Negative candidate settings must remain visible. For example, the COMPAS TVAE 1% real + 9% synthetic candidate has lower accuracy and lower half-sum FairScore than 10% real; CTGAN 1% + 9% has higher accuracy but a large fairness cost. The candidate chain does not support a blanket improvement claim for synthetic augmentation or the specific original TVAE 1% + 9% example. These statements describe the recovered old export, not new same-checkpoint performance evidence.

## Generator implementation boundary

The archived `pca_gaussian_augment` source samples a multivariate Gaussian fitted to the clean root mean/full covariance, with diagonal shrinkage and output-column projection; it does not explicitly perform PCA decomposition or dimensionality reduction. It can be described as the project's PCA-Gaussian statistical control. The archive source is not certified as the executed byte version of each historical run because immutable per-run source/dependency/model/cache hashes were not recorded.

The finite search covered the three known backups, explicit project source/output paths and reachable Git objects. It found ForestDiffusion in driver method lists and reports/logs, but did not recover its historical adapter/call implementation, dependency version, fitted-data/model/cache identity, or historical checkpoint binary SHA. Missing virtual environments/caches were outside the backup scope; this is not evidence that the generator was never installed or executed.

## Status change for the evidence matrix

- PCA separate-suite raw-record/config/metric-round/aggregation recovery: **completed**.
- Historical Fig.3 candidate locations and description conflicts: **audited; original script/input receipt still missing**.
- Historical ForestDiffusion execution and per-run generator/model/cache identity: **still open**.
- Previously verified 840-record synthetic suite and current nine-method CelebA 900: **unchanged by this task**.

Machine receipts: `verification.json`, `acceptance.json`, `raster_compatibility.json`, `pca_source_receipt.json`, and the final `FILES_SHA256.json`.
