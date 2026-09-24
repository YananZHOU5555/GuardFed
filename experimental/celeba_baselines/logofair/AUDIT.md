# LoGoFair official score adapter — exploratory, not integrated

Source: https://github.com/liizhang/LoGofair ; commit `7044815cf813cdad53fca7bab426c4d2194ab505` (retrieved 2026-09-24). No LICENSE/COPYING file exists in this commit. The clone is local research material, not vendored into the GuardFed repository. Adapter executes the pinned official method ASTs without copying their implementation into our source. LF-normalized `FedFairPostClient.py` SHA256: `244e8bd3f3a6f5b4945cd8cf93035d5996954acc7f5b61e41e89651d9f88fba2`.

## Actual mechanism and callable path

Official `fedlearn/algorithm/FedFairPost.py:Server.train` first trains/reuses FedAvg, calls per-client `get_val_score` and optional `calibration`, then for each post round calls every client's `local_fair_post`, averages the returned global multiplier equally, and calls `local_post_eval`. DP uses global lambda and each client's two nonnegative local mu variables. The latter performs five times as many local refinement steps on `true_H`, then derives distinct thresholds for each client and sensitive group. `adapter.py` follows that order and directly invokes those official routines. It preserves the official DP accumulated gradients (the upstream optimizer does not zero them between substeps).

Input: 1D positive-class probabilities, binary labels, binary sensitive attributes and stable integer client IDs. Pass `softmax(logits, dim=1)[:,1]` or `sigmoid(logit_1-logit_0)` from the existing CNN. `fit` takes calibration/validation samples only. `predict` takes probabilities, sensitive groups and client IDs, never evaluation labels. Client IDs at prediction must correspond to fitted local calibration populations; assigning every evaluation point a dummy client would erase the local mechanism.

```python
from adapter import OfficialLoGoFairDP
post = OfficialLoGoFairDP(calibration=True)
post.fit(calibration_probability, calibration_y, calibration_sensitive, calibration_client_id)
pred = post.predict(evaluation_probability, evaluation_sensitive, evaluation_client_id)
```

`calibration=True` calls the official per-client, per-group `netcal.scaling.BetaCalibration` fitter before the fairness optimization; it is the upstream default. It requires `netcal` and its dependencies. CPU-only PyTorch and NumPy suffice for `calibration=False`, but that setting must be identified as the uncalibrated variant. Each client must contain both sensitive groups; for beta calibration each client/group must have both labels. No automatic pooling/fallback silently changes the method. Empty groups, missing IDs, NaN updates and exact score/threshold ties raise errors.

## Explicit upstream problems and adaptation boundary

1. `data_utils.py:117,125,136` accumulates `A_info` over test, validation and train, while `FFPClient.py:96` divides it by only `val_num`. The resulting purported global group probabilities can sum above one, and reference test attributes. This prototype explicitly computes global group probabilities **from the calibration split alone**, consistent with its probability denominator. This is a disclosed statistics correction; do not claim bitwise equivalence with unmodified upstream main.py.
2. `data_utils.py:127-128` overwrites global Y/A counts per client instead of accumulating; EO global counts thus only describe the final client. DP does not use these quantities.
3. `FFPClient.py:300` updates EO's second local multiplier using the first multiplier gradient, which was zeroed. Its EO feasibility loop has `a or b and i < max_iter`, so the first branch can bypass the iteration cap. **EO is not exposed by this adapter.** Do not add EO to a formal table until these issues have a reviewed, separately versioned repair and reference-equation checks.
4. Official `r_beta` uses naive `log(1+exp(beta*x))`; beta=1000 may overflow (especially small-client configurations). The adapter raises on resulting nonfinite updates, retaining failed evidence. It does not silently tune beta or substitute a numerical repair. 20-client synthetic canary below remained finite.
5. Upstream prediction returns 0.5 at exact equality via sign; adapter refuses such ties until a binary tie policy is frozen. Away from exact ties, binary predictions match the upstream rule.

## CPU verification completed

`python check_adapter.py`: PASS on PyTorch 2.8.0+cpu, 20 clients, 1600 synthetic scores, 3 post rounds, 3 mu/lambda substeps, calibration disabled. Checks deterministic replay; differing per-client thresholds; changing the local constraint changes local thresholds; changing the global constraint changes global lambda; unknown evaluation clients and missing sensitive groups rejected. This establishes that both local and global mechanisms are active, not scientific accuracy/fairness on CelebA. The beta-calibration branch has **not** been runtime-tested because netcal is absent locally. No GPU training or real-data end-to-end run was started.

## Minimum next integration

1. Freeze the calibration/client-ID mapping: current GuardFed evaluation is central and lacks evaluation client IDs. Hold out clean per-client train samples, or partition the clean root by a predeclared mapping; do not fit thresholds on the same validation labels later used to rank configurations. If labels/attributes are needed for a synthetic evaluation partition, explicitly disclose that protocol construction and never use evaluation labels in the postprocessing optimizer.
2. Save unchanged FedAvg checkpoint probability arrays plus stable calibration and evaluation client IDs. Reuse each checkpoint for all postprocessing settings rather than retrain its CNN.
3. Install netcal into an isolated environment, run the same small calibration=True canary, then one real CelebA CPU postprocess canary (e.g. 20 rounds and global/local tolerances 0.02/0.05) against a frozen score cache.
4. Record original checkpoint identity, official source commit, adapter hash, corrected statistics policy, postprocessing configuration, client mapping and failure outcomes. Validate-select settings then freeze before multiseed confirmation.

Existing GuardFed `evaluate_for_reporting` sends `LoGoFair` through GuardFed's global two-threshold calibration and aggregation only labels it `LoGoFair-style`. It is not an implementation of the official local/global constrained algorithm and must not populate the new official baseline row.
