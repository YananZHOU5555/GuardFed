### R3.1 — DFA novelty

We agree that combining existing primitives does not by itself establish a new attack mechanism. DFA's contribution is a coordinated threat setting and evaluation of simultaneous utility and fairness degradation. S-DFA applies the two roles to the same malicious clients; Sp-DFA splits a fixed malicious-client budget between them. In the executed four-malicious-client setting, Sp-DFA assigns two clients to each role. The update attack is explicitly described as a root-reference FedSA-inspired implementation, including its attacker-information assumption. We do not claim a new primitive or super-additive synergy without matched-budget component evidence. The supplied contribution and threat-model replacement text adopts this narrower positioning consistently (also R1.1).

### R3.2 — Individual components

We replaced the grouped interpretation with individual-deletion evidence. On Adult and COMPAS under S-DFA, the analysis separately removes U, C, A, F, V and norm handling N, for both distribution anchors and ten seeds per condition. It contains 280 records: 260 new runs and 20 historical Adult Full controls. Masks are applied after internal candidate overrides, preventing a candidate from restoring a deleted score term.

The deletion scope matters: C/A deletion removes the scoring contribution while geometric filtering remains; F/V deletion retains prediction calibration. We therefore report raw and calibrated outputs from the same checkpoint for the 260 available new models. Historical Adult Full binaries are unavailable, and their missing counterfactual outputs are not imputed.

The image study separately checks U, C and A using matched model seeds and round-70 checkpoints. U and C each cover all ten IID/non-IID–scenario cells with 100 deletion/Full pairs. A currently covers eight cells with 80 pairs. The [A table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_eight_scenes80_20261011/TABLES.md) includes every raw/native/shared view and fixed 10/9/6-seed panel. Non-IID A S-DFA/Sp-DFA and the other five image controls remain unfinished; this response is not an all-component completion claim.

These experiments do not support calling every term indispensable. All 12 COMPAS deletion conditions have higher mean accuracy than Full, and six improve all three means. R3.7 discusses these counterexamples and the prediction-rule dependence. Mixed historical environments and replay devices limit causal attribution; deleting a score term is not an isolated removal of every related pipeline operation.

### R3.3 — Sensitive attributes and fairness definitions

The measured scope is a single annotated binary sensitive attribute. Multi-valued and intersectional fairness have not been experimentally validated. The score interface could accept a risk over a set of groups, including intersections, but this is a proposed extension rather than a completed result.

The present AEOD is the absolute binary true-positive-rate gap: it measures equal opportunity, not full equalized odds. A possible equalized-odds risk is the maximum of the between-group TPR range and FPR range. Estimating it requires positive and negative root examples in every group. With sparse or missing support, the rates and fitted thresholds can be noisy or unidentifiable; a numerical fallback does not solve that statistical problem. Group-dependent prediction also requires the relevant sensitive annotation at inference. The supplied method and limitations text makes these requirements explicit and separates conceptual extensibility from the supported binary-group experiments.

### R3.4 — IID/non-IID motivation and component behavior

We interpret α=5000 as an approximately uniform allocation anchor and α=5 as a milder heterogeneous anchor, rather than a universal model of practical FL heterogeneity. The additional Adult/COMPAS study evaluates α=1, 0.5 and 0.1 in 360 runs (R2, stronger non-IID settings). This splitter allocates sensitive groups; it is not label-Dirichlet or general feature heterogeneity.

We also checked realized allocations. In CelebA, replay of the frozen splitter exactly matches all 400 client totals over 20 partitions. Male proportions range from 40.495–42.969% under IID and 9.832–74.303% under non-IID. The training-only joint audit verifies 1,600 Male×Smiling client counts: every cell has support, but target-label variation is narrower than sensitive-group variation. These reconstructed counts are distinguished from originally logged measurements. The stronger-α tabular audit covers 60 partitions and retains empty and single-group clients rather than resampling them away.

The individual-deletion study covers the two original distribution anchors; it is not a complete component-by-α factorial study. Root–client mismatch, correlated scores and estimation variability may contribute to different behavior, but the experiments do not isolate a unique cause. We report the observed utility/disparity trade-offs and treat these explanations as hypotheses, rather than claiming that stronger heterogeneity establishes universal component benefit.

### R3.5 — Notation table

A notation table is supplied in the companion manuscript text. It distinguishes global, client and root parameters and update signs; participating, benign, malicious and retained sets; U/C/A/F/V signals and norm handling; median/MAD normalization; candidate index and temperature; configured malicious count; group prediction thresholds; client-allocation α; and the separate root-skew controls. It also distinguishes a frozen run recipe from the per-round selection among that recipe's internal candidates. Equation references and the final location still need alignment with the matching submitted manuscript source; the supplied table is an insertion candidate, not a claim that the submitted project has already been compiled.

### R3.6 — Exact trusted root update

Inspection of the executed code revealed a description error: root training minimizes ordinary cross-entropy, not an additional fairness-aware root loss. We supply a correction to match the implementation that produced the results.

At the start of each round, the server copies the current global model and creates a fresh optimizer using the run's configured optimizer and learning rate. It shuffles root minibatches and makes one pass over the root set. The root update equals the resulting parameter vector minus the initial global vector. This update is computed once before aggregation and reused as the alignment and norm reference. GuardFed-AD2+ does not aggregate it as an extra pseudo-client.

Fairness-related client reweighting, risk signals, internal candidate selection and group-threshold fitting are separate operations. The companion pseudocode specifies their order, optimizer initialization and parameter-difference sign convention. This corrects the method description without claiming that an unexecuted root objective was evaluated.

### R3.7 — Adult/COMPAS differences

We agree that emphasizing Adult alone would overstate the mechanism evidence. The new individual-deletion study confirms dataset dependence. Under the train-only COMPAS protocol, all 12 deletion conditions have higher mean accuracy than Full, and six improve all three means. For IID, Full ACC/AEOD/ASPD are 0.64838/0.06130/0.04196; removing A gives 0.65999/0.05352/0.03752. We retain this counterexample to universal component necessity.

Prediction calibration can change the interpretation. For non-IID COMPAS, removing F raises raw AEOD from 0.239597 to 0.250984, but lowers calibrated AEOD from 0.049143 to 0.044831. A calibrated table alone therefore cannot identify the contribution of aggregation.

CelebA provides a corresponding example. In non-IID FedSA, the ten-seed paired minus_A−Full differences in raw prediction are ACC +0.078±0.950 percentage points, AEOD −0.00508±0.01328 and ASPD −0.00170±0.01643: all three means improve after deletion. With native/shared calibration, the differences are +0.053±0.948, +0.00372±0.01267 and +0.00207±0.02164, showing an accuracy–disparity trade-off. These are paired means ± sample SD, not significance or every-seed claims; the fixed nine-/six-seed panels are retained alongside them.

Correlated scores, compensation by candidate selection and root-estimation variability are plausible explanations, not isolated causes. The supplied discussion presents both datasets and the raw/calibrated comparisons, and narrows the claim to conditional trade-offs. No best Full seed is compared with deletion means.

### R3.8 — Explain attacks and comparison methods

We supply an implementation/access table covering each method's objective, trusted-root and sensitive-attribute access, update/gradient interface, aggregation or postprocessing operation, implementation source, adaptation and selection procedure. The attack descriptions specify the sensitive-attribute and update-poisoning operations, their budgets and the synchronous/split allocation. F Flip changes the annotation/reweighting path; it does not change CelebA pixels or task labels.

The completed CelebA comparison contains ten native methods across two distributions, five scenarios and ten model seeds (1,000 records). Nine of these methods also have a separate 900-record raw/native/shared comparison. These scopes are kept distinct. FairFed, FairGuard, their combination, FedAA-DDPG and LASA retain project-adaptation labels. Seven target methods still lack complete accepted coverage; a simplified inspired branch is not presented as the original algorithm.

Huber uses the author-approved identity projection on CNN parameters and is labelled an empirical CNN adaptation, without the original constrained-domain guarantee. LoGoFair uses the official demographic-parity postprocessor with 20 fixed image-ID virtual cohorts and root-only fitting; these are not real training clients, and DP here does not mean differential privacy. Its reported native output is the fitted postprocessor's prediction, not the FedAvg backbone cache. The all-negative IID Benign run is retained, so zero disparity is not misrepresented as useful prediction. The [native table](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_ten_method_native_20261010/TABLES.md) preserves the resulting large seed spread and unfavorable comparisons. Full method coverage and a frozen final evaluation remain pending.

### R3.9 — Source code and configurations

A concrete revision snapshot is available at [GuardFed, commit 817f6fd](https://github.com/YananZHOU5555/GuardFed/tree/817f6fd395564b044ed05e360771817f2f4585de). The release includes the accepted comparison tables, per-seed results, frozen configurations, adapter/source records, aggregation code and documented limitations. Large models and raw array archives are retained separately with archive/member hashes; publishing code is not itself a model backup.

This snapshot covers completed evidence, including the A80 response update, and does not certify completion of the remaining benchmark or final evaluation. The final revision release must additionally contain any subsequently accepted methods/results, the frozen evaluation protocol, matching manuscript source and runnable setup instructions. Missing historical checkpoint binaries and adaptation boundaries remain disclosed.

### R3.10 — Limitations

The supplied limitations section covers four substantive boundaries. First, the method requires a representative, annotated root set and training-population information; sparse groups can make risk estimates and thresholds unreliable, and group-dependent prediction needs sensitive attributes at inference. Second, the defense uses a configured malicious count and has been evaluated with small participating populations and binary groups, not million-device or general intersectional deployments.

Third, the softmax result is conditional on score separation in the actual retained set. It does not certify filtering, candidate selection, convergence or population fairness. The executed top-k operation can re-admit intermediate-gate exclusions; this occurs in 116 observed Adult deletion rounds, so no unconditional hard-exclusion guarantee is claimed.

Fourth, performance depends on dataset, prediction rule and selection history. The ablations retain unfavorable effects, and common calibration shows that end-to-end fairness differences cannot be assigned wholly to aggregation. The expanded CelebA evidence is validation-only; selection seeds and the official test partition have already been exposed. A future frozen evaluation cannot be described as an untouched holdout. Historical Table II also has mixed repeat counts, selection conventions and incomplete model provenance. These limitations are stated with the evidence rather than hidden behind a general robustness claim.

### R3.11 — Architecture specification and additional models

We supply the executed model and training specifications. The supplementary tabular MLP is input dimension → Linear(16) → ReLU → Linear(2), using two-logit cross-entropy; encoded columns and preprocessing are specified by dataset. The CelebA CNN has three 3×3 convolution/ReLU/2×2 max-pool blocks with channels 3→32→64→128, adaptive spatial average pooling and Linear(128,2). It receives RGB 64×64 inputs scaled by 1/255, with no batch normalization, dropout or image augmentation.

The supplementary tabular protocol uses Adam, learning rate 0.005, batch size 256, one local epoch and 70 rounds. The image protocol uses batch size 64, one local epoch and 70 rounds; method-specific learning rates are fixed in the selected recipes. Root training uses a fresh configured optimizer and one pass each round.

CelebA adds both a non-tabular dataset and a CNN, addressing the original single-MLP evidence gap. It does not isolate an architecture-only effect or establish generality to arbitrary models. The original submitted MLP must be matched to its archived source before assigning its final manuscript specification; it is not assumed identical merely because both models are called MLPs.
