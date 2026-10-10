# Response to the Associate Editor and Reviewers

**Manuscript:** TDSC-2026-07-3058, “To Kill Two Birds with One Stone: Defending Both Utility and Fairness in Federated Learning Systems”

**Author-review draft — revision incomplete.** This version presents the current responses without internal batch histories. All 24 comments are reproduced verbatim. Proposed manuscript changes have not yet been applied to the matching submitted source, and the unfinished items are listed once at the end. CelebA results remain validation evidence; the official test partition was previously exposed. No final evaluation or submission-ready completion is claimed.

Dear Associate Editor and Reviewers,

Thank you for identifying the gaps in contribution scope, theory and experimental evidence. The response below clarifies the implemented method, presents additional datasets and individual-component evidence, and retains unfavorable results. Reviewer 1 and Reviewer 3 keep their original numbering. Reviewer 2's descriptive headings are navigation aids, not original reviewer numbers.

## Associate Editor

**Original comment (verbatim).**

> This paper proposes a federated learning system that aims to achieve predictive utility and group fairness in the presence of malicious clients. Though the topic is interesting, several concerns were raised by the reviewers regarding the unclear positioning of DFA, the lack of detailed theoretical analysis, and insufficient benchmarking and ablation studies. Please revise the manuscript carefully based on the reviewers' comments.

**Response.** We agree that the revision must clarify the contribution, narrow the theoretical claim, and strengthen the empirical evidence. DFA is a coordinated threat model built from existing attack primitives. The theorem describes conditional suppression of malicious softmax weight; it does not establish that GuardFed achieves the required score separation. Our proposed method description also makes hard filtering, internal configuration selection, update-norm handling, and group-dependent prediction thresholds explicit.

The added evidence includes ACSIncome and CelebA with a CNN. The nine-method CelebA comparison covers two distributions, five scenarios, and ten shared seeds: 900 records. Its three prediction views expose utility–disparity trade-offs and show that native disparity advantages cannot be attributed wholly to aggregation. Individual-component evidence includes complete U- and C-deletion comparisons and eight complete A-deletion scenes. Contrary COMPAS and CelebA findings are retained; these results do not establish that every component is indispensable.

The response and proposed insertions also address imperfect root data, experimental uncertainty, and historical synthetic-result limitations. The remaining A scenes, other image controls, broader method comparability, synthetic provenance, and frozen final evaluation remain unresolved. These additions are validation evidence, not a completed 17-method benchmark or untouched-test confirmation. Manuscript integration and final section/table references remain pending author review.

## Reviewer 1

### R1.1 — Novelty and positioning of DFA

**Original comment (verbatim).**

> 1. Clarification of the novelty and positioning of DFA.
> The proposed Dual-Facet Attack provides a useful setting for jointly examining utility and fairness degradation, but its relationship to existing attack mechanisms could be articulated more clearly. In particular, S-DFA essentially combines sensitive-attribute manipulation with update-level poisoning, while Sp-DFA distributes these two components across different malicious clients. The authors are encouraged to clarify whether the main novelty of DFA lies in a new attack mechanism or in a coordinated threat model and evaluation framework that exposes interactions between the two objectives. A more precise positioning would help readers better understand the contribution without requiring substantial methodological changes.

**Response.** We agree that DFA should be positioned as a coordinated threat model and evaluation framework, rather than a newly invented attack primitive. Its purpose is to examine how attacks on predictive utility and group fairness interact when their roles are colocated or distributed across malicious clients.

S-DFA assigns sensitive-attribute manipulation and model-update poisoning to the same malicious clients. Sp-DFA divides a fixed malicious-client budget between the two roles. In the executed four-malicious-client setting, two clients perform sensitive-attribute manipulation and two perform update poisoning. Sp-DFA therefore changes the allocation of a common budget; it does not give each constituent attack the entire budget. The existing attack configurations support this distinction, but they do not establish super-additive synergy. Such a claim would require matched-budget component controls and is not made.

We also distinguish the implemented root-reference, FedSA-inspired variant from the original algorithm. Access to that reference is an explicit attacker-information assumption. The proposed contribution and threat-model text will consistently describe these primitives, role assignments, budget constraints, and information assumptions. This correction preserves the executed experiments while removing language that could suggest a fundamentally new poisoning mechanism. The defense is evaluated against this defined threat setting; its results are not evidence of effectiveness against arbitrary coordinated adversaries.

### R1.2 — Scope of the theoretical analysis

**Original comment (verbatim).**

> 2. The theoretical analysis could be interpreted more carefully.
> Theorem 1 shows exponential suppression of malicious aggregation mass under the assumption that every benign client's normalized score exceeds every malicious client's score by a margin \mu_t. The result is mathematically reasonable, but the key property required by the theorem is exactly the property that a robust scoring rule is expected to achieve. Therefore, the current result is more naturally viewed as an analysis of the soft aggregation mechanism than as a complete robustness guarantee for GuardFed. The manuscript would benefit from making this scope explicit and, if possible, briefly reporting the empirical score separation observed in representative experiments to better connect the theoretical result with practice.

**Response.** We agree: the theorem analyzes soft aggregation conditional on score separation; it does not prove that the scoring rule achieves that separation. The corrected statement uses the actual final retained clients, selected configuration, and temperature. With h>0 benign clients, m malicious clients, temperature τ, and every retained benign score exceeding every retained malicious score by at least μ≥0, malicious coefficient mass is bounded by m/(h exp(μ/τ)+m). It is zero when m=0; no protection follows when h=0. A perturbation bound additionally requires bounded scaled-update norms. Filtering, configuration selection, root-estimation error, calibration, convergence, accuracy, and population fairness are outside this guarantee.

The Adult/S-DFA diagnostics cover 120 deletion runs and all 8,400 rounds. Strict pre-gate separation holds in 6,065 rounds and fails in 2,335. Actual final sets retain no malicious client in 8,278 rounds; the remaining 122 contain both classes, with positive margin in two and negative margin in 120. Both nonempty nonnegative-margin cases satisfy the conditional bound. Empty-malicious sets are not counted as evidence of strict separation between two nonempty classes.

The intermediate gate is not a final-set exclusion guarantee: top-k can reintroduce excluded clients when k exceeds the gate size, after which softmax uses original scores. This occurs in 116 deletion rounds; malicious mass can reach 1.0. These failures remain reported. The 20 historical Full runs record only rounds 1 and 70; all 40 observed final sets exclude malicious clients, but missing intermediate rounds cannot be inferred and weights are reconstructed rather than directly logged.

The proposed theory revision states these conditions and exceptions alongside the diagnostics. Minimum/maximum separation, not average score difference, is the relevant premise. Round counts are descriptive, correlated observations, not independent replicates or evidence for unmeasured image mechanisms.

### R1.3 — Additional dataset and architecture

**Original comment (verbatim).**

> 3. The experimental coverage could be further strengthened.
> The evaluation is primarily conducted on Adult and COMPAS, both of which are tabular binary-classification benchmarks, with an MLP as the learning model. While these datasets are standard for group-fairness studies and the current experiments already cover both IID and non-IID settings, the generality of GuardFed would be more convincing with at least one additional dataset or model architecture. This is particularly relevant because the method is presented as a general-purpose defense for federated learning systems rather than a defense specialized to tabular fairness benchmarks.

**Response.** We added ACSIncome (120 runs) and CelebA image classification with a CNN, extending the evaluation beyond the original tabular/MLP setting. The initial 240-run CelebA cohort and later validation comparison remain separate because their configurations and selection histories differ. The latter contains 900 records: nine methods × two distributions × five scenarios × ten shared seeds.

Averaging the ten distribution–scenario cells within each seed, then reporting mean±sample SD across ten seeds, native GuardFed-AD2+ obtains ACC 88.420±0.623%, AEOD 0.00967±0.00297, and ASPD 0.06105±0.00466. FLTrust obtains 89.629±0.652%, 0.04585±0.00406, and 0.11456±0.00468. Thus the comparison shows lower disparity with lower accuracy, rather than universal superiority.

We distinguish raw argmax predictions, each method's native output rule, and a shared calibration rule fitted separately to each model's clean-training-root predictions. GuardFed's native and shared outputs coincide. Its native cross-scenario means improve ACC/AEOD/ASPD relative to 6/8, 8/8, and 7/8 baselines, respectively; shared calibration changes these counts to 7/8, 4/8, and 1/8. These are paired-mean directions, not seed win rates or significance tests. Calibration therefore contributes materially to the native disparity comparison; the advantage cannot be assigned wholly to aggregation.

The proposed dataset/model and evaluation sections will report the architecture, prediction rules, separate selection histories, and unfavorable comparisons. Changing modality and model establishes applicability to this tested image/CNN setting, without isolating an architecture-only effect. The frozen split allocates sensitive Male groups, not label-Dirichlet groups; its realized joint Male×Smiling counts have been audited. Complete target-method comparability and frozen final evaluation remain pending, so these validation results do not establish arbitrary-architecture generality or untouched-test confirmation.

### R1.4 — Imperfect root data and synthetic construction

**Original comment (verbatim).**

> 4. The evaluation of the server-side root data could be made more complete.
> The paper usefully studies distributional skew and synthetic augmentation, and the results already show that root-data imbalance can noticeably affect fairness assessment. However, other practically relevant imperfections of the root set, such as label noise, sensitive-attribute noise, or severe underrepresentation of a protected group, are not considered. In addition, the synthetic-root experiment would benefit from a clearer description of how the synthetic generators are trained and exactly what data are available to the server, since this affects the practicality of the proposed root-data construction strategy.

**Response.** We added root-only label-noise and sensitive-attribute-noise studies, plus protected-group underrepresentation using a fixed reservoir that keeps client data unchanged. In benign COMPAS, increasing root sensitive-attribute noise from 0% to 40% raises mean AEOD from 0.05106 to 0.15761. Separately, reducing protected-group share from 50% to 0% raises AEOD from 0.03804 to 0.24644. Zero support prevents identification of the missing group's risk despite an implementation fallback. Under S-DFA, changing the root can also change the attack reference; these are end-to-end sensitivity results, not defense-only effects.

The proposed construction description corrects the server-information assumption. The inspected generator fits dependence on clean root rows but uses training-population support information in projection; the Gaussian-copula path additionally uses training-population empirical marginals. The setup therefore requires more than the small root sample. Synthetic augmentation neither supplies differential privacy automatically nor reconstructs genuinely unobserved groups.

Historical synthetic reporting also needs correction: 797/840 expanded-ratio triplets and 250/260 construction-suite triplets combine metric-wise extrema unattainable at one checkpoint. The latter suite uses one seed; its ten scenarios are not ten independent seeds. A coherent round-70 redraw candidate retains negative evidence: COMPAS TVAE with 1% real plus 9% synthetic root data is worse on all three metrics than 10% real.

This candidate has not repaired the submitted figure or resolved provenance. Recovered coordinates suggest a scenario-aggregation and FairScore-formula discrepancy, but do not identify the original plotting source. The archived PCA-labelled function implements a shrinkage Gaussian without explicit PCA; its historical executed identity and the ForestDiffusion adapter remain unresolved. Missing checkpoint identities and historical evaluation exposure also remain. The proposed text removes blanket synthetic-improvement claims; final figure inputs, formula, scope, and redraw require author review.

## Reviewer 2

### Opening concern — What is new in DFA?

**Original comment (verbatim).**

> The paper is built around the idea that poisoning attacks in FL are usually studied either from the point of view of damaging utility or group fairness. Here, the authors instead propose an attack that combines these two goals by implementing a fairness and utility attack. The motivation is convincing and clear to me. However, the novelty of DFA could be discussed more explicitly. The authors should better clarify what is fundamentally new in DFA compared with the joint use of existing fairness- and utility-oriented attacks.

**Response.** The contribution of DFA is a coordinated threat model for evaluating utility and group-fairness degradation, rather than a new poisoning primitive. Sensitive-attribute manipulation and model-update poisoning already exist. DFA specifies how their roles are combined and how a defense is assessed against both objectives.

S-DFA applies both roles to the same malicious clients. Sp-DFA divides a fixed malicious-client budget between the two roles: in our executed four-malicious-client setting, two clients manipulate sensitive attributes and two poison model updates. Sp-DFA therefore does not give each component a separate full adversarial budget. This distinction matters when interpreting comparisons between the attack settings.

We do not claim that combining the components produces super-additive harm or establishes a fundamentally new attack mechanism. Such claims would require matched-budget component controls beyond the evidence presented. We also identify the implemented update attack as a root-reference, FedSA-inspired variant and disclose its access to that reference. The novelty claim is consequently limited to the coordinated threat setting and its utility–fairness evaluation.

### Client population and participation scale

**Original comment (verbatim).**

> - It was also interested to read the impact of malicious-client Ratio section. I was wondering how many clients are needed for this attack and this section explained this to me. One comment that I have about this is that my guess is that when I have only a few clients (like in the experiments) these kinds of attacks are more doable than in real-life scenarios where I have millions of clients. In this case, the presence of malicious clients would be probably hidden by the amout of clients and by the selection of the clients done by the server. A discussion about this could be interesting.

**Response.** Our experiments use a controlled population of 20 clients, all participating in every round, with four nominal malicious clients in the attack settings. They do not simulate sampling 20 clients from a larger population or validate deployment across millions of devices. The filtering implementation also uses the configured malicious-client count, which is an additional practical assumption.

A larger population does not by itself determine the number of attackers selected. If a round samples m clients uniformly from N clients, of whom M are malicious, the malicious participant count follows a hypergeometric distribution with expectation mM/N. Increasing N reduces this expectation when M is fixed, but not when the malicious fraction M/N is fixed. The probability of selecting no attacker is C(N−M,m)/C(N,m), provided m≤N−M. Device availability and nonuniform selection can change these probabilities.

Our results therefore describe performance under the stated participating-client budget. They do not establish resilience to realistic availability patterns, adaptive selection, or large-population scaling. These distinctions address the reviewer's concern without treating population size as evidence that poisoning becomes harmless.

### Broader benchmark, including non-tabular data

**Original comment (verbatim).**

> - Authors only tested the methodology on two (simple) tabular benchmarks. I'd recommend 1) introducing other tabular benchmarks 2) introduce at least a non tabular benchmark (Celeba for instance if they want to use images) to make the paper stronger and to show that the method generalizes also on non-tabular datasets. Moreover. I'd avoid saying "Mini-Benchmark", for a journal paper I'd expect an extended benchmark.

**Response.** We have added ACSIncome as a third tabular task and CelebA as an image-classification task using a CNN. The expanded CelebA validation comparison contains 900 records covering nine methods, two client distributions, five scenarios, and ten shared model seeds. A separate native-method table adds LoGoFair, yielding 1,000 records across ten methods; it does not expand the nine-method common-calibration comparison to ten methods.

CelebA uses 162,770 training and 19,867 validation images, with Smiling as the target, Male as the annotated grouping variable, and RGB images resized to 64×64. These results provide evidence beyond tabular inputs, but do not establish generalization to arbitrary vision tasks or complete the 17-method target comparison.

The additions also contain counterexamples to uniform superiority. On ACSIncome non-IID S-DFA, GuardFed's mean ACC/AEOD/ASPD are 0.768731/0.043193/0.014080, compared with FLTrust's 0.796770/0.023964/0.044124: GuardFed has lower ASPD, but lower accuracy and higher AEOD. This ACSIncome task uses the documented 2018 California split, encoded features, and customized SEX grouping. We retain these tradeoffs and propose replacing “Mini-Benchmark” with a description of the actual datasets and comparison scope. Remaining method coverage and final evaluation are separate unfinished items.

### Stronger non-IID settings

**Original comment (verbatim).**

> - With the goal of having an extended benchmark, it would also be beneficial to have a more non-IID experiment with values closer to 0.

**Response.** We completed 360 additional runs at Dirichlet α=1, 0.5, and 0.1: Adult/COMPAS × GuardFed/FedAvg/FLTrust × Benign/S-DFA × ten seeds. These extend the historical α=5000 and α=5 settings. The implemented allocation operates on sensitive groups, so a smaller α should not be interpreted as simultaneously testing every form of label, feature, and participation heterogeneity.

The stronger settings expose substantial performance losses. On Adult benign data, GuardFed accuracy changes from 0.81568±0.01296 at α=1 to 0.78040±0.02401 at α=0.1, reported as mean±sample SD. Low disparity is therefore not sufficient evidence of preserved utility. New COMPAS runs use train-only preprocessing and are kept separate from historical results requiring different preprocessing controls.

We also checked realized allocations. At α=0.1, the 200 client observations across ten Adult partitions contain 29 empty clients and 104 nonempty single-sensitive-group clients; COMPAS has 44 and 78, respectively. These are allocation observations, not 200 independent training replicates. A fixed four-of-20 malicious-client ratio consequently need not imply a fixed fraction of attacked examples. Empty clients, sparse groups, low-performing seeds, and nearly constant predictions remain in the evidence. These experiments broaden the heterogeneity assessment, but are not a complete component-by-α factorial study.

### Practicality of a server-side root dataset

**Original comment (verbatim).**

> - I know that in the literature papers assume the presence of this server-side validation dataset but this is not always a realistic scenario. Usually in FL clients do not want to share their data with an external server. A discussion about this and about solutions that could be used to build this dataset would be beneficial for the paper.

**Response.** GuardFed assumes access to trusted, sufficiently representative root data; federated learning does not guarantee that this resource exists. Public data, independently collected data, or authorized contributions are possible sources, but each introduces distribution and governance constraints. We have not implemented a privacy-preserving acquisition protocol.

The actual information requirements extend beyond the root sample. The simulator uses training-population statistics for client reweighting, and inspected synthetic routines use population support information; the Gaussian-copula path additionally uses empirical population marginals. Root threshold fitting requires sensitive labels, and applying group-specific thresholds requires the group attribute at inference. Synthetic generation and keeping client records local do not themselves provide differential privacy.

The completed sensitivity experiments demonstrate why this assumption matters. On benign COMPAS, increasing root sensitive-attribute noise from 0% to 40% raises AEOD from 0.05106 to 0.15761. In the separate fixed-reservoir experiment, reducing protected-group root share from 50% to 0% raises AEOD from 0.03804 to 0.24644. A fallback cannot identify the risk of a group absent from the root.

For CelebA, all ten unique 16,277-example roots contain every Male×Smiling cell, with minimum cell support 2,666. This describes our stratified experimental resource, not a guarantee of practical availability. Historical synthetic-generator execution identities and the final figure correction remain unresolved.

### Table II — Report all computed values

**Original comment (verbatim).**

> - I think it would also be useful to have in Table II all the values that you computed and not only the ones over the threshold. Having N/E in the table is something that seems strange to me.

**Response.** We agree that a well-defined disparity should remain visible when accuracy is poor. Hiding it below an accuracy threshold prevents readers from assessing the utility–fairness tradeoff; numerical zeros and low-utility outcomes should instead be reported together.

We traced all 480 metric cells of the submitted Adult Table II (436 displayed numbers and 44 N/E cells) and recovered all 44 values suppressed as N/E. The source records match 435 displayed values and identify one discrepancy: FairGuard/IID/FedSA accuracy is printed as 59.13%, whereas its source summary gives 54.13%. We preserve both records and do not infer an unsupported explanation for that discrepancy.

Two distinct reconstruction artifacts are available: one retains the historical selection rules; the correction candidate takes all metrics together from round 70 within each run. A finite stored fairness value alone does not prove that its subgroup denominator was nonzero, so unavailable denominator evidence remains explicit.

The recovery also exposes attribution problems: the historical FedWA row maps to AdaAggRL, and the cosine/fairness row mixes GuardFed-AD2 and GuardFed implementations across scenarios. Numerical recovery does not validate those method labels. The correction candidate must be distinguished from the submitted table; final manuscript replacement and decisions on unsupported attributions have not yet been completed.

### Standard deviations and repeat counts

**Original comment (verbatim).**

> - Authors wrote that the experiments are run 10 times with 10 different seeds, however, they only report the avg and not the std. I'd recommend reporting it in the tables. This is important for the fairness metrics, where some values are extremely small and differences between methods can also be very small.

**Response.** The new repeated-training tables report mean±sample SD (ddof=1) over the predefined model seeds, with all metrics taken from the same final checkpoint. The CelebA nine-method three-view comparison has ten seeds per method/distribution/scenario cell; the separate ten-method native table uses the same reporting principle. Matching nine-seed summaries exclude configuration-selection seed 91001, and the six-seed subset is descriptive, not an untouched test. Cross-scenario summaries first average within each model seed.

The submitted Table II does not support a blanket ten-repeat statement. Among its 480 source metric cells, 294 have n=1, 180 have n=10, and six have n=3. Sample SD is computed only when matching repeated records exist. For n=1 it is unavailable, not zero; neither another cohort's SD nor variation across training rounds supplies missing seed uncertainty.

Historical selection also matters. The single-seed exporter independently selected accuracy and disparity extrema over the final ten rounds; 94/98 conditions have no single round attaining all three values. The separate root-generation suite likewise has 250/260 mixed-checkpoint triplets, and its n=10 denotes scenarios at one seed, not ten seeds. A round-70 correction candidate removes metric/checkpoint mixing but cannot create missing repeats or validate ambiguous method identities. We do not treat small mean differences or overlapping SDs as significance tests.

### Recommended literature and unequal subgroup benefit

**Original comment (verbatim).**

> There exists a lot of others unfairness reduction methods that are not cited in this paper:
> - https://arxiv.org/abs/2012.02447
> - https://journals.sagepub.com/doi/abs/10.3233/FAIA240671
> - https://arxiv.org/abs/2503.15163
> - https://ieeexplore.ieee.org/document/9378043/
> - https://arxiv.org/abs/2108.08435
> - https://arxiv.org/abs/2109.08604
>
> Some of them involve the use of Differential Privacy, which is not used in this paper.
>
> Moreover, there are recent studies that also tried to highlight how unfairness reduction can be beneficial only for a subgroup of people while harming others; these can also be interesting considering the topic of the paper: 1) https://ojs.aaai.org/index.php/AIES/article/view/36730 and 2) https://dl.acm.org/doi/full/10.1145/3715275.3732152

**Response.** We checked all eight suggested identifiers and prepared bibliographic entries and targeted related-work text. They cover federated bias mitigation (Abay et al.), privacy–utility–fairness tradeoffs (PUFFLE), global fairness function tracking (Rychener et al.), FairFL, client-wise constrained optimization (FCFL), private fairness-constrained learning (FPFL), harm-centered evaluation (Taik et al.), and client benefits under differing fairness objectives (Corbucci et al.). The IEEE identifier corresponds to FairFL, not FairFed. These works inform positioning and limitations; we do not present them all as implemented poisoning-defense baselines or transfer their theoretical guarantees to GuardFed.

Differential privacy is an additional requirement in some of these settings. GuardFed has no differential-privacy guarantee or privacy budget. In discussing LoGoFair, demographic parity must also be distinguished from differential privacy: satisfying a fairness constraint does not establish privacy protection.

We agree that a smaller aggregate disparity need not benefit every group. In the accepted COMPAS Full non-IID same-checkpoint comparison, calibration raises one group's mean TPR from 0.417638 to 0.575338 while also raising its FPR from 0.161773 to 0.297088. Reporting both error rates, subgroup support, and utility therefore matters. This observation does not establish universal subgroup benefit, causal real-world harm, or protection for unmeasured attributes. The supplied literature text retains these distinctions; final manuscript integration remains separate.

## Reviewer 3

### R3.1 — DFA novelty

**Original comment (verbatim).**

> 1. The novelty of DFA should be further clarified and strengthened. The current formulation appears to combine two existing attack mechanism, rather than introducing a fundamentally new joint attack strategy.

**Response.** We agree that combining existing primitives does not by itself establish a new attack mechanism. DFA's contribution is a coordinated threat setting and evaluation of simultaneous utility and fairness degradation. S-DFA applies the two roles to the same malicious clients; Sp-DFA splits a fixed malicious-client budget between them. In the executed four-malicious-client setting, Sp-DFA assigns two clients to each role. The update attack is explicitly described as a root-reference FedSA-inspired implementation, including its attacker-information assumption. We do not claim a new primitive or super-additive synergy without matched-budget component evidence. The supplied contribution and threat-model replacement text adopts this narrower positioning consistently (also R1.1).

### R3.2 — Individual components

**Original comment (verbatim).**

> 2. The ablation study better isolates the contribution of each individual component of GuardFed. The current ablation study groups several components together. For example, C and A, as well as F and V are removed together, it is difficult to determine the individual contribution of each component. So, I suggest evaluating each component separately, or providing further justification for why these components are evaluated jointly.

**Response.** We replaced the grouped interpretation with individual-deletion evidence. On Adult and COMPAS under S-DFA, the analysis separately removes U, C, A, F, V and norm handling N, for both distribution anchors and ten seeds per condition. It contains 280 records: 260 new runs and 20 historical Adult Full controls. Masks are applied after internal candidate overrides, preventing a candidate from restoring a deleted score term.

The deletion scope matters: C/A deletion removes the scoring contribution while geometric filtering remains; F/V deletion retains prediction calibration. We therefore report raw and calibrated outputs from the same checkpoint for the 260 available new models. Historical Adult Full binaries are unavailable, and their missing counterfactual outputs are not imputed.

The image study separately checks U, C and A using matched model seeds and round-70 checkpoints. U and C each cover all ten IID/non-IID–scenario cells with 100 deletion/Full pairs. A currently covers eight cells with 80 pairs. The [A table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_eight_scenes80_20261011/TABLES.md) includes every raw/native/shared view and fixed 10/9/6-seed panel. Non-IID A S-DFA/Sp-DFA and the other five image controls remain unfinished; this response is not an all-component completion claim.

These experiments do not support calling every term indispensable. All 12 COMPAS deletion conditions have higher mean accuracy than Full, and six improve all three means. R3.7 discusses these counterexamples and the prediction-rule dependence. Mixed historical environments and replay devices limit causal attribution; deleting a score term is not an isolated removal of every related pipeline operation.

### R3.3 — Sensitive attributes and fairness definitions

**Original comment (verbatim).**

> 3. The applicability of the proposed framework to different sensitive-attribute settings and fairness definitions should be clarified. The current formulation assumes a binary sensitive attribute a\in{0,1} and the experiments also appear to focus on a single sensitive attribute. It would be helpful to clarify whether GuardFed can support multi-valued sensitive attributes or multiple sensitive attributes simultaneously. In addition, the current fairness-risk term is based on AEOD and ASPD. It would be helpful for the authors to discuss whether GuardFed can be extended to other fairness metrics, e.g. equalized odds.

**Response.** The measured scope is a single annotated binary sensitive attribute. Multi-valued and intersectional fairness have not been experimentally validated. The score interface could accept a risk over a set of groups, including intersections, but this is a proposed extension rather than a completed result.

The present AEOD is the absolute binary true-positive-rate gap: it measures equal opportunity, not full equalized odds. A possible equalized-odds risk is the maximum of the between-group TPR range and FPR range. Estimating it requires positive and negative root examples in every group. With sparse or missing support, the rates and fitted thresholds can be noisy or unidentifiable; a numerical fallback does not solve that statistical problem. Group-dependent prediction also requires the relevant sensitive annotation at inference. The supplied method and limitations text makes these requirements explicit and separates conceptual extensibility from the supported binary-group experiments.

### R3.4 — IID/non-IID motivation and component behavior

**Original comment (verbatim).**

> 4. The IID and non-IID settings require more motivation and analysis. The experiments use Dirichlet parameters \alpha=5000 and \alpha=5 to represent IID and non-IID settings, respectively. The authors should explain why these particular values were selected and whether \alpha=5 represents a sufficiently heterogeneous practical FL scenario. It would strengthen the evaluation to include additional heterogeneity levels, or at least provide a more detailed analysis of how data heterogeneity affects GuardFed and its individual components. In particular, some components appear to behave differently between IID and non-IID cases, which deserves further explanation.

**Response.** We interpret α=5000 as an approximately uniform allocation anchor and α=5 as a milder heterogeneous anchor, rather than a universal model of practical FL heterogeneity. The additional Adult/COMPAS study evaluates α=1, 0.5 and 0.1 in 360 runs (R2, stronger non-IID settings). This splitter allocates sensitive groups; it is not label-Dirichlet or general feature heterogeneity.

We also checked realized allocations. In CelebA, replay of the frozen splitter exactly matches all 400 client totals over 20 partitions. Male proportions range from 40.495–42.969% under IID and 9.832–74.303% under non-IID. The training-only joint audit verifies 1,600 Male×Smiling client counts: every cell has support, but target-label variation is narrower than sensitive-group variation. These reconstructed counts are distinguished from originally logged measurements. The stronger-α tabular audit covers 60 partitions and retains empty and single-group clients rather than resampling them away.

The individual-deletion study covers the two original distribution anchors; it is not a complete component-by-α factorial study. Root–client mismatch, correlated scores and estimation variability may contribute to different behavior, but the experiments do not isolate a unique cause. We report the observed utility/disparity trade-offs and treat these explanations as hypotheses, rather than claiming that stronger heterogeneity establishes universal component benefit.

### R3.5 — Notation table

**Original comment (verbatim).**

> 5. A notation summary table would improve readability, as the manuscript introduces many symbols and hyperparameters across different sections, making it somewhat difficult for readers to quickly track their meanings and roles.

**Response.** A notation table is supplied in the companion manuscript text. It distinguishes global, client and root parameters and update signs; participating, benign, malicious and retained sets; U/C/A/F/V signals and norm handling; median/MAD normalization; candidate index and temperature; configured malicious count; group prediction thresholds; client-allocation α; and the separate root-skew controls. It also distinguishes a frozen run recipe from the per-round selection among that recipe's internal candidates. Equation references and the final location still need alignment with the matching submitted manuscript source; the supplied table is an insertion candidate, not a claim that the submitted project has already been compiled.

### R3.6 — Exact trusted root update

**Original comment (verbatim).**

> 6. The manuscript states that the trusted root update is obtained through fairness-aware training on the server-side root dataset and is then used as the reference direction for computing root alignment. However, the exact procedure for obtaining this root update is not sufficiently clear.

**Response.** Inspection of the executed code revealed a description error: root training minimizes ordinary cross-entropy, not an additional fairness-aware root loss. We supply a correction to match the implementation that produced the results.

At the start of each round, the server copies the current global model and creates a fresh optimizer using the run's configured optimizer and learning rate. It shuffles root minibatches and makes one pass over the root set. The root update equals the resulting parameter vector minus the initial global vector. This update is computed once before aggregation and reused as the alignment and norm reference. GuardFed-AD2+ does not aggregate it as an extra pseudo-client.

Fairness-related client reweighting, risk signals, internal candidate selection and group-threshold fitting are separate operations. The companion pseudocode specifies their order, optimizer initialization and parameter-difference sign convention. This corrects the method description without claiming that an unexecuted root objective was evaluated.

### R3.7 — Adult/COMPAS differences

**Original comment (verbatim).**

> 7. The ablation results require further analysis. For example, when the reward terms U, C, and A are removed, the Adult non-IID accuracy decreases, whereas the corresponding COMPAS accuracy is slightly higher than that of Full GuardFed. The manuscript currently mainly emphasizes the Adult result. The authors should explain this dataset-dependent behavior.

**Response.** We agree that emphasizing Adult alone would overstate the mechanism evidence. The new individual-deletion study confirms dataset dependence. Under the train-only COMPAS protocol, all 12 deletion conditions have higher mean accuracy than Full, and six improve all three means. For IID, Full ACC/AEOD/ASPD are 0.64838/0.06130/0.04196; removing A gives 0.65999/0.05352/0.03752. We retain this counterexample to universal component necessity.

Prediction calibration can change the interpretation. For non-IID COMPAS, removing F raises raw AEOD from 0.239597 to 0.250984, but lowers calibrated AEOD from 0.049143 to 0.044831. A calibrated table alone therefore cannot identify the contribution of aggregation.

CelebA provides a corresponding example. In non-IID FedSA, the ten-seed paired minus_A−Full differences in raw prediction are ACC +0.078±0.950 percentage points, AEOD −0.00508±0.01328 and ASPD −0.00170±0.01643: all three means improve after deletion. With native/shared calibration, the differences are +0.053±0.948, +0.00372±0.01267 and +0.00207±0.02164, showing an accuracy–disparity trade-off. These are paired means ± sample SD, not significance or every-seed claims; the fixed nine-/six-seed panels are retained alongside them.

Correlated scores, compensation by candidate selection and root-estimation variability are plausible explanations, not isolated causes. The supplied discussion presents both datasets and the raw/calibrated comparisons, and narrows the claim to conditional trade-offs. No best Full seed is compared with deletion means.

### R3.8 — Explain attacks and comparison methods

**Original comment (verbatim).**

> 8. The related-work discussion should provide a more detailed description of the benchmark attacks and compared methods. The manuscript covers a large number of robust, fairness-aware, and root-data-based FL methods, but many are introduced only briefly.

**Response.** We supply an implementation/access table covering each method's objective, trusted-root and sensitive-attribute access, update/gradient interface, aggregation or postprocessing operation, implementation source, adaptation and selection procedure. The attack descriptions specify the sensitive-attribute and update-poisoning operations, their budgets and the synchronous/split allocation. F Flip changes the annotation/reweighting path; it does not change CelebA pixels or task labels.

The completed CelebA comparison contains ten native methods across two distributions, five scenarios and ten model seeds (1,000 records). Nine of these methods also have a separate 900-record raw/native/shared comparison. These scopes are kept distinct. FairFed, FairGuard, their combination, FedAA-DDPG and LASA retain project-adaptation labels. Seven target methods still lack complete accepted coverage; a simplified inspired branch is not presented as the original algorithm.

Huber uses the author-approved identity projection on CNN parameters and is labelled an empirical CNN adaptation, without the original constrained-domain guarantee. LoGoFair uses the official demographic-parity postprocessor with 20 fixed image-ID virtual cohorts and root-only fitting; these are not real training clients, and DP here does not mean differential privacy. Its reported native output is the fitted postprocessor's prediction, not the FedAvg backbone cache. The all-negative IID Benign run is retained, so zero disparity is not misrepresented as useful prediction. The [native table](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_ten_method_native_20261010/TABLES.md) preserves the resulting large seed spread and unfavorable comparisons. Full method coverage and a frozen final evaluation remain pending.

### R3.9 — Source code and configurations

**Original comment (verbatim).**

> 9. It is hoped that the authors will make the source code and experimental configurations publicly available and facilitate follow-up research based on this work.

**Response.** A concrete revision snapshot is available at [GuardFed, commit 817f6fd](https://github.com/YananZHOU5555/GuardFed/tree/817f6fd395564b044ed05e360771817f2f4585de). The release includes the accepted comparison tables, per-seed results, frozen configurations, adapter/source records, aggregation code and documented limitations. Large models and raw array archives are retained separately with archive/member hashes; publishing code is not itself a model backup.

This snapshot covers completed evidence, including the A80 response update, and does not certify completion of the remaining benchmark or final evaluation. The final revision release must additionally contain any subsequently accepted methods/results, the frozen evaluation protocol, matching manuscript source and runnable setup instructions. Missing historical checkpoint binaries and adaptation boundaries remain disclosed.

### R3.10 — Limitations

**Original comment (verbatim).**

> 10. The authors are encouraged to add a brief discussion of the limitations of the proposed method.

**Response.** The supplied limitations section covers four substantive boundaries. First, the method requires a representative, annotated root set and training-population information; sparse groups can make risk estimates and thresholds unreliable, and group-dependent prediction needs sensitive attributes at inference. Second, the defense uses a configured malicious count and has been evaluated with small participating populations and binary groups, not million-device or general intersectional deployments.

Third, the softmax result is conditional on score separation in the actual retained set. It does not certify filtering, candidate selection, convergence or population fairness. The executed top-k operation can re-admit intermediate-gate exclusions; this occurs in 116 observed Adult deletion rounds, so no unconditional hard-exclusion guarantee is claimed.

Fourth, performance depends on dataset, prediction rule and selection history. The ablations retain unfavorable effects, and common calibration shows that end-to-end fairness differences cannot be assigned wholly to aggregation. The expanded CelebA evidence is validation-only; selection seeds and the official test partition have already been exposed. A future frozen evaluation cannot be described as an untouched holdout. Historical Table II also has mixed repeat counts, selection conventions and incomplete model provenance. These limitations are stated with the evidence rather than hidden behind a general robustness claim.

### R3.11 — Architecture specification and additional models

**Original comment (verbatim).**

> 11. The evaluation appears to use only a single MLP model, while its architecture is not specified. I would suggest including evaluation with additional model architectures to demonstrate the generalizability of GuardFed.

**Response.** We supply the executed model and training specifications. The supplementary tabular MLP is input dimension → Linear(16) → ReLU → Linear(2), using two-logit cross-entropy; encoded columns and preprocessing are specified by dataset. The CelebA CNN has three 3×3 convolution/ReLU/2×2 max-pool blocks with channels 3→32→64→128, adaptive spatial average pooling and Linear(128,2). It receives RGB 64×64 inputs scaled by 1/255, with no batch normalization, dropout or image augmentation.

The supplementary tabular protocol uses Adam, learning rate 0.005, batch size 256, one local epoch and 70 rounds. The image protocol uses batch size 64, one local epoch and 70 rounds; method-specific learning rates are fixed in the selected recipes. Root training uses a fresh configured optimizer and one pass each round.

CelebA adds both a non-tabular dataset and a CNN, addressing the original single-MLP evidence gap. It does not isolate an architecture-only effect or establish generality to arbitrary models. The original submitted MLP must be matched to its archived source before assigning its final manuscript specification; it is not assumed identical merely because both models are called MLPs.

## Work still required before submission

| Item | Remaining requirement |
|---|---|
| P1 — Complete benchmark | Complete accepted coverage for the seven remaining target methods. FLGMM and the cosine/fairness hybrid have active fixed-recipe queues; Fed-NGA/Huber validation search is ongoing. FedWA, SmartFL and FedDNA still require faithful source/specification resolution. Preserve adaptation labels, all seeds and negative results; do not substitute simplified branches for original methods. |
| P2 — Image mechanisms | Finish the eight-control CelebA study: 800 new runs with 100 explicitly reused Full controls. At this draft's accepted cutoff, 280 new models have native and three-view evidence. U100 and C100 each cover ten cells; A80 covers eight. Finish non-IID A S-DFA/Sp-DFA and the other five controls, then produce complete matched-seed tables. |
| P3 — Frozen final evaluation | Decide the primary prediction/evaluation endpoint and freeze the final protocol before evaluation. Retain raw/native/shared interpretation and common-calibration controls. Official partition2 contains 19,962 images, but prior test exposure prevents an untouched-holdout claim. No final-test performance is supplied here. |
| P4 — Historical corrections | Finalize coherent Table II replacement and actual repeat counts, without inventing unavailable SD. Resolve unsupported method attributions and synthetic/Fig.3 provenance where possible; otherwise explicitly qualify or withdraw unsupported claims. The terminal redraw remains a candidate, not recovery of the original execution. |
| P5 — Submitted manuscript | Obtain the matching submitted LaTeX project, apply the verified method/theory/discussion insertions and accepted tables, and compile/render the revision. Assign final section, equation, table and page references after integration. The located older source is not assumed to match the submission. |
| P6 — Final reproducibility release | Publish the completed accepted scope, faithful adapters, frozen protocols, per-seed results and usable setup instructions. Retain large-artifact hashes and access information, failure/negative evidence, environment differences and unavailable legacy checkpoints. The current Git snapshot is an intermediate evidence release. |

## Supporting material

- [Ten-method native IID/non-IID tables](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_ten_method_native_20261010/TABLES.md), with the [three-page paper-table PDF](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/celeba_ten_method_native.pdf).
- [Nine-method raw/native/shared comparison](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_three_view_20261009/README.md) and [paired calibration attribution](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_view_attribution_20261009/REPORT.md).
- [Complete U-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md), [complete C-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_full100_20261010/snapshot/TABLES.md) and [eight-scene A-deletion table](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_eight_scenes80_20261011/TABLES.md).
- [Method, root-update pseudocode, theory, notation and manuscript insertion candidates](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A80_reader_20261011/manuscript_insertions_integrated_20261011.md).
- [Detailed evidence and historical audit version](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A80_reader_20261011/rebuttal_integrated_20261011.md), including legacy Table II, synthetic reporting, source histories and reporting conventions.
- [Verified recommended-literature audit](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_20261009/recommended_literature_audit.md).

All repeated-model tables use sample SD (ddof=1), matched seed sets and one terminal checkpoint for all metrics. Cross-scenario summaries first average within model seed. Selection history, mixed environments/devices and historical test exposure remain disclosed. Sensitivity subsets and smaller disparities do not imply significance, untouched confirmation or uniform subgroup benefit.
