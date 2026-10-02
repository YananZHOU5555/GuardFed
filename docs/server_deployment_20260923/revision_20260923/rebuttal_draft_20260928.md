# Response to the Editor and Reviewers — reviewable draft

**Manuscript:** TDSC-2026-07-3058, “To Kill Two Birds with One Stone: Defending Both Utility and Fairness in Federated Learning Systems”  
**Draft date:** 28 September 2026  
**Status:** Internal response draft for author review; not a submission-ready response letter.

> 2 October 2026 evidence update: [Table II provenance audit](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/table2_trace_20261002/追溯报告.md) identifies numerical sources for all 480 cells and restores all 44 threshold-suppressed values. The [four-page table packet](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/table2_recovered_20261002/table2_complete_review.pdf) provides historical reconstruction and a final-round-70 correction candidate for both distributions. Actual sample sizes are mixed: 294 cells n=1, 180 n=10, and 6 n=3. Independent checks reproduce the available statistics. Method-dependent historical checkpoint selection, mixed method identities and one published/source discrepancy are disclosed. This completes the available numerical recovery, not missing-seed experiments, baseline-fidelity certification or manuscript insertion; the response letter remains an internal draft.

The responses below distinguish completed experiments from proposed manuscript changes. **No statement in this draft certifies that the corresponding manuscript passage has already been revised.** Every manuscript insertion, location, unresolved analysis, and unfinished experiment is marked **TODO**. Comment summaries are paraphrases of the supplied decision letter; Reviewer 2's unnumbered comments receive descriptive labels rather than invented original numbers. The final scientific claims require author approval.

## Evidence available when preparing this draft

| Evidence | Completed scope | Reporting boundary |
|---|---|---|
| E1 — supplementary tabular experiments | 1,150 new runs: 260 individual-component ablation runs, 360 stronger-heterogeneity runs, 250 root-noise runs, 160 fixed-reservoir protected-group-share runs, and 120 ACSIncome runs; matching historical controls are identified separately | Fixed final round 70; ten distinct predefined seeds per complete condition; mean and sample standard deviation. New COMPAS runs use train-only preprocessing; historical controls are not interchangeable across protocols |
| E2 — original CelebA formal cohort | 240 runs: four methods × two distributions × three scenarios × ten seeds, CNN, official train/test partitions | Preserve the frozen original configuration and results. This cohort and later tuned validation cohorts are not pooled |
| E3 — expanded CelebA Stage A | 700 accepted records: seven methods × IID/non-IID × five scenarios × ten seeds; 644 newly trained records and 56 explicitly reused records | Validation-only; all 70 condition groups have ten seeds. Additional nine-seed and six-seed summaries expose selection history. This is seven-method coverage, not all 17 manuscript rows |
| E4 — tabular prediction-calibration attribution | 260 verified same-checkpoint raw/calibrated pairs, covering 26 ten-seed conditions | No retraining. Twenty historical Adult Full checkpoints are unavailable; no replacements are fabricated. Six CPU replay mismatches were resolved on the original GPU and retained in the audit |
| E5 — theory diagnostics | Strict score-margin diagnostics for 140 Adult ablation/Full runs, with 8,440 visible round records | Historical Full records contain only the first and last rounds. Negative margins occur; this is not evidence that the separation assumption always holds |
| E6 — CelebA shared-calibration control | All 700 Stage A checkpoints accepted; 1,400 raw/shared evaluation outcomes, zero new training; maximum native-replay metric difference is zero | Same final-round weights, shared root-only rule, validation-only; calibration effects within each checkpoint are identifiable, but the entire across-method difference is not thereby isolated to aggregation |

The E3 matrix uses official CelebA train/validation sizes 162,770/19,867, Smiling as target, Male as the annotated sensitive attribute, a 64×64 RGB CNN, 70 rounds, 20 participating clients, and four nominal malicious clients. Its recipes were selected previously on non-IID Benign/S-DFA validation conditions and then held fixed across both distributions and all five scenarios. The primary metrics are ACC, AEOD, and ASPD; here AEOD is the implemented absolute TPR gap, not full equalized odds. The primary summaries use all predefined seeds, not a favorable seed for GuardFed and averages for other methods.

**Completed controls and bounded integration checks:** E6 reuses all 700 Stage A checkpoints to compare raw predictions with one common root-only calibration rule; all 1,400 evaluation outcomes have passed acceptance with zero new training. FedAA and LASA have each passed full-CelebA three-round Benign/S-DFA pilots, and the FedAA GPU resume check reproduced the complete saved state bitwise. These pilots are pipeline/recovery gates and do not enter the result tables. Remaining formal baseline comparisons, CelebA mechanism ablations, and a final frozen evaluation remain TODO.

### Completed shared-calibration result used in this draft

The following statistics first average the ten distribution/scenario conditions within each seed, then report the mean and sample SD over ten seeds. Raw and shared-calibrated rows use the same trained checkpoints; the common calibration rule is fitted on root data, not validation labels.

| Method/output | ACC (%) | AEOD | ASPD |
|---|---:|---:|---:|
| GuardFed, raw | 88.739 ± 0.620 | 0.03614 ± 0.00440 | 0.10103 ± 0.00505 |
| FLTrust, raw | 89.629 ± 0.652 | 0.04585 ± 0.00406 | 0.11456 ± 0.00468 |
| GuardFed, shared calibration | 88.420 ± 0.623 | 0.00967 ± 0.00297 | 0.06105 ± 0.00466 |
| FLTrust, shared calibration | 89.264 ± 0.719 | 0.00924 ± 0.00287 | 0.07125 ± 0.00666 |

For shared-calibrated GuardFed minus shared-calibrated FLTrust, the paired-seed differences are −0.844 ± 0.371 percentage points in ACC, +0.00043 ± 0.00414 in AEOD, and −0.01020 ± 0.00407 in ASPD. GuardFed is better on ACC/AEOD/ASPD in 0/6/10 of the ten seed-level comparisons, respectively. The ASPD direction remains in the nine-seed and six-seed subsets; AEOD does not show a stable advantage. Calibration is therefore an important contributor to the native comparison. Raw GuardFed still has lower mean AEOD and ASPD than raw FLTrust, with lower accuracy, but this does not establish that aggregation alone explains the between-method difference. No statistical-significance claim is made.

## Response to the Associate Editor

**Comment summary.** Clarify DFA's positioning, the scope of the theoretical analysis, benchmarking, and ablations.

**Draft response.** Thank you for identifying these central concerns. We conducted additional tabular experiments covering individual scoring components, stronger client heterogeneity, and imperfections in the server-side root data, together with ACSIncome and CelebA/CNN evaluations. We also completed a seven-method CelebA validation matrix spanning both IID and non-IID settings, five scenarios, and ten shared seeds per condition. These additions broaden the evidence but do not establish universal superiority: the results reveal utility–fairness trade-offs, dataset-dependent component effects, and root-data failure boundaries.

We agree that DFA should be positioned in relation to its existing attack primitives and that the theorem should be interpreted as a conditional analysis of soft aggregation. The detailed responses below specify the evidence and the corresponding manuscript changes still required.

**TODO — manuscript and response:** align the contribution, threat-model, method, theory, experiments, and limitations sections; insert verified tables and diagnostic figures; provide final section/page/line references after revision. Complete the remaining baseline and attribution work before describing the expanded benchmark as complete.

## Reviewer 1

### R1.1 — Novelty and positioning of DFA

**Comment summary.** S-DFA combines sensitive-attribute manipulation and update poisoning; Sp-DFA separates these roles across malicious clients. Clarify whether the contribution is a new attack mechanism or a coordinated threat model and evaluation framework.

**Draft response.** We agree with this distinction. The defensible contribution is the coordinated evaluation of utility and group-fairness degradation, including the difference between colocating the two attack components and distributing them across malicious clients. The constituent poisoning primitives are not claimed to be newly invented. The benchmark asks whether defenses that address one objective preserve the other under this coordinated threat setting.

Additional attack coverage does not, by itself, prove a novel synergistic attack mechanism. If a stronger synergy claim is retained, it requires a matched-budget comparison against the individual primitives and their appropriate combination. That comparison is not claimed as completed here. We must also distinguish the implemented FedSA-inspired variant and its root-reference access from the original method cited in the literature.

**TODO — manuscript:** revise the abstract/contributions/threat model to use this scope consistently; state the attacker's information and corruption budget; distinguish original attacks from project variants. A stronger mechanism claim remains conditional on additional evidence and author approval.

### R1.2 — Interpretation of the theoretical analysis

**Comment summary.** The suppression bound assumes that every benign score exceeds every malicious score by a margin; this is an analysis of the weighting mechanism rather than an end-to-end guarantee. Connect it to empirical score separation where possible.

**Draft response.** We agree. The score-separation assumption is substantive and is not guaranteed by the theorem. Under the stated separation condition, the result characterizes how soft weighting suppresses malicious aggregation mass. It does not establish that the complete GuardFed scoring, filtering, candidate selection, and prediction-calibration pipeline necessarily satisfies that condition.

We collected strict-margin diagnostics using the minimum benign normalized score minus the maximum malicious normalized score. The available diagnostic set contains 140 Adult runs and 8,440 visible round records; historical Full records provide only the first and last rounds. Negative margins occur. Consequently, an average benign–malicious score difference cannot replace the theorem's strict assumption, and selected-client mass must be distinguished from pre-filter score separation.

**TODO — analysis/manuscript:** summarize representative strict-margin satisfaction rates and malicious aggregation mass with the correct denominators; align the bound with the actual selected set and temperature; state failure conditions and the limited historical temporal coverage. No claim of unconditional robustness is supported.

### R1.3 — Additional dataset or architecture

**Comment summary.** Adult and COMPAS with an MLP do not sufficiently establish generality beyond tabular binary classification.

**Draft response.** We conducted 120 ACSIncome runs and an initial 240-run CelebA/CNN evaluation. We subsequently completed a separate 700-record CelebA validation matrix with seven methods, two distributions, five scenarios, and ten shared seeds. This provides evidence from an additional tabular task and from image classification with a CNN rather than only the original MLP.

The expanded CelebA results exhibit a measurable trade-off. Averaging the ten distribution/scenario conditions within each seed and then summarizing across ten seeds, native GuardFed achieves ACC 88.420% ± 0.623, AEOD 0.00967 ± 0.00297, and ASPD 0.06105 ± 0.00466. Native FLTrust achieves 89.629% ± 0.652, 0.04585 ± 0.00406, and 0.11456 ± 0.00468, respectively. Thus, GuardFed has lower disparity in this native end-to-end comparison but lower accuracy. The completed E6 shared-calibration control shows that calibration accounts for an important part of this disparity difference: with the same calibration rule, ASPD remains lower for GuardFed, but AEOD has no stable advantage. These are descriptive validation results, not a claim of significance, aggregation-only causality, or universal dominance.

The image experiment changes both modality and architecture, so it demonstrates applicability to this CNN/image task rather than isolating an architecture-only causal effect. The ACSIncome task uses an explicitly customized SEX grouping and a person-row split; its scope is also stated narrowly.

**TODO — manuscript:** insert the separate datasets/protocols and complete tables; disclose validation selection and environment history; do not combine tuned validation results with the earlier test cohort into one undifferentiated table.

### R1.4 — Imperfections and construction of the root data

**Comment summary.** Add root label noise, sensitive-attribute noise, and severe protected-group underrepresentation; explain synthetic-generator training and server-visible data.

**Draft response.** We conducted root-only label and sensitive-attribute noise experiments, as well as protected-group-share experiments using a fixed root reservoir. The latter keeps client data fixed while varying which reservoir examples form the root set. Completed low-performing outcomes and the zero-support boundary are retained.

For example, in the COMPAS benign setting, increasing root sensitive-attribute noise from 0% to 40% changes mean ACC from 0.65821 to 0.65232, AEOD from 0.05106 to 0.15761, and ASPD from 0.04538 to 0.15814. In the separate fixed-reservoir experiment, reducing protected-group share from 50% to 0% changes AEOD from 0.03804 to 0.24644 and ASPD from 0.03271 to 0.23001. The experiments therefore expose a limitation rather than establish robustness to arbitrary root corruption. A fallback value at zero group support is not an identifiable estimate of fairness.

For S-DFA, the implemented utility-poisoning component also uses the root update. Root changes can therefore affect both the attack reference and the defense, so these results describe end-to-end sensitivity, not a defense-only causal effect.

**TODO — manuscript:** insert the complete mean/SD tables and group-support diagnostics; describe each existing synthetic generator's actual training data and access assumptions. We do not claim that synthetic data recover an unobserved group or automatically provide differential privacy.

## Reviewer 2

### R2 — Opening concern: what is new in DFA?

**Comment summary.** Explain what is fundamentally new relative to using existing fairness and utility attacks jointly.

**Draft response.** We agree that the attack primitives and the coordinated evaluation should be separated. Our proposed positioning is a coordinated dual-objective threat setting and comparative evaluation, including synchronous and split allocation of the attack roles. It does not treat a composition of existing primitives as an independently established new primitive. See R1.1 for the implementation and matched-budget limitations.

**TODO — manuscript:** harmonize this positioning across the paper and related work; remove unsupported stronger novelty language.

### R2 — Scale and client participation

**Comment summary.** Discuss whether attacks with a small client population remain relevant when the total population and server-side sampling are much larger.

**Draft response.** This is an important scope distinction. Our actual supplementary protocol uses 20 clients participating in every round, with four nominal malicious clients. It is not a million-client evaluation and should not be described as a protocol that samples 20 clients from 100 if that is not the executed configuration.

Under uniform sampling without replacement from a population of N clients containing M malicious clients, the number K of malicious participants in a round of m clients follows a hypergeometric distribution and has expectation mM/N. Therefore, increasing the population size has different consequences when the absolute malicious count is fixed versus when the malicious fraction is fixed. The current experiments hold a small participating population and do not test availability, sampling bias, or large-scale cross-device deployment.

**TODO — manuscript:** correct the actual participation protocol and add this scope discussion, including the role of known malicious-client counts in the implemented filtering rule. No large-population experiment is claimed.

### R2 — Broader benchmark: another tabular and a non-tabular task

**Comment summary.** Add another tabular benchmark and a non-tabular benchmark such as CelebA; avoid framing a journal evaluation as a Mini-Benchmark.

**Draft response.** We conducted the ACSIncome and CelebA/CNN experiments described in R1.3, addressing both requested data types. The initial additional formal experiments comprise 120 ACSIncome runs and 240 CelebA runs. The later 700-record CelebA validation cohort expands coverage to seven methods and all five specified scenarios under both distributions.

The added data do not uniformly favor GuardFed. For example, on ACSIncome non-IID S-DFA, mean ACC/AEOD/ASPD are 0.768731/0.043193/0.014080 for GuardFed and 0.796770/0.023964/0.044124 for FLTrust: only ASPD is lower for GuardFed. We retain this result and describe the task-dependent trade-off.

**TODO — manuscript:** replace the Mini-Benchmark framing with a precise description of evaluated scope. The manuscript's full method list contains 17 rows including GuardFed; seven-method Stage A does not complete every row. Remaining methods require faithful integration and evaluation, not relabeling simplified implementations.

### R2 — Stronger non-IID settings

**Comment summary.** Evaluate Dirichlet parameters closer to zero.

**Draft response.** We conducted 360 new runs at client-allocation α values 1, 0.5, and 0.1 across Adult/COMPAS, GuardFed/FedAvg/FLTrust, Benign/S-DFA, and ten seeds. These supplement the historical α=5000 and α=5 anchors. The implemented partition is based on sensitive-group allocation, so α does not summarize every form of label or feature heterogeneity.

The results reveal a cost of stronger heterogeneity. For Adult benign runs, GuardFed mean ACC decreases from 0.81568 ± 0.01296 at α=1 to 0.78040 ± 0.02401 at α=0.1. Low-disparity results are reported together with accuracy, including failed learning or poorly supported groups rather than filtering them away.

**TODO — manuscript:** add the full three-method results, measured client-level sample/group/label distributions, and explanations of empty or sparse groups. Keep new COMPAS train-only preprocessing separate from legacy controls.

### R2 — Practicality of the server-side validation/root dataset

**Comment summary.** Explain why the trusted root dataset is a realistic assumption and how it could be constructed without violating the FL data-sharing premise.

**Draft response.** A representative trusted root set is an explicit resource assumption, not something guaranteed by federated learning. Possible sources include public data, separately collected data, or explicitly authorized contributions, each with distribution and governance limitations. These possibilities should not be described as an implemented private acquisition protocol.

Our experiments expose the consequences of imperfect root data rather than eliminating this requirement. Root-derived threshold fitting also requires sensitive-attribute labels, and group-dependent prediction thresholds require the attribute at inference. The implementation's use of training-population statistics must be disclosed separately from the root-data assumption.

**TODO — manuscript:** state actual server data access, disjointness and statistics assumptions; describe the synthetic-root setup accurately; discuss representativeness, privacy, and missing-group limitations. No differential-privacy guarantee is claimed.

### R2 — Table II should show computed values instead of threshold-based N/E

**Comment summary.** Display all calculated values, including low-performing cases, rather than suppressing them with a threshold.

**Draft response.** We agree that low utility does not make a well-defined fairness metric unreportable. The completed supplementary cohorts retain low-accuracy and low-disparity outcomes. These metrics must be interpreted jointly: for example, low disparity can accompany almost constant prediction and is not by itself evidence of useful fairness.

The completed reconciliation identifies numerical sources for all 480 historical cells, restoring all 44 values previously hidden as N/E. It matches 435 printed values at their displayed precision and identifies one discrepancy: FairGuard/IID/FedSA ACC was printed as 59.13%, whereas its source summary contains 54.13%. The original submission remains preserved. The recovered packet includes both historical selections and a correction candidate using the final-round-70 record for all three metrics within each run. Numerical zeros are retained; where subgroup denominators were not preserved, we do not infer well-defined group rates from the presence of a finite recorded value.

**Completed — numerical recovery:** all cells and hidden values traced, statistics independently checked, both distribution tables generated; no training was rerun. **TODO — manuscript/table:** insert the corrected table and accurately disclose sample sizes, selection history, metric implementation and method identities. The FedWA historical label maps to AdaAggRL, and the Cosine/fairness row combines GuardFed-AD2 and GuardFed branches; these labels cannot certify faithful implementations of the named baselines.

### R2 — Standard deviations across seeds

**Comment summary.** Add standard deviations, especially where fairness differences are small.

**Draft response.** The completed new cohorts report the mean and sample standard deviation (ddof=1) over the actual predefined seeds at the fixed final checkpoint. E3 includes ten-seed primary tables, a nine-seed subset excluding configuration-selection seed 91001, and a six-seed subset 91005–91010 that was unobserved before the coverage stage. Cross-scenario summaries first average within each seed, preserving the seed as the independent unit rather than treating multiple scenarios as extra seeds.

The completed Table II audit establishes actual n per cell: 294 single-seed cells, 180 ten-seed cells and six three-seed cells. The recovered tables include the mean and sample SD only where matching repeat records exist; single-seed entries explicitly mark SD unavailable. Historical single-seed exports select ACC and fairness metrics independently over the final ten rounds, and the ten-seed F-Flip/FedSA exports change selection direction by method group. These selected statistics are not uniform final-round results. We provide a separate final-round-70 correction candidate, with all metrics from the same run/round record. Different recipes and baseline-identity limitations remain disclosed; a different cohort's SD is never attached to a selected point value. Retained logs do not establish binary checkpoint hashes or create missing repeated runs.

**Completed — available historical statistics:** numerical source reconciliation and true-n mean/SD tables. **TODO — manuscript/table:** replace ambiguous statistical labels, resolve method names and state paired comparison rules. The record does not support describing the full historical table as ten-seed means; any uniform-repeat requirement remains a separate missing-experiment decision. No significance claim is inferred solely from small mean differences or overlapping/nonoverlapping SDs.

### R2 — Recommended literature and subgroup harms

**Comment summary.** Discuss additional unfairness-reduction approaches, including work involving differential privacy, and research showing that aggregate fairness improvements can harm subgroups.

**Draft response.** We agree that a smaller group disparity need not improve every group's error rates. In our verified COMPAS Full non-IID calibration comparison, one group's mean TPR increases from 0.417638 to 0.575338 while its FPR increases from 0.161773 to 0.297088. This same-checkpoint result illustrates why group-specific TPR/FPR and utility must accompany aggregate disparity metrics; it does not establish a broader causal conclusion beyond this postprocessing comparison.

**TODO — literature verification and manuscript:** read and verify all eight recommended entries, determine their relevance, and add accurate bibliographic records and substantive discussion. The supplied identifiers are arXiv:2012.02447, DOI:10.3233/FAIA240671, arXiv:2503.15163, IEEE document 9378043, arXiv:2108.08435, arXiv:2109.08604, AIES article 36730, and DOI:10.1145/3715275.3732152. Titles, authors, and claims are deliberately not invented here. Mention of DP in these works does not imply that GuardFed provides DP or that a new DP experiment has been completed.

## Reviewer 3

### R3.1 — Novelty of DFA

**Comment summary.** The formulation appears to combine existing attack mechanisms rather than introduce a fundamentally new joint strategy.

**Draft response.** We agree that this distinction must be explicit. The proposed scope is a coordinated dual-objective threat model and evaluation, with synchronous and split allocation of the components, rather than a claim that the constituent attack primitives are new. A stronger synergy claim would require additional matched-budget evidence. See R1.1 and R2's opening concern.

**TODO — manuscript:** revise the contribution and attack sections consistently, including the implemented attack variant and information assumptions.

### R3.2 — Isolate the individual components in the ablation

**Comment summary.** Removing C/A or F/V jointly prevents attribution to each component.

**Draft response.** We conducted individual deletions of U, C, A, F, V, and N under S-DFA for Adult and COMPAS, both IID and non-IID, and ten seeds per condition. The evidence matrix contains 280 records including Full controls: 260 new runs and 20 identified historical Adult Full records. Ablation masks apply to every internal adaptive candidate, preventing a candidate override from silently restoring a deleted scoring term.

These are scoring/normalization ablations with stated boundaries. Removing C or A does not automatically remove geometric filtering; removing F or V does not remove prediction-threshold calibration. We additionally verified 260 same-checkpoint raw/calibrated pairs. Twenty historical Adult Full checkpoints are unavailable, so those missing raw/calibrated pairs were not invented or recreated by retraining. Historical Adult Full and new ablations also have environment differences that limit causal interpretation.

The results do not support uniform necessity of every component. Component contributions vary by dataset and distribution, as discussed under R3.7. The CelebA shared-calibration control is now complete for all 700 checkpoints. Under the same root calibration rule, GuardFed retains a lower mean ASPD than FLTrust but not a stable AEOD advantage, and accuracy remains lower. This further cautions against assigning the native comparison's entire fairness difference to the aggregation score. Additional image-specific hard-filter/candidate-selection ablations are planned, not completed.

**TODO — manuscript:** replace grouped-only attribution with the complete individual-component table, paired seed differences, and explicit retained mechanisms. Report contrary outcomes alongside favorable ones.

### R3.3 — Multiple sensitive attributes and alternative fairness definitions

**Comment summary.** Clarify support for multi-valued or multiple attributes and metrics such as equalized odds.

**Draft response.** The executed experiments use one binary sensitive attribute, so empirical support is limited to that setting. A conceptual extension can index groups by a multi-valued attribute or by intersections of attributes and define disparity as a maximum pairwise difference. For example, equalized-odds risk could use the maximum of the cross-group TPR range and the cross-group FPR range, with an explicit policy for unsupported groups. This is a proposed formulation, not an implemented and validated result.

Such extensions increase the number of group/label cells and may make root estimates and group-specific thresholds unreliable when cells are sparse. The current metric named AEOD in the implementation is an absolute TPR gap; it does not jointly constrain TPR and FPR and must not be described as full equalized odds.

**TODO — manuscript:** provide the exact supported setting, clearly labeled extension definitions, sample-support requirements, and calibration limitations. Do not claim completed multi-attribute or full-EO experiments.

### R3.4 — Motivate α=5000/5 and analyze heterogeneity

**Comment summary.** Explain whether α=5 is meaningfully non-IID and how heterogeneity changes component behavior.

**Draft response.** We treat α=5000 and α=5 as historical allocation anchors rather than claiming that α=5 captures all realistic heterogeneity. We conducted the additional α=1/0.5/0.1 experiments described in R2, and individual-component ablations under the original two distributions. The actual partition uses sensitive-group Dirichlet allocation; client composition, support, and imbalance must therefore be measured rather than inferred from α alone.

The completed experiments demonstrate sensitivity to heterogeneity, including the Adult accuracy reduction reported above. They do not establish a single monotonic mechanism across all datasets. In particular, the existing experiments do not form a full component-by-every-α factorial study, so interactions beyond the measured combinations cannot be claimed.

**TODO — analysis/manuscript:** explain the historical anchor choice, add observed partition statistics, and connect measured component effects and score diagnostics with appropriate uncertainty. State which interactions remain untested.

### R3.5 — Notation summary

**Comment summary.** Add a notation table for symbols and hyperparameters.

**Draft response.** We agree that a consolidated notation table is needed, covering clients and rounds, global/local/root updates, root support, raw and normalized U/C/A/F/V signals, normalization and temperature, retained-client sets, candidate selection, and group thresholds. It should distinguish fixed hyperparameters from per-round selected quantities and distinguish client α from root-distribution controls.

**TODO — manuscript:** create and insert the table, check each symbol against the actual implementation and equations, and supply the final location. The table has not been claimed as inserted in this draft.

### R3.6 — Exact procedure for the trusted root update

**Comment summary.** Explain the fairness-aware root training used as the alignment reference.

**Draft response.** The implementation audit identifies a discrepancy that must be corrected: the current root update is obtained through ordinary cross-entropy training on the root data for one pass, not an additional fairness-aware root objective. The precise initialization, optimizer, number of steps, update-difference convention, and reuse across candidate evaluation must be stated from the executed configuration.

Fairness-related client weighting, aggregation scoring, and group-threshold calibration are distinct operations and should not be conflated with the root-update loss. The method description must also disclose the actual information used, including training-population statistics and inference-time sensitive attributes where applicable.

**TODO — manuscript/code documentation:** replace the inaccurate fairness-aware root-training description with verified pseudocode and configuration details; align the alignment definition with the actual update sign and normalization. This is a correction of description, not a claim that a new root-training method was evaluated.

### R3.7 — Explain Adult/COMPAS differences in ablation

**Comment summary.** Explain why removing reward terms can reduce Adult accuracy but improve COMPAS accuracy, instead of emphasizing only Adult.

**Draft response.** The new component-wise experiments confirm that the effect is dataset- and distribution-dependent. In the new train-only COMPAS protocol, all 12 component-deletion conditions have higher mean ACC than Full, and six have better means on all three reported metrics. For example, IID Full has ACC/AEOD/ASPD 0.64838/0.06130/0.04196, while removal of A yields 0.65999/0.05352/0.03752. These descriptive results do not support a claim that every component is indispensable.

Postprocessing can also change the apparent component ordering. For COMPAS non-IID, the raw AEOD after removing F is 0.250984 versus 0.239597 for Full; after group-threshold calibration, the values are 0.044831 and 0.049143, respectively. Adaptive compensation, signal redundancy, and root-estimation variability are plausible explanations, but the current evidence does not isolate one as the cause.

**TODO — manuscript:** discuss both datasets using complete paired-seed results and raw/calibrated outputs. Retain contrary outcomes, describe the limits of historical Adult controls, and replace universal necessity language with the supported conditional findings.

### R3.8 — Explain benchmark attacks and comparison methods

**Comment summary.** Give more detail about the many robust, fair, and root-based methods.

**Draft response.** We agree that method names alone are insufficient. The comparison should identify each method's objective, server data and attribute access, attack assumptions, training/aggregation/postprocessing interface, implementation source, deviations from the source algorithm, and hyperparameter-selection rule.

The completed CelebA Stage A contains seven methods. FairFed, FairGuard, and their current combination must retain project-adaptation labels; they are not automatically certified reproductions of the original methods. The full manuscript list has 17 rows, so ten rows remain outside Stage A. Existing lite/core/inspired branches cannot be substituted without disclosure. FedAA and LASA each passed full-CelebA three-round Benign/S-DFA pilots, and FedAA passed a bitwise complete-state GPU resume check. These are integration gates, not completed ten-seed comparisons, and their results are excluded from the scientific tables.

**TODO — manuscript/experiments:** complete the implementation-fidelity table, resolve hybrid ordering and method mapping, integrate and validate remaining methods, and perform frozen matched comparisons. LoGoFair-style postprocessing can reuse matching trained models, so 1,000 missing matrix records do not necessarily require 1,000 additional CNN trainings.

### R3.9 — Public source code and experimental configurations

**Comment summary.** Make code and configurations available for follow-up research.

**Draft response.** The project retains source snapshots, explicit manifests, per-seed outputs, final checkpoint identities, and acceptance/backup records for the completed new cohorts. These support a reproducible release, but a final public revision artifact must be checked as a whole rather than inferred from a previous code upload.

**TODO — release/manuscript:** verify the accessible repository URL and immutable revision/tag, include environment and data-preparation instructions, publish the exact configuration and selection manifests, and document adapted baselines and unavailable legacy checkpoints. Insert the verified artifact link and availability statement only after the release is checked. No repository URL is invented in this draft.

### R3.10 — Limitations

**Comment summary.** Add a brief limitations discussion.

**Draft response.** The limitations supported by the evidence include dependence on representative trusted root data and sensitive attributes; unstable or unidentifiable fairness estimates with sparse groups; known malicious-count assumptions in the filtering implementation; candidate-selection and calibration overhead; dependence on the tested binary-group setting; stronger-heterogeneity performance losses; and the conditional nature of the theoretical result.

The comparison also contains implementation and evaluation limitations: some baselines are project adaptations, tuning used validation data, the final independent evaluation is not yet complete, and the legacy and new cohorts differ in selected preprocessing/runtime details. In Stage A, 14 reused records were trained with cu130 and 686 with cu128. Four matched first-round migration checks do not establish full 70-round equivalence. These limitations should accompany the results rather than be obscured by favorable aggregate scores.

**TODO — manuscript:** add a concise limitations paragraph tied to the relevant evidence and distinguish measured failure boundaries from untested deployment scenarios.

### R3.11 — Specify the MLP and evaluate another architecture

**Comment summary.** The MLP architecture is unspecified and an additional architecture would strengthen generality.

**Draft response.** The supplementary tabular model is an input-dependent feature layer followed by a 16-unit hidden layer, ReLU, and a two-logit output; feature lists and preprocessing must be specified for each dataset. We additionally evaluated a 64×64 RGB CNN on CelebA, with three 3×3 convolutional stages of widths 32/64/128, activation/pooling, global average pooling, and a two-class output, without BatchNorm. The frozen source and configuration determine the exact layer and optimizer details.

The completed original 240-run and separate expanded 700-record image cohorts provide evidence beyond the MLP. Because dataset and architecture change together, they establish applicability to the tested image/CNN setting rather than architecture invariance or generality to arbitrary models.

**TODO — manuscript:** verify the final architecture description against the frozen implementation, insert the architecture/training table, and keep separate cohort and evaluation-split labels. No additional architecture-only experiment is claimed.

## 简短中文状态清单（内部，不随英文回复直接提交）

- **已完成证据：** 原1,390次正式补充实验；CelebA阶段A 700条、70组完整10seed及10/9/6seed表；260个表格同checkpoint校准配对；已有严格margin诊断；CelebA共享校准700/700验收、1,400个raw/shared评价、0次新训练，native replay最大差异为0。旧1,390次不重跑。
- **新增归因结果：** 同校准后GuardFed相对FLTrust的ACC差为−0.844个百分点、AEOD均值略高0.00043、ASPD低0.01020；ASPD方向在9/6seed子集保留，AEOD无稳定优势。校准贡献重要，不能把方法间差异全部归因于聚合。
- **已通过但不入表：** FedAA/LASA各自full-CelebA三轮Benign/S-DFA pilot；FedAA真实GPU恢复全state逐位一致。正式十seed基线比较仍未完成。
- **旧Table II本项已完成：** 480格数值来源、44隐藏值、真实n统计、IID/non-IID历史复原和同终轮修正候选已交付；不虚构单seed SD，不认证混用基线身份。最终排表脚本仍未找到，数值生成链已核对。**P0：** Table II正文插入与统计/方法身份更正；DFA贡献定位、理论条件、真实AD2+与root更新、COMPAS负消融解释、符号表、逐条回复与正文位置。
- **P1：** 把已验收共享校准归因表写入正文；基线忠实度与正文剩余10行。共享校准与pilot本地备份已通过整体SHA及2,904个内容文件校验。缺1,000个匹配记录不等于全部新增训练，不用简化分支冒充原法。
- **P2：** CelebA机制消融、最终冻结评价、可复现公开版本、8条推荐文献核验和局限段。规划不等于启动，更不等于完成。
- **核心边界：** 公平性降低伴随准确率代价；不能承诺全面胜出，不能选择GuardFed最佳seed对比基线均值；validation不改名为未触碰test，旧最佳值不伪装成十次均值。
- **提交前：** 逐项关闭TODO，核对全文具体位置、数字与源记录、指标定义、原法/适配名称、统计单位和环境披露；由作者确认最终主张。

## Internal evidence sources

1. [Original decision letter](D:/CodexHome/attachments/1e68c864-5e9d-4736-b278-adc131d2de37/已粘贴的文本.txt).
2. [Initial response plan and implementation/statistical audit](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/返修回复与补充实验计划.md).
3. [Analysis of the 1,390 completed supplementary runs](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/补充实验结果分析_20260924.md).
4. [Verified tabular calibration pairs](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/calibration_final_acceptance/README.md).
5. [Stage A acceptance](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_fullcoverage_v1/acceptance_summary.json), [independent result analysis](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_fullcoverage_v1/final_audit/analysis_zh.md), and [complete table artifacts](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_fullcoverage_final_20260928/README.md).
6. [Stage A protocol](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_fullcoverage_v1/PROTOCOL.md) and [remaining baseline/mechanism plan](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_fullcoverage_v1/BASELINE_AND_MECHANISM_PLAN.md).
7. [Shared-calibration protocol](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_shared_calibration_v1/PROTOCOL.md), [completed result analysis](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_shared_calibration_v1/final/analysis.md), and [verified local backup receipt](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_shared_calibration_v1/sharedcal_backup.json). The archive SHA256 and all 2,904 content-member hashes were verified locally; the archive contains 2,905 members including its inventory. Small reports were extracted; evaluation caches and pilot model/state files remain in the verified archive. The original Stage A models retain their previous backup chain and were not redundantly archived here. This draft's shared-calibration numbers were checked against the extracted analysis.
8. [Four image pipeline gates and two recovery checks](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/baseline_image_gates_v1/queue_complete.json), retained as pipeline-only evidence rather than formal table results.

These are internal audit links, not final public artifact citations. This draft does not modify the manuscript, code, experiments, or historical results.
