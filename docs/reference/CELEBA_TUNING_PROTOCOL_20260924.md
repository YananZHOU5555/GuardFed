# CelebA validation tuning v1

Authorized September24 after the first240-image cohort and its test results were reviewed. Those results remain immutable. This is explicitly post-initial-test development, not a never-seen-test claim.

Bounded first screen: 40 new70-round full-training jobs, development seed91001, non-IID alpha5, Benign/S-DFA. GuardFed: learning rate {0.0005,0.001,0.002,0.003} x selector/calibration accuracy drop {0,0.005}, 16 jobs. FedAvg/FairFed/FLTrust: same four learning rates and two conditions,24 jobs. Each method retains its original prediction interface. Training162770; validation19867; official test excluded from this screen and candidate ranking. Batch64,20clients,4 malicious slots, root10%, original CNN and deterministic FP32 unchanged.

Internal AD2+ candidate pool overrides outer U/C/A/F/V weights, temperature, keep ratio, fairness metric, score clip and norm mode. Those outer values are not search dimensions. Drop0 vs .005 changes the root selector. Values above .005 only alter final calibration under this candidate scheme and should be evaluated on the same checkpoint, not retrained. Learning rate affects both clients and root (and thus root-dependent attack); this is whole-system tuning.

Frozen screen ranking: arithmetic mean across the two conditions of ACC - .35*(.45*AEOD+.45*ASPD+.10*max(AEOD,ASPD)) - .10*max(0,max(AEOD,ASPD)-.06). Also retain all three-metric Pareto candidates and each method's accuracy champion. Ranking is validation-only; all candidates, failures and negative results remain recorded. The target is improved utility/fairness, with no promised dominance. GuardFed has an extra method-specific search dimension, disclosed rather than counted as equal search size.

A single development seed is exploratory, not a10-seed statistical result. Before a later formal confirmation, freeze selected configurations and the comparison rule; use paired multiple seeds and a common terminal checkpoint rule. Current script launches no test confirmation or COMPAS retraining. The old test has already been observed, and later results must carry that chronology rather than being described as pristine confirmation.

COMPAS reporting: validation-selected hyperparameters may be optimized. For component evidence, Full and ablations require the same seed/round/statistical reporting rule. A best observed run may be shown separately with its selection rule and all metrics from the same checkpoint; it is not a replacement for the Full10-seed mean or proof of component necessity.

Implementation: unchanged reproduce_paper_tables.py SHA256 cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed. Runner change only groups tuning_candidate separately in summary tables; otherwise candidate configurations with the same seed would collide. Two same-method/same-seed canaries check this grouping through real completed outputs. The frozen jobs hash every data/core file plus their preparer.

Use existing supervisor service guardfed_celeba_tuning with8workers; failure stops dispatch and never retries forever. Source, manifests and canary acceptance are backed up off-server. Further tuning changes get their own recorded protocol, not overwritten results.
