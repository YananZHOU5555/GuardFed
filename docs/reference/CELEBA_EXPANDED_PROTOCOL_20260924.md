# CelebA expanded validation v2 — frozen 2026-09-24
User authorized expanding baselines and optimizing GuardFed; superiority is a research target, not a promised or selected outcome. Prior1390 formal runs and40 validation screen jobs remain immutable.

Stage2a:82 NEW70-round jobs, official train162770 and valid19867 only, seed91001, nonIID alpha5, Benign/S-DFA. Existing40 jobs reused analytically, never retrained. Three new execution branches: Median, project FairGuard-root adaptation and its sequential FLTrust hybrid, each7LR x2conditions=42. Original FedAvg/FairFed/FLTrust gain3 smaller LR x2conditions=18. GuardFed expands to LR{.00025,.000375,.0005,.00075,.001} x drop{0,.0025,.005}; omit8 already completed jobs, leaving22. Total82.

All baseline LR sets combine to {.00025,.000375,.0005,.00075,.001,.002,.003}. GuardFed-specific drop search is an explicitly unequal additional search dimension. Unchanged CNN64, batch64,20clients/4nominal malicious,root10%,deterministicFP32,70round. Learning rate affects both clients/root and root-dependent attacks. Root calibration uses training-root data; external valid ranks whole recipes. No test ranking.

Select using unchanged v1 score averaged over Benign/S-DFA: ACC-.35*(.45*AEOD+.45*ASPD+.10*max(AEOD,ASPD))-.10*max(0,max(AEOD,ASPD)-.06). Retain every candidate, raw70round metrics, accuracy champions and 3metric Pareto set including poor candidates; flag low-accuracy zero-gap degeneration. No bestseed/mixedcheckpoint reporting. n=1 is exploratory.

Implementation fidelity is part of acceptance. FairGuard and hybrid here are existing root-data project adaptations, not independently certified original-paper reproductions. Existing FairFed is also root-fairness weighting, an adapted comparison. LoGoFair/FedAA/Fed-NGA approximations must NOT be added as original baselines. Their official adapters are under separate development; see BASELINE_FIDELITY.md. No silent relabeling or retroactive modification of prior results.

Three real2round pipeline canaries for newly dispatched methods must pass checked_result/config/trajectory/checkpoint checks before launch. Existing8worker concurrency retained from benchmark; do not increase to chase CPU utilization in GPU-bound work.

Remaining authorized work: finish faithful baseline adapters and tests; run their own versioned exploratory canaries/search after implementation validation. Freeze comparison protocols before multi-seed evidence covering IID/nonIID and Benign/S-DFA/Sp-DFA. Do not auto-launch a600-job campaign or test confirmation from this bounded manifest. Persist concrete follow-up in state; monitor supervises and backs up only the active frozen82job queue.

Code/core unchanged from2fbf9d7 except this preparer/protocol. Source/data/config/checkpoint identities verified per job. Service guardfed_celeba_expanded, no auto-retry. Off-server incremental SHA-verified backups. Research aim is joint superiority; report explicitly whether obtained.

