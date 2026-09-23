# E5 ACSIncome protocol — prepared, not formally launched

Worktree `/workspace/GuardFed-acs`, branch `codex/revision-acs-20260923`, base `47eb2f69a6d6b990894da467ffa665650aa2a8fc`.
Code reuses the existing 9→16→2 tabular MLP, training, attacks, AD2+, FedAvg and FLTrust without algorithm changes. Adult and COMPAS legacy behavior remain unchanged.

## Data and explicit departures from the official task

Official source: [Census CA2018 one-year PUMS person file](https://www2.census.gov/programs-surveys/acs/data/pums/2018/1-Year/csv_pca.zip). CSV SHA256 `dc2187fc90df2c5f6b546ee89a2b41c9a97379c9e7136461b6a6c8de871b43e0`, 378817 person rows.

The [official folktables task definition](https://github.com/socialfoundations/folktables/blob/ebace19d8065efc43589f0067bf183398eb156b9/folktables/acs.py) retains `AGEP>16`, `PINCP>100`, `WKHP>0`, `PWGTP>=1`, and labels `PINCP>50000`. This leaves 195665 rows. We explicitly customize its sensitive group from the official default RAC1P to SEX, Male/Census1→1 and Female/Census2→0. SEX is excluded from model input. This is an ACSIncome **SEX fairness variant**, not an unmodified benchmark result.

Features in order: `AGEP,COW,SCHL,MAR,OCCP,POBP,RELP,WKHP,RAC1P`. Retain Census numeric category codes, no one-hot or new learned category mapping. Train-only StandardScaler fits all nine columns. No input feature is missing in this filtered source. The official `np.nan_to_num(x,-1)` uses the positional copy argument; it does not mean NaN→-1. Our explicit zero fill matches its numeric behavior here. PWGTP is used only in the official filter, not as a loss/evaluation sample weight.

Per prespecified seed, stratified person-row 70/30 train/test split uses the joint `SEX × income` stratum. This project-specific split is explicit; the folktables README example uses another split. No household-level separation is claimed. Preprocessing includes all training rows, including root, and never fits heldout rows. Root sampling follows the existing `stratified_sensitive` rule, 10% from the training set. Root rows are removed before client allocation. Dirichlet client allocation is sensitive-group based, preserving the existing implementation. Client/root/test row identities are checked, not just sample counts.

Seed123: 136965 train, 58700 test, 13697 root, 123268 client-pool rows. Both labels and both sensitive groups occur in train/test. All method/attack conditions with a given seed and alpha share identical split/root/client assignment. Alpha changes client allocation only; different seeds change train/test partitions.

## Frozen candidate formal queue

120 jobs: GuardFed-AD2+ (Full), FedAvg, FLTrust × Benign/S-DFA × IID(alpha5000)/non-IID(alpha5) × seeds `[123,456,789,1001,2024,3141,4242,5050,6060,7070]`.

70 rounds; 20 clients; 4 malicious slots for attacks; one local epoch; Adam lr0.005, batch256; real root0.1; synthetic0; no root noise. Existing attack-strength profile: fflip `all_unprivileged`, S-DFA/Sp-DFA FOE `fedsa`, gain4.5, norm-ratio3.0. Existing AD2+ adaptive weights, calibration and other algorithm settings are retained. Device is frozen CPU. Formal scheduling/concurrency is controlled by the parent task and has not been started by this adapter task.

Queue: `deployment/acs_income_v1/manifest.json`; 120 separate `jobs/*.json`. Every job freezes the raw CSV and training/loader/worker/preparer SHA256s. The preparer refuses to overwrite an existing manifest. It prepares jobs only and never launches them. The generic worker verifies source hashes and saves checkpoint SHA, configuration, all rounds and data-contract metadata. Use all 10 prespecified seeds, final round70 metrics at the same checkpoint, mean and sample standard deviation. No test-selected checkpoint, failed-run omission or partial-group formal averages.

## Validation and pilot limitations

15 unit/regression tests passed, including exact frozen-Adult comparison, preserved COMPAS legacy path, ACS filter boundary/labels/SEX/features, heldout-feature perturbation leaving train/scaler identical, and root separation. `scripts/audit_revision_acs_partitions.py` additionally verifies all 20 seed×alpha partitions by passing original CSV row IDs through the same client partition function, checking each real client tensor against its identified rows. It checks unique coverage, pairwise client/root/test disjointness, paired configs and shared split/root across alphas. This is pipeline evidence, not scientific evidence.

Four CPU pilots (maximum four simultaneous single-thread workers): the three benign IID methods for five rounds and GuardFed-AD2+ S-DFA non-IID for two rounds, all seed123. All completed and checkpoint/source hashes matched. Existing reporting **did access the test set** after each round and at final evaluation. Training/calibration used train/root only. No configuration, checkpoint, protocol or method selection was made from these test metrics; the formal queue was frozen before pilots and remained unchanged. All outcomes are retained in `pilot_runs/` and `pilot_review.json`; these runs are never counted among the 120 formal jobs.

Pilot final metrics (accuracy, AEOD, ASPD), not a method ranking:

| Condition | Rounds | Accuracy | AEOD | ASPD |
|---|---:|---:|---:|---:|
| FedAvg benign IID | 5 | 0.786797 | 0.055142 | 0.014115 |
| FLTrust benign IID | 5 | 0.786899 | 0.051595 | 0.013790 |
| GuardFed-AD2+ benign IID | 5 | 0.762385 | 0.026680 | 0.032394 |
| GuardFed-AD2+ S-DFA non-IID | 2 | 0.758910 | 0.003777 | 0.045042 |

All pilot predictions contain both classes (positive-rate0.255–0.401) and accuracy exceeds the measured majority-class0.589421; this only establishes a working learning path. Short single-seed results do not establish comparative performance or robustness. Lower or unfavorable values are preserved.

## Protocol review

Parent review on 2026-09-23 accepted the explicit70/30 joint-SEX×income split and nine numeric Census features with train-only scaling. The prepared120-job CPU queue is handed off for parent-controlled launch, initially the first three seeds with concurrency16, then the remaining seeds after acceptance checks. This subtask has stopped training and source edits.

The parent task should accept or revise the explicit protocol choices before formal launch: SEX variant, numeric Census-code inputs with all-column train-only scaling, person-row joint-stratified70/30 split, equal person weighting, inherited attack-strength/method profile. Any substantive change requires a new versioned manifest; retain the current pilots as belonging to the original candidate protocol. No new algorithm claim is made.

Result entry points: `deployment/acs_income_v1/data_readiness.json`, `partition_identity_audit.json`, `official_source.json`, `pilot_review.json`, `manifest.json`. Raw source is an ignored read-only-use symlink to revision data; shared training environment is only reused, not modified. No formal jobs or old main experiments were started.
