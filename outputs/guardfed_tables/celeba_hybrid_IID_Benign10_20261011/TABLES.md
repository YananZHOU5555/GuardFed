# Hybrid native validation: IID / Benign

Frozen project CNN Hybrid adaptation (CosineFairnessHybrid),recipe learning_rate=0.001,fairness_lambda=20.0,threshold=0.1;native valid-only terminal outputs. The recipe is unchanged from the original four-condition search.

Mean ± sample SD (ddof=1), using each seed’s same terminal70-round checkpoint for all three metrics. ACC is displayed in percent; AEOD and ASPD remain fractions. All source checkpoints and seed IDs are retained in [records.json](records.json); fixed panels are listed in [COVERAGE.json](COVERAGE.json).

| Fixed panel | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |
|---|---:|---:|---:|---:|
| all10 | 10 | 88.026879 ± 1.386421 | 0.02920037 ± 0.00903244 | 0.09250594 ± 0.01410153 |
| exclude_selection_seed9 | 9 | 88.284872 ± 1.188960 | 0.02799785 ± 0.00868993 | 0.09399973 ± 0.01409278 |
| fixed_last6 | 6 | 88.109260 ± 1.360318 | 0.02646322 ± 0.00981595 | 0.08912790 ± 0.01495052 |

Ten seeds = original selected-screen reuse91001 + nine root-adopted formal records91002–91010. The9seed panel excludes selection seed91001;the6seed panel is fixed91005–91010. The panels use identical seed/checkpoint sets across all metrics;all outcomes remain available.

These are exposed-validation development results. Seed91001 participated in the original four-condition recipe search. Prior official-test results had been viewed;no new final-test evaluation is performed here. This is one complete native scene,not whole100 completion,a three-view comparison,statistical significance or a claim of superiority. Training provenance and saved-check platform limits are recorded in [ENVIRONMENT_SCOPE.json](ENVIRONMENT_SCOPE.json);no uniform unmeasured runtime is asserted.

Sources and actual root acceptance pins: [INPUTS.json](INPUTS.json). The earlier pending auxiliary table remains unchanged;this table uses the subsequently adopted native9 chain.
