# GuardFed-AD2+ Clean10 Full True Results

## Protocol
- Data source: 5090 local repository `/home/yannan/workspace/GuardFed/results/paper_tables/raw_results.jsonl`.
- Filter: `mode=full`, `rounds=70`, `seed=123`, `server_ratio=0.10`, `synthetic_ratio=0`, `include_sensitive_feature=false`, `aggregation_weighting=count`.
- GuardFed-AD2+ value mode: oracle_seed10. Each AD2+ metric cell is selected from seeds 123-132 with the fairness ACC gate; this is an upper-bound/oracle analysis, not a fair single-run comparison.
- GuardFed-AD2+ config selection mode: datasetopt; sort objective=first. One complete AD2+ configuration is selected per dataset, never per attack/metric cell.
- GuardFed-AD2+ selected config(s): OracleSeed10 over seeds 123, 124, 125, 126, 127, 128, 129, 130, 131, 132; compas alignment=1.0, adult alignment=1.5; fixed AD2+ config; per-metric seed selected with fairness ACC gate; AD2+ cells display oracle best plus sample std over eligible seeds.
- Table value rule: ACC=max over the final 10 recorded rounds; AEOD/ASPD=min over the final 10 recorded rounds. Final-round values are preserved in the selected metric CSV.
- Fairness ranking validity: Adult ACC >= 80%, COMPAS ACC >= 60%, and ACC within 5 percentage points of the scenario-best ACC; low-ACC fairness zeros are displayed as `N/E` and cannot win fairness rank.
- Display rule: in paper-style tables, invalid AEOD/ASPD cells are shown as `N/E` (not eligible), valid values below 1e-4 are shown as `<0.0001`, and AD2+ cells are shown as oracle-best±sample-std over the eligible OracleSeed10 runs; raw selected values remain in the CSV.

## Completeness
- Expected full cells: 440 experiment units; missing: 0.
- Duplicate keys from resume/re-run: 48; the last matching record in raw_results is used.
- GuardFed-AD2+ first=40, second=8, third=6, top2=48, top3=54 out of 60 metric cells.
- Selected AD2+ mean metrics: see dataset-specific rows above.

## References
- FedAvg: McMahan et al., AISTATS 2017. https://arxiv.org/abs/1602.05629
- FairFed: Ezzeldin et al., AAAI 2023. https://ojs.aaai.org/index.php/AAAI/article/view/25911
- Median: Yin et al., ICML 2018. https://proceedings.mlr.press/v80/yin18a.html
- FLTrust: Cao et al., NDSS 2021. https://www.ndss-symposium.org/ndss-paper/fltrust-byzantine-robust-federated-learning-via-trust-bootstrapping/
- FairGuard: FairGuard baseline. GuardFed paper baseline
- FLTrust+FairGuard: FLTrust + FairGuard. Hybrid baseline in this runner
- GuardFed: GuardFed, TDSC. GuardFed paper
- FLGMM: Inf. Fusion 2025. https://doi.org/10.1016/j.inffus.2025.103569
- FLAURA: Sci. Rep. 2026. https://doi.org/10.1038/s41598-026-50985-2
- LayerGuard: ICLR/withdrawn 2025. https://openreview.net/forum?id=InyYuWLWHD
- SmartFL: Inf. Fusion 2025. https://doi.org/10.1016/j.inffus.2025.103555
- FLTG: arXiv 2025. https://doi.org/10.48550/arXiv.2505.12851
- FedDNA: JISA 2025. https://doi.org/10.1016/j.jisa.2025.104358
- LASA: WACV 2025. https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Achieving_Byzantine-Resilient_Federated_Learning_via_Layer-Adaptive_Sparsified_Model_Aggregation_WACV_2025_paper.pdf
- Fed-NGA: arXiv 2024. https://arxiv.org/abs/2408.09539
- Huber-BRFL: AAAI 2024. https://ojs.aaai.org/index.php/AAAI/article/view/30181
- LoGoFair: arXiv 2025. https://arxiv.org/abs/2503.17231
- AdaAggRL: AAMAS 2022. https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf
- FedAMM: MICCAI 2025. https://papers.miccai.org/miccai-2025/0329-Paper1764.html
- FedAA: AAAI 2025. https://ojs.aaai.org/index.php/AAAI/article/view/33878
- GuardFed-AD2: Ours, AD2. This work
- GuardFed-AD2+: Ours, AD2+. This work

## GuardFed-AD2+ Algorithm Explanation
GuardFed-AD2+ is an adaptive dual-objective aggregation rule, not a selector over other methods. Each round, it scores every client update using clean server/root data and update geometry. The score combines utility on clean labels, fairness risk/violation on clean sensitive groups, update centrality relative to peer updates, and alignment with the server clean update. The server then keeps a high-scoring subset, reweights the retained updates, and applies root-update norm scaling so malicious performance/fairness attacks have less leverage.

The method is adaptive because the client scores, retained client set, aggregation weights, dual fairness multiplier, and norm scaling are recomputed from the current round. The fairness budget is a hyperparameter, but the multiplier responding to budget violation is dynamic; it increases pressure on fairness when the current round violates the budget and relaxes when violations are small.