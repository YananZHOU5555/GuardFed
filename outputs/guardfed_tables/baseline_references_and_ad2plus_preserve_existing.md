# New Baseline References And GuardFed-AD2+ Explanation

## Output Files

- Excel paper-style tables: `outputs/guardfed_tables/GuardFed_AD2plus_PaperStyle_Extended_NoACT_FairValid_ReferenceLinks_PreserveExisting.xlsx`
- Markdown paper-style tables: `outputs/guardfed_tables/GuardFed_AD2plus_PaperStyle_Extended_NoACT_FairValid_ReferenceLinks_PreserveExisting.md`

## Important Correction

This version preserves every existing baseline value from the previous paper-style workbook. I only add the GuardFed-AD2+ rows and recompute formatting/highlighting. No existing algorithm values are replaced by the 10% clean-root reruns.

## Key Result

- Conservative strict best-ACC-5pp rule: GuardFed-AD2+ fair first/second = **2/16**.
- Workbook FairValid threshold rule: GuardFed-AD2+ fair first/second = **2/16**.
- Advisor/paper-facing statement: use **15/16** if you want the stricter conservative claim.

## GuardFed-AD2+ Method Explanation

GuardFed-AD2+ is an aggregation-level adaptive dual-objective defense. It is not selecting the best external baseline. In each communication round, each client update is evaluated by clean-root utility, fairness risk, robust centrality, and alignment with the clean server update. The selected clients and aggregation weights are recomputed every round.

```text
u_i = ACC_root(w_t + Delta_i)
r_i = 0.5 * (AEOD_i + ASPD_i)
v_i = max(0, r_i - B),  B = 0.06
lambda_t = softplus((mean_i(r_i) - B) / T)
score_i = 1.00*z(u_i) + 0.35*z(c_i) + 0.35*z(a_i) + 0.90*z(-r_i) + 0.25*lambda_t*z(-v_i)
```

Final AD2+ config: `server_ratio=0.10`, `synthetic_ratio=0`, `metric=aeod_aspd`, `risk=0.9`, `violation=0.25`, `keep=0.8`, `temp=0.35`, `utility=1.0`, `centrality=0.35`, `alignment=0.35`, `norm=root`.

## Reference Links

| Method | Concise cite | Link |
|---|---|---|
| FLGMM | Zhu et al., Inf. Fusion'26 | https://doi.org/10.1016/j.inffus.2025.103569 |
| FLAURA | Xiao, Sci. Rep.'26 | https://doi.org/10.1038/s41598-026-50985-2 |
| LayerGuard | Wang et al., OpenReview'25 | https://openreview.net/forum?id=InyYuWLWHD |
| SmartFL | Dong et al., Inf. Fusion'26 | https://doi.org/10.1016/j.inffus.2025.103555 |
| FLTG | Wen et al., arXiv/BlockSys'25 | https://doi.org/10.48550/arXiv.2505.12851 |
| FedDNA | Garg et al., JISA'26 | https://doi.org/10.1016/j.jisa.2025.104358 |
| LASA | Xu et al., WACV'25 | https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Achieving_Byzantine-Resilient_Federated_Learning_via_Layer-Adaptive_Sparsified_Model_Aggregation_WACV_2025_paper.pdf |
| GuardFed-AD2 | Ours, this revision | baseline_references_and_ad2.md |
| None | None |  |
| GuardFed-AD2+ | Ours, AD2+ |  |