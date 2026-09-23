# High-impact Baseline Candidates For GuardFed-AD2

## Recommendation

如果目标是让审稿人看到“我们不是只和旧 baseline 比”，我建议加入一个高影响力 baseline pack：

1. **AdaAggRL, AAAI 2025**：高影响力、强相关、主打 adaptive robust aggregation。
2. **Huber-BRFL, AAAI 2024**：高影响力、理论强、实现相对直接。
3. **Fed-NGA, NeurIPS 2025 main**：高影响力、适合 non-IID Byzantine robustness。
4. **LoGoFair, AAAI 2025**：高影响力 fairness baseline，虽然不是攻击防御，但能回应 fairness reviewer。
5. **FedAMM, IEEE TIFS 2025**：顶尖安全期刊，强调高恶意客户端比例。

如果表格不能再加太多方法，最小强版本是：

| Priority | Method | Venue | Why include |
|---|---|---|---|
| 1 | AdaAggRL | AAAI 2025 | Recent top-conference adaptive poisoning defense; directly challenges our "adaptive trust" claim. |
| 2 | Fed-NGA | NeurIPS 2025 main | Recent top-conference Byzantine + heterogeneity defense; strong non-IID baseline. |
| 3 | Huber-BRFL | AAAI 2024 | Top-conference theoretically grounded robust aggregator; easy to explain and implement. |
| 4 | LoGoFair | AAAI 2025 | Top-conference group-fair FL baseline; useful fairness-only comparison. |

## Best Candidates

| Method | Venue | Type | Fit to our setting | Implementation cost | Reference |
|---|---|---|---|---|---|
| **AdaAggRL** | AAAI 2025 | Robust FL against sophisticated poisoning | Very high. It is an adaptive aggregation method for model poisoning, so it directly competes with our adaptive reliability design. | Medium-high. Needs distribution simulation/MMD/history and policy-learning component; a simplified reproducible version can use MMD-based adaptive weights without full RL retraining. | Wang et al., "Defending Against Sophisticated Poisoning Attacks with RL-based Aggregation in Federated Learning." DOI: https://doi.org/10.1609/aaai.v39i24.34733 |
| **Fed-NGA** | NeurIPS 2025 main | Byzantine robust FL under heterogeneity | Very high. It explicitly targets Byzantine attacks and data heterogeneity/non-IID, which matches our IID/non-IID comparison. | Low-medium. Normalize client gradients/updates and aggregate weighted normalized directions. | Zuo et al., "Efficient Federated Learning against Byzantine Attacks and Data Heterogeneity via Aggregating Normalized Gradients." NeurIPS 2025: https://neurips.cc/virtual/2025/loc/san-diego/poster/118753 |
| **Huber-BRFL** | AAAI 2024 | Robust aggregation by Huber loss minimization | High. It is a clean, top-conference robust aggregation baseline with theoretical support. | Medium. Need solve vector Huber M-estimator or use iterative reweighted update aggregation. | Zhao et al., "A Huber Loss Minimization Approach to Byzantine Robust Federated Learning." DOI: https://doi.org/10.1609/aaai.v38i19.30181 |
| **LoGoFair** | AAAI 2025 | Local/global group fairness in FL | High for fairness comparison, but not an attack defense. It should be labeled "fairness-only baseline". | Medium. Post-processing or threshold adjustment using sensitive groups. | Zhang et al., "LoGoFair: Post-Processing for Local and Global Fairness in Federated Learning." DOI: https://doi.org/10.1609/aaai.v39i21.34404 |
| **FedAMM** | IEEE TIFS 2025 | Robust FL against majority malicious clients/backdoor | High venue impact; attack type is backdoor/majority malicious rather than fairness attack. | Medium-high. PCA/critical-parameter similarity/clustering/noise perturbation. | Gai et al., "FedAMM: Federated Learning Against Majority Malicious Clients Using Robust Aggregation." DOI: https://doi.org/10.1109/TIFS.2025.3607273 |
| **FedHAN** | IJCAI 2025 | Poisoning defense under semi-asynchronous heterogeneous FL | Good venue and poisoning-defense relevance, but our current setting is synchronous. | Medium-high if implemented faithfully; lower if using synchronous cache approximation. | Wang et al., "FedHAN: A Cache-Based Semi-Asynchronous Federated Learning Framework Defending Against Poisoning Attacks in Heterogeneous Clients." DOI: https://doi.org/10.24963/ijcai.2025/379 |
| **FedInv** | AAAI 2022 | Byzantine robust FL via inversing local updates | High venue but older. Good if we need another AAAI robust baseline. | High. Requires inversion/dummy dataset reconstruction and Wasserstein filtering. | Zhao et al., "FedInv: Byzantine-Robust Federated Learning by Inversing Local Model Updates." DOI: https://doi.org/10.1609/aaai.v36i8.20903 |
| **GAS** | ICML 2023 | Robust FL on heterogeneous/non-IID data via gradient splitting | Very strong venue and non-IID relevance, but older than 2025. | Medium. Split gradients and plug into existing robust AGRs. | Liu et al., "Byzantine-Robust Learning on Heterogeneous Data via Gradient Splitting." ICML 2023: https://icml.cc/virtual/2023/poster/24965 |

## Methods I Would Not Put In The Main Table

| Method | Venue/status | Reason |
|---|---|---|
| D-Byz-SGDM / DeMoA | NeurIPS 2025 OPT workshop | Strong idea, but workshop rather than NeurIPS main; also focused on partial participation. Better as related work, not main baseline. |
| Rob-FCP | ICML 2024 | It is Byzantine-robust federated conformal prediction, not standard model aggregation for ACC/AEOD/ASPD tables. |
| Rank-Core-Fed / PVC | ICML 2024 | It targets client-level fairness/welfare, not sensitive-attribute group fairness. |
| FedIT safety defense | ICLR 2025 | High venue but about LLM safety alignment under federated instruction tuning; not comparable to tabular Adult/COMPAS fairness metrics. |
| LayerGuard | OpenReview 2025 | Technically relevant, but venue authority is weaker than AAAI/NeurIPS/IJCAI/TIFS unless accepted venue is clarified. |

## How To Present Them In The Paper

I suggest using two groups in the table:

1. **High-impact robust FL baselines**: Huber-BRFL, AdaAggRL, Fed-NGA, FedAMM.
2. **High-impact fairness FL baselines**: FairFed, LoGoFair.

Then state clearly:

> Robust FL baselines are designed primarily for utility-preserving Byzantine or poisoning defense, while fairness FL baselines are designed primarily for group fairness. GuardFed-AD2 is evaluated against both categories because DFA jointly attacks utility and fairness.

This framing helps us avoid an unfair reviewer criticism: LoGoFair is not expected to beat poisoning attacks, and Fed-NGA/Huber/AdaAggRL are not expected to optimize AEOD/ASPD. The point is to show that single-objective high-impact methods fail under dual-objective attacks, while AD2 handles both.

## Practical Integration Order

| Order | Add method | Why first |
|---|---|---|
| 1 | Fed-NGA | Easiest high-impact recent method to implement in a standard synchronous runner. |
| 2 | Huber-BRFL | Strong AAAI baseline; robust M-estimator can be implemented without changing data protocol. |
| 3 | LoGoFair | Gives a recent AAAI fairness-only baseline and strengthens fairness comparison. |
| 4 | AdaAggRL | Strongest rhetorical baseline, but full implementation is more complex. |
| 5 | FedAMM | Strong TIFS journal baseline; add if we want a top security journal comparison. |

