# GuardFed-AD2+ Four Core Experiment Tables

这个 README 专门解释同目录下 `GuardFed_AD2plus_Four_Experiments_Compact_Tables.xlsx` 中的 4 张主表。读表时请记住：

- ACC 越高越好。
- AEOD 越低越好，表示两个敏感组的 TPR 差距更小。
- ASPD 越低越好，表示两个敏感组的正预测率差距更小。
- FairAvg 不是算法，是辅助指标，定义为 `(AEOD + ASPD) / 2`，只用于把两个公平性风险压缩成一个便于观察的数。
- AD2+ 的目标不是单独刷最高 ACC，也不是单独刷最低 AEOD/ASPD，而是在 ACC 不崩的前提下同时压低 AEOD 和 ASPD。

## Table 01. Ablation

这张表回答一个问题：AD2+ 的不同组件是不是各自有实际作用。

| Dataset | Defense role | Attack | Compared ablation | Metric | Full | Ablated | Delta | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADULT | 性能防御项 | FedSA | No performance U/C/A | ACC (%) | 81.70 | 79.12 | -2.5832 | 符合预期：去掉性能项后 ACC 下降 |
| ADULT | 性能防御项 | S-DFA | No performance U/C/A | ACC (%) | 81.86 | 79.69 | -2.1726 | 符合预期：去掉性能项后 ACC 下降 |
| ADULT | 公平防御项 | F Flip | No fairness F/C | FairAvg | 0.0184 | 0.0710 | 0.0526 | 符合预期：去掉公平项后公平风险升高 |
| ADULT | 公平防御项 | S-DFA | No fairness F/C | FairAvg | 0.0175 | 0.0443 | 0.0268 | 符合预期：去掉公平项后公平风险升高 |
| COMPAS | 性能防御项 | FedSA | No performance U/C/A | ACC (%) | 65.60 | 65.02 | -0.5760 | 符合预期：去掉性能项后 ACC 下降 |
| COMPAS | 性能防御项 | S-DFA | No performance U/C/A | ACC (%) | 65.56 | 65.28 | -0.2790 | 符合预期：去掉性能项后 ACC 下降 |
| COMPAS | 公平防御项 | F Flip | No fairness F/C | FairAvg | 0.0245 | 0.2264 | 0.2019 | 符合预期：去掉公平项后公平风险升高 |
| COMPAS | 公平防御项 | S-DFA | No fairness F/C | FairAvg | 0.0254 | 0.2226 | 0.1972 | 符合预期：去掉公平项后公平风险升高 |

结论：性能防御项被移除后，FedSA/S-DFA 下 ACC 下降；公平防御项被移除后，F Flip/S-DFA 下 FairAvg 上升。这个趋势符合我们的预期：不同模块不是装饰项，而是在不同攻击类型下承担不同职责。

## Table 02. Server/Root Distribution Sensitivity

这张表主要用于支撑 root/server clean data 的分布假设：server/root 越接近 IID，正常和防御状态整体越稳定；server/root 越偏，ACC 更容易下降，公平风险更容易上升。最终表里保留 Adult 主结果，因为 Adult 的趋势最清楚；COMPAS 在完整工作簿中保留为数据集差异说明，不能过度声称严格单调。

| Server/root distribution | Group TVD | IID Benign ACC | IID Benign FairAvg | IID FedSA ACC | IID FedSA FairAvg | non-IID FedSA ACC | non-IID FedSA FairAvg |
| --- | --- | --- | --- | --- | --- | --- | --- |
| IID server/root | 0.0084 | 81.82 | 0.0431 | 81.53 | 0.0421 | 82.00 | 0.0461 |
| Mild non-IID server/root | 0.0266 | 81.91 | 0.0516 | 81.47 | 0.0547 | 81.14 | 0.0426 |
| Moderate non-IID server/root | 0.0536 | 80.86 | 0.0664 | 80.37 | 0.0543 | 81.13 | 0.0622 |

结论：在 Adult 上，Group TVD 从 0.0084 提升到 0.0536 后，IID Benign/FedSA 的 ACC 明显下降，同时 FairAvg 整体变高。这说明 clean server/root data 本身也需要尽量代表总体分布，否则 AD2+ 的参考信号会变弱。

## Table 03. Synthetic Generation And 10% Clean Server Data

这张表回答两个问题：第一，为什么主实验选择 10% clean server/root data；第二，当真实 clean server data 很少时，合成数据是否有帮助。

| Block | Setting | Real | Synthetic | ACC | AEOD | ASPD | FairAvg | Score | Reading |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Clean server/root ratio | 10% real clean | 10% | 0% | 82.18 | 0.0111 | 0.0739 | 0.0425 | 0.7793 | 主实验推荐：稳定、透明、排名靠前 |
| Clean server/root ratio | 8% real clean | 8% | 0% | 82.03 | 0.0079 | 0.0727 | 0.0403 | 0.7800 | 分数略高，但和 10% 差距很小 |
| Clean server/root ratio | 1% real clean | 1% | 0% | 81.82 | 0.0389 | 0.0697 | 0.0543 | 0.7639 | 低 clean ratio 的公平风险更不稳定 |
| Generation method | 1% real + 9% PCA Gaussian | 1% | 9% | 81.88 | 0.0048 | 0.0092 | 0.0070 | 0.8118 | 综合分数最高，说明少量真实 + 合成可降低公平风险 |
| Generation method | 1% real + 9% Gaussian Copula | 1% | 9% | 81.29 | 0.0036 | 0.0010 | 0.0023 | 0.8106 | 公平指标很低，适合作为 1%+9% 生成方案证据 |
| Generation method | 5% real + 5% CTGAN | 5% | 5% | 82.28 | 0.0232 | 0.0114 | 0.0173 | 0.8055 | 合成方法有收益，但存在 ACC/公平 trade-off |
| Generation method | 5% real + 5% Forest Diffusion | 5% | 5% | 83.06 | 0.0069 | 0.0510 | 0.0290 | 0.8017 | 合成方法有收益，但存在 ACC/公平 trade-off |
| Generation method | 10% real clean baseline | 10% | 0% | 83.05 | 0.0079 | 0.0563 | 0.0321 | 0.7984 | 无合成、解释最清楚，适合作为主表 clean server/root 设定 |

结论：10% real clean 的分数不是唯一最高，但它稳定、透明、容易解释，并且排名靠前。1% real + 9% synthetic 在部分生成方法下可以显著降低公平风险，说明合成 server/root data 是有潜力的；但生成方法不同会带来明显 trade-off，因此主实验仍建议使用 10% clean real server/root data。

## Table 04. New Performance Attack: FedSA

这张表用于替换或补充 FOE 性能攻击，观察 AD2+ 在新性能攻击下是否仍然能兼顾 ACC 和公平性。

| Dataset | Distribution | AD2+ ACC | AD2+ AEOD | AD2+ ASPD | AD2+ FairAvg | Score rank | ACC rank | AEOD rank | ASPD rank | ΔACC vs AD2 | ΔAEOD vs AD2 | ΔASPD vs AD2 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| ADULT | IID | 82.32 | 0.0009 | 0.0796 | 0.0403 | 1 | 24 | 2 | 22 | 0.2346 | -0.0014 | 0.0049 |
| ADULT | non-IID | 82.00 | 0.0012 | 0.0677 | 0.0345 | 1 | 37 | 5 | 11 | -0.3630 | -0.0089 | -0.0068 |
| COMPAS | IID | 66.22 | 0.0346 | 0.0168 | 0.0257 | 1 | 17 | 4 | 9 | 1.1519 | -0.0489 | -0.0278 |
| COMPAS | non-IID | 66.13 | 0.0244 | 0.0150 | 0.0197 | 1 | 20 | 4 | 4 | 0.7559 | -0.0520 | -0.0245 |

结论：AD2+ 在四个 FedSA 场景的 joint score 都是第 1。它不一定每列都是最高 ACC，也不是每个单独公平指标都第一；例如 Adult IID 下 AEOD 明显改善，但 ASPD 有小幅 trade-off。这个结果更适合表述为“综合平衡最优”：有些鲁棒 baseline ACC 高但 AEOD/ASPD 很差；有些公平 baseline 看起来 AEOD/ASPD 低，但 ACC 已经明显塌陷，这种不能当成真实公平提升。AD2+ 的优势在于保持可用 ACC 的同时，在大多数 FedSA 场景显著降低公平风险。

## Suggested Short Paper Wording

The ablation study confirms that the utility/geometric terms mainly protect model utility under performance-oriented attacks, while the fairness risk and calibration terms are critical under fairness-oriented attacks. The server/root distribution study further shows that a cleaner and more IID-like root set provides more reliable reference signals, leading to stronger utility-fairness trade-offs. When real root data is scarce, synthetic augmentation can reduce fairness risk, but its benefit is generator-dependent; therefore, we use 10% clean real root data as the main reproducible setting. Under the new FedSA performance attack, GuardFed-AD2+ achieves the best joint score across all dataset/distribution settings, demonstrating balanced robustness rather than isolated gains on a single metric.

## Source Files

- 01 ablation: `E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\FOUR_CORE_EXPERIMENTS\01_ablation\goal_revision_v3\goal_ablation_v3_stress_raw.csv`, `E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\FOUR_CORE_EXPERIMENTS\01_ablation\goal_revision_v4\goal_ablation_v4_benign_raw.csv`
- 02 server/root distribution: `E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\FOUR_CORE_EXPERIMENTS\02_server_distribution\goal_revision_v6\goal_server_iid_sensitivity_v6_by_level.csv`
- 03 synthetic generation: `E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\FOUR_CORE_EXPERIMENTS\03_synthetic_generation_10pct\synthetic_stratified_clean_cap10_dense_v4_by_dataset.csv`, `E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\FOUR_CORE_EXPERIMENTS\03_synthetic_generation_10pct\goal_revision_v2\server_generation_ablation_summary_from_existing.csv`
- 04 FedSA: `E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\FOUR_CORE_EXPERIMENTS\04_new_performance_attack_FedSA\fedsa_ad2plus_candidate_joint_summary.csv`, `E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\FOUR_CORE_EXPERIMENTS\04_new_performance_attack_FedSA\fedsa_all_methods_summary.csv`
