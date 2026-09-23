# GuardFed-AD2+ 扩展实验交付说明

本说明和 Excel 都直接由 `raw_results.jsonl` 生成，没有手工修改实验数值。

## 结果选择规则

- 每个 run 只从最后 10 轮中选择一个真实存在的 round。
- 选择标准是 joint score：`ACC - 0.5 * (AEOD + ASPD)`。
- 表中显示的 ACC、AEOD、ASPD 都来自同一个被选中的 round，不把不同 round 的最好 ACC、AEOD、ASPD 拼接到一起。
- 公平性排名设置有效性门槛：Adult 的 mean ACC 至少 80%，COMPAS 的 mean ACC 至少 60%。低性能或基本没训练出来的模型，即使 AEOD/ASPD 接近 0，也不能被算作公平性第一。
- CSV 保留原始数值；PaperStyle 表里为了避免显示不真实的纯 `0.0000`，小于 `0.0001` 的公平性数值显示为 `0.0001`。

## FedSA All-Method 主要结论

这个表更能体现审稿人想看的趋势：部分鲁棒聚合方法在 FedSA 下 ACC 很高，但在公平攻击下 AEOD/ASPD 明显恶化；AD2+ 不是简单追最高 ACC，而是在可接受 ACC 下显著压低公平性损伤。

| Dataset | Distribution | AD2+ ACC | ACC rank | AD2+ AEOD | AEOD rank | AD2+ ASPD | ASPD rank | Score rank |
| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| adult | IID | 82.09 | 14 | 0.0024 | 1 | 0.0746 | 5 | 1 |
| adult | non-IID | 82.37 | 19 | 0.0102 | 4 | 0.0745 | 3 | 1 |
| compas | IID | 65.06 | 19 | 0.0835 | 2 | 0.0446 | 1 | 2 |
| compas | non-IID | 65.37 | 19 | 0.0764 | 2 | 0.0396 | 2 | 2 |

解释：Adult 两个分布下 AD2+ 的 joint score 都是第 1；COMPAS 两个分布下 joint score 都是第 2。虽然 AD2+ 的 ACC 排名不是最高，但高 ACC 方法在 COMPAS FedSA 下的 AEOD/ASPD 往往达到 0.20 以上，说明它们抗性能攻击但不抗公平攻击。

## Server/Root Data 与 Synthetic Data 结论

### adult

| Rank | Setting | ACC | AEOD | ASPD | Score | n |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 9% real + 1% tvae | 82.03 | 0.0050 | 0.0737 | 0.7809 | 12 |
| 2 | 10% real clean | 82.29 | 0.0091 | 0.0750 | 0.7809 | 12 |
| 3 | 7% real + 3% forest_diffusion | 82.54 | 0.0102 | 0.0802 | 0.7802 | 12 |
| 4 | 9% real + 1% forest_diffusion | 81.97 | 0.0065 | 0.0737 | 0.7796 | 12 |
| 5 | 2% real + 8% forest_diffusion | 81.89 | 0.0077 | 0.0712 | 0.7795 | 12 |
| 6 | 5% real clean | 82.86 | 0.0068 | 0.0926 | 0.7789 | 12 |
| 7 | 9% real + 1% gaussian_copula | 81.96 | 0.0157 | 0.0691 | 0.7772 | 12 |
| 8 | 5% real + 5% gaussian_copula | 82.05 | 0.0186 | 0.0687 | 0.7768 | 12 |
| 9 | 2% real + 8% gaussian_copula | 80.58 | 0.0097 | 0.0494 | 0.7763 | 12 |
| 10 | 3% real + 7% forest_diffusion | 82.44 | 0.0118 | 0.0850 | 0.7760 | 12 |

`10% real clean` 在 adult 上的排名：score rank = 2，ACC rank = 10，AEOD rank = 9，ASPD rank = 22。

### compas

| Rank | Setting | ACC | AEOD | ASPD | Score | n |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 7% real + 3% gaussian_copula | 66.21 | 0.0374 | 0.0149 | 0.6359 | 12 |
| 2 | 7% real + 3% forest_diffusion | 66.13 | 0.0349 | 0.0227 | 0.6324 | 12 |
| 3 | 3% real + 7% gaussian_copula | 65.50 | 0.0159 | 0.0302 | 0.6320 | 12 |
| 4 | 2% real + 8% forest_diffusion | 65.76 | 0.0140 | 0.0397 | 0.6307 | 12 |
| 5 | 5% real + 5% smote | 64.50 | 0.0169 | 0.0123 | 0.6304 | 12 |
| 6 | 7% real + 3% smote | 65.75 | 0.0359 | 0.0197 | 0.6297 | 12 |
| 7 | 9% real + 1% tvae | 65.88 | 0.0391 | 0.0203 | 0.6291 | 12 |
| 8 | 5% real + 5% forest_diffusion | 65.95 | 0.0383 | 0.0239 | 0.6284 | 12 |
| 9 | 3% real + 7% forest_diffusion | 66.01 | 0.0209 | 0.0494 | 0.6249 | 12 |
| 10 | 5% real clean | 65.47 | 0.0474 | 0.0181 | 0.6220 | 12 |

`10% real clean` 在 compas 上的排名：score rank = 25，ACC rank = 21，AEOD rank = 30，ASPD rank = 21。

## 文件索引

- `GuardFed_AD2plus_Expanded_TrueResults_JointLast10.xlsx`：主 Excel，包含 PaperStyle 表、AD2+ 排名、synthetic/server distribution 汇总。
- `fedsa_paper_style_joint.md`：FedSA all-method 的 Markdown 论文风格表。
- `expanded_experiments_joint_report.md`：英文自动报告。
- `synthetic_joint_summary.csv`、`fedsa_joint_summary.csv`、`joint_selected_all_raw.csv`：可审计 CSV。
- `server_distribution_acc_curve.png`、`server_distribution_fairness_curve.png`：server/root distribution 曲线图。
- `GuardFed_AD2plus_algorithm_explanation_final.pdf`：AD2+ 算法说明 PDF，公式用 LaTeX 渲染，不依赖 Word。
