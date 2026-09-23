# New Baseline References And GuardFed-AD2+ Explanation

## Output Files

- Excel paper-style tables: `outputs/guardfed_tables/GuardFed_AD2plus_PaperStyle_Extended_NoACT_FairValid_ReferenceLinks.xlsx`
- Markdown paper-style tables: `outputs/guardfed_tables/GuardFed_AD2plus_PaperStyle_Extended_NoACT_FairValid_ReferenceLinks.md`

## Key Result To Report

GuardFed-AD2+ should be reported with the conservative double-sided-attack fairness summary:

| Method | Conservative fair first/second | Scope | Rule |
|---|---:|---|---|
| GuardFed-AD2+ | **15/16** | Adult/COMPAS, IID/non-IID, S-DFA/Sp-DFA, AEOD/ASPD | Fairness metric is counted only when the method remains within 5 percentage points of the best ACC in the same scenario. |

The Excel workbook uses the older FairValid table style, where Adult fairness cells are ranked when ACC >= 80% and COMPAS fairness cells are ranked when ACC >= 60%. Under that table-format rule, GuardFed-AD2+ is 16/16. For advisor-facing or paper-facing claims, use the stricter and safer **15/16** statement.

## New Baseline References

| Method | Concise cite for table | Full reference / status | Link | Why it is relevant |
|---|---|---|---|---|
| Fed-NGA | Fed-NGA, arXiv'24 | arXiv 2024 | https://arxiv.org/abs/2408.09539 | Normalized-gradient style robust aggregation baseline. |
| Huber-BRFL | Huber-BRFL, AAAI'24 | AAAI 2024 | https://ojs.aaai.org/index.php/AAAI/article/view/30181 | Huber-loss Byzantine robust federated learning aggregation. |
| LoGoFair | LoGoFair, AAAI'25 | AAAI 2025 | https://arxiv.org/abs/2503.17231 | Recent fairness-aware federated learning baseline. |
| AdaAggRL | AdaAggRL, AAMAS'22 | AAMAS 2022 | https://www.ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf | Adaptive aggregation via reinforcement-learning inspired policy. |
| FedAMM | FedAMM, MICCAI'25 | MICCAI 2025 | https://papers.miccai.org/miccai-2025/0329-Paper1764.html | Recent adaptive robust aggregation baseline. |
| FedAA | FedAA, AAAI'25 | AAAI 2025 | https://ojs.aaai.org/index.php/AAAI/article/view/33878 | High-impact adaptive aggregation baseline. |

## GuardFed-AD2+ 中文详细解释

### 1. 方法定位

GuardFed-AD2+ 是一个单独的自适应双目标聚合算法，不是从候选 baseline 中挑最优方法。它在每一轮联邦训练中对每个客户端更新进行 clean-root utility、公平风险、鲁棒中心性和 server-root 对齐度评估，然后动态计算客户端得分、筛选集合和聚合权重。

最终表中的 AD2+ 使用 10% clean server/root data，不使用 synthetic data。模型输入不包含敏感属性。

### 2. 为什么它仍然可以叫 adaptive

AD2+ 的部分超参数是固定的，例如公平预算 `B=0.06`、保留比例 `keep_ratio=0.8` 和 softmax temperature。但算法的聚合行为不是固定的。每一轮都会重新计算：

- 每个客户端的 clean-root ACC；
- 每个客户端的 AEOD/ASPD fairness risk；
- 每个客户端到鲁棒 median update 的距离；
- 每个客户端与 clean server update 的 cosine alignment；
- 公平违反项 `v_i = max(0, r_i - B)`；
- 动态 dual multiplier `lambda_t = softplus((mean_i(r_i) - B) / T)`；
- 最终保留的客户端集合；
- softmax 聚合权重；
- root norm scaling 系数。

因此，GuardFed-AD2+ 是 **aggregation-level adaptive**。它不是外层自动调参 selector，而是在每一轮根据当前客户端更新和 clean-root 评估信号动态改变聚合结果。

### 3. 关键公式

对第 `t` 轮第 `i` 个客户端更新 `Delta_i`：

```text
u_i = ACC_root(w_t + Delta_i)
r_i = 0.5 * (AEOD_i + ASPD_i)
v_i = max(0, r_i - B),  B = 0.06
lambda_t = softplus((mean_i(r_i) - B) / T)
c_i = - distance(Delta_i, median_update)
a_i = max(0, cosine(Delta_i, Delta_server))
```

最终客户端分数为：

```text
score_i =
    1.00 * z(u_i)
  + 0.35 * z(c_i)
  + 0.35 * z(a_i)
  + 0.90 * z(-r_i)
  + 0.25 * lambda_t * z(-v_i)
```

其中：

- `z(u_i)` 奖励 clean-root accuracy 高的客户端；
- `z(c_i)` 奖励接近鲁棒中心的客户端；
- `z(a_i)` 奖励与 clean server update 方向一致的客户端；
- `z(-r_i)` 惩罚公平风险高的客户端；
- `lambda_t * z(-v_i)` 动态惩罚超过公平预算的客户端。

### 4. 最终参数

| 参数 | 数值 | 含义 |
|---|---:|---|
| `server_ratio` | `0.10` | 10% clean server/root data |
| `synthetic_ratio` | `0` | 不使用 synthetic data |
| `act_fairness_metric` | `aeod_aspd` | 同时考虑 AEOD 和 ASPD |
| `act_fairness_budget` | `0.06` | 公平风险预算 |
| `act_risk_weight` | `0.90` | 公平风险惩罚强度 |
| `act_violation_weight` | `0.25` | 超预算违反项惩罚强度 |
| `act_keep_ratio` | `0.80` | 每轮保留 80% 客户端 |
| `act_temperature` | `0.35` | softmax 权重温度 |
| `ad2_utility_weight` | `1.00` | clean-root ACC 权重 |
| `ad2_centrality_weight` | `0.35` | 鲁棒中心性权重 |
| `ad2_alignment_weight` | `0.35` | server-root 对齐权重 |
| `ad2_norm_mode` | `root` | 更新范数归一到 clean server update |

### 5. 可直接写给导师或论文的方法描述

英文版本：

> GuardFed-AD2+ introduces an adaptive dual-objective aggregation mechanism that jointly evaluates clean-root utility, fairness risk, robust centrality, and server-update alignment for each client update. A round-adaptive dual multiplier increases the penalty on fairness-violating updates when the current communication round exhibits higher clean-root fairness risk. This allows GuardFed-AD2+ to suppress updates that are both attack-suspicious and fairness-harmful while preserving clean-root utility.

中文版本：

> GuardFed-AD2+ 在每轮训练中动态评估每个客户端更新对干净 server/root 数据的性能、公平性、鲁棒中心性和方向一致性影响。当某一轮整体公平风险升高时，算法会自动增大公平性惩罚，从而更强地抑制可能造成公平性攻击的恶意更新。相比固定规则筛选，AD2+ 的客户端选择和聚合权重会随训练状态和攻击强度动态变化，因此更适合防御同时影响性能和公平性的双重攻击。

## 表格说明

- Excel 的 `Table II COMPAS` 和 `Table III Adult` 按旧版论文表结构组织。
- Methods 后的 Citation 是精简 cite。
- `Reference Links` sheet 放正式 URL。
- `N/R` 表示该 baseline 在最终 10% clean-root 协议下没有对应攻击结果，未用旧协议数值混填。
- 灰色 fairness 单元格表示 ACC 未达到公平排名阈值，不参与 AEOD/ASPD 第一/第二排名。
- GuardFed-ACT 和 Class-B FL 均已排除。
