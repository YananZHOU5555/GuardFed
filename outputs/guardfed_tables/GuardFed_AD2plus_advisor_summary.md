# GuardFed-AD2+ 算法解释与结果汇报

> 说明：本文件保留为早期算法解释和结果摘要。论文超参数范围、论文协议与当前扩展实验的 seed-average 汇报口径，以 [GuardFed_paper_hyperparameter_search_and_reporting.md](../../docs/GuardFed_paper_hyperparameter_search_and_reporting.md) 为准；不要将本文件中的早期单 seed 摘要或简化参数表当作最终实验设置。

## 1. 一句话结论

我们在 GuardFed 的基础上设计了 **GuardFed-AD2+**，一个面向双重攻击的自适应双目标防御聚合方法。它同时考虑模型性能、防御鲁棒性和公平性风险，在每一轮联邦训练中动态评估客户端更新、筛选可信更新并自适应分配聚合权重。

在 Adult 和 COMPAS 两个数据集、IID 和 non-IID 两种划分、S-DFA 和 Sp-DFA 两类双重攻击下，GuardFed-AD2+ 在主表 16 个公平性指标中有 **15/16 个达到第一或第二**，整体上比基础 GuardFed-AD2 的 **13/16** 更稳定。

## 2. 实验设置

- 数据集：Adult、COMPAS
- 数据划分：IID alpha=5000，non-IID alpha=5
- 攻击：S-DFA、Sp-DFA
- 客户端数：20
- 恶意客户端数：4
- 训练轮数：70
- 本地 epoch：1
- batch size：256
- learning rate：0.005
- seed：123
- clean server/root data：10%
- synthetic data：0%
- 模型输入：不包含敏感属性
- Adult 敏感属性：sex，Male=1，Female=0
- COMPAS 敏感属性：race，African-American=1，Others=0

## 3. GuardFed-AD2+ 的动机

原始 GuardFed 的思想是利用 server clean data 检查客户端更新是否可信，并同时考虑公平性。但审稿人可能认为原来的分数设计偏启发式，尤其是不同分数组件之间如何组合缺少更清晰的约束解释。

GuardFed-AD2+ 的目标是把这个过程包装成一个更明确的 **自适应双目标约束聚合问题**：

- 性能目标：客户端更新不能显著损害 clean-root accuracy。
- 公平目标：客户端更新不能显著扩大 AEOD/ASPD。
- 鲁棒目标：客户端更新应接近正常客户端的鲁棒中心，并与 clean server update 方向一致。

因此，GuardFed-AD2+ 不是简单地“选公平性最好的客户端”，也不是从已有 baseline 中挑最好的方法，而是在每一轮对每个客户端更新进行多维度动态评分。

## 4. 为什么它可以叫 Adaptive

GuardFed-AD2+ 的超参数是固定的，但算法行为是动态的。它的 adaptive 体现在每一轮训练中以下变量都会随当前模型、当前客户端更新和当前攻击状态变化：

1. 每个客户端的 clean-root utility 动态变化。
2. 每个客户端的 AEOD/ASPD fairness risk 动态变化。
3. 每个客户端相对鲁棒中心的 centrality 动态变化。
4. 每个客户端与 clean server update 的 alignment 动态变化。
5. 公平性违反项 `v_i` 动态变化。
6. dual multiplier `lambda_t` 根据当前轮整体 fairness violation 动态变化。
7. 被保留的客户端集合动态变化。
8. 聚合权重动态变化。
9. norm scaling 根据当前 clean server update 动态变化。

所以更准确的表述是：

> GuardFed-AD2+ is adaptive at the aggregation level. It uses fixed hyperparameters but dynamically computes client trust scores, fairness penalties, selected client sets, aggregation weights, and normalization scales in every communication round.

## 5. 算法细节

对第 `t` 轮中第 `i` 个客户端更新 `Delta_i`，GuardFed-AD2+ 先在 clean server/root data 上临时评估该更新带来的效果。

### 5.1 Clean-root utility

记客户端更新后的 clean-root accuracy 为：

```text
u_i = ACC_root(w_t + Delta_i)
```

`u_i` 越高，说明该客户端更新对干净 server data 的性能越有利。

### 5.2 Fairness risk

我们同时考虑 AEOD 和 ASPD：

```text
r_i = 0.5 * (AEOD_i + ASPD_i)
```

其中：

```text
AEOD = |TPR_group0 - TPR_group1|
ASPD = |P(pred=1 | group0) - P(pred=1 | group1)|
```

`r_i` 越大，说明该客户端更新越可能引入或放大公平性偏差。

### 5.3 Fairness violation

给定公平预算：

```text
B = 0.06
```

每个客户端的公平违反项为：

```text
v_i = max(0, r_i - B)
```

这里 `B` 是固定预算，但 `r_i` 每一轮、每个客户端都会变化，所以 `v_i` 是动态的。

### 5.4 Dynamic dual multiplier

GuardFed-AD2+ 用当前轮所有客户端的平均 fairness risk 自动调整 dual penalty：

```text
lambda_t = softplus((mean_i(r_i) - B) / T)
```

如果当前轮攻击导致整体公平风险升高，则：

```text
mean_i(r_i) > B  =>  lambda_t 增大
```

公平惩罚自动变强。

如果当前轮客户端更新较干净，则：

```text
mean_i(r_i) <= B  =>  lambda_t 减小
```

公平惩罚自动变弱。

因此，虽然 `B` 是固定的，`lambda_t` 仍然是动态的。

### 5.5 Robust centrality

为了抵抗性能攻击，GuardFed-AD2+ 计算每个客户端更新到鲁棒中心的距离：

```text
c_i = - distance(Delta_i, median_update)
```

客户端更新越接近多数客户端的鲁棒中心，`c_i` 越高。

### 5.6 Root alignment

GuardFed-AD2+ 同时计算客户端更新和 clean server update 的方向一致性：

```text
a_i = max(0, cosine(Delta_i, Delta_server))
```

如果客户端更新方向和 clean server update 一致，说明它更可能是有益更新。

### 5.7 最终客户端评分

所有分数组件先做 robust z-score 标准化，然后计算：

```text
score_i =
    1.00 * z(u_i)
  + 0.35 * z(c_i)
  + 0.35 * z(a_i)
  + 0.90 * z(-r_i)
  + 0.25 * lambda_t * z(-v_i)
```

其中：

- `z(u_i)`：clean-root utility，越高越好。
- `z(c_i)`：robust centrality，越高越好。
- `z(a_i)`：root alignment，越高越好。
- `z(-r_i)`：公平风险惩罚，风险越低越好。
- `lambda_t * z(-v_i)`：动态 fairness violation 惩罚。

### 5.8 客户端筛选和聚合

GuardFed-AD2+ 先进行 hard gate，过滤明显异常更新：

```text
distance_i <= median_distance + 2.5 * MAD
or
alignment_i > 0
```

然后保留评分最高的 80% 客户端：

```text
keep_ratio = 0.8
```

最后使用 softmax 生成聚合权重：

```text
w_i = softmax(score_i / T)
```

并把选中客户端更新归一到 clean server update 的 norm：

```text
Delta_i <- Delta_i * ||Delta_server|| / ||Delta_i||
```

最终聚合：

```text
Delta_global = sum_i w_i * normalized(Delta_i)
```

## 6. 最终采用的参数

| 参数 | 数值 | 含义 |
| --- | ---: | --- |
| `server_ratio` | 0.10 | 使用 10% clean server/root data |
| `synthetic_ratio` | 0 | 不使用 synthetic data |
| `act_fairness_metric` | `aeod_aspd` | 同时优化 AEOD 和 ASPD |
| `act_fairness_budget` | 0.06 | clean-root fairness risk 预算 |
| `act_risk_weight` | 0.90 | 公平风险惩罚权重 |
| `act_violation_weight` | 0.25 | 超预算惩罚权重 |
| `act_keep_ratio` | 0.80 | 每轮保留 80% 客户端 |
| `act_temperature` | 0.35 | softmax 权重温度 |
| `ad2_utility_weight` | 1.00 | clean-root accuracy 权重 |
| `ad2_centrality_weight` | 0.35 | 鲁棒中心性权重 |
| `ad2_alignment_weight` | 0.35 | server update 对齐权重 |
| `ad2_norm_mode` | `root` | 使用 clean server update norm 归一化 |
| `ad2_score_clip` | 0 | 不裁剪 z-score，保留分数区分度 |

## 7. 主结果总结

主表比较范围为：FedAvg、FairFed、Median、FLTrust、FairGuard、FLTrust+FairGuard、GuardFed，以及新增 baseline Fed-NGA、Huber-BRFL、LoGoFair、AdaAggRL、FedAMM、FedAA，最后加入 GuardFed-AD2+。

排名规则：

- ACC 越高越好。
- AEOD/ASPD 越低越好。
- 公平性排名时，排除 ACC 比该场景最优 ACC 低超过 5 个百分点的方法。
- 对于 ACC 明显偏低且 AEOD/ASPD 接近 0 的“未充分训练”结果，不计为公平性最好。

| 方法 | Fair First/Second | Fair First | Avg ACC | Avg AEOD | Avg ASPD |
| --- | ---: | ---: | ---: | ---: | ---: |
| GuardFed-AD2 | 13/16 | 9/16 | 73.12% | 0.075 | 0.050 |
| GuardFed-AD2+ | **15/16** | 7/16 | **73.15%** | **0.075** | 0.059 |

解释：

- GuardFed-AD2+ 的优势主要体现在稳定性：16 个公平性指标中有 15 个达到第一或第二。
- GuardFed-AD2 的第一名数量更多，但部分场景稳定性略弱。
- GuardFed-AD2+ 更适合作为论文主方法，因为它在不同数据集、不同分布和不同双重攻击下表现更稳。

## 8. 论文表述建议

可以在论文中这样描述：

> GuardFed-AD2+ introduces an adaptive dual-objective aggregation mechanism that jointly evaluates clean-root utility, fairness risk, robust centrality, and server-update alignment for each client update. A round-adaptive dual multiplier automatically increases the penalty on fairness-violating updates when the current communication round exhibits higher clean-root fairness risk. This enables GuardFed-AD2+ to suppress updates that are simultaneously attack-suspicious and fairness-harmful while preserving clean-root utility.

中文解释：

> GuardFed-AD2+ 在每轮训练中动态评估每个客户端更新对干净 server 数据的性能、公平性、鲁棒中心性和方向一致性影响。当某一轮整体公平风险升高时，算法会自动增大公平性惩罚，从而更强地抑制可能造成公平性攻击的恶意更新。相比固定规则筛选，AD2+ 的客户端选择和聚合权重会随训练状态和攻击强度动态变化，因此更适合防御同时影响性能和公平性的双重攻击。

## 9. 文件位置

- 详细 Markdown 结果表：`E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/GuardFed_AD2plus_final_paper_tables.md`
- Excel 结果表：`E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/GuardFed_AD2plus_final_paper_tables.xlsx`
- 本汇报文件：`E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/GuardFed_AD2plus_advisor_summary.md`
