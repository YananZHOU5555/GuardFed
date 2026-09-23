# GuardFed-AD2+ 算法说明

**副标题：** 自适应双目标聚合：同时防御性能攻击与公平性攻击

## 1. 一句话概括

GuardFed-AD2+ 是一个自适应双目标聚合算法。它不直接平均客户端更新，而是在每一轮用少量干净 server/root data 检查每个客户端更新是否同时满足“提升性能、不过度破坏公平性、方向不异常、幅度不过大”这几个条件，然后根据综合得分分配聚合权重。

- 如果一个客户端更新让 clean accuracy 变高，而且 AEOD/ASPD 没有明显变坏，它会获得更高权重。
- 如果一个客户端更新看起来像性能攻击，例如方向和 clean root update 相反，或者离大多数客户端很远，它会被降权。
- 如果一个客户端更新导致敏感群体之间的 TPR 或 positive prediction rate 差距变大，它也会被降权。

## 2. 为什么需要 AD2+

传统鲁棒聚合方法通常重点防御性能攻击，例如模型投毒、反向更新或离群更新。这类方法可以保护 accuracy，但不一定能保护 fairness。另一方面，公平性算法通常关注不同敏感群体之间的预测差异，但面对 FOE 或 DFA 这类性能攻击时，accuracy 可能明显下降。

AD2+ 的目标是同时处理这两类风险：既不能让性能攻击破坏模型可用性，也不能让公平性攻击让某个敏感群体受到系统性伤害。

**例子：** 假设某一轮有两个客户端更新。客户端 A 让 accuracy 从 83% 提到 84%，但 AEOD 从 0.02 变成 0.12；客户端 B 让 accuracy 从 83% 提到 83.7%，AEOD 只从 0.02 变成 0.03。普通 FedAvg 可能更偏向 A，因为它看起来性能更好；AD2+ 会识别 A 的公平性风险，并更倾向于 B。

## 3. 符号定义

| 符号 | 含义 | 解释 |
|---|---|---|
| w^t | 第 t 轮全局模型 | 服务器当前持有的模型参数 |
| Δ_i^t | 客户端 i 上传的更新 | Δ_i^t = w_i^{{t+1}} - w^t |
| D_r | clean server/root data | 服务器保留的一小部分干净数据，含标签和敏感属性 |
| a | 敏感属性 | Adult 中可为 sex，COMPAS 中可为 race |
| Δ_r^t | clean root update | 只用 root data 得到的可信更新方向 |
| U_i^t | utility 分数 | 客户端更新在 root data 上的 clean accuracy |
| R_i^t | fairness risk | AEOD 和 ASPD 组成的公平风险 |
| C_i^t | centrality | 客户端更新是否靠近正常更新中心 |
| A_i^t | alignment | 客户端更新是否和 clean root update 方向一致 |
| s_i^t | AD2+ 总分 | 决定客户端聚合权重的综合得分 |

## 4. Clean utility

临时候选模型：

```text
w_i^t = w^t + Δ_i^t
```

Root utility：

```text
U_i^t = Acc(w_i^t; D_r) = (1 / |D_r|) Σ 1[ŷ_{w_i^t}(x) = y]
```

这里的 U_i^t 不是客户端自己报告的训练准确率，而是服务器用干净 root data 独立评估得到的。

## 5. Fairness risk

AEOD：

```text
AEOD(w; D_r) = | TPR_{a=0}(w) - TPR_{a=1}(w) |
TPR_{a=g}(w) = Pr(ŷ_w = 1 | y = 1, a = g)
```

ASPD：

```text
ASPD(w; D_r) = | Pr(ŷ_w = 1 | a = 0) - Pr(ŷ_w = 1 | a = 1) |
```

当前 AD2+ 同时考虑 AEOD 和 ASPD：

```text
R_i^t = 0.5 · AEOD(w_i^t; D_r) + 0.5 · ASPD(w_i^t; D_r)
```

## 6. 公平预算与 violation

```text
V_i^t = max(0, R_i^t - B)
```

当前表格中的 AD2+ 配置使用 B = 0.12。若 R_i^t=0.08，则 V_i^t=0；若 R_i^t=0.18，则 V_i^t=0.06。

## 7. Robust centrality

```text
Δ_med^t = median{{Δ_1^t, ..., Δ_m^t}}
d_i^t = || Δ_i^t - Δ_med^t ||_2
C_i^t = - d_i^t / (median_j d_j^t + ε)
```

越离群的更新，C_i^t 越低。

## 8. Root alignment

```text
A_i^t = cos(Δ_i^t, Δ_r^t)
      = <Δ_i^t, Δ_r^t> / (||Δ_i^t||_2 ||Δ_r^t||_2 + ε)
```

如果一个更新和 clean root update 方向一致，A_i^t 较高；如果方向相反或近似正交，A_i^t 较低。

## 9. AD2+ score

```text
s_i^t = α U_i^t + β C_i^t + γ A_i^t - λ_r R_i^t - λ_v V_i^t
```

当前表格配置：

| 参数 | 当前配置 | 含义 |
|---|---:|---|
| B | 0.12 | 公平风险预算 |
| α | 3.0 | utility 权重 |
| β | 0.2 | centrality 权重 |
| γ | COMPAS=1.0, Adult=1.5 | root alignment 权重 |
| λ_r | 0.1 | fairness risk 惩罚权重 |
| λ_v | 0.02 | violation 惩罚权重 |
| τ | 0.8 | softmax temperature |
| c | 5.0 | score clipping 范围 |
| norm | root | 聚合更新范数缩放到 clean root update 尺度 |

## 10. Softmax aggregation

```text
s̃_i^t = clip(s_i^t, -c, c)
p_i^t = exp(s̃_i^t / τ) / Σ_{j∈K_t} exp(s̃_j^t / τ)
Δ_AD2+^t = Σ_{i∈K_t} p_i^t Δ_i^t
```

当前配置 keep=1，因此所有客户端都可以进入候选集合，但权重不同。

## 11. Root-norm scaling

```text
Δ̂_AD2+^t = Δ_AD2+^t · min(1, ||Δ_r^t||_2 / (||Δ_AD2+^t||_2 + ε))
w^{t+1} = w^t + Δ̂_AD2+^t
```

如果恶意客户端让聚合更新范数变得远大于 clean root update，这一步会把整体更新缩回到可信尺度。

## 12. 为什么 AD2+ 是 adaptive

AD2+ 的 adaptive 不是指每个超参数都自动调参，而是指每一轮的聚合行为会根据当前训练状态重新计算。

- 每轮重新计算每个客户端的 root utility。
- 每轮重新计算 AEOD/ASPD 和 fairness violation。
- 每轮重新计算客户端更新的 centrality 和 alignment。
- 每轮重新计算 softmax 聚合权重。
- 每轮根据 clean root update 重新进行 norm scaling。

因此，即使 B、α、β、γ、λ_r、λ_v 是固定超参数，AD2+ 的实际客户端权重和全局更新仍然会随攻击强度、数据分布和训练阶段动态变化。

## 13. 论文可用中文表述

GuardFed-AD2+ 是一种面向性能攻击与公平性攻击的自适应双目标聚合机制。与传统鲁棒聚合方法主要依赖更新几何异常检测不同，GuardFed-AD2+ 在服务器端引入少量干净 root data，对每个客户端更新同时评估其 clean utility 和 fairness risk。具体而言，服务器将每个客户端更新临时作用到当前全局模型上，并在 root data 上计算 accuracy、AEOD 和 ASPD。随后，算法根据公平预算构造 violation term，用于额外惩罚超过可接受公平风险的更新。与此同时，GuardFed-AD2+ 还计算客户端更新相对于本轮更新中心的 centrality，以及其与 clean root update 的 cosine alignment，从而识别方向异常或离群的恶意更新。最终，utility、fairness risk、fairness violation、centrality 和 alignment 被整合为统一的 AD2+ score，并通过 temperature-controlled softmax 转化为聚合权重。聚合后的全局更新进一步通过 clean root update 的范数进行缩放，以限制恶意更新造成的模型漂移。由于上述所有信号均在每一轮根据当前客户端更新和 root-data 反馈重新计算，GuardFed-AD2+ 能够动态调整客户端权重，在保持模型性能的同时抑制公平性退化。

## 14. 边界说明

- Root data 只用于服务器端评估和校准，不等于把敏感属性作为模型输入特征。
- AD2+ 的公平预算 B 是超参数；adaptive 体现在每一轮客户端得分、权重和更新尺度会动态变化。
- 若需要更强的理论表达，可以把 AD2+ score 解释为 utility-robustness-fairness constrained aggregation 的拉格朗日型近似。
