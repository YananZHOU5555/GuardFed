# GuardFed / AD2+ 超参数范围、设置与结果汇报口径

> 5090 三项扩展计划的完整审计版请以 [GuardFed_AD2plus_Hyperparameter_Audit.md](../outputs/hyperparameter_audit/GuardFed_AD2plus_Hyperparameter_Audit.md) 及其 Excel 为准。本文件负责把论文口径、实际三计划配置、离线 grid 和 seed-average 汇报方式整理到一起。

## 1. 文档目的

本文档面向论文实验设置部分，说明三件事：

1. 论文正文中给出的超参数范围、离散设置和实验协议；
2. GuardFed-AD2+ 实现中使用的固定参数和内部候选配置；
3. 最终表格如何由多次随机实验得到。

所有最终结果均应表述为：在固定实验协议下运行多个随机 seed，保存每个 seed 的完整结果，并对这些 seed 的结果取 arithmetic mean。结果表不是通过挑选某一个 seed、某一个指标或某一轮的极端结果构造的。

## 2. 论文正文明确给出的实验设置

附件论文的实验部分明确给出了以下范围和取值。

| 参数 | 论文中的范围/候选集合 | 论文实验设置 |
|---|---|---|
| 数据集 | Adult、COMPAS | Adult、COMPAS |
| 客户端池 | 100 个客户端 | 每轮选择 20 个客户端 |
| 每轮参与客户端数 | 离散值 20 | 20 |
| 恶意客户端比例 | 主 benchmark 为 20%；比例实验为 10%--50% | 主 benchmark 20%；比例实验 10%、20%、30%、40%、50% |
| 客户端异质性 | Dirichlet concentration 的离散候选 | IID: `alpha=5000`；non-IID: `alpha=5` |
| 学习率 | 固定值 | `0.005` |
| 通信轮数 | `50--100` | 根据实验设置使用范围内的轮数 |
| batch size | `16--256` | 根据实验设置使用范围内的 batch size |
| root data | 训练数据的 5% | `5%`，论文主实验协议 |
| synthetic root data | 合成数据实验中将 root data 扩充至总训练数据的 10% | 另行比较 CTGAN、FD、GC、PCA/PPCA、SMOTE、TVAE 等方法 |
| 随机重复 | 多个随机 seed | 10 个随机 seed，报告 mean |

论文还说明：模型使用 multilayer perceptron，所有实验默认重复 10 个随机 seed，并报告均值。论文没有在正文中为每个 α、β、γ、λ_f、λ_v、B 和 τ 印出一个完整的外部 grid 表；这些参数的实现值应与代码和实验日志保持一致，不能事后凭空补充一个未运行过的搜索范围。

## 3. 当前扩展复现实验的实际运行设置

为了生成当前 GuardFed-AD2+ 扩展表及三组补充实验，实际运行记录采用下面这一套固定协议。它是对论文主协议的具体复现实例，其中 root data 使用 10%，通信轮数和 batch size 取论文给出范围内的具体值。论文主结果仍应按第 2 节的 5% root-data 口径表述。

| 项目 | 实际运行值 |
|---|---:|
| 数据集 | Adult、COMPAS |
| 每轮客户端数 | 20 |
| 主实验恶意客户端数 | 4，即 20% |
| 本地 epoch | 1 |
| 通信轮数 | 70 |
| batch size | 256 |
| learning rate | 0.005 |
| optimizer | Adam |
| device | CUDA GPU |
| IID | Dirichlet `alpha=5000` |
| non-IID | Dirichlet `alpha=5` |
| clean server/root data | 10% |
| synthetic root data | 0% |
| root sampling | `stratified_sensitive` |
| aggregation base weighting | `count`，按客户端样本数加权 |
| sensitive feature | 不输入模型 |
| seed 数量 | 10 |
| seed | 123、456、789、1001、2024、3141、4242、5050、6060、7070 |
| Class-B FL | 不复现 |
| GuardFed-ACT | 不纳入最终比较表 |

### 3.1 5% 与 10% root data 的论文表述

论文正文的默认方法协议是 5% root data；当前 AD2+ 扩展表的实际运行记录使用 10% clean root data。论文中应明确写成：

> The manuscript protocol reserves 5% of the training data as the clean root set. For the extended AD2+ tables reported in this revision, we use a fixed 10% clean root set, with no synthetic root samples, while keeping the same client partition, optimizer, attack definitions, and evaluation protocol.

这样既不修改论文原始协议，也不会把当前扩展表中的 10% 误写成论文所有实验都使用的 5%。

### 3.2 离线范围、离散候选与最终 setting

超参数可以在论文中按“搜索范围/候选集合”和“最终 setting”分开报告。这里的 offline search 指训练前确定候选范围或固定配置；它不使用测试集结果，也不根据最终表格中的某个指标挑选 seed。

| 参数族 | offline 范围或候选 | 当前论文主口径的 setting |
|---|---|---|
| Dirichlet `alpha` | 离散候选 `{5, 5000}` | IID=5000，non-IID=5 |
| communication rounds | 论文范围 `50--100` | 按具体实验记录固定在该范围内 |
| batch size | 论文范围 `16--256` | 具体实验记录固定一个值，不在同一张结果表中混用 |
| learning rate | 论文固定 `0.005` | 0.005 |
| root ratio | 主协议固定 5% | 0.05 |
| malicious ratio | 主 benchmark 20%；敏感性实验 `{10%,20%,30%,40%,50%}` | 主表 20%，比例实验按离散序列报告 |
| random seed | 多个独立 seed | 10 个 seed，最终取 arithmetic mean |
| fairness metric | `AEOD`、`ASPD`，联合风险使用等权平均 | 同时报告 AEOD 和 ASPD |
| synthetic root generator | CTGAN、FD、GC、PCA/PPCA、SMOTE、TVAE | 只在 synthetic-root 补充实验中比较 |

论文正文没有给出 `alpha`、`beta`、`gamma`、`lambda_f`、`lambda_v` 和 `tau` 的外部连续搜索区间。对于这些方法内部参数，应报告“实现中的固定 setting”或 AD2+ 的离散 candidate grid；不要把未实际运行的区间写成已经完成的 grid search。

## 4. 数据与公平性定义

| 数据集 | 标签 | 敏感属性 | 编码 | 模型特征 |
|---|---|---|---|---|
| Adult | `income` | `sex` | Male=1，Female=0 | 不含标签和敏感属性 |
| COMPAS | `two_year_recid` | `race` | African-American=1，Others=0 | 不含标签和敏感属性 |

模型训练时不直接使用敏感属性。敏感属性只用于 root-data 评估、客户端公平风险计算和最终公平性指标报告。

\[
ACC=\frac{\#\{\hat y=y\}}{N}
\]

\[
AEOD=|TPR_{group0}-TPR_{group1}|
\]

\[
ASPD=|P(\hat y=1|group0)-P(\hat y=1|group1)|.
\]

AEOD 和 ASPD 越低表示组间差异越小。若某一敏感组没有正类样本，指标函数应记录 warning，而不是静默产生一个看似很小的公平值。

## 5. GuardFed 方法中的超参数

论文中 GuardFed 的客户端评分由 reward 和 penalty 两部分组成：

\[
R_i=\alpha U_i+\beta C_i+\gamma A_i,
\]

\[
P_i=\lambda_f F_i+\lambda_v V_i,
\]

\[
s_i=R_i-P_i.
\]

其中：

- (U_i)：client update 作用于 root model 后的 root utility；
- (C_i)：update centrality，衡量 update 与 coordinate-wise median 的距离；
- (A_i)：root alignment，衡量 update 与 trusted root update 的 cosine similarity；
- (F_i)：fairness risk，默认是 AEOD 与 ASPD 的等权平均；
- (B)：fairness budget；
- (V_i=\max(0,F_i-B))：超过预算的额外违反项；
- τ：softmax temperature。

最终聚合权重为：

\[
p_i=\frac{\exp(s_i/\tau)}{\sum_j\exp(s_j/\tau)}.
\]

root-norm calibration 为：

\[
\widetilde g= g\cdot\min\left(1,\frac{\|g_r\|_2}{\|g\|_2+\epsilon}\right).
\]

### 5.1 GuardFed 实际实现配置

| 参数 | 实际值 | 说明 |
|---|---:|---|
| `guardfed_fairness_lambda` | 20.0 | 原始 GuardFed fairness penalty 的实现系数 |
| `trust_threshold` | 0.2 | trust score 的选择阈值 |
| `fairguard_mode` | `server_aeod` | FairGuard/GuardFed root fairness 评估模式 |
| root-norm calibration | 开启 | 使用 trusted/root update 控制聚合范数 |
| base aggregation weighting | `count` | 按客户端样本数加权 |

论文正文的符号 α、β、γ、λ_f、λ_v、B、τ 与代码中的具体实现字段应在最终提交版本中保持一一对应。若某个符号只在理论公式中出现、没有单独打印数值，应以实现配置和运行日志为准，不声称存在未执行的 grid 搜索。

## 6. GuardFed-AD2+ 的固定参数

AD2+ 在 AD2 基础上加入 clean-root candidate evaluation。下面的参数是当前运行记录中的顶层配置。

| 参数 | 实际值 | 作用 |
|---|---:|---|
| `act_fairness_metric` | `aeod_aspd` | 顶层公平风险采用 AEOD/ASPD 等权平均 |
| `act_fairness_budget` | 0.06 | 公平风险预算 (B) |
| `act_temperature` | 0.35 | softmax 温度 |
| `act_keep_ratio` | 0.80 | 基础 AD2 默认保留比例 |
| `act_risk_weight` | 1.00 | 基础公平风险项权重 |
| `act_violation_weight` | 1.00 | 基础超预算违反项权重 |
| `ad2_utility_weight` | 1.00 | root utility 权重 |
| `ad2_centrality_weight` | 0.35 | update centrality 权重 |
| `ad2_alignment_weight` | 0.35 | root/server alignment 权重 |
| `ad2_score_clip` | 5.0 | 顶层 robust score clip |
| `ad2_norm_clip_scale` | 2.5 | MAD 范数控制系数 |
| `ad2_norm_mode` | `adaptive` | 基础 AD2 范数模式 |
| `ad2_plus_mode` | `adaptive` | 开启 AD2+ 自适应 candidate evaluation |
| `ad2_calibration_enabled` | `true` | 开启 root calibration |
| `ad2_calibration_budget` | 0.06 | calibration fairness budget |
| `ad2_calibration_temperature` | 0.03 | threshold calibration 温度 |
| `ad2_calibration_quantiles` | 41 | 每个敏感组的 threshold 候选数 |
| `ad2_calibration_max_acc_drop` | 0.03 | 配置层面的 ACC drop 上限 |
| `ad2_calibration_objective` | `acc_floor` | 普通 calibration 的 utility floor 目标 |

## 7. AD2+ 的离散候选池和内部 grid

AD2+ 不从外部 baseline 中选择一个“最好方法”。它在同一个 AD2 加性双目标框架内预先定义 10 个内部候选 configuration。候选配置只使用当前 clean root data 做一步模型评估，随后生成本轮聚合更新。

因此，论文中可以把它说明为一个有限离散 candidate grid，而不是“根据最终测试表挑选方法”。候选集合如下：

| candidate | fairness metric | risk weight | violation weight | keep ratio | temperature | utility weight | centrality | alignment |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `balanced` | `aeod_aspd` | 0.75 | 0.20 | 0.80 | 0.35 | 1.00 | 0.35 | 0.35 |
| `balanced_open` | `aeod_aspd` | 0.75 | 0.20 | 0.90 | 0.35 | 1.00 | 0.35 | 0.35 |
| `fair_stable` | `aeod_aspd` | 0.90 | 0.25 | 0.80 | 0.35 | 1.00 | 0.35 | 0.35 |
| `utility_fair` | `aeod_aspd` | 0.60 | 0.10 | 0.80 | 0.35 | 1.20 | 0.50 | 0.50 |
| `dual_strict` | `aeod_aspd` | 1.10 | 0.35 | 0.80 | 0.35 | 0.70 | 0.25 | 0.25 |
| `dual_sharp` | `aeod_aspd` | 1.30 | 0.50 | 0.70 | 0.20 | 0.60 | 0.20 | 0.20 |
| `aeod_focus` | `aeod` | 0.85 | 0.25 | 0.80 | 0.35 | 0.80 | 0.35 | 0.35 |
| `aspd_focus` | `aspd` | 0.85 | 0.25 | 0.80 | 0.35 | 0.80 | 0.35 | 0.35 |
| `aspd_strict` | `aspd` | 1.10 | 0.35 | 0.70 | 0.20 | 0.70 | 0.25 | 0.25 |
| `max_guard` | `max` | 0.85 | 0.25 | 0.80 | 0.35 | 0.80 | 0.35 | 0.35 |

对于上述 AD2+ candidate，代码固定使用 `ad2_norm_mode=root`、`ad2_score_clip=0` 和 `ad2_norm_clip_scale=2.5`。这三个值属于 candidate 内部实现设置，不应与顶层 AD2 默认值混写。

## 8. AD2+ 自适应机制

对客户端 (i) 的 update Δ_i，root utility、fairness risk、centrality 和 alignment 都在当前轮重新计算：

\[
U_i=ACC_{root}(w_t+\Delta_i),
\]

\[
F_i=\frac{AEOD_i+ASPD_i}{2},
\qquad
V_i=\max(0,F_i-B).
\]

当前候选的评分为：

\[
s_i=\omega_u z(U_i)+\omega_c z(C_i)+\omega_a z(A_i)
 +\omega_r z(-F_i)+\omega_v\lambda_t z(-V_i).
\]

这里的 (z(\cdot)) 是基于当前客户端集合的 robust median/MAD 标准化。公平风险预算 (B=0.06) 是固定目标，但 (F_i)、(V_i)、客户端 score、保留集合、softmax 权重和范数缩放都会随每一轮状态变化，因此 AD2+ 是 aggregation-level adaptive method。

AD2+ 的 10 个 candidate 都在 clean root data 上进行一步评估；候选的 root fairness loss 为：

\[
L_{fair}^{root}=0.45AEOD+0.45ASPD+0.10\max(AEOD,ASPD).
\]

实际实现还设置 root accuracy floor：候选最高 root accuracy 与可接受候选之间最多允许 0.5 个百分点的差距。这个 floor 是 AD2+ 算法内部防止公平值下降到“模型没有有效训练”状态的保护条件，不是最终论文表的 seed 或指标挑选规则。

## 9. 攻击参数的范围、离散候选与最终设置

### 9.1 F Flip

| 项目 | 候选/范围 | 实际设置 |
|---|---|---|
| attack mode | `invert`, `label_conditioned`, `label_conditioned_reverse`, `all_privileged`, `all_unprivileged` | `all_unprivileged` |
| sensitive attribute | 二值属性翻转/重编码 | 仅修改敏感属性 |
| label | 不改变 | 不改变 |
| calibration seed | 固定 | 314159 |

### 9.2 FedSA

| 项目 | 候选 grid | 实际设置 |
|---|---|---:|
| `gain` | `{1.75, 2.5, 3.5, 4.5}` | 4.5 |
| `norm_ratio` | `{2.0, 3.0, 4.0}` | 3.0 |
| attack direction | 对 local/server update 施加性能偏移 | 保留范数约束 |

### 9.3 DFA 变体和恶意比例

| 攻击 | 组成 | 比例设置 |
|---|---|---|
| S-DFA | 同一恶意客户端执行 F Flip + FedSA | 主实验 4/20；比例实验 2/4/6/8/10 |
| Sp-DFA | 公平攻击组与性能攻击组分离 | 主实验 4/20；比例实验 2/4/6/8/10 |

恶意比例敏感性实验的横轴为 10%、20%、30%、40%、50%，即 20 个参与客户端中的 2、4、6、8、10 个恶意客户端。

## 10. Ablation 和 root-data 实验的参数组织

论文消融实验围绕 6 个二元组件：

| 符号 | 组件 |
|---|---|
| U | root utility |
| C | update centrality |
| A | root alignment |
| F | fairness-risk penalty |
| V | fairness-budget violation penalty |
| N | root-norm calibration |

完整模型为 `U+C+A+F+V+N`；消融实验关闭对应组件，其余训练数据划分、seed、optimizer、rounds、batch size 和攻击保持一致。根数据分布实验在不同 root distribution 下重复相同的 benign/FedSA protocol；合成数据实验比较不同生成器，并把 root data 扩充到总训练数据的 10%。

## 11. 结果表的统计汇报口径

论文和导师汇报中统一使用以下表述：

> Unless otherwise specified, every experiment is repeated with 10 independent random seeds. We retain the complete result of each seed and report the arithmetic mean over the 10 runs. The reported table entries are therefore seed-averaged ACC, AEOD, and ASPD values under the same protocol; no individual seed or metric-specific checkpoint is used to manufacture the final comparison.

中文表述：

> 除非另有说明，所有实验均使用 10 个相互独立的随机种子重复运行。我们保留每个 seed 的完整训练结果，并对 10 次运行的最终指标取算术平均，最终表格报告的是 seed-averaged ACC、AEOD 和 ASPD。不同方法使用相同的数据划分、攻击配置、训练轮数和评价协议，不通过挑选某个 seed 或某个指标的极端结果构造表格。

表格展示规则保持简单：ACC 越高越好，AEOD/ASPD 越低越好；若展示公平指标小于 `0.0001`，显示为 `0.0001` 仅是显示下限，原始 JSONL/CSV 中仍保存完整精度。

## 12. 可直接放入论文的 Implementation Details 段落

> We evaluate Adult and COMPAS under IID and non-IID client partitions generated with Dirichlet concentration parameters 5000 and 5, respectively. In each communication round, 20 clients are selected from a pool of 100 clients, and the main benchmark uses a 20% malicious-client ratio. We use a multilayer perceptron with a learning rate of 0.005, communication rounds within 50--100, batch sizes within 16--256, and a clean root set containing 5% of the training data. GuardFed-AD2+ uses a fairness budget of 0.06, a temperature of 0.35, a default keep ratio of 0.80, utility/centrality/alignment weights of 1.00/0.35/0.35, and a robust norm scale of 2.5. Unless otherwise specified, each experiment is repeated with 10 independent seeds and the arithmetic mean is reported.

当前 revision 的 AD2+ 扩展表若使用 10% clean root data、70 rounds 和 batch size 256，应在表题或实验设置脚注中单独标注为 extended protocol，而不替换论文主协议的 5%、50--100 和 16--256 表述。

## 13. 记录文件

- 论文与超参数汇总：[GuardFed_paper_hyperparameter_search_and_reporting.md](GuardFed_paper_hyperparameter_search_and_reporting.md)
- 三组实验索引：[three_plans/README.md](../outputs/attack_strength/three_plans/README.md)
- F Flip 报告：[01_FFlip_FairnessAttack_Last10Selected/report.md](../outputs/attack_strength/three_plans/01_FFlip_FairnessAttack_Last10Selected/report.md)
- FedSA 报告：[02_FedSA_PerformanceAttack_Last10Selected/report.md](../outputs/attack_strength/three_plans/02_FedSA_PerformanceAttack_Last10Selected/report.md)
- AD2+ 恶意比例报告：[03_AD2plus_MaliciousRatio/report.md](../outputs/attack_strength/three_plans/03_AD2plus_MaliciousRatio/report.md)
- 攻击校准配置：[attack_config.json](../results/attack_strength/attack_config.json)
- 原始 seed 运行记录：[raw_results.jsonl](../results/attack_strength/raw_results.jsonl)
