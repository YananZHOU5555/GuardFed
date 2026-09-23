# GuardFed-AD2+ 完整超参数、攻击空间与三计划实验协议

## 1. 给导师的核心结论

当前实验需要区分三类参数：

1. **固定训练协议**：数据集、客户端数量、训练轮数、学习率、batch size、root/server 数据比例等。同一组实验内统一固定，并没有对所有训练参数做笛卡尔积网格。
2. **实际执行的攻击校准 grid**：F Flip 搜索 5 个候选模式；FedSA 搜索 4 个 gain 与 3 个 norm_ratio，共 12 组组合。
3. **AD2+ 内部自适应候选机制**：AD2+ 每轮生成 10 个候选评分视角，用 clean root/server 数据做 one-step 评估，在 accuracy floor 约束下选择一个候选。它不是从外部 baseline 中选择最优，也不是事后根据测试集结果挑选。

准确的论文表述是：**AD2+ 使用固定的基础训练协议，并在每轮通过 clean root/server 数据对预定义的内部候选族进行自适应选择；攻击强度则通过独立 calibration grid 统一确定。**

## 2. 最新 attack-strength 三计划的固定训练协议

| 参数 | 当前采用值 | 允许/声明范围 | 作用 |
|---|---:|---|---|
| 数据集 | Adult、COMPAS | 固定 | 两个公平联邦学习数据集 |
| 客户端数 | 20 | 正整数 | 每轮客户端总数 |
| 主实验恶意客户端数 | 4 | 0..20 | 主实验为 20% 恶意客户端 |
| 恶意比例实验 | 2、4、6、8、10 | 10%..50% | 对应 10%、20%、30%、40%、50% |
| IID Dirichlet alpha | 5000 | 大于 0 | 近似 IID 客户端划分 |
| non-IID Dirichlet alpha | 5 | 大于 0 | 非 IID 客户端划分 |
| 训练轮数 | 70 | 正整数 | 最终指标评估轮 |
| local epochs | 1 | 正整数 | 每个客户端每轮本地训练次数 |
| batch size | 256 | 正整数 | 本地 mini-batch 大小 |
| learning rate | 0.005 | 大于 0 | 本地 Adam 学习率 |
| optimizer | Adam | Adam 或 SGD | 当前核心 runner 使用 Adam |
| clean root/server ratio | 10% | 0..1 | 最新攻击强度和三计划实验使用 |
| synthetic root ratio | 0% | 0..1 | 最新三计划不使用 synthetic root |
| synthetic method | none | none、gaussian_copula、bootstrap、noisy_bootstrap、smote、pca、ctgan 等 | 三计划不生成 synthetic root |
| synthetic epochs | 50 | 正整数 | 仅 synthetic generator 开启时使用 |
| server sampling | stratified_sensitive | 离散选项 | 按敏感组分层抽取 clean root/server data |
| server alpha | `None` | optional float | 三计划不额外改变 server Dirichlet alpha |
| server target sensitive | `None` | optional target | 三计划不强制指定 server sensitive proportion |
| server target label | `None` | optional target | 三计划不强制指定 server label proportion |
| model input | 不包含 label 和 sensitive feature | 布尔开关 | 所有方法统一，避免算法间输入不一致 |
| aggregation weighting | count | count/equal | 按客户端样本数加权 |
| seeds | 123, 456, 789, 1001, 2024, 3141, 4242, 5050, 6060, 7070 | 整数列表 | 正式结果使用 10 个 seeds |

### 关于 5% 与 10% root/server data

工作区同时保留两条实验轨迹：

- 旧 paper-table runner 默认 server_ratio=0.05，对应早期 Table II/III 复现实验；
- 最新 attack-strength 三计划采用 server_ratio=0.10、synthetic_ratio=0.0。

这两条协议不能混写在同一张结果表中。新的实验表应在标题或表注中明确写出 10% clean root/server。

## 2.1 三个扩展计划分别使用的设置

下面这张表是三项扩展计划的实验单元定义。三项计划共享本节第 2 节的训练、数据和 seed 协议；差异只在方法范围、攻击类型和恶意比例设置。

| 计划 | 目的 | 方法范围 | 攻击/对照 | 恶意客户端 | 数据集/分布 | seed 汇报 |
|---|---|---|---|---:|---|---|
| Plan 01 | 强化 F Flip 公平性攻击 | 当前完整方法列表，包含 GuardFed-AD2+ | Benign + F Flip，统一 `fflip_mode=all_unprivileged` | 主实验 4/20 | Adult/COMPAS × IID/non-IID | 10 seeds，保存 raw seed，汇总 mean |
| Plan 02 | 强化 FedSA 性能攻击 | 当前完整方法列表，包含 GuardFed-AD2+ | Benign + FedSA，统一 `gain=4.5`、`norm_ratio=3.0` | 主实验 4/20 | Adult/COMPAS × IID/non-IID | 10 seeds，保存 raw seed，汇总 mean |
| Plan 03 | AD2+ 恶意比例敏感性 | 仅 GuardFed-AD2+ | S-DFA 与 Sp-DFA | 2/4/6/8/10，即 10%--50% | Adult/COMPAS × IID/non-IID | 每个比例 10 seeds，汇总 mean |

Plan 01 和 Plan 02 的当前完整方法列表包含 22 个方法：FedAvg、FairFed、Median、FLTrust、FairGuard、FLTrust+FairGuard、GuardFed、FLGMM、FLAURA、LayerGuard、SmartFL、FLTG、FedDNA、LASA、Fed-NGA、Huber-BRFL、LoGoFair、AdaAggRL、FedAMM、FedAA、GuardFed-AD2 和 GuardFed-AD2+。Class-B FL 与 GuardFed-ACT 不纳入统计。

Plan 03 的比例序列固定为：

```text
malicious_ratio = {0.10, 0.20, 0.30, 0.40, 0.50}
malicious_clients = {2, 4, 6, 8, 10}
```

三个计划的结果表应描述为相同协议下的 10-seed mean。每个 seed 的 ACC、AEOD、ASPD、攻击审计和逐轮轨迹均保留在明细文件中；导师版超参数说明不需要展开某个指标使用哪一轮或哪个 seed。

## 2.2 模型、随机性和本地优化细节

5090 runner 实际使用的模型是一个二分类 SimpleMLP：

```text
input dimension = dataset feature count
Linear(input_dim, 16)
ReLU
Linear(16, 2)
```

其他实现级设置如下：

| 参数 | 实际 setting | 说明 |
|---|---:|---|
| model | SimpleMLP | 所有方法使用同一模型结构 |
| hidden dimension | 16 | 单个隐藏层 |
| output dimension | 2 | 二分类 |
| loss | weighted cross-entropy | 使用客户端重加权时应用 sample weights |
| client optimizer | Adam | learning rate=0.005 |
| server/root optimizer | Adam | learning rate=0.005 |
| local epochs | 1 | 每个客户端每轮一次本地 epoch |
| seed control | Python/NumPy/PyTorch/CUDA | 每个 seed 统一设置 |
| cuDNN benchmark | `true` | 5090 runner 实际开启 |
| DataLoader shuffle | `true` | 客户端和 server/root 本地训练均使用 |

## 2.3 三计划之外的完整攻击场景

runner 的攻击枚举包括：

```text
Benign, F Flip, FOE, S-DFA, Sp-DFA, FedSA
```

其中 Plan 01 重点汇总 F Flip，Plan 02 重点汇总 FedSA，Plan 03 汇总 S-DFA/Sp-DFA 的恶意比例变化；FOE 作为原始性能攻击实现和 DFA 性能侧实现保留在 raw result 与 audit 中。

## 3. 实际运行过的外部 grid

### 3.1 F Flip 校准 grid：5 个候选模式

实际候选为：

```text
invert
label_conditioned
label_conditioned_reverse
all_privileged
all_unprivileged
```

校准 seed 为 314159。结果文件记录的统一选择是：

```text
fflip_mode = all_unprivileged
```

选择规则是用统一 FedAvg calibration 批次比较公平性影响，并排除无效或退化运行。正式主表使用同一 F Flip 配置，不为不同 baseline 单独改攻击参数。

### 3.2 FedSA 校准 grid：12 个强度组合

两个维度为：

```text
gain       = {1.75, 2.5, 3.5, 4.5}
norm_ratio = {2.0, 3.0, 4.0}
```

因此实际运行了 4 x 3 = 12 组组合。最终选择为：

```text
fedsa_gain       = 4.5
fedsa_norm_ratio = 3.0
```

攻击校准文件还保留了每组候选的 calibration score，可以审计为什么选中这一组。

### 3.3 FOE / 性能侧攻击参数

5090 代码保留原 Git 实现的 `attack_acc_0.5` 语义，核心缩放常数为：

```text
FOE_SCALE = -0.5
```

FOE 的实现模式为离散候选：

| `foe_mode` | 操作 | 使用情况 |
|---|---|---|
| `state` | 将恶意客户端 local state 的参数乘以 -0.5 后再形成 update | 默认 FOE mode；当前三计划的基础配置 |
| `delta` | 将 local update delta 乘以 -0.5 | runner 支持的替代实现 |
| `zero` | 将恶意 update 置为零 | runner 支持的替代实现 |
| `fedsa` | 使用 FedSA 的方向偏移和范数约束 | FedSA、S-DFA、Sp-DFA 的性能侧实现 |

当前三计划 runner 的明确配置为：

```text
foe_mode       = state
sdfa_foe_mode  = fedsa
spdfa_foe_mode = fedsa
```

因此，S-DFA 和 Sp-DFA 中的性能侧不是另设一个未记录的攻击强度，而是调用统一的 FedSA 参数：`gain=4.5`、`norm_ratio=3.0`。

### 3.4 F Flip 的数据修改范围

F Flip 的候选模式是离散集合：

```text
{invert, label_conditioned, label_conditioned_reverse,
 all_privileged, all_unprivileged}
```

所有 F Flip 模式都满足：

- 只修改恶意客户端的敏感属性；
- 不修改真实标签；
- audit 记录敏感属性实际变化比例；
- audit 记录 `label_changed_count`，正式配置要求为 0；
- 正式三计划统一使用 `all_unprivileged`。

### 3.5 FedSA 的实际更新公式参数

对恶意客户端原始 update `g`，代码使用 clean server update 方向进行性能偏移：

```text
post_vec = pre_vec - gain × ||pre_vec|| × clean_direction
```

随后使用：

```text
max_norm = max(||pre_vec||, norm_ratio × ||pre_vec||)
```

对超过上限的 update 做范数缩放。所有方法共享同一组 `gain` 和 `norm_ratio`，攻击审计记录 pre/post norm ratio 与 cosine 变化。

## 4. Baseline-specific settings

除各方法自身的聚合规则外，所有 baseline 共用同一个数据预处理、模型、optimizer、客户端划分、攻击配置和指标实现。代码中记录的 baseline 相关参数为：

| 方法/组件 | 参数 | 实际 setting |
|---|---|---:|
| FedAvg | aggregation weighting | `count` |
| FairFed | `fairfed_beta` | 1.0 |
| FairFed | `use_reweighting` | `true` |
| FLTrust | `trust_threshold` | 0.2 |
| FairGuard | `fairguard_mode` | `server_aeod` |
| GuardFed | `guardfed_fairness_lambda` | 20.0 |
| GuardFed | trust selection threshold | 0.2 |
| Median | coordinate aggregation | coordinate-wise median |
| AD2/AD2+ | root calibration | enabled |

新增的 FLGMM、FLAURA、LayerGuard、SmartFL、FLTG、FedDNA、LASA、Fed-NGA、Huber-BRFL、LoGoFair、AdaAggRL、FedAMM 和 FedAA 使用仓库中的对应 core reproduction implementation；它们不改变统一训练协议。

## 5. AD2+ 的实际参数与动态机制

### 5.1 动态公平风险和 dual multiplier

对客户端 i，当前轮从 clean root/server 数据得到公平风险 r_i。公平预算为：

$$B=0.06.$$

公平违反项为：

$$v_i=\max(0,r_i-B).$$

当前轮平均风险为 r_bar，temperature 为 0.35，base weight 为 1.0 时：

$$\lambda_t=1.0\cdot\operatorname{softplus}\left((\bar r-B)/0.35\right).$$

所以 0.06 是固定的公平风险预算，不是每轮自动变化的超参数；但 r_i、v_i、r_bar、lambda_t 都会随当前模型和当前客户端更新变化。因此 AD2+ 的 adaptive 体现在：公平惩罚、客户端评分、保留集合和聚合权重都会逐轮变化。

### 5.2 AD2 基础参数

| 参数 | 基础值 | AD2+ 内部候选范围 |
|---|---:|---:|
| act_fairness_budget | 0.06 | 固定公平预算 |
| act_fairness_metric | aeod_aspd | aeod、aspd、aeod_aspd、max |
| act_temperature | 0.35 | 0.20 或 0.35 |
| act_keep_ratio | 0.80 | 0.70、0.80、0.90 |
| act_risk_weight | 1.00 | 0.60、0.75、0.80、0.85、0.90、1.10、1.30 |
| act_violation_weight | 1.00 | 0.10、0.20、0.25、0.35、0.50 |
| ad2_utility_weight | 1.00 | 0.60、0.70、0.80、1.00、1.20 |
| ad2_centrality_weight | 0.35 | 0.20、0.25、0.35、0.50 |
| ad2_alignment_weight | 0.35 | 0.20、0.25、0.35、0.50 |
| ad2_norm_mode | adaptive | adaptive；AD2+ candidate 内部使用 root |
| ad2_score_clip | 5.0 | AD2+ candidate 内部为 0 |
| ad2_norm_clip_scale | 2.5 | 固定正数 |

AD2+ 顶层 calibration 参数也固定记录如下：

| 参数 | 实际 setting | 作用 |
|---|---:|---|
| `ad2_calibration_enabled` | `true` | 开启 clean-root calibration |
| `ad2_calibration_base_weight` | 1.0 | calibration 的基础权重 |
| `ad2_calibration_budget` | 0.06 | calibration fairness budget |
| `ad2_calibration_temperature` | 0.03 | threshold calibration 温度 |
| `ad2_calibration_quantiles` | 41 | 每个敏感组的 quantile 候选数 |
| `ad2_calibration_max_acc_drop` | 0.03 | 配置层面的最大 ACC drop |
| `ad2_calibration_objective` | `acc_floor` | 普通 calibration 目标 |
| `ad2_plus_mode` | `adaptive` | 启用 AD2+ 内部 candidate selector |

### 5.3 AD2+ 的 10 个内部候选

| 候选 | fairness metric | risk | violation | keep | temp | utility | centrality | alignment |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| balanced | aeod_aspd | 0.75 | 0.20 | 0.80 | 0.35 | 1.00 | 0.35 | 0.35 |
| balanced_open | aeod_aspd | 0.75 | 0.20 | 0.90 | 0.35 | 1.00 | 0.35 | 0.35 |
| fair_stable | aeod_aspd | 0.90 | 0.25 | 0.80 | 0.35 | 1.00 | 0.35 | 0.35 |
| utility_fair | aeod_aspd | 0.60 | 0.10 | 0.80 | 0.35 | 1.20 | 0.50 | 0.50 |
| dual_strict | aeod_aspd | 1.10 | 0.35 | 0.80 | 0.35 | 0.70 | 0.25 | 0.25 |
| dual_sharp | aeod_aspd | 1.30 | 0.50 | 0.70 | 0.20 | 0.60 | 0.20 | 0.20 |
| aeod_focus | aeod | 0.85 | 0.25 | 0.80 | 0.35 | 0.80 | 0.35 | 0.35 |
| aspd_focus | aspd | 0.85 | 0.25 | 0.80 | 0.35 | 0.80 | 0.35 | 0.35 |
| aspd_strict | aspd | 1.10 | 0.35 | 0.70 | 0.20 | 0.70 | 0.25 | 0.25 |
| max_guard | max | 0.85 | 0.25 | 0.80 | 0.35 | 0.80 | 0.35 | 0.35 |

这 10 行不是 10 个 baseline，也不是事后挑最优结果。每轮都对候选更新做一次 clean-root/server one-step evaluation，在 accuracy floor 约束下选择一个候选。

候选公平损失为：

$$L_{fair}=0.45\,AEOD+0.45\,ASPD+0.10\max(AEOD,ASPD).$$

候选选择分数为：

$$S=ACC-0.35L_{fair}-6.00\max(0,ACC_{floor}-ACC)-0.10\max(0,\max(AEOD,ASPD)-B).$$

其中，只有满足 accuracy floor 的候选优先参与选择；如果没有候选满足，才从全部候选中选择最高分。测试集不参与训练过程中的候选选择。

## 6. Group threshold calibration 的 grid

AD2+ 还对两个敏感组分别构造 threshold candidates：

```text
quantiles = 41
quantile interval = 0.02 ... 0.98
additional candidates = 0, min(margin)-1e-6, max(margin)+1e-6
```

因此每个敏感组最多产生 44 个候选值，去重后进行两组阈值组合搜索；两个组的组合规模最多约为 44 x 44。该搜索使用 clean root/server labels 和 margins，不使用测试集标签。

默认 calibration objective 是 acc_floor：

- base accuracy 为 root/server 上未调整阈值的 accuracy；
- ad2_calibration_max_acc_drop=0.03 给出最多 3 个百分点的基础上限；
- AD2+ 每轮进一步把严格候选 floor 收紧到当前候选最高 ACC 下最多 0.5 个百分点的差距；
- fairness risk 超过 ad2_calibration_budget=0.06 时增加 violation 惩罚。

## 7. 哪些参数没有做外部 grid search

以下参数在当前完整结果中是固定值，而不是已经完成全范围搜索的结果：

- learning_rate=0.005；
- rounds=70；
- local_epochs=1；
- batch_size=256；
- server_ratio=10%；
- act_fairness_budget=0.06；
- ad2_norm_clip_scale=2.5；
- ad2_calibration_max_acc_drop=0.03；
- 10 个正式随机种子集合。

这些值有明确的实现范围，但不能写成“我们搜索过所有可能值并选出最优”。如果导师要求完整的 AD2+ 超参数敏感性实验，应另外运行一个标注清楚的新实验，例如：

```text
budget ∈ {0.03, 0.06, 0.10}
temperature ∈ {0.20, 0.35, 0.50}
keep_ratio ∈ {0.70, 0.80, 0.90}
centrality/alignment weight ∈ {0.20, 0.35, 0.50}
norm_clip_scale ∈ {1.5, 2.5, 3.5}
```

这会形成一个新的 hyperparameter sensitivity 实验，而不是回溯性地把当前固定配置说成已搜索。

## 8. 建议放入论文 appendix 的英文表述

> We use a fixed training protocol with 20 clients, one local epoch, batch size 256, learning rate 0.005, 70 communication rounds, and 10% clean root/server data. The non-IID partition uses Dirichlet alpha 5, while IID uses alpha 5000. Attack strengths are calibrated independently using a five-mode F-Flip grid and a 12-point FedSA grid. GuardFed-AD2+ does not select among external methods. Instead, at each communication round it evaluates ten predefined internal scoring candidates on the clean root set, applies an adaptive accuracy floor and fairness-risk penalty, and selects the candidate update with the highest constrained clean-root score. All fixed values and candidate ranges are reported in the supplementary hyperparameter audit.

## 9. 可核查文件

5090 上的实际仓库路径为 `/home/yannan/workspace/GuardFed`。以下是 5090 上直接核对的源文件；Windows workspace 中的同名副本只作为整理和导出材料：

- `/home/yannan/workspace/GuardFed/scripts/run_attack_strength_study.py`
- `/home/yannan/workspace/GuardFed/scripts/reproduce_paper_tables.py`
- `/home/yannan/workspace/GuardFed/results/attack_strength/attack_config.json`
- `/home/yannan/workspace/GuardFed/results/attack_strength/raw_results.jsonl`

- `E:\OneDrive\文档\GuardFed\.codex_remote\reproduce_paper_tables.py`
- `E:\OneDrive\文档\GuardFed\.codex_transfer\run_attack_strength_study.py`
- `E:\OneDrive\文档\GuardFed\results\attack_strength\attack_config.json`
- `E:\OneDrive\文档\GuardFed\results\attack_strength\raw_results.jsonl`
- `E:\OneDrive\文档\GuardFed\results\attack_strength\seed_results.csv`
- `E:\OneDrive\文档\GuardFed\results\attack_strength\audit.csv`
