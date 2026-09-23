# GuardFed-AD2+ 完整超参数、攻击空间与三项实验计划

## 1. 实验参数应如何理解

**问题：我们的实验是否对所有超参数进行了大规模笛卡尔积搜索？**

不是。当前实验中的参数分为三类：

1. **固定训练协议**：包括数据集、客户端数量、训练轮数、学习率、批大小、clean root/server 数据比例等。同一组实验中的这些参数保持一致。
2. **攻击强度校准网格**：F Flip 搜索 5 种候选攻击模式；FedSA 搜索 4 个 `gain` 与 3 个 `norm_ratio`，一共 12 种组合。
3. **AD2+ 的内部自适应候选机制**：AD2+ 每轮生成 10 个预定义的内部评分候选，用 clean root/server 数据进行一次评估，并在准确率约束下选择一个候选。它不是从外部基线方法中选择最好的方法，也不是根据测试集结果事后挑选。

因此，论文中可以准确表述为：

> AD2+ 使用统一且固定的基础训练协议；在每一轮通信中，它利用 clean root/server 数据对预定义的内部候选族进行自适应选择；攻击强度则通过独立的校准网格统一确定。

## 2. 最新三项实验计划的固定训练协议

| 参数 | 当前采用值 | 允许或声明范围 | 作用 |
|---|---:|---|---|
| 数据集 | Adult、COMPAS | 固定 | 两个公平联邦学习数据集 |
| 客户端数 | 20 | 正整数 | 每轮参与的客户端总数 |
| 主实验恶意客户端数 | 4 | 0 到 20 | 主实验中为 20% 恶意客户端 |
| 恶意比例实验 | 2、4、6、8、10 | 10% 到 50% | 分别对应 10%、20%、30%、40%、50% |
| IID Dirichlet alpha | 5000 | 大于 0 | 近似独立同分布的客户端划分 |
| non-IID Dirichlet alpha | 5 | 大于 0 | 非独立同分布的客户端划分 |
| 训练轮数 | 70 | 正整数 | 最终指标评估轮数 |
| 本地训练轮数 | 1 | 正整数 | 每个客户端每轮进行一次本地训练 |
| 批大小 | 256 | 正整数 | 本地 mini-batch 的大小 |
| 学习率 | 0.005 | 大于 0 | 当前本地 Adam 优化器的学习率 |
| 优化器 | Adam | Adam 或 SGD | 当前核心 runner 使用 Adam |
| clean root/server 数据比例 | 10% | 0 到 1 | 最新攻击强度实验和三项计划使用 |
| 合成 root 数据比例 | 0% | 0 到 1 | 最新三项计划不使用合成 root 数据 |
| 合成数据方法 | none | none、gaussian_copula、bootstrap、noisy_bootstrap、smote、pca、ctgan 等 | 三项计划不生成合成 root 数据 |
| 合成数据训练轮数 | 50 | 正整数 | 只有启用合成数据生成器时使用 |
| server 抽样方式 | stratified_sensitive | 离散选项 | 按敏感属性分组抽取 clean root/server 数据 |
| server alpha | `None` | 可选浮点数 | 三项计划不额外改变 server 侧的 Dirichlet alpha |
| server 目标敏感属性比例 | `None` | 可选目标值 | 三项计划不强制指定 server 敏感属性比例 |
| server 目标标签比例 | `None` | 可选目标值 | 三项计划不强制指定 server 标签比例 |
| 模型输入 | 不包含标签和敏感属性特征 | 布尔开关 | 所有方法统一，避免不同算法使用不同输入 |
| 聚合权重 | count | count 或 equal | 按客户端样本数量加权 |
| 随机种子 | 123、456、789、1001、2024、3141、4242、5050、6060、7070 | 整数列表 | 正式结果使用 10 个随机种子 |

### 2.1 关于 5% 与 10% 的 clean root/server 数据

工作区中同时保留两条实验轨迹：

- 旧的论文表格复现实验默认使用 `server_ratio=0.05`，即 5% clean server/root data；
- 最新的攻击强度实验和三项扩展计划使用 `server_ratio=0.10`，即 10% clean server/root data，同时 `synthetic_ratio=0.0`。

这两条协议不能混写在同一张结果表中。最新三项实验的标题或表注应明确写出：**10% clean root/server data**。

## 3. 三项扩展计划分别做什么

**问题：三个计划的实验单元如何定义？**

三项计划共享上一节中的训练协议、数据处理方式和 10 个随机种子。差异只在方法范围、攻击类型和恶意客户端比例。

| 计划 | 目的 | 方法范围 | 攻击或对照 | 恶意客户端 | 数据集与分布 | 种子汇报方式 |
|---|---|---|---|---:|---|---|
| 计划 01 | 强化 F Flip 公平性攻击 | 当前完整方法列表，包含 GuardFed-AD2+ | Benign + F Flip，统一使用 `fflip_mode=all_unprivileged` | 主实验 4/20 | Adult、COMPAS × IID、non-IID | 10 个种子，保留原始值并汇总平均值 |
| 计划 02 | 强化 FedSA 性能攻击 | 当前完整方法列表，包含 GuardFed-AD2+ | Benign + FedSA，统一使用 `gain=4.5`、`norm_ratio=3.0` | 主实验 4/20 | Adult、COMPAS × IID、non-IID | 10 个种子，保留原始值并汇总平均值 |
| 计划 03 | AD2+ 恶意比例敏感性 | 仅 GuardFed-AD2+ | S-DFA 与 Sp-DFA | 2、4、6、8、10，即 10% 到 50% | Adult、COMPAS × IID、non-IID | 每个比例使用 10 个种子并汇总平均值 |

计划 01 和计划 02 的完整方法列表包含 22 个方法：

```text
FedAvg、FairFed、Median、FLTrust、FairGuard、FLTrust+FairGuard、
GuardFed、FLGMM、FLAURA、LayerGuard、SmartFL、FLTG、FedDNA、LASA、
Fed-NGA、Huber-BRFL、LoGoFair、AdaAggRL、FedAMM、FedAA、
GuardFed-AD2、GuardFed-AD2+
```

`Class-B FL` 和 `GuardFed-ACT` 不纳入统计。

计划 03 的恶意客户端比例固定为：

```text
malicious_ratio = {0.10, 0.20, 0.30, 0.40, 0.50}
malicious_clients = {2, 4, 6, 8, 10}
```

三个计划的结果表应描述为：**相同实验协议下 10 个随机种子的平均结果**。每个随机种子的 ACC、AEOD、ASPD、攻击审计信息和逐轮训练轨迹均保留在明细文件中。

## 4. 模型、随机性与本地优化

**问题：所有方法是否使用同一个模型和训练设置？**

是。5090 runner 实际使用一个二分类 `SimpleMLP`：

```text
输入维度 = 数据集特征数量
Linear(input_dim, 16)
ReLU
Linear(16, 2)
```

其他实现级设置如下：

| 参数 | 实际设置 | 说明 |
|---|---:|---|
| 模型 | SimpleMLP | 所有方法使用同一模型结构 |
| 隐藏层维度 | 16 | 单个隐藏层 |
| 输出维度 | 2 | 二分类输出 |
| 损失函数 | 加权交叉熵 | 使用客户端重加权时应用样本权重 |
| 客户端优化器 | Adam | 学习率为 0.005 |
| server/root 优化器 | Adam | 学习率为 0.005 |
| 本地训练轮数 | 1 | 每个客户端每轮进行一次本地 epoch |
| 随机性控制 | Python、NumPy、PyTorch、CUDA | 每个随机种子统一设置 |
| cuDNN benchmark | `true` | 5090 runner 实际开启 |
| DataLoader shuffle | `true` | 客户端和 server/root 本地训练均使用 |

## 5. 所有攻击类型

5090 runner 中保留的攻击枚举为：

```text
Benign、F Flip、FOE、S-DFA、Sp-DFA、FedSA
```

计划 01 重点汇总 F Flip；计划 02 重点汇总 FedSA；计划 03 汇总 S-DFA 和 Sp-DFA 在不同恶意比例下的结果。FOE 作为原始性能攻击实现以及 DFA 中的性能侧实现保留在原始结果和攻击审计文件中。

## 6. 实际执行过的攻击校准网格

### 6.1 F Flip：5 个候选模式

实际候选模式为：

```text
invert
label_conditioned
label_conditioned_reverse
all_privileged
all_unprivileged
```

攻击校准使用的随机种子为 `314159`。最终统一选择的配置为：

```text
fflip_mode = all_unprivileged
```

选择过程使用统一的 FedAvg 校准批次比较公平性影响，并排除无效或退化运行。正式主表对所有基线方法使用同一套 F Flip 配置，不为不同基线单独调整攻击参数。

### 6.2 FedSA：12 个强度组合

FedSA 的两个搜索维度为：

```text
gain       = {1.75, 2.5, 3.5, 4.5}
norm_ratio = {2.0, 3.0, 4.0}
```

因此实际校准网格包含：

```text
4 × 3 = 12 组组合
```

最终统一采用：

```text
fedsa_gain       = 4.5
fedsa_norm_ratio = 3.0
```

攻击校准文件保留每组候选的校准分数，可以审计最终为什么使用这一组参数。

### 6.3 FOE 与性能侧攻击参数

5090 代码保留原 Git 实现中的 `attack_acc_0.5` 语义，核心缩放常数为：

```text
FOE_SCALE = -0.5
```

FOE 支持以下离散模式：

| `foe_mode` | 具体操作 | 使用情况 |
|---|---|---|
| `state` | 将恶意客户端的本地模型状态乘以 -0.5，然后形成 update | 默认 FOE 模式，也是三项计划的基础配置 |
| `delta` | 将本地 update 乘以 -0.5 | runner 支持的替代实现 |
| `zero` | 将恶意 update 置为零 | runner 支持的替代实现 |
| `fedsa` | 使用 FedSA 的方向偏移和范数约束 | FedSA、S-DFA、Sp-DFA 的性能侧实现 |

当前三项计划的明确配置是：

```text
foe_mode       = state
sdfa_foe_mode  = fedsa
spdfa_foe_mode = fedsa
```

因此，S-DFA 和 Sp-DFA 中的性能攻击侧使用统一的 FedSA 参数：`gain=4.5`、`norm_ratio=3.0`。

### 6.4 F Flip 实际修改什么

F Flip 的候选集合为：

```text
{invert, label_conditioned, label_conditioned_reverse,
 all_privileged, all_unprivileged}
```

所有 F Flip 模式均满足以下约束：

- 只修改恶意客户端的敏感属性；
- 不修改真实标签；
- 审计文件记录敏感属性实际变化比例；
- 审计文件记录 `label_changed_count`，正式配置要求为 0；
- 正式三项计划统一使用 `all_unprivileged`。

### 6.5 FedSA 的实际更新公式

对恶意客户端的原始更新向量 `g`，代码沿着 clean server update 的方向进行性能偏移。核心形式为：

```text
post_vec = pre_vec - gain × ||pre_vec|| × clean_direction
```

随后设置范数上限：

```text
max_norm = max(||pre_vec||, norm_ratio × ||pre_vec||)
```

超过上限的 update 会被缩放回范数范围内。所有方法共享同一组 `gain` 和 `norm_ratio`。攻击审计记录攻击前后范数比例以及余弦方向变化。

## 7. 各基线方法的实际设置

除方法自身的聚合规则外，所有 baseline 共用同一套数据预处理、模型、优化器、客户端划分、攻击配置和指标实现。

| 方法或组件 | 参数 | 实际设置 |
|---|---|---:|
| FedAvg | 聚合权重 | `count` |
| FairFed | `fairfed_beta` | 1.0 |
| FairFed | `use_reweighting` | `true` |
| FLTrust | `trust_threshold` | 0.2 |
| FairGuard | `fairguard_mode` | `server_aeod` |
| GuardFed | `guardfed_fairness_lambda` | 20.0 |
| GuardFed | trust selection threshold | 0.2 |
| Median | 聚合方式 | 按坐标取中位数 |
| AD2/AD2+ | root calibration | enabled |

新增的 FLGMM、FLAURA、LayerGuard、SmartFL、FLTG、FedDNA、LASA、Fed-NGA、Huber-BRFL、LoGoFair、AdaAggRL、FedAMM 和 FedAA 使用仓库中对应的复现实实现，不改变统一训练协议。

## 8. AD2+ 的实际参数与动态机制

### 8.1 公平风险、违反项与动态 multiplier

**问题：`act_fairness_budget=0.06` 是固定值，那么 AD2+ 为什么仍然可以叫 adaptive？**

答案是：0.06 是固定的公平风险预算阈值，但客户端风险、风险违反程度、动态 multiplier、客户端评分、保留集合和聚合权重都根据当前轮的模型和更新实时计算。

对于客户端 `i`，当前轮在 clean root/server 数据上得到公平风险 `r_i`。公平预算为：

$$B=0.06.$$

公平违反项为：

$$v_i=\max(0,r_i-B).$$

当前轮平均风险记为 `r_bar`。当 temperature 为 0.35、基础权重为 1.0 时，动态 multiplier 为：

$$\lambda_t=1.0\cdot\operatorname{softplus}\left((\bar r-B)/0.35\right).$$

因此，0.06 本身不是每轮变化的超参数；但 `r_i`、`v_i`、`r_bar` 和 `lambda_t` 会随着当前模型及每个客户端更新而变化。AD2+ 的自适应性具体体现在：

- 每轮重新计算公平风险；
- 每轮重新计算公平违反项和动态 multiplier；
- 每轮重新计算客户端的 utility、centrality、alignment 与 risk；
- 每轮重新决定保留哪些客户端；
- 每轮重新计算聚合权重；
- 每轮根据 clean root/server update 调整范数缩放。

### 8.2 AD2 基础参数与候选范围

| 参数 | 基础值 | AD2+ 内部候选范围 |
|---|---:|---:|
| `act_fairness_budget` | 0.06 | 固定公平预算 |
| `act_fairness_metric` | `aeod_aspd` | `aeod`、`aspd`、`aeod_aspd`、`max` |
| `act_temperature` | 0.35 | 0.20 或 0.35 |
| `act_keep_ratio` | 0.80 | 0.70、0.80、0.90 |
| `act_risk_weight` | 1.00 | 0.60、0.75、0.80、0.85、0.90、1.10、1.30 |
| `act_violation_weight` | 1.00 | 0.10、0.20、0.25、0.35、0.50 |
| `ad2_utility_weight` | 1.00 | 0.60、0.70、0.80、1.00、1.20 |
| `ad2_centrality_weight` | 0.35 | 0.20、0.25、0.35、0.50 |
| `ad2_alignment_weight` | 0.35 | 0.20、0.25、0.35、0.50 |
| `ad2_norm_mode` | adaptive | `adaptive`；AD2+ 内部候选使用 `root` |
| `ad2_score_clip` | 5.0 | AD2+ 内部候选为 0 |
| `ad2_norm_clip_scale` | 2.5 | 固定正数 |

### 8.3 AD2+ 顶层校准参数

| 参数 | 实际设置 | 作用 |
|---|---:|---|
| `ad2_calibration_enabled` | `true` | 开启 clean-root 校准 |
| `ad2_calibration_base_weight` | 1.0 | 校准的基础权重 |
| `ad2_calibration_budget` | 0.06 | 校准阶段的公平预算 |
| `ad2_calibration_temperature` | 0.03 | 阈值校准温度 |
| `ad2_calibration_quantiles` | 41 | 每个敏感组的分位点候选数量 |
| `ad2_calibration_max_acc_drop` | 0.03 | 配置层面允许的最大准确率下降 |
| `ad2_calibration_objective` | `acc_floor` | 普通校准目标 |
| `ad2_plus_mode` | `adaptive` | 启用 AD2+ 内部候选选择器 |

### 8.4 AD2+ 的 10 个内部候选

**问题：这 10 个候选是不是 10 个 baseline？**

不是。这 10 个候选是 AD2+ 内部的 10 种评分视角，不是 10 个外部算法。每轮都用 clean root/server 数据对这些候选做一次 one-step 评估，并在准确率约束下选择一个候选。

| 候选名称 | 公平性指标 | 风险权重 | 违反项权重 | 保留比例 | 温度 | utility 权重 | centrality 权重 | alignment 权重 |
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

候选公平性损失为：

$$L_{fair}=0.45\,AEOD+0.45\,ASPD+0.10\max(AEOD,ASPD).$$

候选选择分数为：

$$S=ACC-0.35L_{fair}-6.00\max(0,ACC_{floor}-ACC)-0.10\max(0,\max(AEOD,ASPD)-B).$$

满足准确率下限的候选优先参与选择。如果没有候选满足准确率下限，则从所有候选中选择分数最高者。测试集不参与训练过程中的候选选择。

## 9. 敏感属性分组阈值校准网格

AD2+ 还会对两个敏感属性组分别构造阈值候选：

```text
quantiles = 41
quantile interval = 0.02 ... 0.98
additional candidates = 0, min(margin)-1e-6, max(margin)+1e-6
```

因此每个敏感组最多生成 44 个候选阈值。去重后，对两个敏感组的阈值组合进行搜索，组合规模最多约为：

```text
44 × 44
```

该搜索只使用 clean root/server 数据的标签和模型输出边际值，不使用测试集标签。

默认校准目标是 `acc_floor`：

- 基础准确率是在 root/server 数据上使用未调整阈值时的准确率；
- `ad2_calibration_max_acc_drop=0.03` 给出基础层面最多 3 个百分点的准确率下降上限；
- AD2+ 每轮会进一步把严格候选下限收紧到当前候选最高准确率最多低 0.5 个百分点；
- 当公平风险超过 `ad2_calibration_budget=0.06` 时，增加违反项惩罚。

## 10. 当前哪些参数没有做外部网格搜索

以下参数在当前完整结果中是固定值，而不能写成“已经对所有可能值搜索并选择最优”：

- `learning_rate=0.005`；
- `rounds=70`；
- `local_epochs=1`；
- `batch_size=256`；
- `server_ratio=10%`；
- `act_fairness_budget=0.06`；
- `ad2_norm_clip_scale=2.5`；
- `ad2_calibration_max_acc_drop=0.03`；
- 正式使用的 10 个随机种子集合。

如果后续需要完整的 AD2+ 超参数敏感性实验，应另行运行并明确标注为新的实验，例如：

```text
budget ∈ {0.03, 0.06, 0.10}
temperature ∈ {0.20, 0.35, 0.50}
keep_ratio ∈ {0.70, 0.80, 0.90}
centrality/alignment weight ∈ {0.20, 0.35, 0.50}
norm_clip_scale ∈ {1.5, 2.5, 3.5}
```

这会形成新的超参数敏感性实验，不能回溯性地把当前固定配置描述为已经完成全范围搜索。

## 11. 可直接用于论文的英文原意中文翻译

当前实验采用固定训练协议：20 个客户端、每个客户端 1 个本地 epoch、批大小 256、学习率 0.005、70 轮通信以及 10% clean root/server 数据。non-IID 划分使用 Dirichlet alpha=5，IID 划分使用 alpha=5000。攻击强度通过 5 模式 F Flip 网格和 12 点 FedSA 网格独立校准。GuardFed-AD2+ 不在外部方法之间进行选择，而是在每一轮通信中，利用 clean root 数据评估 10 个预定义的内部评分候选，在自适应准确率下限和公平风险惩罚约束下，选择 clean root 评分最高的候选更新。所有固定值和候选范围均在补充超参数审计文件中报告。

## 12. 可核查文件

5090 上的实际仓库路径为：

```text
/home/yannan/workspace/GuardFed
```

以下文件已经在 5090 上直接核对：

```text
/home/yannan/workspace/GuardFed/scripts/run_attack_strength_study.py
/home/yannan/workspace/GuardFed/scripts/reproduce_paper_tables.py
/home/yannan/workspace/GuardFed/results/attack_strength/attack_config.json
/home/yannan/workspace/GuardFed/results/attack_strength/raw_results.jsonl
```

Windows workspace 中对应的整理文件为：

```text
E:\OneDrive\文档\GuardFed\.codex_remote\reproduce_paper_tables.py
E:\OneDrive\文档\GuardFed\.codex_transfer\run_attack_strength_study.py
E:\OneDrive\文档\GuardFed\results\attack_strength\attack_config.json
E:\OneDrive\文档\GuardFed\results\attack_strength\raw_results.jsonl
E:\OneDrive\文档\GuardFed\results\attack_strength\seed_results.csv
E:\OneDrive\文档\GuardFed\results\attack_strength\audit.csv
```

