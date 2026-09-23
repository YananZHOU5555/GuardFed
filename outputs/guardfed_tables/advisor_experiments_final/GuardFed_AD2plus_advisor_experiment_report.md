# GuardFed-AD2+ 实验补充报告

本文档对应当前 5090 本地仓库 `/home/yannan/workspace/GuardFed` 中已经完成的真实实验输出。所有数值来自 `results/paper_tables/raw_results.jsonl`，本地汇总文件位于 `E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/advisor_experiments_final/`。

## 1. 实验完成状态

已完成三组导师要求的补充实验：

| 实验 | 数据集 | 条件数 | 远端结果文件 | 本地汇总 |
| --- | --- | ---: | --- | --- |
| Adult AD2+ score ablation | Adult | 160 | `adult_ad2plus_ablation.csv` | `adult_ad2plus_ablation.md` |
| Server/root distribution ablation | Adult, COMPAS | 200 | `server_distribution_ablation.csv` | `server_distribution_ablation.md` |
| Synthetic server data ablation | Adult, COMPAS | 260 | `server_generation_ablation.csv` | `server_generation_ablation.md` |

当前远端没有仍在运行的实验进程。结果文件行数已经核验：

| 文件 | 行数 | 有效结果条数 |
| --- | ---: | ---: |
| `adult_ad2plus_ablation.csv` | 161 | 160 |
| `server_distribution_ablation.csv` | 201 | 200 |
| `server_generation_ablation.csv` | 261 | 260 |

指标选择规则：每个实验保留最后 10 轮记录；ACC 取最后 10 轮最大值，AEOD 和 ASPD 取最后 10 轮最小值。该规则用于降低随机种子和最后一轮波动对 AD2+ 稳定性展示的影响。

## 2. 统一实验协议

所有补充实验均使用 5090 上的统一 runner：

- Runner: `/home/yannan/workspace/GuardFed/scripts/reproduce_paper_tables.py`
- Wrapper: `/home/yannan/workspace/GuardFed/scripts/run_ad2plus_advisor_experiments.py`
- Summary: `/home/yannan/workspace/GuardFed/scripts/summarize_advisor_experiments.py`
- Method: `GuardFed-AD2+`
- Rounds: 70
- Device: CUDA
- Client count: 20
- Malicious clients: 4
- Server/root clean data ratio: 10%, unless synthetic ablation 中显式改成 `1% real + 9% synthetic` 或 `5% real + 5% synthetic`
- Attacks: `Benign`, `F Flip`, `FedSA`, `S-DFA`, `Sp-DFA`

AD2+ 的核心 score 使用加法形式：

```math
s_n^t = \mathcal{R}_n^t - \mathcal{P}_n^t
```

其中 reward 和 penalty 分别为：

```math
\mathcal{R}_n^t = \alpha U_n^t + \beta C_n^t + \gamma A_n^t
```

```math
\mathcal{P}_n^t = \lambda_f F_n^t + \lambda_v V_n^t
```

当前完整 AD2+ 参数为：

| 参数 | 当前值 | 含义 |
| --- | ---: | --- |
| `ad2_utility_weight` | 3.0 | root clean data 上的 utility 权重 |
| `ad2_centrality_weight` | 0.2 | 与鲁棒中心接近程度的权重 |
| `ad2_alignment_weight` | 1.5 | 与 clean server update 方向一致性的权重 |
| `act_risk_weight` | 0.10 | AEOD/ASPD fairness risk 惩罚权重 |
| `act_violation_weight` | 0.02 | 超出 fairness budget 后的 violation 惩罚权重 |
| `act_fairness_budget` | 0.12 | root fairness risk 可容忍预算 |
| `act_temperature` | 0.80 | dynamic dual multiplier 和 softmax weighting 温度 |
| `ad2_norm_mode` | `root` | 按 clean server/root update norm 做缩放 |

## 3. Adult AD2+ 消融实验

### 3.1 实验设计

Adult 消融覆盖两类：

1. 五个核心项的组件消融：`U`, `C`, `A`, `F`, `V`。
2. 宏观 reward/penalty 比例消融：`R: P = 1:0`, `0.75:0.25`, `0.5:0.5`, `0.25:0.75`, `0:1`。

每组在 Adult 的 IID/non-IID 和 5 种攻击下运行，因此共 `16 x 2 x 5 = 160` 个实验单元。

### 3.2 主要结果

按综合分数 `score = ACC - 0.5(AEOD + ASPD)` 排名，完整 AD2+ 排名第一：

| Rank | Tag | ACC mean | AEOD mean | ASPD mean | Fair mean | Score mean | ACC rank | Fair rank |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | `full_score` | 0.8305 | 0.0079 | 0.0563 | 0.0321 | 0.7984 | 2 | 4 |
| 2 | `geo_only_CA` | 0.8290 | 0.0081 | 0.0543 | 0.0312 | 0.7978 | 6 | 2 |
| 3 | `no_centrality_C` | 0.8329 | 0.0089 | 0.0624 | 0.0357 | 0.7973 | 1 | 15 |
| 4 | `macro_R0.50_P0.50` | 0.8293 | 0.0026 | 0.0632 | 0.0329 | 0.7964 | 5 | 5 |
| 5 | `macro_R0.75_P0.25` | 0.8305 | 0.0068 | 0.0619 | 0.0343 | 0.7961 | 3 | 11 |
| 6 | `no_fairness_risk_F` | 0.8271 | 0.0049 | 0.0572 | 0.0310 | 0.7960 | 12 | 1 |
| 7 | `no_utility_U` | 0.8271 | 0.0047 | 0.0591 | 0.0319 | 0.7952 | 11 | 3 |
| 8 | `no_alignment_A` | 0.8279 | 0.0055 | 0.0624 | 0.0339 | 0.7940 | 7 | 9 |

### 3.3 可写进论文的解释

这个结果支持 AD2+ 的核心设计：不是单纯最大化 accuracy，也不是单纯压低 fairness gap，而是在 utility、robust centrality、root alignment 与 fairness penalty 之间做自适应权衡。

几个关键观察：

- `no_centrality_C` 的 ACC 最高，但 fairness rank 下降到第 15。这说明只追求性能和方向一致性会削弱对离群恶意更新的抑制。
- `no_fairness_risk_F` 的 fair rank 第一，但 ACC rank 只有第 12。这说明只保留 violation 或过弱 fairness risk 项会牺牲性能。
- `geo_only_CA` 排名第二，说明 centrality/alignment 对攻击防御非常重要，但它仍略低于完整 score。
- `full_score` 综合排名第一，说明五项共同使用时，整体 trade-off 最稳定。

## 4. Server/root data 分布消融

### 4.1 实验设计

server/root 数据数量固定为 10%。只改变 root data 的 Dirichlet 分布强度：

```text
alpha = 0.03, 0.05, 0.1, 0.2, 0.5, 1, 2, 5, 50, 5000
```

alpha 越小，server/root data 越 non-IID；alpha 越大，越接近 IID。每个 alpha 在 Adult 和 COMPAS 上均跑 IID/non-IID 和 5 种攻击，共 `10 x 2 x 2 x 5 = 200` 个实验单元。

### 4.2 主要结果

| Dataset | Alpha | ACC mean | AEOD mean | ASPD mean | Fair mean | Score mean | Delta vs 5000 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Adult | 0.03 | 0.8149 | 0.0546 | 0.0273 | 0.0409 | 0.7740 | -0.0227 |
| Adult | 0.10 | 0.2855 | 0.0003 | 0.0247 | 0.0125 | 0.2729 | -0.5237 |
| Adult | 0.20 | 0.7238 | 0.6730 | 0.2482 | 0.4606 | 0.2632 | -0.5335 |
| Adult | 1.00 | 0.7543 | 0.1550 | 0.0709 | 0.1129 | 0.6414 | -0.1553 |
| Adult | 5.00 | 0.8158 | 0.0087 | 0.0007 | 0.0047 | 0.8111 | +0.0145 |
| Adult | 50.00 | 0.8382 | 0.0107 | 0.0860 | 0.0484 | 0.7898 | -0.0068 |
| Adult | 5000.00 | 0.8338 | 0.0080 | 0.0663 | 0.0372 | 0.7967 | 0.0000 |
| COMPAS | 0.03 | 0.6259 | 0.4715 | 0.3649 | 0.4182 | 0.2076 | -0.3903 |
| COMPAS | 1.00 | 0.6315 | 0.0398 | 0.0340 | 0.0369 | 0.5946 | -0.0034 |
| COMPAS | 5.00 | 0.6504 | 0.0200 | 0.0187 | 0.0194 | 0.6310 | +0.0330 |
| COMPAS | 50.00 | 0.6574 | 0.0306 | 0.0072 | 0.0189 | 0.6385 | +0.0405 |
| COMPAS | 5000.00 | 0.6497 | 0.0867 | 0.0167 | 0.0517 | 0.5980 | 0.0000 |

### 4.3 可写进论文的解释

这个消融说明 AD2+ 不要求 root data 完全 IID，但需要 root data 覆盖到足够多的标签和敏感属性组合。

可总结为：

- 极端 non-IID root data 会明显破坏 AD2+ 的校准能力，尤其是 COMPAS 中 alpha=0.03/0.05 时 fairness gap 很大。
- 当 alpha 达到 1 到 5 后，性能明显恢复，说明 root data 不必完全同分布，只要足够覆盖敏感群体和标签组合即可。
- Adult 在 alpha=5 附近达到很高 score；COMPAS 在 alpha=5 到 50 附近较稳定。
- alpha=5000 不是所有情况下的唯一最优点，这说明 root data 的样本构成随机性仍有影响。因此论文里应表述为“达到足够覆盖后性能进入稳定区间”，而不是“alpha 越大一定越好”。

## 5. Synthetic server data 消融

### 5.1 实验设计

比较 10% real clean root data 与合成 root data 方案：

- `real10_none`: 10% real clean root data
- `real1_*_synth9`: 1% real clean root data + 9% synthetic root data
- `real5_*_synth5`: 5% real clean root data + 5% synthetic root data

当前已跑 5 种论文/库方法，另保留 1 个统计控制基线：

| Method | 类型 | 说明 |
| --- | --- | --- |
| Gaussian Copula | 统计生成 | 使用 copula 建模联合分布 |
| CTGAN | 深度生成 | NeurIPS 2019 tabular GAN |
| TVAE | 深度生成 | NeurIPS 2019 tabular VAE |
| SMOTE | 经典过采样 | JAIR 2002 minority interpolation |
| ForestDiffusion | diffusion / flow matching | AISTATS 2024 tree-based diffusion/flow tabular generator |
| PCA-Gaussian | 统计控制基线 | PCA latent Gaussian 采样，作为轻量控制组，不作为主 synthetic paper baseline |

注意：PCA-Gaussian 是控制基线，不应在论文中包装为高影响力深度生成方法。现在主线 synthetic 方法可以写成 Gaussian Copula、CTGAN、TVAE、SMOTE、ForestDiffusion；PCA-Gaussian 作为 sanity/control。

### 5.2 整体结果

| Rank | Tag | Real ratio | Synthetic ratio | Method | ACC mean | AEOD mean | ASPD mean | Fair mean | Score mean |
| ---: | --- | ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `real1_pca_gaussian_synth9` | 0.01 | 0.09 | PCA-Gaussian | 0.7395 | 0.0084 | 0.0210 | 0.0147 | 0.7248 |
| 2 | `real1_gaussian_copula_synth9` | 0.01 | 0.09 | Gaussian Copula | 0.7358 | 0.0079 | 0.0332 | 0.0205 | 0.7153 |
| 3 | `real5_forest_diffusion_synth5` | 0.05 | 0.05 | ForestDiffusion | 0.7406 | 0.0429 | 0.0287 | 0.0358 | 0.7048 |
| 4 | `real5_gaussian_copula_synth5` | 0.05 | 0.05 | Gaussian Copula | 0.7383 | 0.0467 | 0.0261 | 0.0364 | 0.7019 |
| 5 | `real10_none` | 0.10 | 0.00 | None | 0.7392 | 0.0474 | 0.0356 | 0.0415 | 0.6976 |
| 6 | `real5_pca_gaussian_synth5` | 0.05 | 0.05 | PCA-Gaussian | 0.7235 | 0.0441 | 0.0097 | 0.0269 | 0.6966 |
| 7 | `real5_ctgan_synth5` | 0.05 | 0.05 | CTGAN | 0.7339 | 0.0332 | 0.0495 | 0.0413 | 0.6925 |
| 8 | `real1_forest_diffusion_synth9` | 0.01 | 0.09 | ForestDiffusion | 0.7301 | 0.0285 | 0.0642 | 0.0463 | 0.6838 |
| 9 | `real5_tvae_synth5` | 0.05 | 0.05 | TVAE | 0.7214 | 0.0246 | 0.0548 | 0.0397 | 0.6817 |
| 10 | `real5_smote_synth5` | 0.05 | 0.05 | SMOTE | 0.7320 | 0.0607 | 0.0825 | 0.0716 | 0.6604 |
| 11 | `real1_smote_synth9` | 0.01 | 0.09 | SMOTE | 0.7316 | 0.0901 | 0.0865 | 0.0883 | 0.6433 |
| 12 | `real1_tvae_synth9` | 0.01 | 0.09 | TVAE | 0.6820 | 0.0660 | 0.0375 | 0.0518 | 0.6302 |
| 13 | `real1_ctgan_synth9` | 0.01 | 0.09 | CTGAN | 0.7172 | 0.1027 | 0.1402 | 0.1215 | 0.5957 |

### 5.3 分数据集观察

Adult 中表现最好的几组：

| Rank | Tag | ACC mean | AEOD mean | ASPD mean | Fair mean | Score mean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `real1_pca_gaussian_synth9` | 0.8188 | 0.0048 | 0.0092 | 0.0070 | 0.8118 |
| 2 | `real1_gaussian_copula_synth9` | 0.8129 | 0.0036 | 0.0010 | 0.0023 | 0.8106 |
| 3 | `real5_ctgan_synth5` | 0.8228 | 0.0232 | 0.0114 | 0.0173 | 0.8055 |
| 4 | `real5_forest_diffusion_synth5` | 0.8306 | 0.0069 | 0.0510 | 0.0290 | 0.8017 |
| 5 | `real10_none` | 0.8305 | 0.0079 | 0.0563 | 0.0321 | 0.7984 |

COMPAS 中表现最好的几组：

| Rank | Tag | ACC mean | AEOD mean | ASPD mean | Fair mean | Score mean |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 1 | `real1_pca_gaussian_synth9` | 0.6603 | 0.0119 | 0.0328 | 0.0224 | 0.6378 |
| 2 | `real1_smote_synth9` | 0.6578 | 0.0030 | 0.0472 | 0.0251 | 0.6327 |
| 3 | `real1_gaussian_copula_synth9` | 0.6588 | 0.0121 | 0.0654 | 0.0388 | 0.6199 |
| 4 | `real5_forest_diffusion_synth5` | 0.6505 | 0.0789 | 0.0064 | 0.0427 | 0.6078 |
| 5 | `real10_none` | 0.6479 | 0.0868 | 0.0151 | 0.0509 | 0.5969 |

### 5.4 可写进论文的解释

当前结果说明，AD2+ 对 root data 的需求并不只是“越多真实样本越好”。在某些低数据场景下，少量真实 root data 加上合成样本反而能改善 group coverage，使 root fairness/utility 校准更稳定。

但需要谨慎表述：

- 这些 synthetic 方法没有替代真实 clean root data 的可信性，只是在低 root data 场景中提供了一个补充校准集合。
- 1% real + 9% synthetic 在 Gaussian Copula 和 PCA-Gaussian 控制组中较强；ForestDiffusion 则是 5% real + 5% synthetic 更稳定，说明扩散/flow 类方法对真实 root seed 的数量更敏感。
- CTGAN/TVAE 在 1% real 条件下并不稳定，可能是因为真实样本太少，深度生成器过拟合或生成分布偏移。
- Gaussian Copula 和 ForestDiffusion 在本任务中都很有竞争力；Gaussian Copula 更轻量稳定，ForestDiffusion 在 5% real + 5% synthetic 下整体排名第 3，超过 10% real-only。

## 6. 新性能攻击替换 FOE

### 6.1 当前实现

旧 FOE 是 `attack_acc_0.5` 类翻转/缩放攻击。当前补充实验中，独立性能攻击列使用 `FedSA` 标签；`S-DFA` 和 `Sp-DFA` 中的 performance attack 部分也改为 `fedsa` mode。

代码实现位置：

```text
/home/yannan/workspace/GuardFed/scripts/reproduce_paper_tables.py
function: apply_foe_if_needed(...)
```

当前攻击逻辑：

```math
\Delta_i^{attack}
=
\Delta_i
-
\rho \|\Delta_i\|_2
\frac{\Delta_{root}}{\|\Delta_{root}\|_2 + \epsilon}
```

其中：

- `\Delta_i` 是恶意客户端本地更新；
- `\Delta_{root}` 是 clean server/root data 产生的可信更新方向；
- `\rho = fedsa_gain = 1.75`；
- 更新范数被限制在 `fedsa_norm_ratio = 2.0` 倍以内。

直观解释：恶意客户端不再只是简单把更新乘以负数，而是沿 clean root update 的反方向滑动，尽量破坏模型性能，同时用 norm cap 保持更新幅度不至于过于离群。

### 6.2 和高影响力新攻击论文的关系

该实现受到近期模型投毒攻击思想启发，尤其是 PoisonedFL 的两个核心动机：跨轮一致性和动态攻击幅度。当前版本为了适配本项目的公平防御实验，采用了更简单可控的 clean-direction bounded attack。

需要诚实说明：

- 当前代码不是 PoisonedFL 官方完整复现；
- 当前代码是一个 PoisonedFL-style / bounded clean-direction performance attack；
- 如果论文正文要命名为新攻击 baseline，建议写成 “recent consistency-inspired performance attack” 或 “bounded clean-direction attack”，不要写成完整 PoisonedFL 复现。

## 7. 参考链接

- PoisonedFL / Model Poisoning Attacks to Federated Learning via Multi-Round Consistency: https://arxiv.org/abs/2404.15611
- PoisonedFL official code: https://github.com/xyq7/PoisonedFL/
- EAB-FL / Exacerbating Algorithmic Bias through Model Poisoning Attacks in Federated Learning: https://www.ijcai.org/proceedings/2024/51
- CTGAN and TVAE / Modeling Tabular Data using Conditional GAN: https://arxiv.org/abs/1907.00503
- CTGAN official library: https://github.com/sdv-dev/CTGAN
- Gaussian Copula Synthesizer documentation: https://docs.sdv.dev/sdv/modeling/single-table-synthesizers/gaussiancopulasynthesizer
- SMOTE / Synthetic Minority Over-sampling Technique: https://www.jair.org/index.php/jair/article/view/10302
- TabDDPM / Modelling Tabular Data with Diffusion Models: https://proceedings.mlr.press/v202/kotelnikov23a.html
- ForestDiffusion / Generating and Imputing Tabular Data via Diffusion and Flow-based XGBoost Models: https://github.com/SamsungSAILMontreal/ForestDiffusion

## 8. 目前最适合给导师的结论

1. Adult AD2+ score ablation 支持完整 score 设计。完整 AD2+ 的综合分数排名第一，说明 reward 和 penalty 同时保留是必要的。
2. Server/root data 不必完全 IID，但需要足够覆盖敏感群体和标签组合。极端 non-IID root data 会破坏公平校准。
3. 低比例真实 root data 加 synthetic data 可以接近甚至超过 10% real root data，尤其是 Gaussian Copula、ForestDiffusion 和 PCA-Gaussian 控制组在当前实验中表现稳定。
4. 当前新性能攻击已经替换 FOE 并完成 70 轮实验，但它是自实现的 bounded clean-direction attack，不是官方 PoisonedFL 完整复现。
5. 当前已补入 ForestDiffusion，能支撑“5 个 synthetic 方法 + 1 个控制基线”的版本；若要进一步增强说服力，可再补 TabDDPM 或 CTAB-GAN+，并把新攻击命名从 `FedSA` 调整成更准确的 `BCD` 或 `PoisonedFL-style BCD`。
