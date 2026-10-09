# CelebA实际分区与root支持审计

2026-10-09。**20份分区、400个client样本总数逐个精确匹配；实际敏感组分配已经恢复，未运行新训练。** 本审计取Stage A中GuardFed-AD2+的Sp-DFA、IID/non-IID、seed91001–91010共20份70轮原结果。它报告的是攻击前的真实分区，而非运行时被改写的敏感注释。

## 证据与精确重建

- [acceptance.json](acceptance.json)记录20个member及原inventory SHA/bytes、6份增量归档SHA、20个冻结job SHA、源码与AST/helper SHA，以及全部门检。原result只安全恢复到本目录raw_results，未恢复/改动模型或旧结果。
- [audit_partitions.py](audit_partitions.py)可复现统计；[冻结函数摘录](frozen_source_function_excerpts.txt)保留create_client_data_dict、load、root和runtime的具体源码。core与CelebA loader的完整SHA均匹配Stage A冻结manifest。
- 由global_group_counts减root的敏感组计数得客户端池：Male=0共85058、Male=1共61435，合计146493。对每个seed重新建立default_rng，按敏感组1、0顺序，先shuffle同长度int64数组，再dirichlet/cumsum/floor切分。shuffle的随机状态只与数组长度/类型有关；40个同长度不同取值的精确RNG状态及swap-pattern检验通过。无需图像或属性metadata。
- 400个重建样本总数与原image_data_contract.client_sample_counts完全一致，也与attack_audit.samples一致。另40个原Sp-DFA属性攻击客户端的fflip_changed与重建Male=1数量精确一致：冻结all_unprivileged模式将敏感注释设0，因此该计数提供额外的原始组数交叉核验。
- load_bundle/load_celeba_bundle及分区函数不接收attack参数，所读config也不依赖攻击模式或experiment tag；run_experiment先load_bundle，再调用client_runtime_data。root/分区先于攻击，且敏感数组在runtime中复制后才改写。相同seed/alpha/配置的攻击前分区由此不依赖attack选择；这不等于另行提取了所有方法/场景的原文件。

## 跨seed实际分布

每个分区先在20个client内计算描述量，再对10个共享seed报告mean±sampleSD(ddof=1)。以下n=10，不能把400个client或两个分布中重复的root当成独立seed。百分比的SD单位为百分点。

| 分区描述量 | IID（α=5000） | non-IID（α=5） |
|---|---:|---:|
| 每seed最小client样本数 | 7195.5±42.4 | 3979.2±644.2 |
| 每seed最大client样本数 | 7461.5±39.5 | 12053.1±1670.8 |
| 20client样本数CV（populationSD/mean） | 0.0095±0.0013 | 0.2987±0.0604 |
| 每seed最小client男组比例（%） | 40.949±0.238 | 16.312±3.047 |
| 每seed最大client男组比例（%） | 42.811±0.149 | 68.285±4.918 |
| 20client男组比例populationSD（百分点） | 0.473±0.052 | 13.756±1.554 |
| 前4恶意client覆盖客户端池样本（%） | 19.964±0.102 | 18.874±3.132 |
| 前4覆盖客户端池Male=0样本（%） | 19.944±0.118 | 18.289±3.396 |
| 前4覆盖客户端池Male=1样本（%） | 19.992±0.169 | 19.684±5.583 |

全部200个IID client观测的男组比例范围为40.495%–42.969%，non-IID为9.832%–74.303%；样本数分别为7126–7532、2926–15638。400个client中没有缺Male=0或Male=1的情况。非IID下前4恶意client的样本覆盖率范围为15.173%–24.661%，所以“4/20恶意client”不是每个seed恰好20%的样本覆盖。不能把这些敏感组统计扩展为未经恢复的client标签联合异质性。

## 原20分区明细

[per_partition_20.csv](per_partition_20.csv)给出完整数值/来源SHA；[per_client_400.csv](per_client_400.csv)给出每个client的两组计数。下表root四格顺序为Male0/Smiling0、0/1、1/0、1/1。

| 分布 | seed | client样本min–max | client男组% min–max | 前4样本覆盖% | root四格支持 |
|---|---:|---:|---:|---:|---|
| IID | 91001 | 7233–7412 | 41.114–42.789 | 20.009 | 4318/5133/4107/2719 |
| IID | 91002 | 7208–7516 | 40.495–42.892 | 19.963 | 4300/5151/4108/2718 |
| IID | 91003 | 7163–7532 | 41.112–42.775 | 19.951 | 4315/5136/4067/2759 |
| IID | 91004 | 7226–7447 | 40.719–42.876 | 20.156 | 4403/5048/4144/2682 |
| IID | 91005 | 7216–7450 | 41.080–42.873 | 19.877 | 4356/5095/4138/2688 |
| IID | 91006 | 7136–7442 | 41.204–42.793 | 19.886 | 4373/5078/4092/2734 |
| IID | 91007 | 7126–7419 | 40.690–42.969 | 19.817 | 4364/5087/4095/2731 |
| IID | 91008 | 7175–7445 | 41.111–42.915 | 19.968 | 4431/5020/4160/2666 |
| IID | 91009 | 7227–7461 | 40.890–42.426 | 20.098 | 4395/5056/4094/2732 |
| IID | 91010 | 7245–7491 | 41.073–42.803 | 19.916 | 4349/5102/4121/2705 |
| non-IID | 91001 | 4683–10128 | 16.845–68.061 | 20.481 | 4318/5133/4107/2719 |
| non-IID | 91002 | 3560–14251 | 19.314–74.303 | 18.358 | 4300/5151/4108/2718 |
| non-IID | 91003 | 2926–15638 | 15.883–70.814 | 18.092 | 4315/5136/4067/2759 |
| non-IID | 91004 | 4583–11504 | 9.832–71.845 | 24.661 | 4403/5048/4144/2682 |
| non-IID | 91005 | 4135–11833 | 17.421–71.211 | 15.952 | 4356/5095/4138/2688 |
| non-IID | 91006 | 4663–11106 | 18.585–71.359 | 15.173 | 4373/5078/4092/2734 |
| non-IID | 91007 | 4110–11045 | 20.117–69.790 | 15.618 | 4364/5087/4095/2731 |
| non-IID | 91008 | 3056–11890 | 17.037–63.602 | 18.638 | 4431/5020/4160/2666 |
| non-IID | 91009 | 4371–12266 | 14.506–58.481 | 23.162 | 4395/5056/4094/2732 |
| non-IID | 91010 | 3705–10870 | 13.582–63.383 | 18.606 | 4349/5102/4121/2705 |

## root四格与代表性

每个root有16277图像；敏感组计数Male=0为9451、Male=1为6826，均由敏感组分层抽样固定。训练总体四格为43688/50821/41002/27259。相同seed在两分布中的root image-id SHA、四格和代表性指标完全相同，因此下表及统计使用10个独立seed，而非重复的20份。

| root描述量 | 10seed mean±sampleSD | 所有seed范围 |
|---|---:|---:|
| 四格中最小支持数 | 2713.4±28.2 | 2666–2759 |
| sensitive×label联合TVD | 0.003486±0.002155 | 0.000614–0.007495 |
| label边际TVD | 0.002777±0.002500 | 0.000061–0.007495 |
| sensitive边际TVD | 0.000006144±0.000000000 | 固定 0.000006144 |
| root Smiling=1比例（%） | 47.945±0.384 | 47.220–48.504 |

四格均有支持，最小2666；这些是已测root分层抽样的代表性描述，不证明任意root具有同样支持，也不消除真实部署获取root的隐私/可用性假设。所有已存group/sensitive/label TVD均从计数重算并在绝对1e-12内吻合。10个root原始明细见 [root_support_10_unique_seeds.csv](root_support_10_unique_seeds.csv)。

## 不能恢复的部分

**没有恢复每client的Smiling标签数或Male×Smiling四格。** 总体/root四格和client样本数不足以唯一确定client标签联合计数；没有加载metadata，不能据此虚构标签分布、子群TPR/FPR分母或预测指标。当前审计只关闭CelebA敏感组分区及root支持这部分P4；Adult/COMPAS强异质性实测分区及历史synthetic图实现链仍需另外证据。

## Manuscript paragraph candidate

We audited the realized pre-attack CelebA partitions, rather than interpreting heterogeneity solely from the Dirichlet parameter. Replaying the frozen sensitive-group split for 20 partitions (two distributions and ten shared seeds) exactly matched all 400 archived client sample counts. Across the 200 client observations per distribution, the Male=1 fraction ranged from 40.495% to 42.969% under IID and from 9.832% to 74.303% under non-IID; no client lacked either sensitive group. Across seeds, the within-partition sample-count CV was 0.00946±0.00129 versus 0.29870±0.06038. The first four nominally malicious clients covered 19.964±0.102% versus 18.874±3.132% of client-held examples. The 16,277-example root had nonzero support in all four sensitive/label cells (minimum 2,666), with joint total-variation distance from training-population proportions 0.003486±0.002155 across ten unique root seeds. These are sensitive-allocation and root-support diagnostics; client label-by-sensitive counts were not recovered and are not inferred from alpha.

未访问服务器、未训练、未更改旧结果或冻结协议；归档、源码、helper和产物SHA均可追溯。
