# Adult score diagnostics: restored and independently verified

2026-10-09。**恢复与原始结果全量对应已通过；这项是离线证据验收，没有重跑训练。** 条件定理只能解释真实最终集合中的 softmax 系数质量。现有结果不能支持“所有轮都满足严格分离”或“硬筛选保证不会重新纳入 gate 外客户端”。

## 恢复及验证

- 归档：`revision_verified740_20260923T090026Z.tar.gz`；SHA256 `55b50d489a4ee1824a8637c7da61e25157b50f500ddbd2fc5119711ba16c399b`。
- 仅按安全 basename 恢复4个指定普通文件，原文件保持归档逐字节内容；路径/member SHA见 [restore_receipt.json](restore_receipt.json)。
- [独立验收JSON](independent_acceptance_20261009.json) 与 [可复现验证脚本](verify_restored_20261009.py)：140/140输入文件SHA，8440/8440轮、46字段共388240项，与原始结果重算对应；14/14原摘要组对应。数值容差绝对1e-12+相对1e-12，身份/集合/原文字段精确匹配。
- 20个历史Full wrapper逐一对应本地 `results/attack_strength/raw_results.jsonl` 的原行SHA及完整result body；120个新消融文件逐一对应归档member SHA。
- 8400新轮重新由记录的raw distance/alignment计算门禁，由raw/standardized信号重算score，核10候选评分与获选项、top-k及已存权重。40历史轮核已存门禁、top-k、候选和score；它们缺raw几何输入和直接权重。
- frozen `ec419b7` 源码SHA为 `75ba1ffed74ae4ac0b56e876360e04e8e1700b4b3bb34d11cb8ee59f4c1470fe`；softmax/weighted_average/top-k/MAD四函数AST与已检核心一致。对直接权重重算的最大误差为6.66e-16（本地NumPy 2.2.6），独立标量exp计算的最大差为3.85e-13；没有超过1e-12的权重失败。原归档在原环境记录误差0，二者分别披露。
- [独立逐轮CSV](independent_round_checks_20261009.csv)保留逐轮类别、候选、温度、pre-gate/gate/selected margin、恶意质量与条件上界。

## 实测覆盖和统计单位

新队列是Adult/S-DFA下U/C/A/F/V五个评分删除及N范数删除，2分布×6删除×10seed=120 runs，完整8400轮。历史Full为2分布×10seed=20 runs，但仅记录round1/70，共40轮；它的其他1360轮没有补造。每组10个共享seed；轮次是相关观测，不能当成8440个独立seed或据此算显著性。以下是描述性计数。

| 来源 | runs | 可见轮 | pre-gate正/零/负 | 最终无恶意 | 最终良/恶混合（正/零/负） | gate外再纳入轮 | 非空前提上界核验 |
|---|---:|---:|---:|---:|---:|---:|---:|
| new_completed | 120 | 8400 | 6065/0/2335 | 8278 | 122 (2/0/120) | 116 | 2/2 |
| historical_full | 20 | 40 | 31/0/9 | 40 | 0 (0/0/0) | 0 | 0/0 |

“最终无恶意”是m=0的零质量情形，**不能记作非空严格分离前提成立**。新轮122个最终混合集合中仅2个strict margin≥0，120个为负；2个非空条件上界均成立。另3个含恶意的轮存在数值下溢后的零质量，故“无恶意客户端”和“恶意权重为零”也不能混同。

| 来源/分布 | 删除项 | 可见/计划轮 | pre-gate正/负 | 最终无恶意 | 最终正/负margin（两类非空） | gate外轮 | 最大恶意系数质量 |
|---|---|---:|---:|---:|---:|---:|---:|
| 旧Full/IID | none | 20/700 | 17/3 | 20 | 0/0 | 0 | 0 |
| 旧Full/non-IID | none | 20/700 | 14/6 | 20 | 0/0 | 0 | 0 |
| 新消融/IID | A | 700/700 | 568/132 | 689 | 0/11 | 11 | 1 |
| 新消融/IID | C | 700/700 | 435/265 | 632 | 1/67 | 67 | 1 |
| 新消融/IID | F | 700/700 | 620/80 | 700 | 0/0 | 0 | 0 |
| 新消融/IID | N | 700/700 | 566/134 | 698 | 0/2 | 0 | 3.91255e-24 |
| 新消融/IID | U | 700/700 | 434/266 | 685 | 0/15 | 15 | 1 |
| 新消融/IID | V | 700/700 | 631/69 | 694 | 0/6 | 6 | 1 |
| 新消融/non-IID | A | 700/700 | 460/240 | 699 | 0/1 | 1 | 0.999233 |
| 新消融/non-IID | C | 700/700 | 375/325 | 687 | 0/13 | 13 | 0.212037 |
| 新消融/non-IID | F | 700/700 | 579/121 | 699 | 0/1 | 0 | 0 |
| 新消融/non-IID | N | 700/700 | 417/283 | 700 | 0/0 | 0 | 0 |
| 新消融/non-IID | U | 700/700 | 355/345 | 696 | 0/4 | 3 | 0.42689 |
| 新消融/non-IID | V | 700/700 | 625/75 | 699 | 1/0 | 0 | 0 |

删除项中的质量1.0及接近1的负结果完整保留；这些不是Full全部70轮的失效率。40个历史Full可见轮最终都没有恶意客户端，质量由已存score、候选温度、最终集合按经8400新轮交叉核验的公式重建，不能改写成历史权重直接日志。

## 真实门禁行为与理论边界

实际实现先计算几何gate，若小于n−f则回退到全体；将gate外score临时写为−1e9，再在全体n个位置按k=max(n−f,ceil(nq))取top-k。若k大于gate大小，会把gate外位置再纳入；softmax使用原score。此次观测116/8400新轮出现这种再纳入（本地执行冻结top-k与原集合8440/8440精确一致，无tie例外）。因此原实现的“hard_gate”字段是中间门禁，最终集合不保证是其子集。此报告披露既有行为，未修改方法或重解释旧训练。

固定每轮获选内部候选、实际最终集合S和有效温度τ=max(配置温度,1e-6)，设h/m分别为最终良性/恶意数。在h>0、m>0且μ=min良性score−max恶意score≥0时，Ω≤m/(h exp(μ/τ)+m)。m=0时Ω=0且margin因空类未定义；h=0时没有良性保证。

此次2个非空前提实例：

| 分布/删除/seed/round | h/m | strict margin | 实际恶意质量 | 条件上界 |
|---|---:|---:|---:|---:|
| IID/−C/4242/68 | 15/1 | 0.110199347 | 9.65395645e-07 | 0.0464017415 |
| non-IID/−V/7070/28 | 15/1 | 0.339688738 | 0 | 0.0246363002 |

定理不证明scorer自然产生分离、自适应候选选择有效、root估计泛化、几何gate总能剔除攻击、阈值校准有效，亦不保证完整AD2+的准确率、公平性、收敛或任意架构安全。系数作用在norm处理后的更新上，向量贡献界还需要实际scaled-update范数上界；本次没有重建客户端更新tensor，不能把Ω当总攻击损害。

候选选择核验仅由已有root metrics重算分数和winner，不是重新评估每个candidate checkpoint。pre-gate的margin也以已获选候选为条件。该证据只覆盖Adult/S-DFA及既定删除项，不能迁移成CelebA/其他攻击或其他seed的成立率。

## Suggested rebuttal paragraph

We independently re-audited 140 Adult/S-DFA result files against their archived hashes and recomputed all 8,440 visible score-diagnostic records. The 120 deletion runs provide 8,400 complete rounds; the 20 historical Full runs provide only rounds 1 and 70 (40 observations). Among the new rounds, strict pre-gate score separation is positive in 6,065 and negative in 2,335. After the actual candidate selection and retention operation, 8,278 rounds retain no malicious client; the remaining 122 contain both classes, with positive margin in two and negative margin in 120. We do not count empty-malicious sets as satisfying a nonempty score-separation premise. Both nonempty nonnegative-margin cases satisfy the conditional softmax-mass bound. The intermediate gate is not an absolute final-set constraint: top-k selection re-admits gate-excluded clients in 116 observed deletion rounds. These diagnostics connect the conditional statement to its actual implementation while exposing its limits; they do not establish an accuracy, fairness, convergence, or full-pipeline robustness guarantee.

原4文件保持不变；所有新验收产物均在本score_analysis目录。没有运行服务器训练、改旧原始结果、重选seed或对外发消息。
