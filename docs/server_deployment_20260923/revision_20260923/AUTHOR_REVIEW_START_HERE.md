# GuardFed返修总览：从这里看

更新：2026年10月12日14:38（悉尼），对应03:38 UTC实测。

**实验已有大量有效成果，但整项返修尚未完成。全部24条意见已有英文回应稿；完整17方法表、全部图像消融、最终评价和投稿正文尚未交付完。**

## 当前进度

“已训练”表示模型到终轮；“服务器检查通过”表示保存结果通过原评价检查；“已验收”还包括离机核对和备份。前两者不能直接替代最后一项。

| 工作 | 实际状态 | 还差什么 |
|---|---|---|
| 原结果与表格补充 | 原论文结果保留；ACSIncome、更强non-IID、root扰动和表格消融已有验收材料 | 将可靠结果与限制落实到正文；不重跑旧结果 |
| CelebA主比较 | 历史11方法表已覆盖IID/non-IID×5场景×10种子，共1100条评价记录 | 统一CUDA后的完整表；目标17方法的其余6行 |
| 统一CUDA评价 | 已冻结1500条在服务器端全部完成：主比较1100条＋前四组机制控制400条；离机验收仍为407条 | 验收其余1093条并合并出表；这是重做评价，不是重新训练 |
| 图像机制训练 | 800项新增控制中619项到70轮；8项活动、173项等待、失败0。native训练结果已验收400项 | 余181项训练，以及剩余结果的统一评价和验收；这里的400不等于统一CUDA已验收400 |
| Fed-NGA | 96项新增覆盖训练均到终轮；先前仅19项新native结果已验收 | 验收其余结果，显式合并4项复用记录后入表；已有恒负预测保留 |
| 简单组合控制 | 96项新增训练中70项到终轮；先前29项新native结果已验收 | 余26项训练及后续评价、验收；4项screen复用另计 |
| Huber及另外三方法 | Huber适配获准，96项覆盖尚未启动；FedWA/SmartFL/FedDNA缺明确对应来源 | 执行Huber队列；确认三方法论文/代码身份后忠实实现 |
| 旧Table II | 480格均找到数值来源，44个N/E底层数值已恢复 | 294格只有一次记录，无法计算SD；方法标签和打印值冲突须更正 |
| 回复稿与投稿正文 | 24条意见都有英文回应稿；方法、理论、限制插入稿已备 | 更新最终证据，合入匹配送审的LaTeX工程，核表图引用并编译 |

这些数量有复用和包含关系，**不能相加成总训练次数，也不能据此给整个返修一个完成百分比。** 最终17方法目标为1700条主比较记录；加8组控制共2500条，尚未全部齐备。

## 现在看哪些材料

| 想看什么 | 打开哪里 | 怎样使用 |
|---|---|---|
| 最新统一CUDA数值 | [四方法阶段表](E:/OneDrive/文档/GuardFed/tmp/celeba_uniform_cuda_fourmethod_partial_20261011/actual_partial400/TABLES.md) | 四方法各100条；IID/non-IID、10种子均值±SD。已验收，尚非完整比较 |
| 最宽的现有CelebA展示表 | [十一方法论文表PDF](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_eleven_method_native_pdf_20261011/celeba_eleven_method_native.pdf) | 历史CPU/GPU混合评价口径，用于审阅覆盖和布局；待统一CUDA表替代最终比较 |
| 逐条回应reviewer | [24条英文回复稿](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md) | 已写全，仍是工作稿；进度截止早于本页，不代表问题全部解决 |
| 旧表追溯 | [Table II说明](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/table2_trace_20261002/追溯报告.md) | 区分找到了原数值与有足够重复次数计算SD |

## 审稿意见还缺什么

| 审稿关切 | 已有回应或证据 | 提交前收尾 |
|---|---|---|
| 新数据集、非MLP架构、基线广度 | ACSIncome、CelebA CNN及历史11方法比较 | 完成统一评价后的最终表，标清适配和选择历史 |
| 消融与跨数据集差异 | 表格消融、部分图像控制、同模型校准对照 | 收完图像控制；COMPAS反例否定每个评分项都不可或缺，须收窄主张 |
| root实用性与公平性假设 | 噪声、群体欠覆盖已有实测 | 写清代表性、敏感属性和总体信息假设；处理原合成图来源 |
| 理论、创新和方法描述 | 条件性证明、实际流程、符号修订稿 | 将正文公式/伪代码改成实际5090实现；不宣称普遍公平或收敛保证 |
| non-IID与规模 | 更强non-IID补充与CelebA双分布覆盖 | 20客户端全参与不能证明大规模/部分参与能力，保留限制 |
| 统计与复现 | 旧表来源、重复次数、配置及公开快照 | 更正旧表与方法标签，处理Fig.3来源，统一最后发布；不为单次结果补造SD |

**回复写全24条，不等于24条问题全部解决。** 有的需要实验，有的需要纠正描述、收窄结论或撤下无法核验的旧主张；不能全靠增加训练解决。

## 接下来沿三条线收尾

1. **收完当前实验。** 先验收已完成的1500条CUDA评价和NGA结果；让剩余机制/组合训练完成，再统一评价、验收和出表。完成标准是完整网格、同终轮checkpoint、原始三指标、10种子均值±样本SD及可追溯证据齐备。
2. **补齐有明确身份的方法并纠正旧证据。** 目标17方法保留；Huber待执行，另外三方法先确认来源。Table II、原合成图和方法标签逐项更正或限定，保留负结果。17方法是后续扩展目标，不能把行数当成审稿问题解决的标准。
3. **形成最终投稿版本。** 明确最终评价的数据端点、比较规则和主指标；将最终表图、实际方法和逐条回应合入匹配送审的LaTeX工程，完成编译与引用核对。

待作者资料或决定：FedWA、SmartFL、FedDNA的确切文献或官方代码；匹配送审的LaTeX工程；最终评价方案。当前CelebA表是验证集结果，官方test已有历史暴露，不能称从未接触的测试集。

## 当前结果能说到哪一步

已验收四方法表中，GuardFed的AEOD均值在10个场景均最低；按10种子的跨场景汇总，准确率低于FLGMM，ASPD高于LoGo。这是阶段均值比较，不是显著性结论，也不支持所有方法、所有指标全面领先。LoGo和FedAvg各保留1项恒定预测；零差异不能单独等同于有用的公平预测。

服务器89.22.197.55:60350在线。03:38 UTC双5090利用率100%/98%、温度68/63°C，Recovery None、cgroup OOM0、磁盘约1.04TB余；主训练从昨日448增至619。三小时聊天巡检最后核验为暂停，当前无可用编辑工具；服务器Supervisor队列独立运行，不声称有持续聊天巡检。

最近已核验的公开实验快照：[Git提交2166380](https://github.com/YananZHOU5555/GuardFed/commit/2166380072d41423465df0d4e64ba3d2736bd719)，截止407条CUDA验收。大文件留服务器或F盘，Git保存代码、配置和精简证据。详细执行状态见[RUNNING](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/RUNNING.md)。

<details>
<summary>历史审阅入口（2026-10-11固定截止；不代表当前进度）</summary>

# GuardFed 返修审阅入口

2026-10-11。**回复稿已覆盖全部24条意见，但整项返修尚未完成。** 本页只指向已验收的材料；运行中的任务不计入论文结果。

先看[完整逐条英文回复稿](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL100_Hybrid20_native11_20261011.md)。它保留审稿人原话，并逐项说明新增证据、修正和限制。该回复稿的机制截止仍为380条、F80、FLGMM三视图100和组合控制20；新增11方法native表（1100条评估记录、含完整FLGMM100）已写入。其后的机制native400、NGA新增19条验收，以及CPU/GPU校准诊断见实时入口，尚未写入该固定截止稿。十一方法表、三页PDF、当前FL100回复稿、作者CUDA批准与首7项/Hybrid29证据已在Git70发布（4fccc8b8，197个文件blob核SHA）；07:55运行观察、800固定3及恢复源码已在Git71发布（ba302df5，170个blob及remote核SHA）；FL3、NGA19和FedAvg100已在Git72发布（51a27b50，258个blob/remote核SHA）；FL100完整CUDA采用、LoGo2与剩98启动及09:19截止入口已在Git73发布（f1148225，246个blob/remote核SHA）；本次Full100与LoGo100完整采用、CUDA去重407及新入口尚未发布Git74。再看[方法、理论、符号和讨论的正文插入候选](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A100_reader_20261011/manuscript_insertions_integrated_20261011.md)。这些文字还未移植到匹配投稿的LaTeX工程。

| 审阅内容 | 已有材料与准确边界 |
|---|---|
| 其他返修实验 | 完整回复稿同时整理了ACSIncome（120次）、Adult/COMPAS更强non-IID设置（360次）、root扰动和表格消融等证据及限制；本页的CelebA表不代表全部新增实验。 |
| CelebA 主比较表 | [十一方法 native 表](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_eleven_method_native_20261011/TABLES.md)及[三页论文表PDF](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_eleven_method_native_pdf_20261011/celeba_eleven_method_native.pdf)：IID/non-IID各5场景，每条件10种子，共1100条方法–场景–种子评估记录，包含完整FLGMM100；不是1100个独立训练模型。尚非目标17方法完整表。 |
| 预测校准是否解释优势 | [九方法 raw/native/shared 表及说明](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_three_view_20261009/README.md)：900个模型记录的三种预测视图，不是2700次独立训练。共同校准后，原始准确率与差异指标的权衡仍需保留。 |
| 图像机制消融 | 训练U/C/A/F各100新结果已采用，共400/800；历史CPU三视图380（F80）及旧失败保留。统一CUDA去重采用407=首阶段105（Full100+旧5控制）+原八方法102+FLGMM100+LoGo100；Full100只共享一次，旧7与Full100重叠2条，不是411。Full100原Linux whole saved gate及离机900指标/2400计数/300规则、native差0通过；LoGo100使用已采用FedAvg CUDA缓存与原冻结CPU netcal后处理，无新fit/refit，1986700预测/300指标/800计数重算误差0，保留1条恒负结果。09:48:54服务器108/500、09:43:14服务器277/800仅为保存验收观测，不增加离机采用。[四方法CUDA native400阶段表](E:/OneDrive/文档/GuardFed/tmp/celeba_uniform_cuda_fourmethod_partial_20261011/actual_partial400/TABLES.md)已生成并独立验收、根采用，非最终17方法表且无新增科学记录。实际冻结1500条，完整2500目标尚未齐备；不重训、不改参数/容差、不访问test。 |
| 新增基线与简单组合 | [FLGMM十场景100条](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_flgmm_ten_scenes100_20261011/TABLES.md)和[组合控制两场景20条](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_hybrid_three_view_two_scenes20_20261011/TABLES.md)三视图单独展示；FLGMM的完整native100已纳入十一方法表。FLGMM最后10条已完成Linux原完整保存验收、Windows零拟合核对、F盘24成员SHA备份及root采用；旧90条保持不变。该100条仍为历史CPU评价，新统一CUDA表另行产生。Fed-NGA已有19条新训练结果验收，均为恒定负类预测，保留为负结果；64次搜索仍是单种子调参证据，不是完整多种子对照。 |
| 旧 Adult Table II 的值与标准差 | [四页重建表PDF](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/table2_recovered_20261002/table2_complete_review.pdf)及[追溯说明](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/table2_trace_20261002/追溯报告.md)：480格均已找到数值来源，原来44个N/E的底层值已恢复。294格只有1次记录，180格有10次，6格有3次；单次记录的SD不可计算。数值追溯完成不等于忠实方法和统一十种子比较完成。 |
| 各方法真正使用了什么信息 | [中英文实现边界表](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/method_information_access_20261011/TABLE.md)，[英文版](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/method_information_access_20261011/TABLE_EN.md)：14个已集成实现的输入、root/群体访问和适配限制；不是14个方法实验全部完成。 |

Table II还需落实一处正文数值更正：FairGuard/IID/FedSA的投稿准确率是59.13%，历史源汇总是54.13%，同终轮修正候选是53.94%。三个数值属于不同记录，不能互换。历史FedWA行实际来自AdaAggRL，cosine/fairness行也混用了实现；有数值不等于原方法得到忠实复现。

方法按此前5090实现解释，包括内部候选选择、硬筛选、范数处理和群体阈值。Huber采用已同意的恒等投影，明确标为CNN项目适配，不继承约束域理论保证。LoGoFair采用固定图像ID的20个虚拟群体，仅称人口适配；DP指demographic parity，不是差分隐私，也不证明真实训练客户端公平性。

[论文附录候选：14个实现的信息访问与适配边界](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/method_information_access_paper_appendix_20261011/APPENDIX.md)将上述实现说明压缩为英文表和配套文字。表号暂记Table X，尚未移入投稿工程，也未核验IEEE双栏排版。

提交前仍须完成剩余基线和图像控制的验收、完整IID/non-IID表、冻结最终评价、正文整合及最终复现发布。FedWA、SmartFL、FedDNA仍缺足以忠实实现的对应资料；最终评价端点和匹配投稿的LaTeX源仍待确定。现有CelebA表是验证集结果，官方测试集已有历史暴露，不能称未触碰测试集。原论文结果保留，不通过挑种子、混选checkpoint或只删除负结果改善表格规律。

实时任务状态以[RUNNING](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/RUNNING.md)最新带时间的记录为准；本页不是实时监控。统一CUDA协议与恢复边界见[本阶段协议](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_uniform_cuda_evaluation_v1/PROTOCOL.md)。

</details>
