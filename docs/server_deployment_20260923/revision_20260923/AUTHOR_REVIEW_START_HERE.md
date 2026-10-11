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
