# CelebA 九方法完整验证表

2026-10-04（Australia/Sydney）。900/900条记录完成验收：九方法×IID（α=5000）/non-IID（α=5）×Benign、F Flip、FedSA、S-DFA、Sp-DFA×共享种子91001–91010。全部指标取70轮终轮checkpoint；train162770、valid19867。现有表属于验证集结果，不能更名为独立test。

沿用此前正文式三指标分行排版，每格报告10个种子的均值±样本标准差（ddof=1）。ACC为百分比，AEOD/ASPD为[0,1]差值；AEOD实际实现为绝对TPR差。未按指标挑选不同种子或checkpoint，未用综合分数替代主表指标，负结果和恒定预测均保留。

| 交付 | 文件 |
|---|---|
| 三页PDF：IID、non-IID、双分布合并 | [PDF](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_final_20261004/celeba_iid_noniid_ten_seed.pdf) |
| IID主表源 | [LaTeX](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_final_20261004/celeba_iid_ten_seed.tex)、[Markdown](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_final_20261004/celeba_iid_ten_seed.md) |
| non-IID主表源 | [LaTeX](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_final_20261004/celeba_noniid_ten_seed.tex)、[Markdown](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_final_20261004/celeba_noniid_ten_seed.md) |
| 双分布主表源 | [LaTeX](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_final_20261004/celeba_iid_noniid_ten_seed.tex) |
| 排除配置选择种子的9种子表（91002–91010） | [Markdown](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_final_20261004/celeba_iid_noniid_exclude_selection.md)、同名.tex |
| 同口径6种子子集（91005–91010） | [Markdown](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_final_20261004/celeba_iid_noniid_matching_six.md)、同名.tex；不宣称该子集从未观察 |
| 覆盖、来源和独立复核 | [覆盖](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_final_20261004/coverage.md)、[provenance](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_final_20261004/provenance.json)、[独立审计](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_final_20261004/statistics_audit.json) |

原七方法700条原始值保持不变，新增FedAA/LASA200条（192新增＋8显式复用）使用先前冻结recipe。合并900条的来源是836条新增、64条复用，不能将900再次计为新增训练。FedAA使用policy0.001_keep16_local0.001，LASA使用LASA_s0.3_l2_lr0.001；两者均为披露限制的适配实现。

跨场景汇总先在每个seed内平均两分布×五场景，再跨10个seed统计；不将十个场景视为十个独立seed。完整九方法汇总及八个基线配对差异保存在seed_paired_summary.json。主要比较如下：

| 原生输出口径 | ACC (%) | AEOD | ASPD |
|---|---:|---:|---:|
| FLTrust | 89.629 ± 0.652 | 0.04585 ± 0.00406 | 0.11456 ± 0.00468 |
| FedAA-DDPG适配 | 88.623 ± 1.044 | 0.04572 ± 0.00610 | 0.10847 ± 0.00908 |
| LASA适配 | 87.752 ± 1.221 | 0.03963 ± 0.00395 | 0.09785 ± 0.00826 |
| GuardFed-AD2+ | 88.420 ± 0.623 | 0.00967 ± 0.00297 | 0.06105 ± 0.00466 |

GuardFed准确率较LASA高0.668个百分点（配对seed胜7/10），较FedAA低0.204个百分点（胜4/10），较FLTrust低1.209个百分点（胜0/10）；对这三者两项公平差值均更低（各胜10/10）。FairGuard的ASPD均值更低，但准确率约74.04%；该结果同样保留。均值排名和胜率本身不构成显著性结论，也不支持“全部指标胜过所有方法”。

九方法表比较完整GuardFed组校准流程与基线原生输出。七方法共享校准700模型的独立归因结果仍有效：ASPD收益保留、准确率有代价、AEOD均值无明确优势；该对照未扩展到FedAA/LASA，不混入本表。七方法曾用non-IID Benign/S-DFA筛选，新增两方法用两分布Benign/S-DFA筛选，均涉及seed91001。原14条复用记录使用cu130，其余886条使用cu128；迁移首轮一致不证明完整70轮等价。所有适配与局限已保留在表注。

验证完成：90格各有相同10seed；900条配置身份检查；新增200条从已核哈希备份中的原始JSON复核终轮、数据/环境/checkpoint和三指标；810组均值/SD由renderer复算；独立审计1944个统计标量最大误差2.78e-17；五份Markdown和五份LaTeX各1080个数值单元一致。五份LaTeX片段已通过XeLaTeX编译；三页最终PDF均经渲染及视觉检查。verification.json记录最终文件哈希。

模型恢复：新增192条由24＋40＋48＋40＋40的五个无重复增量覆盖，8条复用仍引用原调参恢复链；archive SHA及member hashes在本机核验通过。见[阶段恢复链](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/celeba_baseline_fullcoverage_v1/restore_chain.json)。源码、来源快照及表格都保存在本目录，可用merge_sources.py和build_tables.py复现；固定identity_evidence.json与最终statistics_audit.json分开保存。

本阶段完成，整个返修未完成。17方法目标尚缺其余8方法800格、CelebA机制消融、冻结最终评价，以及正文与逐条rebuttal。现有监控继续检查服务器健康；本次未启动下一队列或test。
