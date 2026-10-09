三项身份均匹配提交版[28]/[33]/[35]；本轮没有一项因现有资料而可直接解锁忠实实现。关键障碍不是“没有代码”本身：FedWA持有的完整三页扩展摘要缺可执行DRL规格；SmartFL/FedDNA尚未取得完整方法正文，不能断言其论文缺公式。若取得完整可执行数学规格，作者代码并非必要前提。

| 方法 | 提交版与原来源 | 当前实际持有 | 判定 |
|---|---|---|---|
| FedWA [28] | Guo / Wang / Wu；AAMAS2022 pp1610–1612；DOI10.5555/3535850.3536051 | [原三页PDF](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf)及本地完整提取文本 | 身份正确；完整extended abstract仍不足实现DRL。 |
| SmartFL [33] | Dong / Gao / Zhou / Yang / Kuang / Fu；DOI10.1016/j.inffus.2025.103555 | 出版社题录XML、作者论文条目及旧报告预览摘要 | 身份正确；未取得完整算法，不能评价正文是否给齐。 |
| FedDNA [35] | Garg / Bansal / Yadav / Kandhoul / Dhurandher / Woungang；DOI10.1016/j.jisa.2025.104358 | 出版社题录XML、作者公告/分享路线及旧报告预览摘要 | 身份正确；未取得指纹/历史/阈值完整公式。 |

提交版依据是[已提取文本](E:/OneDrive/文档/GuardFed/tmp/pdfs/tdsc_submission/submission.txt)第956–962、984–986、991–995行（两栏布局夹有另一栏文字），对应给定PDF SHA549a4191b1ac560bdaba79d9dce3b11693ab69c4a8cc0f7f52be31afa1110071。SmartFL投稿写2025，已保存出版社与Crossref记录为最终卷126/2026年2月；这是online-first与最终卷格式的统一事项，不能误判为另一篇。FedDNA卷97/104358/2026一致。FedWA完整题名含Extended Abstract；本地PDF未印DOI，DOI由已有报告及保存题录绑定，本轮未作新的登记解析。历史paper.md缺正式paper.bib，recommended_references.bib属于审稿推荐文献，不能借来补猜三法条目。

FedWA的可执行性复核

本地[原文](E:/OneDrive/文档/GuardFed/tmp/celeba_baselines/remaining_20261009/group_b/sources/fedwa_primary.txt)第111–122行仅给样本均值局部目标和按样本数加权全局目标；第158–167行Eq(3)给按客户端权重求和的全局**损失目标**。它不是已确定的actor输出映射，也没有足以重建参数更新的完整算法。第190–212行描述上传local models、由DRL学习贡献/决定权重、继续聚合。可核原文短句为“Using the received local models and current global model”（第200行）；此处没有MDP或学习器定义。原文三页随后进入致谢与参考文献，已持有的版本没有算法附录。

必须补齐：state编码/缩放/顺序；action到权重约束及model/delta语义；reward公式、所用数据权限与时点；具体RL算法、网络、优化器/LR、探索、回放或target网络（如适用）、初始化与更新日程。选择DDPG、借用FedAA reward、固定三项cosine权重或自行softmax，都会新增原文未确定的科学操作。最少作者输入是“此引文对应的完整算法/补充规格（若存在），或源码+默认配置+MDP定义”；再取同一份三页PDF不能填缺口。

SmartFL与FedDNA的缺项是材料缺项

本轮直接解析保存的出版社XML：两份顶层均只有coredata，公式与algorithm元素数均为0；作者SmartFL条目也只有citation。它们支持题录身份，不能支持算法实现。既有报告中的“两阶段多数方向”“激活指纹+历史+MAD”保留为机制概述；没有获得公式编号，本包不会编造Eq编号。旧报告提到SmartFL方法节/Appendix C、FedDNA Sections3/4，只是下一份材料应包含的目标范围。

| 方法 | 取得全文后必须固定的定义 | 为什么现在不能推断 |
|---|---|---|
| SmartFL | 成对分数、逐步删除对象/次序/tie、停止条件及是否需要f；完整参数还是update；归一化与pseudo-reference公式；正相关召回集合、90°/零范数边界；最终权重/范数恢复/空集；Appendix C等默认值与模型 | row-mean cosine top-n−f不等于未知的逐对删除；等权、cosine权重和不同归一顺序产生不同输出。 |
| FedDNA | probe来源/数据权限/数量/预处理/固定或重采样；激活层、跨probe与神经元归约/归一/距离；history初始化/ID对齐/更新顺序/衰减与consistency公式；MAD作用分数、中心/系数/零MAD/tie；最终权重、model/delta、空集及默认值 | 参数范数三标量不是行为指纹；“固定探针”旧概述不足冻结采样/数据访问，常用3-MAD和任意历史窗口都不是已确认原法。 |

SmartFL最少需要[精确DOI全文](https://doi.org/10.1016/j.inffus.2025.103555)或accepted manuscript及全部算法/默认值附录；FedDNA最少需要[精确DOI全文](https://doi.org/10.1016/j.jisa.2025.104358)或accepted manuscript及方法/实现附录。若论文给齐，直接据规格独立重实现即可；否则只向作者索取上表仍缺部分或对应strategy/probe/model/default代码。无需笼统要求“必须公开官方仓库”。FedDNA旧share-link的ATP-3不能说明永久不可得，未重复请求；DNA序列/normalization/dynamic-node同名FedDNA、ICLR同名subspace SmartFL和FedAA DDPG均不替代目标。

最短恢复路径与本轮边界

1. Root按上述三份最小材料请求取得source-matched原文/补充内容；无新线索时不重复搜索已完成的作者文件与HTTP路线。
2. 原文到位后逐项绑定公式/伪码/默认值/数据访问。可执行完整规格优先于“是否有代码”，缺项再收窄询问。
3. 只在规格充分后准备隔离adapter及固定输入组件比对，明确caller上传full model、delta或gradient；状态型方法保留客户端ID与跨轮状态。组件检查必须针对原公式及边界，不用旧简化分支作oracle。
4. 组件通过后再交root决定共同图像管线的最小真实门检与正式选择协议。本包不启动实现、CNN、GPU或新搜索，也不改正式方法/协议。

本轮新增primary检索0、下载0、忠实adapter0、科学接受0。发现的新价值是把“原文真的未给规格”（持有FedWA完整摘要）与“尚未拿到原文”（另两篇）分开，明确代码并非唯一恢复路径，并保留每项不能自由补猜的公式/接口。已按路径核对tmp/celeba_baselines内PDF，只有既有FedWA、Huber、Fed-NGA，未发现SmartFL/FedDNA完整PDF；这只界定该包，不推断作者或用户其他位置无源。没有重读31HTTP/42作者文件全集，没有全盘扫描、SSH/推理/训练、STATE/Git/论文修改或性能主张。

实际来源pins、缺项逐条原因、原XML结构检查、恢复接口清单和身份信息保存在[DECISION.json](DECISION.json)。未执行任何下载代码或原实验helper。
