# 剩余10方法来源、忠实度与接入次序（2026-09-28）

本次仅查阅投稿正文、现有代码与原论文/作者代码，未访问服务器、未训练、未修改其他目录。主要证据是 `tmp/pdfs/tdsc_submission/submission.txt` 第295–308行及文末[27]–[35]，旧实现 `tmp/publish-5090/scripts/reproduce_paper_tables.py`。旧 lite/core 分支不能直接成为原法完整复现。这里的“可接入”指下一步工程可执行，不是已通过真实图像实验。

## 名单与引用核对

| 剩余行 | 投稿引用 | 当前准确状态 | 最先要做的事 |
|---|---|---|---|
| LoGoFair | [27] Zhang等 AAAI2025 | 已有独立adapter，本次不重复验收 | 交由现有adapter集成检查；保留本地/全局公平后处理 |
| FedAA | [30] He等 AAAI2025 | 已有独立DDPG adapter，本次不重复验收 | 检查跨轮状态、reward数据来源、策略及RNG恢复 |
| Fed-NGA | [34] Zuo等 NeurIPS | 已有式9聚合组件，非完整训练法 | 同global点梯度路径、独立服务器步长、攻击消息语义 |
| FedWA | [28] Guo/Wang/Wu AAMAS2022 pp1610–1612 | 引用和名称匹配；旧AdaAggRL启发式不匹配 | 获取充分DRL规格，不能把FedAA DDPG或旧启发式改名顶替 |
| LASA | [29] Xu/Zhang/Hu WACV2025 | 找到作者官方代码，最优先算法接入 | 独立移植完整过滤/稀疏化规则并对齐官方输出 |
| Huber-BRFL | [31] Zhao/Yu/Wan AAAI2024 | 对应Huber聚合论文，Huber-BRFL是本文描述性标签 | 实现固定目标的样本数加权Huber求解器，再复用严格梯度路径 |
| FLGMM | [32] Zhu/Huang/Zhang Information Fusion | 作者代码可用，但有论文/代码行为需裁定 | 隔离warmup、GMM、SPC状态并先核对监测判决是否用于聚合 |
| SmartFL | [33] Dong等 Information Fusion | 名称/作者匹配；公开摘要不足以重建所有步骤 | 取得原文算法/作者代码后确定第一阶段删除和最终聚合规则 |
| FedDNA | [35] Garg等 JISA2026 | 名称/作者匹配；必须是行为指纹版本 | 取得激活指纹、历史一致性、探针数据具体定义 |
| Cosine Similarity + Fairness Deviation | 本文自定义hybrid，无独立引文 | 可作为自定义控制优先接入 | 固定并清楚标记旧GuardFed分支与当前AD2+不同 |

正文17行=当前7行+以上10行。三个已有adapter只是代码进度，不能提前计为已完成的科学结果。

书目信息提醒：FLGMM、SmartFL DOI含2025，但出版社最终卷126对应2026，正文2025可能是online-first年份，需统一格式。Fed-NGA当前标题对应NeurIPS2025最终论文；正文写vol38、2026应按最终正式BibTeX复核。不能把早期arXiv标题变化误判为另一种方法。FedDNA有多个同名方法，不能误接2023 PLOS的dynamic node alignment或其它DNA生物任务。

## 优先1：LASA作者代码移植

原始来源：[WACV论文](https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Achieving_Byzantine-Resilient_Federated_Learning_via_Layer-Adaptive_Sparsified_Model_Aggregation_WACV_2025_paper.pdf)，[作者仓库](https://github.com/JiiahaoXU/LASA)。固定commit `8477367a4e8708cde264f7572805040c650af59f`，本目录保存 `lasa_official.py` 与 `lasa_mask_help.py`。

已读到的算法实现：整体范数中位数裁剪；对2D/4D权重作全局top-k稀疏化；逐层使用范数和符号统计相对median/std的偏差分别过滤；取两个集合交集、空交集回退全部；对选中客户端做层聚合。稀疏掩码使用严格大于阈值，tie会影响保留数量。[官方聚合函数](https://github.com/JiiahaoXU/LASA/blob/8477367a4e8708cde264f7572805040c650af59f/algorithms/defense/lasa.py)

旧实现803–830行以 `ReLU(cos_to_layer_median) × sigmoid(norm_deviation)` 排序强保留n−f，既无官方符号过滤，也不是官方两个阈值交集；不能作为同算法。

最小接入：保留三个独立超参 sparsity/lambda_n/lambda_s；移除硬编码 `.cuda()` 用输入device；避免原函数修改调用者字典；保留layer选中索引和mask身份。官方的 `clipped_local_updates` 与随后稀疏操作有对象别名风险，应通过实际行为检查而非凭变量名称假设最后聚合一定为dense。零范数、零std、全零层、NaN客户端删除后的索引须明示修复，不偷偷引入n−f筛选。

门检：固定三层CNN张量，逐步对齐clip、mask、norm/sign分数、集合、最终update；加入top-k ties、全相同层、全零层和单个异常客户端；先CPU固定输入、再双GPU确定性小图像门检。现阶段未运行这些门检。

## 优先2：本文cosine＋fairness hybrid

投稿第306–308、395–409、519行将其定义为自建hybrid，无外部算法复现义务。旧代码1275–1287行 `method == 'GuardFed'` 是候选映射：`t_i = max(0,cos(root_update,update_i)) * exp(-lambda*AEOD_i)`；选`t_i > tau`，空集合取argmax，**最终等权平均**，不是按`t_i`加权。默认lambda20、tau0.2。

接入新标签如 `CosineFairnessHybrid`，保留来源标签 `legacy GuardFed`，不得将其展示成GuardFed-AD2+。核对旧表实际run配置和模型身份后才能说它就是旧行的数值来源；本次只证明代码机制一致候选。门检覆盖等权/信任权重区别、阈值等号、空集fallback、NaN、fairness单位和攻击后root评价。新CelebA可独立有界调lambda/tau并固定测试前规则。

## 优先3：Huber-BRFL明确目标后实现

[AAAI原论文](https://ojs.aaai.org/index.php/AAAI/article/download/30181/32095)式5–6是 `argmin_s sum_i n_i * phi_i(||s-X_i||)`，phi为阈值Ti的向量Huber损失。其式14给不等样本数阈值形式 `Ti=T0+M/sqrt(n_i)`。式29为未加样本权重的示例Weiszfeld更新；按式5推广应乘n_i：`a_i=n_i*min(1,Ti/r_i)`，`s_next=sum(a_i X_i)/sum(a_i)`。Ti需在该次求解内固定。原训练算法上传同一global点的梯度。[arXiv完整论文](https://arxiv.org/abs/2308.12581)

旧849–872行每个内循环重新以residual median+1.345MAD选cutoff，且不接收counts；它不是最小化固定的原文样本加权目标，12次也不等于收敛。

下一步写小型求解器返回收敛/残差/目标轨迹；对独立凸优化数值解比对，检验大Ti退化加权均值、r=0、异样本量和截断迭代未收敛。参数Ti或T0/M用验证集选择。之后复用严格Fed-NGA梯度数据路径；如输入Adam多步delta则明确写aggregation adaptation，不能继承原梯度理论。未找到作者官方代码不代表代码不存在；完整公式足以启动独立重实现。

## FLGMM：代码充足，但先解决状态和上游差异

[原论文出版社](https://www.sciencedirect.com/science/article/pii/S1566253525006414)，[作者仓库](https://github.com/HantaoZhu/FLGMM)。commit `a064c82a4bc460e168ce09e0654924a39d258bcc`，本地 `flgmm_official.py`。

作者实现40–48行选GMM中**样本最多簇**，而不是最小均值簇；324–340行对当轮FedAvg中心算距离、按该簇标准化并累计history；56–66行建立 `UCL=mean+L*std`；warmup后SPC监测。旧642–675行每轮对coordinate-median距离拟合GMM、选最小均值簇并逆距离加权，省略状态阶段，明显不一致。

需裁定的官方代码疑点：417–420行第二次GMM取得bounds_2却仍使用bounds；439–444行当轮超限写入excluded，而459–465行聚合使用旧excluded_clients。不能静默修复后称与官方完全相同，也不能故意保留bug弱化基线。应先按论文伪码确立预期，并同时保留上游版本/修正差异。

最小适配是stateful对象保存标准化距离history、warmup阶段、UCL；剥离绘图和原数据加载。门检必须跨warmup边界，验证某客户端后期新攻击会改变实际聚合集；GMM退化/零std、空集、重启恢复等价也必须覆盖。

## FedWA：身份正确，规格不足不能凭空补算法

[AAMAS2022原文](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf)已下载并全文读。原文明确称FedWA，描述从全局/局部模型学习客户端贡献并由DRL决定权重；因此[28]与FedWA名称匹配。但3页extended abstract没有足够actor/critic结构、状态编码、reward、优化器和训练过程细节；本次未取得作者代码。旧875–892行AdaAggRL是固定0.45/0.35/0.20系数的root cosine、center cosine、stability组合，无DRL状态或学习。

下一步继续寻找作者扩展稿/代码及关联项目；取得细节后再写独立adapter。若只能设计自己的RL权重器，必须独立方法名并披露设计，不能把其结果放FedWA原法行。缺规格是当前证据限制，不是GPU或数据障碍，也不需要重跑旧结果来掩盖。

## SmartFL：不要把摘要级相似机制当完整算法

[Dong等原论文出版社](https://www.sciencedirect.com/science/article/pii/S156625352500627X)公开内容支持两阶段：成对排除分歧大的归一化local models建立pseudo-global；再取与其正相关的local models。旧713–735行是所有pairwise cosine的行均值top-n−f、再正cosine召回和cosine加权；第一阶段删除规则、是否需要f、原模型/差分语义及最终权重尚未对原公式证实。

本次未获取作者算法全文/官方代码，不能将此旧启发式认证为原法。下一步以DOI `10.1016/j.inffus.2025.103555` 取得完整第4节/附录伪码或作者仓库。门检需要手造多数簇+两个反向/高范数客户端，精确比对第一阶段每步移除对、pseudo向量、90度边界、第二次召回和最终加权。不要误接同名的其它SmartFL项目。

## FedDNA：缺的是行为特征，不是再加一个MAD

[Garg等原论文出版社](https://www.sciencedirect.com/science/article/pii/S2214212625003941)说明使用激活层模型指纹、客户端行为历史一致性及MAD阈值。旧762–787行只有模型参数距离、update范数、root cosine三个当轮标量的MAD异常分；无激活forward hooks、指纹探针、历史state，关键机制缺失。

下一步取得DOI `10.1016/j.jisa.2025.104358` 全文具体指纹层、探针数据、归一方式、历史更新、MAD multiplier、筛选与权重；再设计固定CNN激活采集和每客户端历史checkpoint。门检采用参数范数相近但激活行为突变的模型，以及稳定异质客户端，证明实际检测读取行为历史而非参数捷径。无这些定义时不能用猜测实现填原法行。

## 可执行顺序与通用门检

1. **立即可启动实现**：LASA和自定义cosine hybrid；无需新训练证明来源，先固定输入一致性。
2. **公式足够、共享路径有收益**：Huber聚合求解器，与Fed-NGA严格梯度路径一起接入；原法/多步delta适配分开标记。
3. **官方代码足够但需消解歧义**：FLGMM状态机及上游代码差异。
4. **继续补来源规格**：FedWA、SmartFL、FedDNA；公开摘要/标题不够，不能“core reproduction”直接晋级。

每项进入真实图像门检前固定source hash、参数、数据访问、客户端消息语义、输出模型身份和选择规则；小数据门检只能确认实现与接口，不得当性能排名。最终多seed五场景双分布表保留全部负结果，列数扩大不等于忠实基线数量自动扩大。
