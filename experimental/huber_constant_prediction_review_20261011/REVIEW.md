# Huber 恒负预测：有限只读诊断

**未发现明确的符号、维度、样本数缩放或不收敛结果被接受的实现错误；但这不等于证明该 CNN 基线已充分调参。** 当前证据更符合小幅全批梯度更新未在70轮内学出分类边界。此为可检验解释，尚不能定为根因，更不能把零差异指标称为公平性胜出。

## 实测范围与关键证据

本次只读根已接受46项中的14个Huber结果，来自 `tmp/gradient_native_after32_20261011`（7）、`after39`（3）、`after42`（4）的 `RAW_STORAGE_INDEX.json` 所指 F 盘 `restored/runs/*/result.json`；14个小文件分别与既有member SHA相符。未读模型/图像、未重新执行验收或推理。

- **14/14均为server_eta=0.03，seed均91001**；覆盖T0=.01/.1、M=0/1的部分四条件组合。这是单seed、多个配置/条件，绝不是14seed。冻结32项Huber搜索另含eta=.3，尚不在本次接受集合，不能由当前14项判定全部候选失败。
- 14项终轮positive_rate均0、prediction_count均19,867；ACC=.5166859616449389恰等于majority_accuracy，AEOD=ASPD=0。每项70轮所存三指标不变。
- 980条原轮诊断均标收敛；驻点残差最大9.997557364980832e-10（原门1e-9）。每项有70个不同global-point hash，更新L2非零，范围0.0002625169368704572–0.0008413370876390392。**不是参数完全未更新，也不是求解器返回零步。** 驻点收敛只说明本轮聚合目标已解，不能说明CNN训练收敛。
- 更有区分力的对照：IID Benign、T0=.1的M0/M1两项，全部70轮IRLS迭代数为0，最终权重与ni/N最大差6.94e-18；即初始样本加权均值已满足驻点门，仍恒负。故“鲁棒裁剪过强”无法单独解释这两项；未涉及攻击的Benign也失败，不能只归因S-DFA。

## 方法忠实性与尚未证明的内容

冻结源：[adapters.py](E:/OneDrive/文档/GuardFed/tmp/celeba_gradient_screen64_v2_20261010/snapshot/remaining_20261009/group_a/adapters.py:35)第35–79行按客户端样本求完整CE梯度，逐batch求和后只除一次ni；第83–98行按相同trainable parameter顺序加已带符号步长，并检查维数。第108–177行求固定Ti的样本加权向量Huber目标，ni除总量只作公共缩放，Ti=T0+M/sqrt(ni)。[worker.py](E:/OneDrive/文档/GuardFed/tmp/celeba_gradient_screen64_v2_20261010/snapshot/gradient_bridge_20261010/worker.py:160)第160–182行采用 `step=-server_eta*center`，第236–260行检查各客户端确在未改变的同一global点求梯度，再仅应用一次参数步。未见额外除参数维数、batch数或重复除ni。

本地作者论文文本 [huber_arxiv.txt](E:/OneDrive/文档/GuardFed/tmp/celeba_baselines/remaining_fidelity_20260928/huber_arxiv.txt) Algorithm1、式3–6、阈值式14，与上述核心对应；实现节无ni的示例式29针对未加权目标，当前显式ni对应其正式式5。已有 [group_a/REPORT.md](E:/OneDrive/文档/GuardFed/tmp/celeba_baselines/remaining_20261009/group_a/REPORT.md:16)引用AAAI最终稿。此次独立核对的是现存作者文本与冻结代码，并未重新获取最终PDF或官方训练仓库，故不声称完整官方实现逐位复现。

适配器/worker/protocol的小文件与已收集源成员逐一SHA一致（55ca0f…3125c0、f16070…df0ab4、c5010a…aa496）；未修改冻结源。identity_Rp已获作者授权，是CNN实用适配，不继承原论文的约束域/平滑性等理论保证。70轮仅70次完整梯度全局更新；不是70轮localAdam多步优化。Fed-NGA则逐客户端单位化梯度（`snapshot/fednga/fednga.py`第38–44行），因此相同eta并不代表相同更新幅度，不能把它的有效步长直接移给Huber，也不能加范数恢复冒充原法。

## 判断与最小下一步

**最值得检验的是优化预算/有效步长不足，尚非已证实bug。** 非零而很小的参数步、Benign均值梯度极限同样退化、所有已接受项只覆盖小eta，都支持这一解释；现有compact记录没有训练CE和逐样本margin轨迹，不能区分缓慢学习、初始化偏置主导、特征学习停滞或其他数值效应。完整梯度与样本加权正确不保证当前搜索范围足够强；现有搜索草案报告本就明确未证明覆盖真实CNN最佳区间。

先按原冻结队列完成/接受全部64搜索，再原样比较剩余eta=.3候选的prediction rate、步长和Benign结果，不提前选recipe或启动192覆盖。若仍普遍恒负，下一项应是**单个Benign同初始化、训练/root-only的短诊断**：记录更新前后训练CE、margin分布、参数变化，并在T较大时与独立样本加权全局梯度对照；优先辨别下降方向和有效幅度，之后才决定是否需要有界eta/训练预算协议修订。该新增诊断和调参未在本次执行，必须保留原32候选与负结果，不因GuardFed预期领先而删改。无需为当前退化结果停止健康队列；也不得把当前14项当作Huber充分调优后的最终比较。

本次操作：只读代码、原文和小结果/诊断；SSH、模型读取、fit/forward、训练、recipe选择、STATE/Git改动均为0。
