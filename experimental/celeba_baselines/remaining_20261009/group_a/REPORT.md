# Group A：四方法可执行组件与剩余真实图像验收（2026-10-09）

已经完成四法可执行组件、CPU 门检，以及 CosineFairnessHybrid 独立 worker 的实际 CNN 合成图像管线接入。**未启动真实训练，未生成 CelebA 科学结果，未修改原冻结 worker 或任何旧结果**。本目录由 group A 独占。组件接口在 `adapters.py`，Hybrid worker 在 `worker.py`；验收证据在 `gate_results.json`、`logofair_calibration_gate.json`、`integration_acceptance.json` 与 `acceptance_interface_gate.json`。

| 方法 | 当前实际交付 | 正式接入还缺什么 |
|---|---|---|
| Fed-NGA | 同一 global 点的完整客户端梯度；原式9聚合；有符号参数步长 | 梯度上传攻击语义及真实图像/双GPU门检、独立服务器步长搜索 |
| Huber-BRFL | 固定 Ti、样本数加权的向量 Huber 目标；有界收敛求解与诊断 | 同上；还需冻结 Ti/服务器步长/投影策略，拒收不收敛任务 |
| CosineFairnessHybrid | 独立标签 worker、严格验收接口、32条准备清单；实际 CNN/localAdam/攻击/评价合成管线通过，三轮新旧轨迹与 checkpoint 逐位一致 | 真实 CelebA/GPU 门检、协议审阅冻结与70轮验证搜索；目前 PREPARED 未运行 |
| LoGoFair-DP | 调用固定官方本地/全局 DP 优化器及40个 BetaCalibration；原 default 校准分支通过 | **中心验证集没有真实 client ID；映射尚未解决，真实图像门检未完成**；不能使用 dummy ID 或自行改变统一评估协议 |

## 来源与方法身份

Fed-NGA 以 [NeurIPS2025最终论文](https://proceedings.nips.cc/paper_files/paper/2025/file/29c71a71cf3b354036d9b413cd049cbf-Paper-Conference.pdf) 为准，式8–10、Algorithm1。最终稿明确允许同点随机子集梯度；全客户端经验风险梯度是子集取全集的情况。因此我们采用完整数据、按 minibatch 累加而不做任何 local optimizer step。它和旧 `fed_nga_aggregate` 的“Adam多步 delta 归一 + median norm恢复”不是同一算法。此前 `tmp/celeba_baselines/fednga/fednga.py` 式9组件直接复用，不另建公式实现。

Huber 使用 [AAAI2024正式论文](https://ojs.aaai.org/index.php/AAAI/article/download/30181/32095) 式5–6固定目标与阈值族 Ti=T0+M/sqrt(ni)。`huber_center` 求解 `sum_i ni * Huber_Ti(||s-gi||)`，每次求解 Ti 完全固定，更新权重为 `ni*min(1,Ti/ri)`。诊断中的 objective 除以总样本数，仅做公共缩放，最优解不变。原文实现节未加样本权的 Weiszfeld示例，按正式式5的目标显式加入 ni。旧 worker 内循环每步重估 residual MAD cutoff 且忽略 counts，不认证为此原方法。投影若采用 W=R^p即恒等，需要协议明示；没有额外裁剪冒充投影。

LoGoFair 使用[作者官方仓库](https://github.com/liizhang/LoGofair)，commit `7044815cf813cdad53fca7bab426c4d2194ab505`。沿用独立 `logofair/adapter.py`，运行官方 AST方法，不复制无许可证仓库源码进本项目。官方客户端源码LF SHA256=`244e8bd3f3a6f5b4945cd8cf93035d5996954acc7f5b61e41e89651d9f88fba2`。保留每 client 两个 local_mu、每轮平均 global_lambda、本地 true_H细化、每 client/group 两阈值和官方 beta校准。只暴露 DP，不声称 EO实现完成。校准 priors仅由 calibration样本计算，明确修正上游 train+valid+test计数除valid分母问题；非与 unmodified main.py逐位等价。

CosineFairnessHybrid 是本文自建控制。对每个攻击后 client model 在干净 root计算 AEOD比例，构造 `trust=ReLU(cos(root_delta,client_delta))*exp(-lambda*AEOD)`；严格选 `trust>tau`，空集取首个 argmax，最后**等权平均**。新独立标识不得显示为 GuardFed-AD2+。旧 Table II cosine行混用了 GuardFed/AD2身份，本次实现不会追认旧行身份正确。

## 可执行接口与最小接入

```python
from adapters import empirical_gradient, fednga_step, apply_parameter_step

# clients已经使用冻结分区与数据变换；轮内global model不做local optimizer step。
gradient_rows = torch.stack([
    empirical_gradient(global_model, client, batch_size=64, use_reweighting=False)
    for client in clients if client["n"] > 0
])
counts = [client["n"] for client in clients if client["n"] > 0]
# 必须先用已验收、已冻结的梯度上传攻击转换替换恶意rows。
step = fednga_step(gradient_rows, counts, server_eta)
apply_parameter_step(global_model, step)  # step已含负号，不再取反。
```

参数顺序由 `parameter_layout(model)` 固定，只包含 trainable named_parameters，不把 buffer当gradient。现有RGB64 CNN没有BN/dropout。接口拒绝其他模型含BN/dropout时静默采用新状态策略；不改变调用者的 `.grad`、global参数与原训练模式。完整gradient按每batch损失**和**累积，最后除客户端总样本数，不对不等长batch均值再平均。零样本client需显式省略并记录（n=0本来也不占权重）；零梯度客户端保留权重并贡献零方向，这是式9未定义零除的明确扩展。

若为与统一 CelebA 预处理保持可比而 `use_reweighting=True`，目标明确变成 `sum(weight*CE)/sum(weight)`，完整客户端分母只用一次；它不同于原旧worker逐batch做重加权后Adam多步。须写进配置/manifest/方法注释，不能无声称原实验超参复刻。**若关闭重加权且 Male不作为输入，F-Flip仅改变敏感元数据，梯度可能与Benign完全一致，这是预期机制，不能为制造攻击效果另改标签。**

```python
from adapters import huber_center, huber_thresholds
thresholds = huber_thresholds(counts, t0=T0, m=M)
center, diagnostics = huber_center(gradient_rows, counts, thresholds,
                                    max_iter=1000, tolerance=1e-9)
if not diagnostics["converged"]:
    raise RuntimeError("Preserve solver failure; do not accept this run")
apply_parameter_step(global_model, -server_eta * center)
```

Huber solver内部使用float64求解，返回原输入dtype；记录 objective_trace、stationarity_l2、Ti、最终weights、迭代数与converged。正式日志应保存至少 convergence、stationarity、objective、iterations和thresholds；无需把全部trace重复写进每个指标表。验证完整70轮不能以单次CPU小向量测试代替。

```python
from adapters import cosine_fairness_hybrid
update, info = cosine_fairness_hybrid(updates, root_aeod, root_update,
                                     fairness_lambda=20., threshold=.2)
# 常规原 apply_update 做模型state加法，保留其消息语义。

from adapters import fit_logofair
post = fit_logofair(root_prob, root_y, root_sensitive, root_client_id,
                   calibration=True, global_delta=.02, local_delta=.02)
prediction = post.predict(valid_prob, valid_sensitive, valid_client_id)
```

LoGoFair必须使用 CNN正类概率 `softmax(logits)[:,1]`；fit只用干净训练root校准标签，predict不接收评价标签。当前中心valid没有 client_id，因此上例中 `root_client_id/valid_client_id` **尚无被批准的真实映射**。可审阅选择是：保留真实分区client的独立holdout及同源评价分区；或明确新增、冻结、披露虚拟client映射适配。前者涉及训练/评价协议变化，后者改变“local”总体含义，都不能由此组件自行决定。本接口要求caller提供IDs，unknown ID和缺 sensitive group会失败，不把所有样本随意分给dummy client。每组BetaCalibration还要求两种label；失败证据保留，不自动池化、更改beta或关闭校准。

## 已执行的门检及边界

`python tmp/celeba_baselines/remaining_20261009/group_a/check_adapters.py` 已 PASS：

- 线性 CE手算真实梯度、不等长minibatch、重加权分母、输入参数及已有.grad不变；式9有符号参数更新；非有限/溢出参数步长拒收且不留下部分参数修改。
- 使用现有 `CelebACNN` 的5张**合成 uint8 RGB64**输入，整批与分批梯度一致；global checkpoint不变。没有真实CelebA样本，无科学性能主张。
- Fed-NGA非等样本量二维手算、正尺度不变、抵消、零方向分母保留。
- Huber与独立SciPy BFGS目标/梯度解一致，目标单调，大Ti加权均值极限、全同/零残差、截断不收敛及非法Ti/counts。
- Hybrid 60组输入直接执行旧 worker AST做逐位回归；严格阈值等号、空集首argmax、NaN公平值、零向量、等权/信任权区别与AEOD单位检查。
- 官方 LoGoFair DP20client/1600score的确定性重放、不同client阈值、本地与全局约束独立起效、未知client拒收。

默认beta校准门检额外运行：

```powershell
& 'tmp/celeba_baselines/remaining_20261009/group_a/.venv/Scripts/python.exe' `
  'tmp/celeba_baselines/remaining_20261009/group_a/check_logofair_calibration.py'
```

`calibration=True` 的40组官方BetaCalibration MLE拟合全部有限，重放阈值与预测逐位一致，校准改变阈值，PASS。环境是仅在本目录创建的system-site-packages虚拟环境：Python3.10.4 / Torch2.8.0+cpu / netcal1.3.6 / NumPy1.26.4。**主训练环境未修改。** netcal1.3.6依赖旧matplotlib，隔离环境继承的OpenCV声明需NumPy>=2，存在一个未用依赖冲突；这些门检不导入OpenCV。应在服务器独立评分后处理环境复现，不把该venv当完整训练环境复制。

## 最少候选与剩余门检

`candidate_configurations.json` 是明确标 `PROPOSED_NOT_FROZEN_NOT_QUEUED` 的建议，每法8候选；不是manifest、不是训练授权记录。推荐沿用已冻结的 seed91001、IID/non-IID、Benign/S-DFA四条件 valid-only搜索口径，在母任务裁定训练/攻击/映射协议后再冻结。所有方法同一选择规则、终轮同checkpoint三指标；完整多seed覆盖保留负结果。

LoGoFair可在每条件已验收FedAvg同终轮checkpoint的score cache上尝试8套后处理，原则上无需重训CNN；必须先解决ID映射并保存root/calibration/eval样本身份和校准state。Fed-NGA/Huber不能复用FedAvg梯度轨迹；需要真实同点梯度训练。Hybrid需按其筛选逻辑重训，不能把已有AD2+ checkpoint换名。

尚未完成：真实数据攻击消息验收、除 Hybrid 外其余三法的 server worker接线、GPU/70轮/多seed科学训练、LoGoFair中心验证ID适配、冻结final评价。本次无SSH重试、无队列启动、无方法参数/实验协议自动变更。组件通过不等于这些科学任务已完成。

## Hybrid 独立 worker 与32条准备清单

`worker.py` 只在独立载入的冻结 core 内替换 `aggregate_round`，调用上述已门检 Hybrid 组件；非本方法调用原聚合。冻结 core 位于 `tmp/revision-publish-20260928/scripts/reproduce_paper_tables.py`，SHA256=`cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed`。没有复制或修改 core 的 CNN、local Adam、root Adam、数据分区、攻击、模型更新或 native argmax 评价。

公平值必须等于 core 的 `fairness_details[client]["aeod"]`，来自**攻击后 client model** 在干净 root 的评价；NaN 按旧实现置为1。消息是原 core 的 model delta；root 参照是原 root localAdam delta。聚合仍严格阈值筛选后等权平均，不按样本数或 trust 再加权。标签是 `CosineFairnessHybrid`，方法说明明确为自建 legacy GuardFed mechanism control；不冒充 GuardFed-AD2+。

`protocol.json` 与 `screen_jobs_draft/manifest.json` 均为 **PREPARED_NOT_FROZEN**，`execution_started=false`。草案为8候选（LR∈{0.0005,0.001} × lambda∈{5,20} × tau∈{0.1,0.2}）× IID(alpha5000)/non-IID(alpha5) × Benign/S-DFA，共32任务；每任务70轮、seed91001、完整 valid19867/train162770。共同参数来自只读 group B 准备协议，与先前已验收 FedAA 图像设置一致：20client/4malicious、local epochs1、batch64、重加权开启、Male仅作元数据、S-DFA沿冻结 FedSA delta攻击。选择规则为候选四条件既有 score 均值，完全相同则按 candidate 字典序；保留全部候选、准确率冠军与三指标 Pareto，不称四条件为四个 seed。

可执行入口：

```powershell
python tmp/celeba_baselines/remaining_20261009/group_a/worker.py `
  --repo <已核验的仓库根目录> --job <冻结后job.json> --out <全新输出目录>
```

**当前草案会被入口拒绝，且在创建输出前拒绝。** `prepare_jobs.py` 仅生成草案，不会冻结或启动。协议最终冻结会改变 protocol SHA，须由主任务审阅后重新生成对应 job/manifest；不能手改状态后沿用旧 job hashes。worker要求输出目录不存在，保留部分输出与失败证据，不承诺中轮恢复。运行完成保存 checkpoint、逐轮诊断、result、provenance 与验收 SHA receipt；失败保存 `failure.json`。

在调用端将本目录加入 import 路径后，使用 `accept_result.checked_result(Path(job_path), Path(output))`。它核对冻结身份、源码/数据/adapter/protocol/job哈希、完整1..70轨迹与轮摘要、split及样本数、seed/alpha/攻击、终轮三指标一致、所有轮筛选/公平值/参数诊断、攻击审计、非有限值、checkpoint载入与artifact SHA；存在失败证据即拒收。未完成返回None，不把部分任务当完成。

## Hybrid 管线和验收接口的实测证据

`check_worker.py` 已 PASS：新 worker 接口、确定性重放、旧 GuardFed 控制各运行3轮，共9轮。输入仅是20client×4张合成 uint8 RGB64、8张 root、16张评价图，运行实际 `CelebACNN`、localAdam/rootAdam、S-DFA敏感元数据翻转及FedSA消息攻击、native评价。

- 每个三轮运行核验60次 client localAdam与3次rootAdam；逐个核验攻击后模型等于 global+attacked delta，以及60次 rootAEOD来源。敏感元数据翻转不改变真实分类标签，恶意 delta 确实发生攻击转换。
- 聚合消息、strict阈值、空集回退及等权均值与旧 GuardFed逐轮逐位相同；新/重放/旧分支的全部最终模型张量、轨迹、攻击审计逐位相同。
- 三轮 progress完整，终轮指标等于轨迹末行；保存checkpoint重新载入评价完全一致。AD2+校准函数在门检中设为失败钩子，确认本方法未调用该分支。
- 32个草案身份/参数/哈希核验通过；当前 PREPARED 入口拒绝，未生成正式输出。

`check_acceptance.py` 已 PASS，专门验证拒收规则：更改seed/终轮/split/alpha/阈值、缺轮、测试集冒充验证集、终轮指标不一致、选中集合或参数错误、checkpoint SHA损坏、保留失败证据均拒收。测试仅在临时目录构造一个**合成的70行结构fixture**来覆盖完整结构路径，测试后已删除；它不是70轮训练证据，没有修改真实 PREPARED protocol，科学结果计数为0。

首次管线门检在已经执行三轮后，检查脚本错误地要求所有 benign attack audit都有 `label_changed_count`，实际 core只对翻转客户端记录该字段，导致门检KeyError。修复为benign缺省0，同时在每个攻击钩子直接检查原始 y没有改变；失败模型/诊断和修复说明保存在 `failed_gates/20261009_label_audit_assertion/`。修复前准备清单保存在 `prepared_history/`，当前清单已按修复后验收源码重新生成。没有科学结果受到影响。

主任务下一步是核查真实 CelebA/服务器环境，执行真实小轮门检，再审阅并冻结该32任务协议。Fed-NGA/Huber攻击语义与LoGoFair client-ID仍需独立裁定，本次没有绕过这些未决点。
