# Fed-NGA / Huber 真实图像门检：准备交付

状态 **PREPARED_NOT_FROZEN**。本目录只准备4条真实全量CelebA三轮CPU探索门检，当前实际运行0、训练轮次0、科学表记录0。未部署/启动服务器门检、未安装包、未更改64候选正式草案或其5项未决协议。Hybrid PID10424及CPU8–15由现有任务继续使用，本实现拒绝分配这些CPU。

## 有界pilot选择与正式决策边界

| 项目 | 本次门检明确选择 | 依据与尚未批准的正式问题 |
|---|---|---|
| 经验风险 | `original_unweighted_ce`，每client全部样本CE总和/n | 与原算法样本经验风险一致；是否另报共享重加权公平预处理，仍未批准 |
| 上传攻击 | 正梯度上传 `g_attack=-A(-g, r)`；`r`为原clean-root localAdam delta | 保留当前攻击者root算法/访问；不等于旧训练轨迹。根梯度参照是另一个未批准的威胁选择 |
| 全局步长 | 两法独立`eta=.03`；Huber固定`T0=.1,M=1`，`Ti=T0+M/sqrt(ni)` | 直接选已准备网格中的有限内部尺度，用于诊断梯度/步长/求解；不是已调优最佳参数，不从Adam LR推导 |
| Huber参数空间 | 显式`W=R^p`恒等投影 | 当前CNN适配，不主张原凸/有界理论成立；正式空间选择未批准 |
| 真实管线 | IID Benign、non-IID S-DFA各3轮，各法共4运行 | 将来执行后仅证明这四条短程CPU管线；70轮稳定性、GPU/方法比较仍需独立验收 |

原5项正式决定在封存`protocol.json`中全部保持`UNRESOLVED`，本目录不批准它们。正式64草案的`validate_job`仍需先通过原freeze门。本独立门检使用显式探索scope与单独资源/dispatch收据，不能当作正式搜索获准。

当前完整五场景实际路由为：Benign无foe，F Flip仅元数据，FedSA显式fedsa，S-DFA/Sp-DFA用各自fedsa覆盖。默认`foe_mode=state`未进入这些上传路径，**不是五场景矩阵阻塞**；只有未来显式请求state型梯度攻击时才拒绝。未加权CE的F-Flip-only零效果是实际机制含义，不能改标签强造效果。门检不新增F-Flip科学结论。

## 实现与拒收边界

`gate.py`直接调用封存梯度worker的真实经验梯度/官方聚合组件；client optimizer步数为0。完整official train162770拆分clean root16277与client146493，valid19867，RGB64 CNN、20client/4malicious、seed91001、alpha5000/5、70轮旧协议其余相关项保持。本门检只将horizon设3轮和CPU，原root Adam LR=.001仅用于攻击参照。

每轮记录60条累计真实client梯度：共同参数点SHA、梯度/上传SHA、原始样本数、CE分母、攻击类型/模式/范数。攻击oracle在零原点消息坐标调用冻结core的`apply_foe_if_needed`，严格比对符号共轭输出；这是实际上传消息的代数oracle，**不是构造local模型或把Adam差分当梯度**。clientAdam、root候选公平推理和阈值校准路径被显式拒绝。Fed-NGA实际矩阵另算式9并比对；Huber保留固定Ti、全部目标轨迹、终止stationarity、最终权重及FP32落地舍入界；未收敛在参数更新前停止并保留failure。

验收核完整3轮、全部client、真正alpha、完整样本/组标签支持与image-ID SHA、源码/缓存/原800manifest+adapter+协议+dispatch哈希前后一致、有限终轮模型、同模型native三指标重预测一致。每轮保存CPU/RSS/显存温度/正式800真实轮次快照；整体通过须见正式队列增长且无失败。部分输出/失败标记均拒绝盲续跑，不自动删目录或重试，不读取test、不增加服务或GPU进程。

prepare/inspect是纯本机文件检查，不导入torch、不训练。运行必须有独立匹配scope/job/source SHA的`APPROVED_BOUNDED_EXPLORATORY_GATE_ONLY`收据及8个明确、排他CPU编号；当前模板`dispatch_receipt.PENDING.json`不可执行。主代理先确定资源，核其他CPU门检affinity无重叠后才可提供收据，本代理未代填批准。

## 可执行接入

本机准备与检查（已准备时不重跑prepare覆盖快照）：

```powershell
python tmp/celeba_gradient_realimage_gate_20261009/prepare_gate.py --project .
python tmp/celeba_gradient_realimage_gate_20261009/check_preparation.py
python tmp/celeba_gradient_realimage_gate_20261009/gate.py inspect
```

主代理审阅后，将整个目录原字节复制至`/workspace/guardfed_checks/celeba_gradient_realimage_gate_20261009`，完整核`SETUP_SHA256.json`，先读取服务器guide并核scope中所有支持身份。如历史输入缺失，仅准确恢复相同SHA，不能放宽。确认CPU预算后另存匹配批准收据，不改scope/原protocol。以下命令是**待执行入口**：

```bash
PY=/workspace/guardfed_envs/celeba-cu128-20261009/bin/python
G=/workspace/guardfed_checks/celeba_gradient_realimage_gate_20261009
"$PY" "$G/gate.py" inspect
"$PY" "$G/gate.py" run --dispatch-receipt "$G/dispatch_receipt.APPROVED.json"
"$PY" "$G/gate.py" summarize --dispatch-receipt "$G/dispatch_receipt.APPROVED.json"
```

只起这一个前台/受控后台CPU进程，保持8计算线程；命令自身不创建监督机制。运行前仍须核对已运行Hybrid和FLGMM等CPU任务，不能直接套当前示例编号。通过后按actual pilot梯度范数、step/model尺度、Huber收敛及耗时再审阅正式5项决策与搜索尺度；失败则保留证据分析，不能自动调整阈值/方法/seed重试。即使全部PASS，也不自动启动64screen、test或其他队列。

来源与方法适配限制沿用字节封存`gradient_bridge_20261009/REPORT.md`和group_a组件，不重复来源/合成实验审计。本交付的本机检查仅证明准备身份和关闭dispatch门，不声称真实CelebA训练完成。
