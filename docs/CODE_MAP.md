# 代码地图与方法边界

## 共用核心

`scripts/reproduce_paper_tables.py` 集中保存 `ExperimentConfig`、`run_experiment`、攻击、聚合与后处理。阅读时按函数定位，避免沿用旧 README 的公式或行号。

| 问题 | 函数 / 文件 |
|---|---|
| 默认值、消融、root 噪声、图像评价 split | `ExperimentConfig` |
| 数据、root 与客户端分区 | `load_bundle`；`src/data_loader.py`；`src/celeba_data.py` |
| AD2 与 AD2+ 聚合 | `guardfed_act_aggregate`；`guardfed_ad2plus_adaptive_aggregate` |
| 聚合分派、完整训练与保存 | `aggregate_round`；`run_experiment` |
| 指标 | `compute_metrics`：accuracy、AEOD、ASPD |
| 分组阈值校准 | `fit_group_thresholds` |
| 单个任务的来源、checkpoint 与终轮核验 | `scripts/run_revision_ablation.py` 的 `worker` / `checked_result` |

这里的 AEOD 实现为两个敏感组 TPR 之差的绝对值；ASPD 为正类预测率之差的绝对值。不要根据缩写自行改成 TPR/FPR 的平均差。分母为零时保留 NaN 与诊断。

`src/algorithms/` 是早期模块实现，不是共用核心中 AD2+ 的替代入口。

## 研究基线

以下路径相对于 `experimental/celeba_baselines/`；保留原文件结构以便核查源码身份。表中“入口”用于定位实现，不保证在空仓库直接执行历史队列。

| 方法 / 控制 | 实现与入口 | 需要说明的边界 |
|---|---|---|
| FedAA | `fullcoverage_20261003/fedaa/`；官方 DDPG 及适配代码见 `deployment/baseline_adapters_20260928/fedaa/` | CNN、root 奖励、客户端选择及 RNG 等项目适配；需要冻结 job / recipe |
| LASA | `fullcoverage_20261003/lasa/`；参考实现见 `deployment/baseline_adapters_20260928/lasa_20260928/` | 保留原筛选及适配；历史 queue 动态读取部署目录 |
| FLGMM | `remaining_20261009/group_b/worker.py`、`flgmm_adapter.py`、`sources/` | 固定作者源码及许可证；不要与 core 内的同名历史分支互换 |
| Fed-NGA / Huber | `gradient_bridge_20261009/worker.py`、`remaining_20261009/group_a/adapters.py`、`fednga/fednga.py` | 上传同一 global 点的真实梯度；不能把 Adam 多步 delta 当同点梯度。Huber 的投影 / 步长 / Ti 以具体协议为准 |
| CosineFairnessHybrid | `remaining_20261009/group_a/worker.py`、`adapters.py` | 项目组合控制，保留独立方法标签 |
| LoGoFair | `logofair/adapter.py`、`logofair_bridge_20261009/bridge.py` | 调用固定官方优化器；官方客户端源码另取，需 `netcal`。DP 表示 demographic parity |

LoGoFair 官方来源为 [liizhang/LoGofair](https://github.com/liizhang/LoGofair/tree/7044815cf813cdad53fca7bab426c4d2194ab505)。预期文件 `logofair/upstream/fedlearn/models/FedFairPostClient.py` 的 LF SHA256 为 `244e8bd3f3a6f5b4945cd8cf93035d5996954acc7f5b61e41e89651d9f88fba2`。本交付未补入该文件；适配器会核验内容身份。

LoGoFair 历史方案使用图像 ID 定义的虚拟群体，不能据此宣称真实训练客户端公平性。保留的部分协议标记为 `PREPARED_NOT_FROZEN`，它们是对应阶段的准备材料；不能作为最终正式协议。其他 CLI 方法标签也不自动获得“原方法忠实复现”的认证。

## 部署与评价快照

`deployment/evaluate_shared.py` 与 `accept_shared.py` 需要原模型、预测分数、manifest 和固定服务器目录。`deployment/baseline_adapters_20260928/` 及 `experimental/` 下部分脚本同样保留 `/workspace/...`、`tmp/...` 和相对部署路径。

交付仅整理已发布源码，未改写上述路径或重算冻结协议的哈希。重新部署时须恢复对应目录、源码身份、外部数据和 checkpoint，再核对 job；不能仅修改绝对路径后沿用旧验收声明。`reuse_manifest.json` 只记录复用来源，不附所引用的大文件。

`scripts/run_revision_ablation.py prepare` 会读取历史 Adult 结果并复用控制；`run_attack_strength_study.py` 缺少匹配校准记录时会触发校准。二者都不是空仓库安装自检。

未收入此分支的批量 job、增量发布材料、论文、PDF、工作簿、日志和科研结果，可在固定[来源提交](https://github.com/YananZHOU5555/GuardFed/tree/37c35d6c954d0a94e4221c378f9405a278c59df5)核查。只读检查脚本也可能依赖这些材料，未将全部脚本声明为独立可运行。
