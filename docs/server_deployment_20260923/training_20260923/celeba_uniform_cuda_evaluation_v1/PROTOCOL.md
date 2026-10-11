# CelebA 完整比较范围统一 CUDA 重评 v1

作者于 2026-10-11 明确批准统一 CUDA 重评完整比较范围，包括 Full。批准记录见同目录 `AUTHOR_DECISION_20261011.json`。本文件是已冻结机器清单与实际执行记录的阅读入口；不把本文件的生成时间冒充实验前冻结时间。

## 比较与设备

所有方法采用各阶段已经选择并冻结的 recipe、seed 和第 70 轮 checkpoint。只重做评价，不重训、不重新选择 checkpoint、不改参数或原生复现容差 `1e-12`。clean train root 与 valid 图像的 CNN 前向统一为原训练路径的确定性 FP32 CUDA；关闭 TF32、benchmark，保留原图像顺序和 batch 规则。后处理使用原冻结实现及设备：GuardFed 的 NumPy 阈值拟合、LoGo 的 CPU netcal 拟合均保留。

评价集为 valid，19,867 张；原训练数据 162,770 张。只用 clean train root 拟合校准，不能用 valid/test 选择阈值。历史官方 test 已暴露，不能声称本次 valid 是未触碰 test。本阶段不授权 finaltest。

每模型同时保存 native、raw、shared-calibration 三视图的预测、计数与指标，所有视图来自同一个终轮 checkpoint。AEOD 在项目中是绝对 TPR 差，不是完整 equalized odds。保留失败、恒定预测与负结果，不选择有利 seed，不混合 checkpoint 或指标。主端点仍待作者决定；本阶段不会自行把某个视图定为唯一主结果。

## 实际阶段边界

完整目标为 17 方法 × 2 分布 × 5 场景 × 10 seed，加 8 类机制控制 × 100，共 2,500 条逻辑记录；Full 的 100 条已经包含在方法表中，不重复计算。该目标不是已经齐备或已经验收的清单。

当前已经冻结、实际启动的首阶段仅 500 条：Full 100，加 minus_U/minus_C/minus_A/minus_F 各 100。使用 IID（alpha=5000）和 non-IID（alpha=5），Benign、F Flip、FedSA、S-DFA、Sp-DFA，seed 91001–91010。清单源：`tmp/mechanism_uniform_cuda_scope_prepare_20261011/EXACT500_INPUTS.json`，SHA `cebaae3603c391a31db982a31304bd836c8ef88d45574efde87366c110824a2e`。实际物化记录与原 checkpoint/job/result/data/source 身份逐条绑定；不以历史 CPU 已完成为由跳过 CUDA 评价。

原调用器、原 CUDA 科学体和原 Linux 保存结果验收循环保持可追溯。首批固定七项包括 Full、四类控制及此前 CPU 失败模型的配对 Full/控制。七项原生三指标差值均为 0，原 Linux whole 验收后，通过离机零拟合审计及 F 盘归档全部 110 成员 SHA；root 采用记录：`tmp/uniform_cuda7_saved_audit_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json`。这是 7 项结果，不能称 500 项完成。

## 中断与有限恢复

首次队列于 2026-10-11 06:23 UTC 启动，累计 10 项完成原 Linux 保存结果验收。在 Full/IID/Benign/91005 的后置资源检查，读取 `/proc/113052/cwd` 发生 PermissionError。该项已产生原生差值 0 的 receipt，但缺少正式 binding/ACCEPTED，因此保留为未验收，不能直接采用。诊断只证明这个可观测错误，未证明该进程当时一定处于 zombie 状态。

修复仅允许在 PermissionError 后当场再次核实进程已消失或处于 Z/X 时跳过；仍存活、不可读或无法判断时继续拒绝。模型前向、校准、统计、容差及资源配额均不变。恢复须在新 namespace 中，原 10 项 CUDA 结果逐条核原 Linux ACCEPTED、binding、receipt、预测数组、源/数据/模型哈希与 stat 身份后才可复用；原失败输出不覆盖、不直接采用。对失败项仅一次明确恢复，然后依原清单顺序完成其余 490 项，遇到新失败即停止，无自动重试。

## 资源、验收与产物

服务器为 `ssh -p60350 root@89.22.197.55`，repo `/workspace/GuardFed-celeba-expanded`，Python `/workspace/guardfed_envs/celeba-cu128-20261009/bin/python`。执行前遵守 `/etc/vast-agents-guide.md`，核源码、GPU UUID、RecoveryAction、CPU/cgroup、内存、磁盘、保护服务及重复进程。评价采用一个 CPU107 worker/GPU0，nice10/idle IO；原机制训练、Hybrid 和 NGA 队列继续独立运行。不能为提高占用更改数值路径或训练协议。

服务器保存模型与原始结果，本机只按验收差集取最小预测/receipt/配置/日志证据，写入前核 `F:/YananResearchStorage/` 为 Yanan 2TB；不得回退内置盘存放批量数据。归档与每个成员均核 SHA，保存原失败和完整恢复链。

论文表必须等相应场景 10 seed 完整验收再生成，使用均值 ± sample SD（ddof=1）；另列排除配置选择 seed91001 的 9 seed 和未参与早期搜索的 91005–91010 六 seed。跨场景先在 seed 内平均，再跨 seed 统计，不把场景充当独立 seed。披露 2 个 Full 的 cu130 历史训练身份，其余首阶段记录为 cu128；统一重评不证明两种训练环境数值等价。旧 CPU 表完整保留并明确标识，不能与新 CUDA 表拼成同一设备协议。
