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

## 2026-10-11 实际执行增量（07:55 UTC 截止）

以下为实际执行后的追加记录，不追改上述首阶段冻结时间。首阶段后续于 Full/non-IID/Benign/91002 的后置资源检查再次发生 `/proc/117589/cwd` PermissionError。其 receipt 和 binding 已生成，但无最终 ACCEPTED，未作为已完成项复用。原首次10项和 resume1 新47项合计 **57项**；47是新增数，不能作为累计数。故障当时的进程类别与状态未能回溯证明，原因不称 zombie 或非Python。

经独立审查，资源检查仅对明确非Python进程免读 cwd，所有PID/task的CPU affinity和名义预算仍核查；存活但不可读的Python及未知进程仍拒绝。科学worker原文、参数、容差、GPU0/CPU107及单线程未变。新 namespace `uniform_cuda_formal_20261011_resume2` 于07:50 UTC实际启动，逐项重核原57项来源、模型、数组、binding、SHA和stat后复用，只执行原顺序剩余443项。07:55实测累计服务器保存验收58/500，失败模型已在新目录通过、原生三指标最大差0；本地严格离机采用仍为首7项。旧失败证据与部分输出全部保留，无自动重试。

另批原八方法800条库存已冻结，cohort SHA `d39e29333c40e8a5b63f1344487225abac4ba83216dc32e75f64badc87bc0b71`：FedAvg、FairFed、Median、FLTrust、FairGuard、FLTrust+FairGuard、FedAA、LASA各100条。Full不在此800内，不与首500重复计数。原788条cu128、12条cu130训练身份保留；本次评价统一设备不等于训练环境等价。原生实现和第70轮checkpoint不变。

该800批的启动模块命名冲突与一次资源读取失败均保留在旧目录。第三目录 `uniform_cuda_formal_original800_20261011_attempt003` 复用第一次真正通过的FedAvg单例，仅新做固定FedAA/LASA两例。三例均通过原Linux完整保存验收（native三指标差0）、离机零拟合27指标/72计数/9预测规则核对及F盘85成员SHA备份，root记录 `tmp/old800_cuda3_saved_audit_prepare_20261011/mixed_producer_v2/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json`。其余797项于07:46 UTC实际启动；07:54实测服务器累计23/800、无失败。本地严格采用3/800，不能把服务器进度直接等同论文验收。

以上两批共冻结1300个互不重复的逻辑记录；本地严格CUDA采用共10（7+3），完整目标2500尚未齐备。FL100运行调用器和LoGo100冻结后处理缓存适配仍在源审查/输入准备阶段，未据方案称实际运行。LoGo需要完整、根采用的FedAvg CUDA100缓存后才可评价；原已冻结CPU netcal后处理状态保持不变，不通过新缓存重新择优或拟合。其余正在训练的方法和四控制需各自完成后增量接入，不启动未经冻结的finaltest。

## 2026-10-11 08:20 UTC FL100实际执行追加

FL100原终轮输入与七包身份冻结（cohort8c16d111…），未回填此前冻结时刻。固定screen IID Benign91001、fullcoverage non-IID F Flip91001和non-IID Sp-DFA91001均通过原Linux完整验收、native三指标差0、离机零拟合27指标/72计数/9规则及F盘77成员SHA核验。root tmp/FL100_cuda3_saved_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json SHA e6db56a24eda9ea9840caba54958e79ee77bb427a4b11804e0b03b2462f8c905。2026-10-11T08:20:21Z在原namespace实际启动剩余97，service guardfed_uniform_cuda_FL100_remaining_valid；原顺序补集、CPU111/GPU0、单线程、原1e-12不变，无自动重试。共实际冻结1400逻辑记录，Full仍只一次；当前本地CUDA13不代表全比较或finaltest完成。LoGo仅源码已审，须FedAvg CUDA100根采用缓存、原poststate精确恢复和Linux CPU环境通过实际门后才能运行。
