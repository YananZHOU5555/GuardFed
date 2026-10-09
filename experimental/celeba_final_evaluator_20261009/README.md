# CelebA terminal evaluator — local preparation

状态为 **PREPARED_NOT_FROZEN**。此目录只实现终态模型的三种预测视图、指标和冻结拒绝门；不训练、不调度、不读取真实图像或 test 标签。`acceptance.py` 使用真实项目 RGB64 CNN 的合成 CPU 输入，以及八条已有、已接受的 validation margin 缓存做软件验收。不会替代 all900 native valid 图像重放。

本次本地验收 **11/11 组通过**：含冻结core路径选择/拒绝检查；冻结门拒绝45种不完整或漂移输入，并对全部900条已有 job metadata 做纯内存正门检查。八条缓存的三视图指标与共享阈值均逐值一致；272个字段比较的唯一非零数值差是 `FairGuard_IID_Benign_seed91005` 的 `server_adaptive_lambda`，差 `−1.1102230246251565e−16`，小于 `1e-12`。

从仓库根运行本地验收：发布目录用 `python -B experimental/celeba_final_evaluator_20261009/acceptance.py`；本地准备目录用 `python -B tmp/celeba_final_evaluator_20261009/acceptance.py`。仅使用现有 Python、NumPy、PyTorch 与冻结 core 的已有依赖，不安装新包。输入路径由本目录位置推导，不依赖服务器。优先选择仓库根 `scripts/reproduce_paper_tables.py` 且SHA必须等于下述冻结值；否则只接受既有 `tmp/revision-publish-20260928` 下同SHA文件。两处均缺失或SHA错误时明确报缺，不使用错误版本的core。

`evaluator.py` 保留真实预测差异：raw 为 `margin > 0`（精确零归 class0），shared calibration 为 `margin >= group threshold`；current900 native 仅 GuardFed-AD2+ 使用原始 root-only 校准，其余八种方法为原始 argmax。共享配置严格使用已有 shared-calibration recipe。冻结 core SHA 为 `cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed`，通过私有 function globals 注入 cached clean-train-root margins，不修改 core 全局函数或原文件。Target labels 只进入固定预测后的评分函数。

验收包括手算混淆矩阵、AEOD=绝对 TPR 差、ASPD=正预测率差、零分母、恒定预测、坏输入、空视图、zero ties、篡改或非有限阈值、root-only 标签隔离、无 optimizer/无梯度/权重不变的 CNN 推理、权重改变拒绝和冻结门身份漂移拒绝。Fit SHA 用于发现阶段间篡改，不能代替可信来源或授权。

八个预先列明的历史 validation job 在 `acceptance.py:SELECTED`；覆盖七个 StageA 方法、IID/non-IID、Benign/S-DFA。仅从归档提取各自 `margins.npz` 与 `evaluation.json`，校验归档与成员 SHA、原 manifest/config/checkpoint/result 身份；`regression.json` 保存每条三视图完整指标、共享阈值、校准诊断、逐字段预期/实际/差值及输入身份。容差 `1e-12`，不重算全700或产生新科学实验结果。

128MB原归档不在Git中。完整验收仍要求将 `sharedcal700_and_baseline_gates_20260928.tar.gz` 按已发布的 `docs/server_deployment_20260923/training_20260923/celeba_shared_calibration_v1/sharedcal_backup.json` 与 `backup_inventory.json` 恢复链放回同一目录，归档SHA必须为 `f7528394e8888163e157323654ad9cee31f4d32a1b65a0ef57cb6894382e352e`。已核验的八条原始缓存保存在本目录 `accepted_valid_inputs/`，成员SHA及原归档身份保存在 `input_identity.json`；可以单独用于数值复核，但不能替代完整验收中的原归档SHA和成员字节核验。程序不会在原归档缺失时静默改用缓存并报告完整验收通过。

冻结门需要可信的外部 receipt 与调用方独立重算的 observed 身份；当前 draft 必然拒绝。冻结版 protocol 必须明确四个 decisions、完整 selected_views、target_split/target_n/target image IDs/order SHA、model_inventory SHA。Receipt 必须绑定 protocol/job/evaluation source/inventory SHA 以及 all900 native valid replay 的 accepted、n=900、最大指标差≤`1e-12`、acceptance SHA。Observed 必须核对 checkpoint/result/raw job/config/root IDs/source+data/target IDs+order+n+split/inventory SHA，并提供从已核验库存取出的 `model_inventory_record` 与重算的 `adapter_source_hashes`。按真实库存身份核对历史 recipe ID、F-Flip slug 与 FedAA/LASA ID，不能从方法名重新生成 ID。此目录不生成真实授权 receipt，不提供真实 dispatch CLI。

验收结果见 `acceptance.json`、`checks.json`、`checks.log`，源与输入身份见 `input_identity.json`。`FILES_SHA256` 按相对路径排序覆盖持久文件（排除自身与 Python bytecode），可用于复制后核验。

本地修复记录保存在 `initial_failure_and_recovery.json`（两条历史 recipe ID 的测试输入纠正）和 `gate_identity_failure_and_recovery.json`（从错误的 ID 拼接约束改为真实库存身份绑定）。两次失败均发生在本地软件验收，不涉及训练、test 标签或真实最终评估；修复后重跑上述全部检查。

剩余限制：未访问服务器、未读取真实 CelebA 图像/test 标签、未验证 GPU 数值等价、未完成900模型 native valid replay，未冻结正式协议或选择规则，未执行真实最终评估。
