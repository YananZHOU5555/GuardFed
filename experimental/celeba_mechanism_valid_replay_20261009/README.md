# 机制终轮 valid 三视图：仅准备桥接

状态 **PREPARED_NOT_DISPATCHED**。库存仅含已接受且离机的8个真实70轮 `minus_U/IID/Benign/seed91001..91008` 模型，另引用 baseline900 中完全相同的100个Full。其余792个只列待完成ID，没有伪造checkpoint SHA。本目录实际图像推理、新训练、Full权重复制均为0；不构成最终评价或机制900后处理完成。

遵循已封存 `celeba_mechanism_v1/PROTOCOL.md` 的 raw/native/common train-root-only calibration 三视图范围，保留原方法名 `GuardFed-AD2+` 和显式variant/config，未创建新科学协议、服务或派发器。

## 实际输入与本地检查

- `inventory_actual8_Full100refs.json`：8个原始终轮模型/config/source/data/adapter/job/result/checkpoint归档引用，100个Full原库存记录canonical SHA与权重/结果/job/root身份引用，792个仅ID。
- `preparation_check.json`：复用v4 `verify_archive`，5+3两归档124成员hash/大小完整通过，离机verification/receipt/previous receipt/ledger绑定一致；未提取或复制权重。归档内v3首5有效，当前v4接受快照8条有效，没有覆盖旧证明。
- `final_checks/selfcheck.json`：29项拒收检查通过；实际封存 `check_native` 函数体验证1e-13可接受、1e-11拒收，容差仍1e-12。无Torch/Numpy导入、模型加载、margin生成或图像推理。
- `actual108_path_schema_precheck.json`：13份原归档中216个实际raw job/result成员SHA/大小通过；8机制+100Full的108个job都带有output，且literal历史路径/config/source/data/终轮schema一致。执行原v2 `validate_original` 函数体，但 `checked_result` 返回值绑定已接受归档JSON；这是路径与schema预检查，未重新验收108项科学结果或张量，不解包权重。
- `final_checks/reuse_function_hashes.json`：16个实际复用函数逐个source/AST SHA，原函数体与封存源未改。

基线v3的FedAA历史raw job缺output问题不出现在这108项；仍不把基线专属兼容分支带入机制身份校验。

## 最小桥接边界

封存 baseline v2/v3 保持原字节。不能直接调用其批次入口：`validate_inventory` 固定九方法900格且要求 `ablation_component=none`；`storage_bindings` 固定2700文件；adapter查找限制在baseline树；其original结果校验没有候选mask ledger。新桥只处理库存/路径/variant/验收身份和Full引用，不重写训练、阈值或指标。

`bridge.bind_runtime(...)` 延迟导入原v3/v2/evaluator/v4，仅在独立外部批准收据SHA匹配、Linux cu128、CPU-only独占8核且单进程时可创建运行接口。新原始job/config/output保持原字节与历史路径；原v4 `accept_new` +机制 `worker.checked` 负责真实终轮/候选mask/完整诊断/有限张量/adapter/protocol检查。前后完整绑定源、数据及自身原artifact SHA/解析路径/stat；受信缓存symlink沿用原v2处理，未扩展任意外部target。

原样复用：v2 `replay_one`、`metadata`、`rebuild_root`、`check_native`；v3 `private`、`full_hashes`、`check_source_tokens`；evaluator的margin提取、train-root阈值拟合、预测、混淆矩阵/指标。`private(replay_one, ...)`只替换原记录校验和已绑定读取路径。保留native原group阈值、raw margin>0/tie0、shared同一root-only frozen recipe及校准>=threshold。

返回三个接口：

```python
runtime = bridge.bind_runtime(inventory_path, inventory_sha, dependency_paths,
                             repo, model_id, approval_path, approved_receipt_sha)
runtime["validate_original"](original_worker, runtime["record"], repo)
runtime["replay_one"](approved_output, wall_seconds=1800)
accepted = runtime["accept_saved_predictions"](approved_output)
```

其中 `validate_original` 已在绑定闭包中保存实际worker，调用参数只用于统一接口。`replay_one`生成原封存推理body的receipt与独立 `bridge_receipt.json`，不覆盖原body证明。`accept_saved_predictions`复核同checkpoint三视图，重建root和ID、核margin/预测数组SHA、重拟合root阈值、重算全部指标/native差异；不做新图像推理。失败保存独立 `.bridge_failure.json` 与原partial，拒绝覆盖/重试，不修改原队列。

这不是独立runner框架；父代理可在现有前台协调器内调用接口。**当前无批准收据、无CPU派发。** 先审代码，再在现有phase11释放资源后明确分配独占8 CPU/单进程，生成外部批准receipt，才能进行首项有界真实valid replay；成功与退出码仍需严格accept。

批准receipt所需字段：`status=APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY`、scope本桥、inventory/bridge SHA、当前8项的非空selected_ids、device=cpu、compute_threads=8、max_processes=1、8个明确allowed_cpus、target_split=valid、native_tolerance=1e-12、final_test_dispatch=false、具体全新output路径。当前代码不生成此批准收据。

`dependency_paths`显式提供 `v2/v3/evaluator/evidence_v4/baseline_inventory/manifest/protocol` 七个原文件路径，每个均核封存SHA，部署时不能用“最新同名文件”替换。v3只作为原私有函数/输入hash工具复用，不执行其固定900派发逻辑。

## Full去重与解释边界

100Full与baseline900重合，库存标记 `replay_required_here=false`、`weights_repacked_here=false`。`reference_baseline_full(...)`只加入外部SHA绑定的原baseline严格单批次acceptance，核inventory/core/v2/validID及native1e-12并筛出Full引用；重复/外来ID、错误阈值源、仅有累计状态标签均拒收。当前100Full引用不宣称其三视图已全部完成；两个封存v2 canary的复用仍由父代理原baseline恢复链管理，不重复推理。本桥交付不自动消费任何未审阅baseline阶段结果。

原100Full环境98cu128+2cu130，新8cu128；CPU重放须实测native一致，不能假设CUDA环境等价。seed91001参与过选参，其余validation seed也曾观察。原训练loader会materialize含test尾部的Smiling/Male元数据；新重放沿用封存prefix读取只解释train+valid，禁止test图像推理/拟合/评分。不能宣称整个项目从未访问test标签或untouched test。

8个新机制模型尚无重放吞吐/内存/native差异实测；GPU正式800训练、原protocol/结果/所有seal均未改。后续新接受模型需新的库存和独立封存，不能把本库存的8改成800或给pending项补假SHA。
