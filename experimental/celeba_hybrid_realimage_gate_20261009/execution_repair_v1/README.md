# Hybrid 诊断序列化修复：仅准备，未派发

原门检的第三项是 **TERMINAL_FAILURE**：三轮训练和 checkpoint 已保存，但结果 JSON 被未定义的 Pearson 诊断 NaN 拒绝，未完成终轮重预测与验收。第四项未启动。原失败不会重分类为成功。两个 IID/Benign 三轮门检已接受并离机保存，修复只引用它们，不重复训练或推理。

本包只允许原冻结 `non-IID_S-DFA_hybrid_seed91001_cpu_gate3` 和 `non-IID_S-DFA_legacy_reference_seed91001_cpu_gate3` 两项在新隔离输出目录各重做三轮。当前状态 `PREPARED_NOT_APPROVED`，没有运行。原正式 32 候选 screen、参数、科学方法、最终 test 和旧结果均不变。

`writer_policy.py` 仅接受冻结 `all_unprivileged` 将四个恶意客户端敏感属性全部覆盖为常量的已知原因。精确 `attack_audit[0..3].fflip_label_corr_after` 的 Python float NaN 保存为 JSON `null`，并另存原路径、原 NaN 的 IEEE754 位型、未定义原因、冻结 job 和生产源码身份。`null` 表示未定义，绝不解释为零。所有其他非有限数、未知路径、主指标、权重/控制量继续拒绝。终轮模型的有限性继续由原验收器检查。

`repair_execute.py` 复用原 `run_one`、`compare` 的同一个 code object，只通过私有 globals 绑定独立输出路径和精确 writer。原 `checked` 所有条件完整执行，再要求 sidecar 身份与含义一致；原八项 artifact 必须齐全，额外 sidecar 也必须核 SHA。真实 CNN/localAdam、完整 train162770、clean root16277、valid19867、攻击后本地模型 root AEOD、等权筛后聚合、每轮 exact 对照和终轮 native 重预测没有改动。

`writer_checks.json` 是 24 项本地组件检查，包含未知 NaN、主指标 NaN/Inf、权重和控制非有限、错误客户端/字段/原因、sidecar 位型与路径篡改、缺批准收据拒绝，以及有限值和输入结构不变。它们不是新图像训练结果；本包训练、推理次数均为零。

下一步需 root 审阅本包 seal/源码和两项 scope，再提供外部 `APPROVED.json`。收据必须绑定本包 `FILES_SHA256.json`、`REPAIR_SCOPE.json` 的 SHA，精确两项 job SHA、CPU8–15/8线程、当前无计算重叠；同时声明不授权 test/formal32/自动重试。派发前重新实测资源、原源码/数据、旧两项 IID 原始 artifact 与失败备份身份。

建议 normal supervisor 单服务，`autostart=false`、`autorestart=false`，无额外监督/端口；命令为：

```sh
ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python /workspace/guardfed_checks/celeba_hybrid_realimage_gate_20261009/execution_repair_v1/repair_execute.py run --approved /absolute/APPROVED.json
```

`inspect` 仅验证准备文件，不导入 Torch、不读图像、不建立输出。实际 run 另设 CUDA 隐藏和 8 计算线程。任何未知非有限数、身份或 native 对照错误立停并保存失败，不自动重试；两个新 canary 接受后仍需原始模型/结果/sidecar/日志离机 SHA/member 核验，才能形成“两个新项+两个引用项”的门检闭环。不能用三轮门检宣称正式性能、多 seed、70轮或 CPU/GPU 等价。

沿用原 loader 会 materialize 全 split 的 Smiling/Male 元数据，包含 test 尾部；不会读取 test 图像或进行 test 推理、拟合、选参。因此不声称 untouched test。
