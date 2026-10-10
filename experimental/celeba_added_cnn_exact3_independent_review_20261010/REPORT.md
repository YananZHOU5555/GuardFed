# 新增 CNN exact3 三视图入口独立审查

结论：**源码审查 PASS，未发现阻断项；尚未获得 Linux 运行或科学结果证据。** 可交由 root 完成现场 preflight，并在真实授权绑定后执行这三个接口 gate。不能由本 PASS 宣称派发获批、native 回放一致、三方法全覆盖或方法排名。

固定候选：`tmp/celeba_added_cnn_three_view_gate_preparation_20261010/`；FILES seal `49c42b90214eae57d456f8ded1d2a4ff2de2762a78d6bcf5c4ae1a5766de9434`；HANDOFF `99cfa89ff093e4610f1c19d196b8b6a360227f374d845dbe05e38d61174c06bc`。独立读取并核对全部 38 个封存成员（698,601 bytes），没有修改候选。

## 检查结果

- 范围固定为 FLGMM IID/S-DFA/91005、CosineFairnessHybrid IID/Benign/91002、Fed-NGA eta0.01 non-IID/Benign/91001。三条均来自原已接受的 70 轮 full valid 记录。NGA 仍标作原 screen 记录；未改为获胜配置。Huber 和 LoGoFair 不在允许范围。
- 已阅读原 `replay_one` 全体、bridge 的 `science_bindings/identity_record/validate_metadata`、candidate 全体及 evaluator 的 fit/predict 路径。17 个原科学函数源片段 SHA 独立匹配；针对科学与 runtime 函数做静态全局引用检查，无缺失全局变量。三个 identity 补充后的 record 字段满足原 replay 读取需求，config 字段均为原 ExperimentConfig 已声明字段。
- 原 replay 的两处调用仅重绑 external identity 与实际 checkpoint 路径，另有 receipt key 和 claim 文本修改；metadata/root重建/predict/评分/weights identity/native 容差沿用原函数。未复制另一套科学公式。
- 逐项检查 35 个 proof/artifact 映射：来源 F/Windows 字符串仅作来源标识，读取通过登记 map 到 package 或 `/workspace/...`；checkpoint 为固定服务器路径。Linux 上 bridge 自带的来源 F 路径 failure glob 不能实际检测服务器目录，但 candidate 的 `check_inputs` 对真实 `runtime_output` 检查 `failure*.json/FAILED.json`，并检查选中 producer 命令行与全部 runtime artifact SHA，补上该门检。
- 原训练设备分别为 cuda、cuda:0、cuda，Torch 均 cu128；config 原 CUDA 字符串保留。本次进程在 Torch import 前要求 CUDA hidden，模型显式 `.cpu()`、checkpoint `map_location='cpu'`，原 resource gate 验证可见 GPU 数量为 0；不会在 CPU 冒充执行原 GPU strict checker。
- native/raw 对这三个方法均为 `margin > 0`；shared 为 root-only 冻结七字段 fitter，按组 `margin >= threshold`。原 metadata 只物化 train+valid 标签前缀，图像只 materialize root 与 valid；valid 标签在 fit/predictions 固定后进入评分；不运行 test。三视图来自同一终轮 checkpoint，权重前后 identity 相同、无梯度、native tolerance 固定 `1e-12`，不一致先保存 receipt/arrays 后 fail-stop。
- 工程资源独立固定 CPU120–127、8 线程、interop1、nice≥10、idle I/O、1 进程、loader0。fresh preflight≤300s，要求实测 eligible mask、所有线程占用、实际 quota 与 nominal reservation、GPU/服务/内存/磁盘健康及选中 producer quiescent。进程 lock 与逐条检查限制重复运行；该固定组三条顺序执行，无自动扩容或重试。
- source review、manifest、seal、Linux preflight 与 exact3 authorization 由实际 SHA 相互绑定；本候选不生成预填 PASS 授权。输出要求新路径，科学失败保存 FAILURE 和已完成记录，不覆盖失败。成功也只标 `NOT_ROOT_ADOPTED`，仍需离机核验及 root 采用。

## 剩余实测与非阻断注意

Linux 的真实路径存在性/哈希、imports、CPU120–127 是否空闲、cgroup 余量和服务状态均未由本次独审测量。运行期 CPU 扫描着重拒绝窄 affinity（≤8 CPU）重叠，不能替代 root 对宽 affinity 活跃进程及整体预留预算的 preflight。进入科学 try block 之前的拒收由部署命令保留 stdout/stderr，README 已明确要求保存每次 attempt。

`freeze_inputs.py` 保留一次性生成器中的旧 CPU112–119 提示字符串；当前被冻结 MANIFEST、candidate.CPUS、README 和授权检查均为 CPU120–127，且生成器存在拒绝覆盖门。此文字不影响当前 runtime，但不要把旧生成器说明当实际资源状态。

本审查未运行 full metadata tests、SSH、CNN、root重建、fit 或训练；没有证明 CPU 与原 CUDA 数值等价。root 负责独立复跑已有 metadata 检查和执行前现场核验，科学结果只由实际三条 receipt 与后续接受决定。
