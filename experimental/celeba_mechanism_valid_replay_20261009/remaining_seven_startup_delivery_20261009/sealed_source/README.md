# 原8机制终轮中剩余7项：三视图 valid 重放准备

状态 **PREPARED_NOT_APPROVED**。本目录没有派发训练或推理，不覆盖原13成员 bridge 准备、single及其失败/恢复/验收 seal。只处理原封存8个真实70轮终轮中的 `minus_U_IID_Benign_seed91001` 与 seeds91003–91008；已接受并离机的 seed91002 明确排除。不会填充其余792个 pending checkpoint SHA，也不会扩到800或 test。

`batch.py` 是独立外层生命周期：一个轻量 coordinator 顺序派发七个独立子进程，每次只有一个 CPU8线程计算 worker，CPU112–119、nice10/idleIO、CUDA隐藏。每个 worker 直接调用原 `bridge.py` 的 `bind_runtime`、`replay_one`、`accept_saved_predictions`，原 scientific body、evaluator、root-only阈值拟合、native1e-12 和三个视图均未改写。独立子进程避免重复导入原 v2 时重设 PyTorch interop 状态的问题。输出、日志、批准收据各按原ID独占，单 worker POSIX锁避免并行；部分目录/失败 sentinel 一律保留拒绝，不自动跳过或重试。

原 bridge 继续核：原70轮完整性、manifest/job/variant/方法身份、checkpoint与全部指标同终轮、源码/数据/分区/root/train/valid身份、完整group-label支持、保存的margin/预测与root拟合重算、三视图全部指标和原native终轮1e-12一致、全部输入before/after不变。每项结束立即严格接受，错误立停保留。这里不创建 optimizer、gradient或新训练。

Full只使用旧100个严格原训练结果的身份与data contract引用，不重推理、不再备份其权重。已存在的 Full seed91001 三视图 receipt 有显式源/预测/离机 SHA引用；其余 six（91003–91008）三视图明确 **MISSING**，待后续baseline900实际接受后join。不能把旧calibration记录拼成raw/shared，也不能为本批临时补Full推理。配对展示缺失不影响这七个真实机制终轮本身的重放验收。

资源外层复用封存v2 classifier，另显式识别实际 `repair_execute.py` Hybrid8线程和 `release_v2/run_one.py` FL两条各1线程，绑定具体入口、cwd、源/job/package SHA、声明线程及实际affinity；不固定PID。最坏声明并存预算88baseline+8formal+8Hybrid+2FL+本8=114<实际122.87999核配额，实际启动仍须重新实测。114是预约计算线程数，绝不称实际114核利用；coordinator只记轻量调度，不冒充第二个8线程计算 worker。

`selfcheck.json` 是26项本地准备/拒收检查：实际七条库存身份，已接受single/pending/Full/重复ID拒绝，换源/输出/CPU/并发/tolerance/test/retry拒绝，部分目录和failure保留，known资源重复/重叠/超配额拒绝。没有读取图像、Torch科学导入、训练或推理，不代替真实重放验收。

root核本目录 seal 后，需在外部批准收据填 `APPROVAL_TEMPLATE.json` 的绑定SHA并改为 `APPROVED_REMAINING_SEVEN_MECHANISM_VALID_REPLAY_ONLY`。批准精确七ID/输出、原8库存与bridge、CPU112–119、单worker、valid/1e-12、无Full新推理/test/自动重试。root先读服务器guide，核实时资源及新空输出，生成64hex+LF的 `APPROVED.sha256`。normal supervisor建议模板已准备，autostart/autorestart=false；本准备未安装服务。

```text
<cu128_python> batch.py inspect
ionice -c 3 nice -n 10 <cu128_python> -u batch.py manage --approved /absolute/APPROVED.json --approved-sha256 <root-bound-SHA>
```

完成只代表这七条 valid实现重放，不能称final test或完整900机制表。完成后按七项验收差集备份 receipt、bridge_receipt、预测NPZ、strict_acceptance、各ID批准及日志/源码scope并核archive/member SHA；不重包已离机原模型/Full权重。保持原single14成员备份引用独立。原训练loader曾materialize全split属性元数据，重放只核train/root/valid，不宣称untouched test。
