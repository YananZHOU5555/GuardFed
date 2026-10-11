# FL100 CUDA caller：SOURCE-ONLY，尚未部署或执行

复用已审 old800 单 worker：`guard.py/worker.py` 来自 attempt003，`run_queue.py` 来自无复用分支的 attempt002。三个完整 diff 和逐 span raw inverse 在同目录；`install_once.py.patch/.ndiff` 给出原单次 installer 到本提案的完整差异及双向原字节恢复。原 FL formal API、GPU body、七包 native 身份和原 Linux whole saved science 均只按封条引用，不改科学函数。

固定接口 canary 按给定顺序选不同包，不按指标选择，n=3 不进行方法统计比较：

1. `FLGMM_Tg20_L2.0_lr0.001_IID_Benign_seed91001_screen`（identity0，保留早期 screen/选择历史）。
2. `FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91001_fullcoverage`（identity2，必须是冻结 3f1c… candidate/26cf… seal，不替换为后来的 capacity_pool32 包）。
3. `FLGMM_Tg20_L2.0_lr0.001_non-IID_Sp-DFA_seed91001_fullcoverage`（identity5）。

`ROOT_INSTALL_INPUTS.template.json` 故意不可执行。root 源审后另存真实输入：status=`ROOT_AUTHORIZED_FL100_FIXED3_CALLER_V1`、execution_authorized=true、当前 caller seal、原 API/mapping seal、三份实际 review 的本机 path/SHA 与 source_reviews_accepted_by_root=true、服务器 author decision path/SHA、真实 GPU0 UUID、deadline（<=72h）、受保护服务及准许状态。CPU111/GPU0 只是候选；现场不合格就停，不另选设备。固定 CPU111 及其 SMT 不得有其他窄 mask owner，保留全部 PID/task、nominal quota、RAM/disk、GPU free>=2048MiB/Recovery=None、nice10/idleIO、重复目标和诊断进程拒绝。该门检查受限任务与预算，不声称宿主独占。

待 root 审核且真实授权后，唯一安装命令模板（本包未执行）：

```powershell
python -B tmp/FL100_cuda_runtime_caller_prepare_20261011/install_once.py --inputs <实际ROOT_INSTALL_INPUTS.json> --inputs-sha256 <实际输入SHA> --source-seal-sha256 <本包FILES_SHA256实际SHA> --receipt-dir <未存在的root执行目录>
```

installer 只送原 API 的12个小成员、三个 caller、17KB 七包映射和审阅 metadata；不拷贝模型、原始数据、数组或8MB库存。77个 runtime role 使用实际 Linux path/SHA，原11 source/data 在服务器 CPU111/nice10/idle 子进程原字节全 hash 一次并封存 stat receipt；逐项资源门检查 sealed stat 和源 SHA。controller 保持宽 mask，单个科学 worker 由 OS `taskset -c 111` 在导入前固定。原 shared root-only calibration 会实际执行，不能称 zero-fit。固定 train162770/valid19867、70round 原 checkpoint、native 1e-12、margin>0/shared>=、FP32/deterministic 与原后处理设备规则由未改 API 约束。SIGALRM/SIGTERM 转异常保留失败，所有阶段不自动重试、不覆盖。

安装仅启动 `guardfed_uniform_cuda_FL100_canary_valid` 的固定3。三项各经原 native comparison 与原 Linux whole saved check，通过后仍由 root 独立审读并生成外部 `ROOT_CANARY_ACCEPTANCE.json`。至少含 root_accepted=true、ids=固定3、cohort=实际 EXACT100 path/SHA、records[id].acceptance=每项实际 formal_saved_acceptance path/SHA。未有该实际证明不能运行其余97。后续 root 的服务器命令（未执行、外部 SHA 必填）：

```sh
cd /workspace/guardfed_checks/uniform_cuda_FL100_20261011_canary1
nice -n 10 ionice -c 3 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B run_queue.py remaining --canary-gate-sha256 <实际ROOT_CANARY_ACCEPTANCE_SHA>
```

remaining97 是 EXACT100 原顺序去固定3的补集，仍每次只一项；不自动放行、不重做已接受 canary、不重新选 recipe。失败时保留 namespace、partial output、stderr、QUEUE_FAILURE 和 formal_failure，root 确定实际原因后另行授权恢复。本机原3views/旧 CPU 证据与 Windows refit failure不改变；原 Linux whole 通过不代表跨平台 bitwise 或 Windows whole PASS。新 CUDA 输出必须新验收，不拿旧 CPU结果充作新 CUDA。阶段不是训练、test、100已完成或论文性能结果。此提案不生成离机大文件，后续取数组时仍须实际 F: `Yanan 2TB`/Healthy/容量守卫。
