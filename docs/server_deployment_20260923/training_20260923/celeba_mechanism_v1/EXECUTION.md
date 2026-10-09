# CURRENT EXECUTION: GuardFed mechanism formal800 — measured 2026-10-09T09:50:26.636988+00:00

用户重新提供89.22.197.55:60350并明确授权停止sglang，现使用实例52183675开展缺失返修实验。sglang已停止；两张5090实际CUDA张量检查通过。先读/etc/vast-agents-guide.md（SHA42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa）。旧完成GuardFed队列仍禁止重启；213.224.31.105:26712内部当前状态未知，不自动切回。

源码五文件、全部RGB64缓存与官方标签/划分身份已核，100已验收Full对照及缺失历史依赖准确恢复（307成员验收，303新建+4原有相同）。20项cu130真实图像三轮门检全部严格接受，两套Full与原worker同horizon模型张量/全部指标/诊断精确一致；149备份成员及archive SHA在本机通过。原cu130门检完整保留在preflight_history/cu130_20261009，仅原job字节复制回空输出路径。

当前使用独立/workspace/guardfed_envs/celeba-cu128-20261009/bin/python，torch2.11.0+cu128、driver595.84，未改原环境/驱动/方法/seed/指标。实测服务guardfed_celeba_mechanism_formal   RUNNING   pid 9179, uptime 3:31:30；完成24，活动8，等待768，失败0；资源与轮次见server_reactivation_20261009/latest_formal_live.json。800新机制正式队列已启动，70round valid-only，8并发；100Full显式复用；检查dispatch receipt与full identity acceptance。

正式机制设计800新+100复用，IID/non-IID×5场景×10共享seed，按原协议固定recipe。Full98cu128+2cu130且多数原产出driver570.211.01，环境混合须披露；短程门检不能证明70round等价，提供排除两条cu130的配对敏感性口径。PROTOCOL.md为不可改动的准备快照，当前执行事实优先读EXECUTION.md、dispatch receipt、本入口及状态JSON。

原三小时任务guardfed-training-health本地TOML为PAUSED且提示仍指旧阶段；当前没有原生automation_update工具，未修改调度器、未建立替代监督机制。服务器supervisor只管理已启动队列，不等于三小时聊天巡检已恢复。可审阅提示与限制见server_reactivation_20261009/MONITOR_HANDOFF.md。

2454个历史新增完整训练及九方法900验证记录/备份保持原值。返修回复已逐字核24块原意见、40本地引用、209项SHA声明；新增260条生成器/PCA数值追溯与20份CelebA联合分组重建已核，投稿Fig3原脚本/FD执行身份仍缺。其余8基线与正式最终评价仍未完成；准备代码/门检不当作论文科学结果。主入口REBUTTAL_COMPLETION_20261009.md。

九方法900终轮模型/result/raw-job已全部精确接入当前服务器：100Full复用现存路径，其他800恢复至独立artifact_store，共2700文件逐SHA核验，原历史output修改0。两条完整valid19867/root16277原图CPU重放已接受，native三指标误差0，raw/native/shared三个视图的18指标与48混淆计数经主代理独立复核；52封存文件及27归档成员离机通过。初始阶段仅2条重放；当前全批启动和接受数见下面最新执行更新，不称最终评价完成；详见validation900_restore_20261009/README.md。各基线真实图像门检的完整接受及备份状态分开记录，首轮证据不等于完整门检PASS。

机制新结果已有23项通过独立70轮严格验收，23项离机备份，100Full身份复核保持有效；5份增量各SHA/member通过本机验收，原Full权重不重复打包。实时queue完成数与该已验收/备份分母分开。v2首备份因活动日志增长而在preflight拒绝，原检查保留；独立v3处理正常活动目录，新v4修正异常重检的诊断保全路径，经独立审查/回归通过。训练及封存v1/v2/v3不变，首5项归档保持有效。当前不是800或整个返修完成。

CPU端Fed-NGA/Huber四条真实图像三轮探索门检沿原源码/数据路径执行，完整接受和备份见下面更新；原加载器会物化全split标签元数据，包括test尾部，仅训练/验证像素参与运算，不称untouched test。执行附件见tmp/celeba_gradient_realimage_gate_20261009/EXECUTION_HANDOFF.md。

九方法验证重放已有26项吞吐阶段新任务严格接受并离机SHA/member验收，加之前2条共28个实际重放；native误差0，三视图指标/混淆计数独立重算一致。已完成1/2/4/8/11计划中的前5阶段，只报实测吞吐，不称已知最优或受控提速。v3的8并发失败证据完整保留；v4覆盖900条语义身份与2700文件SHA，语义接受不计图像推理。基于已接受28个独立ID，剩余872补集已独立冻结并实际启动；11并发、每worker8线程/nice10，详见下面最新启动更新；test未启动；阶段明细见tmp/celeba_final_valid_replay_20261009/v3/。该CPU重放只读train-root/valid语义标签，完整文件SHA读取包含test所在字节；它不调用会物化全split标签的原完整loader，不能与梯度gate的元数据边界混淆。

新增基线门检实测（2026-10-09T08:40:59.797823+00:00）：celeba_hybrid_realimage_gate_20261009：2/4条单项canary产物PASS；celeba_gradient_realimage_gate_20261009：2/4条单项canary产物PASS。各完整cohort尚未严格汇总/离机验收，科学表记录仍为0；不将三轮canary计入正式70轮结果。原始只读快照及SHA见TRAINING_STATE.json的baseline_gate_live_20261009，保留原startup封存证明。

FLGMM两条完整真实图像CPU三轮canary均已严格接受并离机验收54成员，CPU任务已退出。两条ACC均0.516686、AEOD/ASPD为0的恒定预测负结果保留；Tg1为管线覆盖，不计正式论文结果，不推断CPU/GPU等价。GPU四项跨卡重复门检另行接受；32项搜索尚未启动。凭据见TRAINING_STATE.json的flgmm_cpu_canary_20261009。

FLGMM原v2四项GPU三轮canary均完成，各自身份通过，跨卡训练张量、指标、controller及Torch RNG逐位一致；整体门检按原冻结规则保留REPEAT_MISMATCH。实际差异限于被快照混入的SciPy导入期文档示例default_rng熵状态，原失败报告与121成员已离机核验。记录范围缺陷由独立v3修复；原v2门检不追改，32项搜索未启动；三轮canary不计正式性能结果。

更新：FLGMM v3四项GPU门检已全部严格接受、两组跨卡重复精确，139内容成员+清单及四份旧模型引用离机核验，服务正常EXITED；此前3/4启动快照只作历史记录。32项新搜索包已准备，尚未启动，不计正式性能样本。

更新：Fed-NGA/Huber四项三轮真实图像门检全部严格接受，240条同点client gradient与攻击符号oracle通过；86成员离机及主代理核验，服务EXITED。四项恒定预测负结果保留；五项正式协议选择仍未冻结，科学表记录0。此前2/4快照及“尚无完整四项PASS”为历史观察。

更新：Hybrid第三条non-IID/S-DFA在result严格JSON记录层失败，已定位敏感组零方差使原攻击审计相关系数未定义；两条IID已接受，整体2/4未通过。46成员失败证据离机保留，第三条模型有限且三轮aggregate精确仍不能替代完整接受。仅准备限定字段null与独立原因记录的writer修复；未知NaN和主指标/训练状态仍必须拒绝，未重启队列。

机制统一评价更新：一条minus_U/IID/Benign/seed91002终轮checkpoint已完成raw/native/shared三视图重放，native误差0，9指标/24混淆计数/3规则独立重算一致，14成员离机及主代理核验。它与70轮科学训练验收分母分开；Full仅引用已有严格结果，未重复推理/备份，未运行test。两项工程失败及第三次受限恢复链完整保留。

最新FLGMM搜索启动：guardfed_celeba_flgmm_screen     RUNNING   pid 18899, uptime 0:01:32，32项×70轮valid-only、seed91001、两卡各1任务/CPU1/nice10。首批实际round2/1，无失败；66封存源/输入+seal和4启动凭据已离机及主代理核验。尚无完整70轮接受，不称阶段完成；前文‘32未启动’为门检历史观察。入口tmp/celeba_flgmm_screen_20261009_v2_dispatch/BACKUP_HANDOFF.md。

最新九方法重放启动：guardfed_celeba_valid_remaining872_20261009   RUNNING   pid 19149, uptime 0:01:14，只执行900既有模型减去已接受28的872补集，80批、每批至多11；11个真实CNNworker均nice10、8计算线程，outer nice0仅编排。38成员部署包、只读inspect及启动凭据离机并经主代理核验。首10秒全cgroup实用50.39/122.88核，throttle0，主800仍增长。新批仍待独立三视图和离机接受，不把RUNNING计入已接受；不训练旧模型、不运行test。入口tmp/celeba_final_valid_replay_20261009/v4/remaining872_prepared_v2_20261009/README.md。

最新九方法离机接受：唯一ID collector严格合并83/900实际三视图重放，尚缺817；其中原吞吐/门检28+新补集55，来源版本/原始config/checkpoint/数组/归档SHA均绑定，失败旧批不计样本。首11项64归档成员和99指标/264计数另经主代理独立重算全0；原模型不重复打包，不将该83项称完整最终评价。新collector路径tmp/celeba_final_valid_replay_20261009/v4/remaining872_execution_20261009/cumulative_83_accepted.json。

# HISTORICAL PREPARATION SNAPSHOT — no execution at time of preparation

# Mechanism stage: handoff and execution

This stage is prepared and locally checked. No new real-image or GPU run has started. `prepared_acceptance.json` verifies the complete 800-new/100-reused grid, 18 image gates and two unchanged-worker references, the five frozen core/loader source files, and six CPU/synthetic component checks. It does not verify current server data, real images, GPU numerics or scientific outcomes.

Runtime files are `adapter.py`, `worker.py`, `runner.py`, `prepare.py` under the project-local `tmp/celeba_mechanism_20261009/`. A byte-identical deployment snapshot and setup archive accompany this entry. Preserve file bytes during transfer; remote paths in the manifest are POSIX. A draft Windows-path error was intercepted before launch, corrected and rechecked; see `predispatch_path_failure.json`.

Authorized target: `ssh -p 26712 root@213.224.31.105`, instance52514165. Before remote actions read `/etc/vast-agents-guide.md`, inspect actual GPU/process/resource state, and verify the repository and data. Never start the old server or previously completed queues.

Install runtime files at `/workspace/GuardFed-celeba-expanded/deployment/celeba_mechanism_20261009/` and the prepared stage contents at `/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1/`. Do not replace the frozen core. The three commands below are sequential, with inspection and acceptance at each boundary. They are instructions, not records of execution.

```bash
cd /workspace/GuardFed-celeba-expanded
.venv/bin/python deployment/celeba_mechanism_20261009/runner.py preflight --repo /workspace/GuardFed-celeba-expanded --stage /workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1
.venv/bin/python deployment/celeba_mechanism_20261009/runner.py freeze --repo /workspace/GuardFed-celeba-expanded --stage /workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1
.venv/bin/python deployment/celeba_mechanism_20261009/runner.py run --repo /workspace/GuardFed-celeba-expanded --stage /workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1
```

The preflight queue executes only 20 three-round pipeline jobs: 18 intervention/distribution gates and two unchanged Full references. Formal dispatch needs all gates, exact Full same-horizon regression, the identities of all100 reused Full models/results, source/data/config consistency, no existing training worker and a separate frozen receipt. Eight independent processes use the established two-GPU/one-CPU-thread schedule; no tuning of ablation parameters occurs. Existing accepted outputs are checked before skipping. A failed or partial directory stops recovery; preserve and inspect it. There is no new midround-resume promise.

The runner and its source checks compile locally; the actual queue path remains untested on this currently unreachable server. In particular, live guide/lineage/real-image gates must pass before describing this stage as ready or running. If a legitimate identity or gate mismatch appears, preserve it and fix the deployment/control problem; do not weaken acceptance.

After training, still required: strict900-record merge, per-condition paired ten-seed tables, raw/native/shared-root-only calibration controls, incremental off-server model/result/log backups with member SHA and restore chain. No automatic test evaluation is included. The separate missing-baseline matrix and final evaluation remain outside this stage.
