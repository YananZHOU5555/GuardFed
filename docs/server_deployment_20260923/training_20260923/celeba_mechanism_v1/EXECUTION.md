# CURRENT: GuardFed返修实验

核验入口更新：2026-10-11T05:12:07.556730+00:00。最近合并实测2026-10-11T05:02:03.780720+00:00；NGA窄观测2026-10-11T04:58:03.095044+00:00。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。sglang已按授权停止。

| 阶段 | 严格验收、离机SHA及root已采用 | 最新实测与边界 |
|---|---:|---|
| 机制训练 | 新400/800，Full100显式复用 | 实测400终轮、8活动、392等待、训练失败0；U/C/A/F各100 native完成，另四类400继续 |
| 机制三视图 | U100/C100/A100/F80，共380 | remaining620已远端闭合213，本机已采用200；seed91004去F/non-IID/Sp-DFA CPU重放数值失败，队列已停止；不得宣称F100闭合 |
| FLGMM | 96新+4screen=100 native完整 | 三视图仍90；最后10独立评价因配套服务失败状态被启动检查拒绝，未启动，正在准备隔离资源检查 |
| 主比较表 | 11方法native1100及三页PDF已采用 | IID/non-IID各5场景、每格10seed；9/6seed为敏感性，不是未触碰test；仍非目标17方法 |
| 组合控制 | native19、三视图20已采用 | 实测29新终轮；尚非完整100 |
| Fed-NGA/Huber | 搜索64、部署门检14、NGA新1已采用 | NGA实测7新终轮、第8项seed91009在60轮；96新+4复用，1worker CPU105/GPU1；Huber96未开始 |

本次机制native最后20仅non-IID S-DFA/Sp-DFA，终轮70/valid19867/source/data/config/checkpoint核验、190归档成员与离机SHA通过，旧480对象及52账本前缀保留，canonical账本53项已在本地/服务器同步采用。根：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T050221Z/ROOT_DELTA_VERIFICATION.json。保存三视图380、F80表与既有负结果不变；训练完成不能替代校准评价验收。

已保存故障证据：tmp/fl_final10_completed_aux_runtime_prepare_20261011/ACTUAL_NATIVE_ARTIFACTS.json。失败ID minus_F_non-IID_Sp-DFA_seed91004；训练native ACC/AEOD/ASPD=.8928877032/.0011269696/.0705992338，CPU重放=.8930890421/.0032383875/.0679697308，最大差.0026295030，原容差1e-12拒绝。源码、数据、模型/result/job SHA吻合；尚不能判定原因。不改容差、不抹除失败、不盲重试剩余620；主训练、NGA和Hybrid独立继续。已验收380与原生400仍有效，失败CPU数组不计入论文结果。

FLGMM完整native根 tmp/fl_native_final10_prepared_20261011/ROOT_ADOPTION_REVIEW.json；最后10归档107成员，旧86有序链不变。新增11方法表 outputs/guardfed_tables/celeba_eleven_method_native_20261011/TABLES.md，三页PDF outputs/guardfed_tables/celeba_eleven_method_native_pdf_20261011/celeba_eleven_method_native.pdf。1100为方法–场景–种子评估记录，LoGoFair复用FedAvg checkpoint，不能称1100次独立训练。1980场景统计数值/990显示格独立核验，误差至1.11e-16；旧1000对象和统计精确保留。balanced十seed：GF88.42%/.0097/.0610，FLGMM89.35%/.0439/.1119；准确率代价、LoGo更低ASPD及FLTrust更高准确率都保留，不称全面领先或显著。

当前完整24条英文回复：docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL90_Hybrid20_native11_20261011.md。38条审稿引文和既有统计范围修正保留；该稿机制固定截止380/F80、FL三视图90、Hybrid20，新增native11已写入；后来的native400/本次故障尚未写入固定截止稿。正文插入候选、14方法信息访问附录及作者入口在revision_20260923；尚未移入匹配投稿LaTeX、未做最终IEEE排版验收。旧TableII全部480格均已追溯（包括原44个N/E），实际n1=294/n10=180/n3=6，单次SD不可计算；历史选轮与round70修正候选分开，不能宣称旧值全是同checkpoint。

Huber采用用户同意的恒等投影，标CNN适配、不沿用约束域理论；LoGoFair固定image_id的20虚拟cohort仅人口适配，DP不是差分隐私。剩余六个native方法覆盖、其他图像控制400、共享校准闭合、忠实FedWA/SmartFL/FedDNA资料、主端点/最终评价边界、匹配LaTeX与最终发布仍待完成。官方test曾暴露，当前只valid；不挑seed、不用Full最佳对比消融均值、不混checkpoint指标、不删负结果。

最近双5090为100/100%、64/61°C、约5GB显存，RecoveryAction None；cgroup OOM0，内存约69.4/519.2GB、CPU配额122.87999核、磁盘余1.059TB。累计CPU计数不等于瞬时利用率，不据占用重启健康任务。大型证据留服务器或F:/YananResearchStorage/GuardFed，写前核Yanan2TB/Healthy/容量，无内置盘回退。Git68为f252db2c固定历史截止，本次native400/11方法/新全文尚未发布；不改旧发布回执。三小时聊天automation仍PAUSED，当前无可调用更新工具，未创建替代监督。

本入口只采用上述实际根证明和带时间观测；全部历史尾字节保持，原STATE/当前前缀另存在tmp/root_native400_native11_replay_failure_20261011。整个返修未完成。

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
