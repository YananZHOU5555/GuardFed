# CURRENT: GuardFed返修实验

更新时间：2026-10-10T23:56:43.104332+00:00。服务器 ssh -p60350 root@89.22.197.55，实例52183675；repo /workspace/GuardFed-celeba-expanded。先遵守/etc/vast-agents-guide.md。sglang按授权停止，其他项目不动。

## 已验收证据与实测

| 阶段 | 原strict、离机SHA和root已采用 | 23:50:58 UTC实测（不代替验收） |
|---|---:|---|
| 机制训练 | 350/800新增；Full100另复用 | 352完成、8活动、440等待、失败0；活动28–49轮 |
| 机制三视图 | U100/C100/A100/F50，共350 | remaining620远端闭合172，离机采用170 |
| FLGMM覆盖 | 新增77/96；screen4另复用 | 新增终轮79；三视图81，八完整场景表80，部分场景1另存 |
| Fed-NGA/Huber搜索 | 全64验收并按原规则选参 | 服务自然结束，未重启 |
| 组合基线覆盖 | 新增12/96；screen4另复用 | 新增终轮21；IID Benign三视图10已采用 |

原生机制350：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T232753Z/ROOT_DELTA_VERIFICATION.json。最新增量9项补齐去F IID Sp-DFA十seed，实际102归档成员，旧441记录与48账本前缀不变，当前49条链。三视图350：tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/ROOT_ADOPTION.json。原340对象/顺序保持，新10项原数组重放、零新拟合/训练/CNN，实际92归档成员与离机SHA通过。

F50论文表：docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_five_scenes50_20261011/TABLES.md。IID Benign/F Flip/FedSA/S-DFA/Sp-DFA各10配对；raw/native/shared及固定10/9/6面板，均值±sampleSD(ddof1)。原F40对象、648统计量与324显示单元不变。新Sp-DFA的native/shared去F−Full均值：ACC+0.1173百分点、AEOD−0.00661、ASPD+0.01725；raw两种公平差距均变差。S-DFA的Full均值优势及其他场景的取舍/反向保留，说明预测规则依赖，不作普遍必要性、显著性或独立因果主张。仍缺五个non-IID F场景50条及四类控制400条，共450三视图未接纳；不等于全部尚未训练。

FLGMM八完整场景表：outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/TABLES.md。non-IID FedSA10：native ACC89.15±0.88%、AEOD0.0428±0.0049、ASPD0.1088±0.0055；shared88.77±0.81%、0.0087±0.0079、0.0689±0.0061，准确率代价保留。原71记录不变。Linux原校准通过，Windows仅零拟合审计保存输出；原Windows47重拟合失败、资源拒绝、首个FedSA传输ID拒绝及单个groupKL差−2.168404344971009e−19保留，不称跨平台逐位等价。

梯度64搜索：8配置/方法×IID/non-IID Benign/S-DFA×seed91001，70轮valid。冻结四条件score均值、精确并列字典序选择NGA eta0.001及Huber eta0.3/T0=0.01/M0。Huber27/32、NGA12/32恒负预测，全部负结果、准确率冠军和Pareto保留。n=1无sampleSD/显著性；零差距不当公平性胜出，不推断原算法普遍失效。

192新增+8复用覆盖源码已冻结。原真实图像CPU三轮门检当前3/14单项通过，index3运行；整体验收/离机保存及启动资源复核仍待完成，192覆盖未启动。短门检不是70轮性能或GPU等价证明。FFlip在此梯度桥接中是元数据空效应，保留其边界；初次资源元数据拒绝保留，未更改科学函数。

## 回复稿与边界

当前24条原话统一英文工作稿：docs/server_deployment_20260923/revision_20260923/rebuttal_working_F50_gradient64_20261011.md，固定native350/views350/F50/gradient64/Hybrid表10。原24块38行、标题与顺序精确保持；未纳入FL81或门检结果，供作者审阅，尚未应用提交版正文。历史F40稿固定341/340/F40截面，保留原字节。

Huber按用户决定采用恒等投影，明确CNN项目适配，不沿用原约束域理论保证。LoGoFair固定图像ID映射20虚拟cohort并用官方DP后处理；DP为人口统计平等，标为人口适配，不称真实训练client公平性。

十方法native1000及九方法三视图900/2052项归因已接纳；目标17方法仍未齐。尚需剩余基线忠实覆盖、机制控制、冻结最终评价和提交版正文/rebuttal收尾。FedWA/SmartFL/FedDNA忠实规格、主终点/测试边界、匹配LaTeX和Fig3来源问题待解决，已有问题不重复询问。最终partition2仅核19962 image_id元数据，未读标签/像素或作最终推理。validation选参、旧test暴露及cu128/cu130/driver差异继续披露。

双GPU此次100/100%，65/61°C，6038/5037MiB，RecoveryAction None；cgroup RAM76.27/519.17GB、OOM0，磁盘余1.060TB。CPU有界诊断见tmp/cpu_throughput_readonly_20261011/ROOT_DIAGNOSIS.json；冻结worker/thread结构可解释低总CPU占比，未证明算子瓶颈或最优吞吐，未改数值/并发/实例。

## 保存与接续

大文件只写F:/YananResearchStorage/GuardFed，写前核Yanan 2TB/Healthy/容量；E仅代码/配置/索引/精简报告。模型、数组、数据和归档不进Git，优先留服务器，不重复历史备份。

最近已验证Git63：ccb5207c24762d3ac92ed76b40bd40b3e2330954，191 blob/远端通过；其固定截止native341/views340/F40、FL67/71/table70、gradient46。上述新350/F50、FL77/81/table80、gradient64和新工作稿待下一紧凑增量发布，不能冒称已上传。三小时聊天任务guardfed-training-health仍PAUSED，当前无automation_update接口，未建替代cron/Windows任务；supervisor训练独立继续。

入口写入器：tmp/update_increment64_final_entries_20261011.py。下方历史原字节保持。

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
