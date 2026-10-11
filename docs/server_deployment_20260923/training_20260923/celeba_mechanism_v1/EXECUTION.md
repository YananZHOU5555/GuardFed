# CURRENT: GuardFed返修实验

当前采用范围：native/views370、F70论文表、FL native86/views81、Hybrid views20/论文表20。实际入口写入时间见ROOT_METADATA_ADOPTION；最近完整五队列实测2026-10-11T01:52:32.890213+00:00；补充main观察02:10:55为375终轮/8活动/失败0，部分新worker尚未首轮，观察不等于验收。服务器ssh -p60350 root@89.22.197.55，实例52183675，repo/workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。sglang已按授权停止，其他项目不动。

| 阶段 | strict＋离机SHA＋root已采用 | 最近实测（不能代替验收） |
|---|---:|---|
| 机制训练 | 新370/800，Full100另复用 | 372终轮、8活动、420等待、失败0；活动5–53轮 |
| 机制三视图 | U100/C100/A100/F70，共370 | remaining620远端闭合192，离机采用190；F70七场景配对表已独立核验，旧F60原样保留 |
| FLGMM覆盖 | 新86/96，screen4另复用 | 新86终轮、2活动，64/43轮；三视图仍81，八完整场景表80＋部分1 |
| Fed-NGA/Huber搜索 | 全64已验收并按冻结规则选参 | 原CPU3轮门检固定截止13/14单项PASS，index13因新主worker首轮进度缺失未启动；整组/192覆盖未开始 |
| 组合基线覆盖 | 新19/96，screen4另复用 | 新24终轮、1活动；Benign10＋F Flip10三视图20已采用；两场景论文表20已独立核验，原Benign10表不变 |

最新机制native370根：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T014706Z/ROOT_DELTA_VERIFICATION.json（4ae0a66…）。三视图370根：tmp/celeba_mechanism_remaining_F_nonIID_FFlip10_root_adoption_20261011/ROOT_ADOPTION.json（a69fd809…）。新10仅non-IID F Flip去F，原460 native对象/50账本前缀及360保存结果顺序保持；native110和保存92归档成员已核。原完整inspection371中的额外FedSA1仅pending，accepted scope明确为370；无混入、丢弃或额外重放。保存输出核90指标/240计数/30规则，native差0，零新增训练/拟合/CNN。首次本地H变量错误发生在SSH前，已保留并仅修exec命名空间；原科学代码不变。

FL新增9为non-IID S-DFA seed91002..10，root tmp/fl_native_after77_SDFA9_prepared_20261011/ROOT_ADOPTION_REVIEW.json（904e3ee…）；97成员、9模型/72有限张量，旧77有序前缀保持。86新＋4screen复用为90个native引用，不能写成90三视图；新9三视图source已冻结并部署；启动前资源检查因主队列新worker尚无首轮进度暂缓，未启动。原Windows重拟合失败及后续zero-fit边界保留。

Hybrid F Flip10根：tmp/hybrid_FFlip10_root_adoption_20261011/ROOT_SCIENTIFIC_ADOPTION.json（ed315fe…），原Linux whole重拟合10、24成员离机SHA、Windows保存输出零拟合；90指标/240计数/30规则、native差0。旧Benign10未重放。FFlip seed91005根groupKL诊断差−2.168404344971009e−19原值/hex保留，不改容差、不称跨平台逐位等价。新CPU FP32重放与原CUDA训练身份分开披露。

当前F70论文表：docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_seven_scenes70_20261011/TABLES.md，root b3af4a2a…；原F60保持。五IID(alpha5000)＋non-IID Benign/F Flip(alpha5)，raw/native/shared、固定10/9/6面板、sampleSD(ddof1)。非IID Benign去F−Full在10/9seed准确率略升且两公平差距略增，6seed准确率方向反转；已有IID攻击取舍保留。F70新增non-IID F Flip10已核验：去F−Full native/shared ΔACC+0.006±0.846pp、ΔAEOD−0.00113±0.01122、ΔASPD+0.00518±0.01759；准确率近似平局、AEOD/ASPD取舍保留，9/6及raw面板完整，不称F普遍必要。仍缺三non-IID F场景30与四类控制400，共430保存三视图；未完成全部返修。

FL八场景80表：outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/TABLES.md；non-IID FedSA10 native ACC89.15±0.88%、AEOD0.0428±0.0049、ASPD0.1088±0.0055，shared88.77±0.81%、0.0087±0.0079、0.0689±0.0061，准确率代价保留。不能用综合score替代原三指标。

梯度64按冻结四条件score均值/精确并列字典序选NGA eta0.001、Huber eta0.3/T0=0.01/M0；Huber27/32、NGA12/32恒负预测，全部负结果、accuracy冠军/Pareto保留，n=1不作SD/显著性。14门检短程不是70轮性能/GPU等价。13项individual PASS与index13等待不等于whole14验收；原资源启动窗口拒绝均保留，未放宽round1..65条件。须原整组saved比较、离机完整state和root验收后，才启动已冻结的NGA96、再Huber96。

24条原话英文工作稿：docs/server_deployment_20260923/revision_20260923/rebuttal_working_F70_FL81_Hybrid20_release_20261011.md（9be134d9…），固定native/views370/F70、FL86新＋4screen/81三视图/80表、gradient64、Hybrid表20；仍为作者审阅稿，未应用提交LaTeX。最新Git65 commit8515952b668c3a22427a8d0db5cc0a2d4c7cf6e3保持固定发布截止，237变更blob SHA/远端通过。本轮native/views370、FL86、Hybrid20及新入口尚不在Git65；原回执不改。

Huber恒等投影按用户决定标CNN项目适配，不沿用约束域理论保证；LoGoFair固定image_id映射20虚拟cohort＋官方人口统计平等后处理，标人口适配，不称真实训练client公平性。十方法native1000、九方法三视图900/2052归因已采用；17方法目标未齐。FedWA/SmartFL/FedDNA忠实规格、主终点/最终test边界、匹配LaTeX、Fig3来源仍待解决，已有问题不重复问。partition2仅核19962 image_id元数据；未读最终标签/像素、未最终推理。validation选择、旧test暴露、cu128/cu130/driver/设备差异继续披露。

最近双GPU100/100%，64/63°C，6044/5041MiB，RecoveryAction None；cgroup内存76.44/519.17GB、OOM0，CPU配额122.87999核，磁盘余1.0595TB。快照累计CPU计数不作瞬时利用率；有界吞吐诊断未证明最优配置，未改并发/数值/实例。大型证据留服务器；必要差集写F:/YananResearchStorage/GuardFed，写前核Yanan 2TB/Healthy/容量，无内置盘回退。E仅代码/配置/索引/精简报告；大文件不入Git。

三小时聊天任务guardfed-training-health仍PAUSED，当前无automation_update接口；未另建cron/Windows监督。supervisor独立继续。当前写入器tmp/refresh_F70_Hybrid20_editorial_root_20261011.py仅采用已有root和实测事实；旧STATE/current前缀已另存，本段下方历史原字节保持。

新增论文表根：F70 b3af4a2a…（1134标量/567显示格/1260指标/3360计数）；Hybrid20 c221f009…（108标量/54显示格/180指标/480计数）。旧F60及Hybrid Benign10对象/表行保持；统计最大差1.421e−14≤1e−12。完整24点回复root9246cd58…，24原话块/38引文行顺序exact，24组新增mean±SD来自保存JSON。新稿尚未应用提交源或最终评价。

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
