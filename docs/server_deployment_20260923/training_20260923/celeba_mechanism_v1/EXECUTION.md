# CURRENT EXECUTION: GuardFed返修实验

服务器：ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded。用户已授权停止sglang，文件保留；不自动切回213实例。先遵守/etc/vast-agents-guide.md。

## 当前队列

以下“已验收”均经过原strict、离机SHA和root核验；实时完成文件与验收数量分列。主机制实测时间2026-10-10T17:33:53.592547+00:00；五队列合并观测2026-10-10T17:43:13.117705+00:00。

| 阶段 | 已验收 | 实测活动/剩余边界 |
|---|---:|---|
| 机制训练 | 300/800新增，100 Full另复用 | 五队列观测完成303、活动8、等待489、失败0；固定70轮/8并发 |
| 机制三视图 | 300终轮checkpoint | U100/C100各十场景；A100的完整场景以接受索引及已采用表为准；远端闭合不等于离机验收 |
| FLGMM完整覆盖 | 59/96新增，4复用另计 | 观测终轮62，2个worker有轮次增长 |
| Fed-NGA/Huber搜索 | 42/64 | 观测终轮45，单worker推进；所有候选/恒定预测保留，未选recipe |
| 组合基线完整覆盖 | 9/96新增，4复用另计 | 观测终轮11，单worker推进；7个三轮门检不计正式样本 |

五队列已按真实worker身份、轮次增长、来源和错误检查核验；详见docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/ROOT_FIVE_QUEUE_GROWTH_20261010T1743.json。最近主资源采样为CPU12.01/122.88核、RAM78.08GB、磁盘余1.060TB；GPU瞬时利用率/显存/温度和Recovery原值见server_reactivation_20261009/latest_formal_live.json。这是该采样时刻的值，不代表连续占用；不因瞬时低利用率重启健康任务。冻结8并发/FP32/参数/seed不变。

FLGMM三视图累计61个valid终轮checkpoint：此前48原记录保持，新增13完成Linux原whole检查/root-only拟合、F盘30成员SHA和Windows零拟合保存输出审计（117指标/312计数/39规则，native差0）。新增一条root审计group_kl差−2.168404344971009e−19保留；原Windows47重拟合和whole仍FAIL，新13未在Windows重拟合，不称跨平台逐位等价。来源57新增native训练＋4screen复用；与当前机制300分列；此回放采用未新增训练或test。FL100及17方法未齐。入口tmp/fl_three_view_after48_20261011/ROOT_SCIENTIFIC_ADOPTION.json。 六完整场景（五IID＋non-IID Benign）各十seed的三视图表已独立及root采用，固定10/9/6面板，324统计标量/162展示格、549计数派生指标核验；61条全保留，non-IID S-DFA的单条screen不进完整场景统计。IID alpha5000/non-IID alpha5，验证集选择史和校准取舍披露。表格：outputs/guardfed_tables/celeba_flgmm_six_scenes60_20261011/TABLES.md。

组合基线IID Benign十seed native验证表已独立及root采用，固定10/9/6面板、18个均值/样本SD标量和9个展示格核验；来源为9新增+1screen复用。仅此场景齐备，其他九场景和三视图未齐。入口outputs/guardfed_tables/celeba_hybrid_IID_Benign10_20261011/TABLES.md。

Huber已验收10个终轮候选均恒负预测，ACC=0.5166859616449389、AEOD=ASPD=0，作为退化负结果保留，不据零差距称有效或选冠军。

## 已交付与尚缺

- 十方法native验证表已接受1000格，IID/non-IID各五场景，10/9/6种子；三页PDF：outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/celeba_ten_method_native.pdf。仍缺其余7方法完整覆盖，不能称17方法完成。
- 九方法三视图900记录与2052项校准归因已接受；native和共享校准的优势方向不同，准确率代价及负结果保留。这仍是验证集证据。
- U/C各100对三视图表已接受。A最新100对、10个完整场景：docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_ten_scenes100_20261011/TABLES.md。A100已齐五IID及五non-IID场景；五IID聚合原字节保持，另列五non-IID及平衡十场景seed-first汇总，n始终为seed数。Sp-DFA十/九种子与六种子方向变化均保留。删除项存在指标取舍，不声称每项不可或缺。
- 24条原意见完整英文作者审阅稿，优先阅读清晰版：docs/server_deployment_20260923/revision_20260923/rebuttal_clear_A100_20261011/rebuttal_clear_20261011.md。详细证据版：docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A100_reader_20261011/rebuttal_integrated_20261011.md。A100十场景、固定面板反转及seed-first取舍已纳入；原意见/旧数字/整表保留在详细版；清晰版保留关键结论、反例与全部未完成边界。正文插入稿尚未应用到提交版源项目。
- 旧TableII的480个原值已追溯真实重复数；缺依据的SD不补造。Fig3终轮候选已核260记录/78均值，但执行来源缺口仍保留。

Huber采用作者接受的恒等投影，明确CNN项目适配且不继承原理论；LoGoFair采用固定图像ID的20虚拟cohort，不能称真实client公平性。LoGoFair100已接受，cache原始预测不冒充其DP后处理native。新增CNN三视图身份桥源码/64门检已通过。FLGMM、组合及一个NGA搜索checkpoint的三条真实图像接口已核验采用：Linux完整原检查、F盘原保存数组/校准重拟合块通过，27指标/72基础计数/9规则和原native差0；Windows完整检查的FL审计group_kl约2.2e-19差异保留，不改容差，不称Windows全文检查通过。仅3代表接口，不是三方法100格齐备或最终test。入口tmp/celeba_added_cnn_exact3_root_execution_20261010/ROOT_SCIENTIFIC_ADOPTION.json。

下一步继续冻结队列，按有价值批次增量验收；补齐七方法、其余机制控制及新增CNN三视图。FedWA/SmartFL/FedDNA忠实规格、最终主终点/测试边界、提交版LaTeX源及Fig3来源仍待解决；已提出的作者问题不重复询问。历史test暴露、seed91001选择史、混合推理设备与cu128/cu130/driver差异保持披露，不把短程门检写成70轮等价。

官方最终评价候选分区已作独立ID元数据核验：partition2共19962张、顺序SHA见final_split_metadata_20261011/ROOT_METADATA_VERIFICATION.json。仅解码image_id/split并核整体文件身份；未解码标签数组、读取像素或模型、拟合或test推理。旧准备协议原字节保留；该事实不选择主终点、不冻结最终协议，历史test暴露仍披露。

## 保存、发布与巡检

本机大文件只写F:/YananResearchStorage/GuardFed，写前核F为Yanan 2TB且容量足；服务器大文件优先原地保留。E只保留代码、索引、配置与精简报告，不删除科学原始证据。

最近验证推送5749c96bb2c9a04904bfb56caeeda45c42fe7233（分支codex/revision-evidence-baselines-20260928）；发布截止以publication_closed_increment55_verified_20261011.json为准，后续本机新增不自动算已推送。

三小时聊天任务guardfed-training-health仍PAUSED；本会话无automation_update接口，未建立替代cron/Windows任务。supervisor运行训练不等于聊天定时巡检恢复。后续接续读TRAINING_STATE.json和server_reactivation_20261009/MONITOR_HANDOFF.md。

工程失败、恢复和旧批次完整时间线：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/entry_history/detailed_current_fbac3b4a7bc590b5.md。下方历史原字节保留；当前事实以本段、STATE和实测凭据为准。

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
