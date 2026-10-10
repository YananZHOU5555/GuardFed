# CURRENT: GuardFed返修实验

本段更新时间：2026-10-10T22:42:03.307776+00:00。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。sglang已按授权停止，文件保留。

## 已接纳结果与实时观测

| 阶段 | 原strict、离机SHA及root验收 | 22:39:26 UTC实测（不代替验收） |
|---|---:|---|
| 机制训练 | 341/800新增；100 Full另复用 | 完成344、活动8、等待448、失败0；活动第5–23轮 |
| 机制三视图 | 340终轮checkpoint：U100/C100/A100/F40 | remaining620远端闭合164，离机接受160另列 |
| FLGMM完整覆盖 | 67/96新增；4screen另复用 | 新增终轮77；三视图接受71，七完整场景表70，另保留一个未齐场景记录 |
| Fed-NGA/Huber搜索 | 46/64 | 终轮62/64；未完成全部搜索或选择recipe |
| 组合基线覆盖 | 12/96新增；4screen另复用 | 新增终轮19；IID Benign三视图十seed表已接纳 |

同次资源实测：双GPU利用率100/100%，温度66/62°C，显存6040/6020MiB，RecoveryAction None；cgroup内存76.45/519.17GB、OOM0，磁盘余1.060TB。五队列服务与worker身份通过，small source/data hash未变，大图像只核stat及历史全量SHA。记录：docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_five_queue_20261010T223921Z.raw.json。瞬时采样不证明持续占用或完成全部实验，冻结方法、seed、并发、FP32和统计口径保持。

本批机制native新接纳8项：补齐去F的IID S-DFA十seed，并接纳Sp-DFA seed91001，总341/800。94归档成员SHA及终轮/配置/源码/数据/checkpoint身份通过，旧433记录和47条恢复链前缀不变，当前48条链已索引。S-DFA10保存数组随后核验并接纳，三视图330→340；Sp-DFA一项native尚不计为三视图完成。原始大文件留服务器及F盘，增量证据待Git63发布。

## 新补齐的四个IID场景F消融表

docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_IID_four_scenes40_20261011/TABLES.md

IID Benign/F Flip/FedSA/S-DFA各10共享seed；raw/native/shared三视图及固定10/9/6面板。80记录/40配对，root独立核648均值SD标量、324展示格、720计数派生指标和1920基础计数；旧60记录、486标量、243展示值原样保留。表格制作没有新增拟合、CNN推理或训练。

S-DFA十seed native：Full为ACC88.689±0.809%、AEOD0.00769±0.00463、ASPD0.06377±0.00697；去F为88.412±1.202%、0.01054±0.00697、0.06941±0.01477。去F减Full配对均值为ACC−0.277pp、AEOD+0.00284、ASPD+0.00564；raw/native/shared的10/9/6九个面板三指标均值方向均支持Full，不作显著性、普遍必要性或单独因果主张。

其他场景的取舍保留：F Flip的9/6seed准确率差值反向；FedSA十seed去F准确率均值+0.136pp，六seed native/shared的ASPD差值−0.00560；Benign也有取舍。native/shared相同不作独立确认，混合设备、环境和选择史均披露。F仍缺六场景60项，另四类机制控制400项：共460三视图未接纳；native未接纳459项，不能称这些任务全部尚未训练。

FL七完整场景表70及单列部分记录已接纳，表：outputs/guardfed_tables/celeba_flgmm_seven_scenes70_20261011/TABLES.md。non-IID F Flip十seed native ACC89.28±0.98%、AEOD0.0525±0.0077、ASPD0.1185±0.0068；shared为88.90±1.04%、0.0091±0.0073、0.0713±0.0111，存在准确率代价。旧Windows全量重拟合失败、五次资源拒绝、Linux原校准通过及Windows零拟合计数审计均保留，不称跨平台逐位一致。

Hybrid IID Benign三视图十seed表已接纳，表：outputs/guardfed_tables/celeba_hybrid_three_view_IID_Benign10_20261011/TABLES.md。原screen seed91001身份与选择史、已验收seed91002及八个覆盖checkpoint显式保留；其余场景继续。native ACC88.027%、AEOD0.02920、ASPD0.09251；shared87.708%、0.02321、0.03383，有准确率代价。Hybrid为项目控制，不称外部原方法；原Windows resource导入失败及采样审计微小差异保留。

## 返修边界与接续

十方法native验证表1000格，IID/non-IID各五场景已交付；九方法三视图900记录和2052项归因已接纳。U/C/A各100配对、十场景表已接纳。尚缺七方法完整覆盖、其余机制控制、冻结最终评价及提交版正文/rebuttal收尾。FedWA/SmartFL/FedDNA忠实规格、最终主终点/测试边界、提交版LaTeX源及Fig3执行来源仍待解决；已有作者问题不重复询问。

24条原意见统一英文工作稿：docs/server_deployment_20260923/revision_20260923/rebuttal_working_20261011.md；已root核原话与顺序、F30/Hybrid数值及链接，供作者审阅。该稿内容截止native333/views330/F30；本次F40为单独新证据表，尚未写进该稿。F20旧稿、正文插入候选及F30补充稿保持各自截止；未应用提交版正文，不能称整份返修完成。

Huber按用户同意采用恒等投影，标为CNN项目适配且不继承原约束域理论。已验收14条Huber搜索结果均恒负预测，ACC0.5166859616449389、AEOD/ASPD0，作为退化负结果保留，不称公平性获胜。LoGoFair采用固定图像ID20虚拟cohort，标为人口适配，不称真实client公平性；其100个native结果已接纳。

最终候选partition2的19962个image_id元数据已核，未读标签/像素或作最终模型推理；validation选择史、旧test暴露、cu128/cu130及driver差异披露。NGA/Huber须64项完整验收后按冻结四条件评分规则选recipe，再过真实图像门检，不能以62项观测启动192覆盖。FL下一批non-IID FedSA10已终轮，先按原strict和离机核验接纳，不把77终轮当77验收。

CPU有界诊断：tmp/cpu_throughput_readonly_20261011/ROOT_DIAGNOSIS.json。15.18秒四采样观察约12个活动单线程worker及12.09/122.88核的cgroup平均用量，双GPU均100%；未采到磁盘读取或新增配额节流，符合冻结worker/thread结构。尚不能据此证明具体算子瓶颈或最优吞吐；未改数值、并发或实例设置，利用空闲CPU处理已闭合证据。

## 保存、Git与巡检

大文件只写F:/YananResearchStorage/GuardFed，写前核F为Yanan 2TB/Healthy且容量足；E仅代码、配置、索引与精简报告。模型/原始数组/归档不进Git，服务器大文件优先原地保留。

最近已验证推送fb1f31031aad89206f252969a6c26af6cde9efa4，截止以publication_closed_increment62_verified_20261011.json为准（245提交文件哈希及远端分支通过）。本次341/340/F40、统一工作稿和入口刷新在该提交之后，待Git63实际发布，不能提前称已上传。三小时聊天任务guardfed-training-health仍PAUSED，本会话无automation_update接口；未建替代cron/Windows任务。supervisor训练与聊天巡检分开。

入口写入器：tmp/update_increment63_F40_entries_20261011.py。旧生成器只保留历史截止，下方历史原字节保留。

## Historical accepted increment — 2026-10-09 09:52 UTC

The mechanism cohort now has23 new terminal70 records independently accepted and backed up off server in five linked increments;100 existing Full controls remain explicit reuse. The latest increment contains seven new IDs and86 SHA-verified members, archive `0a751a6676e1cdcdf1154208930110ae3822fc061287da2729d107fe40e64660`. Two minus_U/IID cells have all ten seeds: Benign and F Flip. Relative to Full, mean accuracy changes are−1.10686 and−0.30855 percentage points, while ASPD changes are−0.02272 and−0.00916. These are native validation contrasts with mixed historical/new runtime lineage, not proof that every component is necessary. Other cells remain incomplete.

The existing nine-method cohort has83 actual distinct model replays accepted and backed up off server, not900 inferred from a semantic identity check. All83 preserve native metrics exactly and have independently verified raw/native/shared calibration arrays. This comprises28 prior useful throughput/gate results plus55 new results from the running872-item complement. Root independently checked the next44 new models: four64-member original archives,396 metrics,1056 confusion counts and132 prediction rules all pass, with no old models repacked. The11-worker queue remains active, with817 models not yet accepted off server. The earlier five representative1/2/4/8/11-worker batches show11 workers as the highest observed throughput, not a controlled or global optimum. One new mechanism checkpoint also has a strict off-server three-view replay; its denominator is separate from baseline83 and mechanism23 training results. See `server_reactivation_20261009/VALID_CHUNKS001_004_ROOT_VERIFICATION.json` and `MECHANISM_VALID_ONE_ROOT_VERIFICATION.json`.

FLGMM's two CPU and four GPU three-round real-image canaries are accepted off server. Both GPU repeat conditions match exactly; the original v2 RNG-recording failure remains preserved, with independent v3 fixing only the import/training record boundary. Fed-NGA/Huber's four three-round real-image gates are strictly accepted with240 same-point gradient/sign-oracle checks and86 off-server members. Their constant-prediction negative outcomes remain visible, scientific table records remain zero, and the five formal gradient-protocol choices remain unfrozen. See `server_reactivation_20261009/FLGMM_V3_ROOT_VERIFICATION.json` and `GRADIENT_FOUR_ROOT_VERIFICATION.json`.

Hybrid's first two IID canaries are accepted. Its third run failed during strict JSON serialization of an undefined zero-variance attack diagnostic;46 failure members are preserved off server. A separately reviewed writer represents only four exact undefined correlation fields as null with their original NaN bits and reason; all unknown nonfinite values remain rejected. Only the two unaccepted non-IID canaries are authorized for bounded recovery. The frozen32-item FLGMM validation search is actually running, two GPU workers, with no complete70-round acceptance yet. Seven additional mechanism checkpoint replays are separately approved after a12-member source review, excluding the already accepted single replay; deployment still requires the live preflight. None is a completed baseline cohort. Current measured service/queue state is in `RUNNING.md` and `TRAINING_STATE.json`; the preparation snapshots below are historical.

Closed evidence through23 mechanism records, baseline83 replay, FLGMM/gradient gates, the first mechanism replay, both bulk queues and the seven-replay source/approval was pushed and73 committed blobs independently SHA-verified against the remote branch at commit `c53519a43ce40e8394f1cba077bbe94be4acdee6`. The later two startup deliveries are also root-verified: Hybrid's repaired non-IID gate has a real first round, and the seven-replay worker has processed real validation images; neither startup is a completion claim. See `server_reactivation_20261009/BOUNDED_STARTUPS4_ROOT_VERIFICATION.json`. The full rebuttal and final evaluation remain incomplete.

## Historical implementation and writing snapshots

- Complete CelebA mechanism design:800 missing controls plus100 existing Full records; isolated controls, strict result/ledger checks and bounded fail-stop launcher. Both sets of20 three-round gates (cu130/cu128) are strictly accepted and backed up off server; model tensors, metrics and diagnostics match exactly across all20 short gates. Independent strict Full100 science-input acceptance also passes. The separate dispatch receipt is frozen and the800-new scientific queue started under supervisor at2026-10-09 06:18:58UTC, eight processes,70rounds,validation only. At06:20:44UTC, four workers had reached round1, all eight alive,0completed/0failed; no finaltest. These short checks are not proof of70-round cross-runtime equivalence. Current measured status: `RUNNING.md`, `TRAINING_STATE.json` and `celeba_mechanism_v1/EXECUTION.md`.
- Four additional method components are implemented with CPU checks: true same-point gradients/Fed-NGA, fixed-threshold sample-weighted Huber, the original cosine/fairness control, and official LoGoFair optimization/calibration. These are not completed GPU baselines. LoGoFair still needs an explicit client-ID evaluation adaptation; gradient methods need a declared attack-message/training mapping.
- FLGMM author-code aggregation is integrated into the unchanged CNN/local-Adam/attack path, with a checked full-state-to-delta bridge and a three-round synthetic-image pipeline test. Thirty-two screen jobs are prepared, not frozen or run. FedWA/SmartFL/FedDNA source recovery remains unresolved.
- Updated English rebuttal covers24 original comment blocks with source alignment, manuscript insertions and eight checked recommended references. Entry: `../revision_20260923/rebuttal_20261009/README.md`. The missing Adult score archive is restored and fully checked:140 original files,8440 visible rounds and388240 fields; observed gate re-entry and the conditional theorem's limited scope are retained. See `score_analysis/restoration_and_score_report_20261009.md`. Pending experimental evidence is labeled explicitly.
- CelebA realized-partition audit is complete:20 original Sp-DFA records,400 client counts and40 original sensitive-group counts match exactly. IID Male ratios range40.495%–42.969%, non-IID9.832%–74.303%; root support is counted once per shared seed. See `celeba_partition_audit_20261009/README.md`. The later joint audit below adds metadata-based clean-training reconstruction with an explicit boundary between original contracts and newly reconstructed receipts.

## Additional evidence and historical execution snapshots

- The stronger-alpha Adult/COMPAS audit is now complete:60 partitions,120 original paired jobs,1200 client totals,2400 sensitive-group margins and240 S-DFA flip counts match. Restored archived client label-joint counts are distinguished from independently replayed sensitive counts. Alpha0.1 includes empty and single-group clients, and nominal malicious IDs can hold over65percent of client samples. See `tabular_partition_audit_20261009/README.md`.
- Historical synthetic numerical lineage is checked for840 records. All840 joint-checkpoint rows match; an older independent-column export has797 rows without a common checkpoint. Seventy settings now also have three-seed summaries formed after averaging the four scenarios within each seed. A separate260-record generation suite is recovered, including all40 PCA-control records;250 old triplets lack a common checkpoint. Its26 setting aggregates have ten scenarios each, not ten seeds. Raster compatibility favors a ten-scenario/half-gap candidate over the submitted S-DFA/full-gap description, but the exact original plotting source remains missing. Archived PCA denotes a covariance Gaussian statistical control, not demonstrated PCA dimensionality reduction. ForestDiffusion implementation/fit/cache identity and the submitted Fig.3 exact input chain remain unresolved. See `synthetic_lineage_audit_20261009/README.md` and `synthetic_figure_recovery_20261009/README.md`; the900 CelebA validation records are unaffected.
- Fed-NGA/Huber have an isolated same-point-gradient worker and64 search proposals. Eighteen synthetic CNN rounds,108 attack sign-oracle cases and22 acceptance checks pass. No client Adam difference is presented as a gradient. The64 formal search proposals and five protocol choices remain unfrozen. A separately approved four-job real-image exploratory gate now runs onCPU104–111; the first client gradient uses all7351 local samples, zero optimizer steps, the unchanged common point and an exact attack-sign oracle. Its17-member startup/gradient archive is verified off server; no complete four-job acceptance is claimed. See `../../../tmp/celeba_gradient_realimage_gate_20261009/EXECUTION_HANDOFF.md`. The original loader can materialize test-label metadata; no test images are inferred, fitted or selected, and an untouched-test claim is invalid.
- All900 terminal model/result/raw-job identities have been located in25 existing SHA-verified backups. The900 final-evaluation job proposals are PREPARED_NOT_FROZEN; this is model preservation and planning, not test inference. Live source/data/environment, target sample identity, valid replay and final reporting rules still require verification. See `final_evaluation_prepared_20261009/README.md`.
- A terminal-model evaluator now passes11 local acceptance groups,45 invalid-freeze/drift rejection cases and cached validation replay of eight preserved jobs. This is software evidence; all900 real-image native-valid replay, a final dispatcher and actual final evaluation remain absent. Local entry: `../../../tmp/celeba_final_evaluator_20261009/README.md`; published entry: `experimental/celeba_final_evaluator_20261009/README.md`.
- LoGoFair's official optimization/calibration pipeline passes a three-round synthetic RGB64 check using an accepted FedAvg checkpoint,40 official BetaCalibration fits and repeated/RNG/order/serialization checks. It still needs an explicitly chosen evaluation-client population; synthetic client IDs are not real CelebA clients. The bounded primary-source follow-up located a FedDNA author account but no corresponding implementation; FedDNA/SmartFL/FedWA specifications remain incomplete. No guessed adapters or scientific performance results were produced.
- CelebA clean training joint support is now reconstructed and independently verified for20 partitions,400 clients,1600 label×Male cells and80 root cells. Original client totals/sensitive margins and train/valid/root identities match; no empty or unsupported joint cell occurs in these20 partitions. The realized non-IID split groups byMale, not bySmiling labels. New client-ID/joint-row hashes are reconstruction receipts, not invented historical runtime hashes. See `celeba_joint_partition_20261009/README.md`.
- Mechanism evidence tooling is separately sealed: v1 is the accepted strict Full100 gate; v2 directly binds new controls to the corresponding independent Full root/train/valid identities and sample partition. Four local acceptance groups and34 rejection boundaries pass, including mixed checkpoints, root tampering and duplicate/tampered backups. A live initial v2 inspection correctly reports PARTIAL,0new+100reused; no missing result is filled. Entry: `../../../tmp/celeba_mechanism_evidence_20261009/README.md`, promoted to `experimental/celeba_mechanism_evidence_20261009/README.md` in Git.
- All900 historical terminal models/results/raw jobs are now byte-verified on the selected89 server. Full100 reuses300 existing artifacts; the remaining800 were restored as2400 exact members under a separate artifact store. All2700 file hashes pass, without overwriting old outputs. Two full-image CPU native-valid canaries also pass (19867 valid and16277 root images each; native metric difference0). The root agent independently checked52 sealed files,27 archive members,18 raw/native/shared metrics and48 confusion counts. See `validation900_restore_20261009/README.md`. This is2 actual replays, not900 completed or final-test evaluation.
- Hybrid and FLGMM are running bounded, full-data CPU image gates alongside the frozen GPU queue. The first Hybrid round matches the legacy aggregation tensors/selection/trust/RNG exactly; FLGMM's first round completed its GMM selection and retained the near-chance, zero-gap outcome. Complete three-round gates and subsequent baseline searches remain pending. Early activity is not a paper performance result.
- The first five new mechanism results now pass strict terminal70 acceptance and off-server backup: archive SHA `de867e84fe6185af62368e5ddacbf618d23f432112bb5cc3238dd49d25d08cd8`, all70 members verified. A first backup-tool preflight rejected a growing active log; that diagnostic remains preserved. Separately versioned v3 classifies exact live workers as pending, retains failure/orphan rejection and freezes invalid diagnostic bytes. The training worker and sealed v2 are unchanged. See `../../../tmp/celeba_mechanism_evidence_20261009/README_v3.md`. Current queue-completed counts can exceed this strictly accepted/backed-up snapshot.

The numerical superiority target is a research objective. Current accepted results have accuracy/fairness tradeoffs; no rule change, deleted negative result or selected-seed-versus-mean comparison is justified by that target.

## Current server and monitor boundary

The user explicitly supplied89.22.197.55:60350 again and authorized stopping unrelated `sglang` to free both5090 GPUs. That host is reachable; actual CUDA, frozen source/cache and restored100Full identities have been checked. The newly frozen mechanism queue now runs under supervisor; no old completed queue was restarted. Both20-gate sets and the15-member frozen startup archive have passed off-server SHA/member checks. The other endpoint213.224.31.105:26712 remains unobservable this turn and is not automatically retried or selected. See `server_reactivation_20261009/` for source/data/restore/environment/gate receipts.

The original three-hour chat monitor's local configuration is PAUSED with a stale prompt. No native scheduler update/read tool is available in this session; no replacement scheduler was installed and activation is not claimed. A concrete updated prompt is in `server_reactivation_20261009/MONITOR_HANDOFF.md`. Server supervisor management of a queue is separate from the chat monitor.

## Remaining actual completion criteria

1. SSH/guide/core/cache restoration, both real-image gate sets, strict Full100 acceptance and the separate dispatch freeze are complete on the user-selected89 host. Continue the frozen800 new scientific jobs through strict terminal acceptance and incremental off-server backup; a launch is not completion.
2. Pass real-image gates and complete the missing faithful baselines, validation selection and full paired coverage; disclose any unresolved unavailable original method instead of substituting its old simplified branch.
3. Complete the800 new mechanism jobs and their900-record native/raw/shared-calibration comparison (800 controls +100 Full). The existing nine-method900-model replay is separately complete and accepted off server. These two cohorts share100 Full checkpoints, so together contain1700 unique terminal checkpoints; a completed baseline replay cannot stand in for the mechanism controls. Strictly accept and back up each cohort with explicit reuse references.
4. Freeze the final evaluation identities and rules, evaluate accepted models without test tuning, and disclose prior test exposure. This may reuse terminal models and is not automatically another full training campaign.
5. Replace the rebuttal's pending-evidence fields only with accepted measurements; finalize manuscript tables/text, response and all linked artifacts.

The active goal remains incomplete. Prepared files, working toy adapters and a finished response draft are concrete progress, not evidence that the missing scientific experiments have finished.
