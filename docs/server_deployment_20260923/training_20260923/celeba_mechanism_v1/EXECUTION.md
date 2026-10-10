# CURRENT EXECUTION: GuardFed返修实验 — 实测 2026-10-10T02:06:18.910647+00:00

当前服务器：ssh -p60350 root@89.22.197.55，实例52183675；repo /workspace/GuardFed-celeba-expanded。用户明确授权停止sglang，模型/文件保留。213.224.31.105:26712当前内部状态未知，不自动切换。先遵守/etc/vast-agents-guide.md，既有SHA为42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa。

## 当前执行与验收

| 阶段 | 实际状态与分母 | 接续入口 |
|---|---|---|
| CelebA机制消融 | 当前观测完成172、活动8、等待620、失败0；已独立严格验收并离机170/800新增，另100 Full显式复用 | server_reactivation_20261009/latest_formal_live.json；celeba_mechanism_v1/EXECUTION.md及dispatch receipt |
| FLGMM验证搜索 | 32/32已严格验收并离机；最新来源绑定终轮/活动读STATE对应快照，不把未验收完成项计作接受 | tmp/celeba_flgmm_screen_20261009_v2_dispatch/LATEST_BACKUP.json及accepted_delta_after6_20261009/ROOT_ADOPTION_REVIEW.json |
| FLGMM完整覆盖 | 96新+4复用；状态ROOT_ACTUAL_FLGMM96_VALID_COVERAGE_STARTUP_AND_ROUNDS_VERIFIED，短程5新+2参考已核；新增70轮离机接受16/96 | tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_SEVEN_CANARY_CLOSURE.json及ROOT_COVERAGE_STARTUP.json |
| 组合基线验证搜索 | 22/32项已严格验收、离机并通过本机来源绑定的记录复核；尚未完整选recipe | tmp/celeba_hybrid_screen_execution_20261009/LATEST_BACKUP.json |
| 九方法旧checkpoint三视图评价 | 900/900已严格验收并离机；原CPU872服务因native偏差failstop EXITED，不重启 | tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009/chunk_039/cumulative_900_accepted.json |
| 机制三视图评价 | 累计170份：minus_U完整100，minus_C七场景70对/140记录表已独立采用：五IID及non-IID Benign/F Flip，各10seed；1134统计/567单元/1260计数指标/3360计数与630配对指标通过；旧120记录/972统计/486单元及162个IID seed-first标量保持，不计算不平衡七场景总均值。新增F Flip删除C的native/shared差为ACC+0.943pp、AEOD−0.00308、ASPD+0.01177；全部10/9/6面板及负结果保留。Full5CPU/65GPU、C70CPU与训练环境/选择史披露；其他三non-IID C场景及六变体未完成。最新完整英文稿仍封存C60，尚未合入第七场景，正文未应用、未运行test；U论文表10完整场景 | U表：celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md；C表：docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_seven_scenes_20261010/snapshot/TABLES.md；其他七variant未完成 |

准确11份C补集三视图已正常退出、0残留worker并严格验收离机，累计C12、U100。110个归档成员、99指标/264计数/33规则通过，native偏差全0；原101份及Full不重推。凭据tmp/celeba_mechanism_valid_C_after1_20261009/execution_candidate/backups/incremental_20261009T193419Z/ROOT_ADOPTION_REVIEW.json。C IID Benign十seed评价齐备，该单场景三视图表亦已独立验收；该历史11项增量只含F Flip两seed；后续8项已补齐F Flip十seed并完成两场景表验收，其余八个C场景仍未齐。 C IID Benign三视图表现已核验162统计标量、81展示单元和216计数指标，入口docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_Benign10_20261009/snapshot/TABLES.md。raw删除C后ACC−0.014pp、AEOD−0.00483、ASPD−0.00383；native/shared为−0.083pp、+0.00292、−0.00142。保留10/9/6面板及Full2CPU/8GPU对C10CPU、环境/选择历史；不作C必要性、因果或显著性主张。

历史8项C/IID/F Flip seed91003–91010终轮checkpoint评价状态EXACT8_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED，新增离机接受8。只重建valid三视图，CPU112–119/8线程/nice10/idleIO/CUDA隐藏；旧112与Full不重推。最新C表入口docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_seven_scenes_20261010/snapshot/TABLES.md，其余3个C场景仍待完成。

删除C的IID Benign native十seed论文表已独立核验54统计标量/27展示单元/10对checkpoint，保留9/6seed面板；入口celeba_mechanism_v1/native_C_Benign10_20261009/TABLES.md。十seed配对删除差ACC−0.083个百分点、AEOD+0.00292、ASPD−0.00142，9/6面板方向有变化，不作必要性/因果/显著性结论。该封存单场景快照中的F Flip只有两seed、不纳入其均值；最新完整场景评价与表以本页当前表记录为准。

主机制服务guardfed_celeba_mechanism_formal，固定70round/valid-only/8并发，IID(alpha5000)/non-IID(alpha5)×5场景×10共享seed；100 Full身份已复核，旧权重不重训/重复打包。FLGMM原搜索服务已正常EXITED、0worker，32/32严格离机，冻结规则选Tg20/L2/lr0.001，32评分与32候选均值标量经独立及root复核；前两分差0.00004978，仅n=1搜索不作SD或显著性结论。其后5新+2参考短程已严格离机，FLGMM完整覆盖启动状态True，新96项仍待逐项严格验收，不把短程计入论文样本。组合基线服务guardfed_celeba_hybrid_screen32仍运行，GPU0/CPU104单线程，未选完整recipe，组合100项尚未启动。全程不运行test。

主机制最近实测CPU 11.42/122.88核，RAM 75.60GB，磁盘余1.061TB；GPU/温度/RecoveryAction与近期错误读同一实时JSON。只在真实轮次/日志、进程身份和资源证据支持时判断健康，低瞬时占用不重启。服务标签与完成文件不代替验收。

## 当前恢复与研究选择

原CPU失败为FairGuard/IID/F Flip/seed91009：native超原1e-12，65成员失败现场完整保留，原chunk036的10份strict partial当时未登记，后来通过显式审阅导入派生436账本。独立单模型GPU诊断已复现原三指标，差值全0；当前CPU/GPU native/raw只有image172599一处翻转，共享校准预测无翻转。三份归档与保存数组已独立验收；缺历史GPU逐图数组，不声称唯一历史根因，原424账本保持不变。凭据NATIVE_GPU_DIAGNOSTIC_ROOT_VERIFICATION.json。

旧436来源链及新GPU队列的前2批已严格验收、离机登记，累计460/900（CPU434、GPU26）；import11新增CNN推理0，旧424/425/436账本及原CPU失效现场不改。原464范围按43批次、每批至多11条、单GPU/单线程派发；实际CPU106协调、CPU105 worker/nice10已核。已登记与远端闭合分开统计，保存预测按原规则重建9指标、24混淆计数；root重拟合核验引用原远端strict，未声称本机重新拟合。第三批在worker资源预检、CNN之前触发Protected main800 health failed而自动停下。当前主训练正常，但失败瞬间的三个健康条件未保存原始截图，不能唯一归因为任务交接。原服务未重启，两条成功partial随后通过原strict部分验收与保存数组复核显式登记，新增CNN推理0；仍缺440条，修复需独立版本和不重复已完成项的补集。独立V2工程修复已通过root源码差异审阅、60个健康组合、11个资源拒收边界及两份实际快照schema核验，仅资源预检与输入保全变化，原科学body/strict及24个成员字节不变；新440精确补集已在独立目录启动，2026-10-09T15:24:25.802273+00:00实际CPU106协调、CPU105单GPU/单线程worker、nice10/idle及资源凭据通过；观测440条推理正常退出、远端闭合440条，新增离机接受0，累计仍为460。初次supervisor审批文件名不一致导致contract前退出、未创建输出或推理；原日志与配置保存后仅修正包外路径，现用配置SHA71826d102a628eb9fa0869dfa9f360a71527f2c595d8467d7464f6b720f429f5。V2队列随后已有440条通过原严格验收、全部archive/member SHA和独立保存数组复核，累计900/900（CPU434/GPU466），仍缺0；原460账本不改，离机验收新增CNN推理0。 native1e-12、原model/source/data/map/root/valid、同checkpoint全部视图与失败保留规则不变；混合CPU/GPU来源不能冒充统一设备的最终公平比较。入口tmp/celeba_valid_gpu_recovery_implementation_20261009/README.md。监控不自动启动准备包。

LoGoFair虚拟人口映射提案已独立核验：四条件共用固定image-ID哈希20组，root/valid顺序相同，80个root(label,Male)格最小116人；未解码valid标签/score、未拟合或评价。虚拟群体不是原训练client，人口定义仍待用户裁定；原32草案的mapping SHA仍null。入口tmp/celeba_logofair_population_proposal_20261009/REPORT.md。

## 已完成证据与剩余交付

2454项历史新增训练、九方法900原始验证结果及旧备份保持原值；当前九方法三视图评价已验收900/900。重放为混合CPU/GPU来源的验证集评价，最终test仍未完成。旧TableII的480个原值已追溯实际重复数，缺乏依据的SD不补造。完整入口REBUTTAL_COMPLETION_20261009.md；英文rebuttal已对齐24块原意见、40处本地引用及209项SHA声明，尚不能把待补实验写成完成。

原20项cu130与另20项cu128真实图像三轮门检已严格接受、离机，两套Full与原worker短程张量/指标/诊断精确。Fed-NGA/Huber四条真实图像三轮门检含240梯度/攻击oracle已接受；FLGMM的CPU2/GPU4及组合基线的CPU4/CUDA4门检已接受。所有恒定负类和工程/数值失败保留。三轮门检不证明70轮跨环境等价或科学性能优势；门检服务已EXITED，不重启。

完整17行比较仍缺8方法的完整多seed结果：LoGoFair、Fed-NGA、FedWA、Huber、FLGMM、SmartFL、FedDNA及组合控制。梯度方法正式协议、LoGoFair人口和最终评价主终点/测试边界仍待裁定；FedWA/SmartFL/FedDNA忠实规格仍缺，不能用简化旧分支冒充。主机制800、完整机制三视图、冻结最终评价、正文及最终回复仍未完成。Fig3原脚本/ForestDiffusion执行身份仍缺；已核数值与缺失来源明确区分。

已接受native场景的10/9/6种子中期论文表：celeba_mechanism_v1/native_interim100_20261009/rendered/TABLES.md。仅展示10个齐备的Full–minus_U配对场景，保留所有指标及取舍，不补造未完成场景，不以Full最佳seed对比消融均值。在IID Sp-DFA场景，Full准确率较高、去U的两个公平性差距更低，不能声称每项不可或缺。AEOD为绝对TPR差，不是完整equalized odds；Full98cu128+2cu130、多数旧driver570.211.01和当前driver595.84差异、seed91001选择历史均披露。native含各方法原校准，不能据此单独证明聚合机制。native100十场景表已单独核验540统计标量，旧九场景54展示行不变；所有十场景删除U的ACC/ASPD均更低，AEOD八场景更高、两场景更低。non-IID FedSA/S-DFA删除U后ACC分别下降0.601/0.453个百分点，公平性方向存在取舍。九场景三视图已由另一份root凭据独立闭合，数量与来源见下方；不从native表推断评价完成。

十个完整场景、100对Full–minus_U checkpoint的raw/native/shared三视图表已独立验收：1620均值/样本SD标量、1800组计数指标和810展示单元复算通过；旧九场景243统计行及184记录精确保持。IID/non-IID×五场景×十seed已齐，另外统一保留9/6seed面板，native/shared在200记录相同。新增non-IID Sp-DFA删除U的raw准确率下降0.151个百分点、两个差距均值上升；native准确率下降0.121个百分点但ASPD下降，保留取舍。Full5CPU/95GPU及98cu128/2cu130，control100CPU/cu128；混合设备、历史环境、验证选择和test暴露仍披露。此表只闭合minus_U，不是其余七variant完成或最终test，也不能证明每项不可或缺。入口celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md。

九方法三视图论文表已另行完成并通过root实际900份原receipt连接、8100个组计数指标重建及4860个均值/样本SD核验：outputs/guardfed_tables/celeba_nine_method_three_view_20261009/README.md。完整IID/non-IID、五场景、10/9/6共享种子和三视图均平行保留；旧native三指标及展示表值精确一致，94个旧JSON的SD最后bit差异最大2.78e-17单独保留。英文24意见完整草稿入口docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010/rebuttal_integrated_20261009.md；最新完整C60作者审阅稿纳入五IID及non-IID Benign，U100/900旧证据保持；24原意见逐字、10处可逆修改、36数值pointer/18展示值/27方向/50链接通过，原C50两全文可逐字恢复；其他四non-IID C场景及六变体待完成，正文未应用、最终test未运行；后续C70七场景表已独立采用，新增non-IID F Flip尚未合入该封存全文；当前仅三non-IID C场景及六变体仍缺，不把旧稿范围当实时进度，保留COMPAS反例/真实n/环境/选择史及全部pending。旧封存稿保留。仍为作者审阅稿，正文源文件未应用，最终评价与其余方法/机制未写成完成。

准确11份机制三视图评价已严格验收并离机，累计71；non-IID F Flip十seed及FedSA seed91001，排除已闭合60。实际服务正常EXITED、11完成、0残留worker、无失败；120 archive成员、99指标、264混淆计数和33预测规则通过，native偏差全0。archive SHA cbbaab743385b5c3354a4d93538a66188d0fed5c96df6245e547c484c17e0e87；backups/incremental_20261009T152532Z/ROOT_ADOPTION_REVIEW.json单独核源码/数据/终轮及全部成员。历史六场景表保持不变；新七场景三视图表仍须实际Full配对和独立统计验收，不从71条数量推断表完成。

准确11份after71三视图valid评价已严格验收、离机备份及root登记，累计82份；仅覆盖此前native82减已闭合71，non-IID FedSA seed91002–91009及S-DFA seed91001–91003。实际服务正常EXITED、11完整、无残留/失败，110个archive成员、99指标/264计数/33规则均验证，native偏差全0。Full仅引用900已接受的原三视图身份，原71记录不变。现有七场景均值表保持原封存；新FedSA仍仅九seed，不能称八个完整场景。新批验收入口tmp/celeba_mechanism_valid_incremental_after71_20261009/execution_candidate/backups/incremental_20261009T163050Z/ROOT_ADOPTION_REVIEW.json。该历史评价范围不含其余variant；后续minus_C等native终轮单独验收，不从源准备推断评价完成。

新增准确10份after82模型评价在首worker审批检查停止：固定10项范围仍遇旧11项基数断言。实际EXITED、0worker、0完成、输出目录无结果文件，失败发生于运行时依赖绑定和CNN之前；新增离机接受0。原失败source/log/审批/启动检查完整保留，原82评价及主800训练不变；旧after82禁止重启。证据tmp/celeba_mechanism_valid_incremental_after82_20261009/execution_candidate/ROOT_FAILURE_REVIEW.json。独立工程修复须新namespace和有效10审批正向门检，准备不计启动或接受。 修复版仅改审批数量11→10和独立namespace/pin；有效10审批及拒收门检通过，原科学计算保持。实际新服务guardfed_celeba_mechanism_valid_after82_v2，状态COMPLETE_STRICT_OFFSERVER_NO_NEW_FULL_INFERENCE，新增离机接受10。Full与已接受82不重推，固定CPU112–119/8线程/nice10/idleIO/CUDA隐藏，后续观测和验收按实际凭据。

最后准确8项minus_U三视图评价已正常EXITED、0残留/失败，全部严格验收并离机，累计100份minus_U。89个归档成员、72指标/192计数/24规则验证通过，native偏差全0；原92及Full不重推、C4排除。独立验收凭据tmp/celeba_mechanism_valid_incremental_after92_20261009/execution_candidate/backups/incremental_20261009T183102Z/ROOT_ADOPTION_REVIEW.json；十场景三视图论文表须另有配对统计验收，不从评价数量推断表完成。

九方法校准解释已独立复算2052标量：先每seed平均十场景，再跨seed统计；原native下GuardFed的ACC/AEOD/ASPD均值分别优于6/8、8/8、7/8基线，shared下为7/8、4/8、1/8。这是均值方向，不是显著性或seed胜率。原生公平性优势不能全部归因于聚合。全部正负差、10/9/6面板保留，入口outputs/guardfed_tables/celeba_nine_method_view_attribution_20261009/REPORT.md；英文解释补稿rebuttal_validation900_addendum_20261009.md不替换原封存24意见稿，不改变主终点。

九方法三视图论文表已编译为9页A3横向PDF：outputs/guardfed_tables/celeba_nine_method_three_view_pdf_20261009/celeba_nine_method_three_view.pdf。raw/native/shared各3页10/9/6种子，2430个均值±SD单元及4860个展示数字与原TeX精确一致，九页视觉/页界检查通过。仅label唯一化及wrapper排版，原67成员封条保持；本次无统计重算、推理或test。

提交版源项目尚缺：paper.md虽是IEEEtran LaTeX，但属于不同标题、方法和表结构的历史稿；已核19输入SHA及三工作副本，未找到所查paper.bib/archi.png/inv.pdf，也未认证为提交PDF同版。正文未修改、完整构建未声称。有限检索记录manuscript_source_locator_20261009/REPORT.md；已询问提交版路径，其他训练继续。

Fig3终轮修正候选已交付outputs/guardfed_figures/synthetic_terminal_candidate_20261009/fig3_terminal_candidate.pdf：260条第70轮完整三指标、26设置及78均值独立核验，全部13设置/数据集保留；只一个历史seed，不算场景SD，不用旧逐列最优或争议FairScore。图中明确历史test可见、ForestDiffusion执行身份缺失/PCA标签局限及checkpoint二进制SHA未恢复。候选未采用，不替换原图，不宣称P4关闭。

## Git与巡检

最近已验证推送：3601c9dfca63dc1c6203aceb2c7fc066faa630fa，分支codex/revision-evidence-baselines-20260928，253份committed blob逐SHA及远端分支核验；后续本机变化未自动算作已推送。记录publication_closed_increment39_verified_20261010.json。

三小时聊天任务guardfed-training-health仍PAUSED；本会话没有原生automation_update工具，未编辑调度器或建立替代cron/Windows任务。supervisor持续运行训练不等于聊天巡检恢复。待原生接口可用时按server_reactivation_20261009/MONITOR_HANDOFF.md恢复同一任务；不从历史计划自动派发新队列。

下方为历史记录；当前事实以本入口、TRAINING_STATE.json及对应实际凭据为准。源/数据/冻结配置保持一致、无重复worker且已验收项严格跳过时才有限恢复外部中断；数值/逻辑错误保留证据，不循环重试、不改统计/seed/并发或driver/实例。


历史5项C/IID FedSA增量（seed91001/03/05/06/08）终轮valid三视图评价：EXACT5_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED，新增严格离机接受5。固定CPU112–119/8线程、nice10/idleIO/CUDA隐藏；原120与Full不重推。来源/实际预检与后续验收入口tmp/celeba_mechanism_valid_C_after20_20261009/execution_candidate。该批次闭合时FedSA仅5/10；最新覆盖以本页当前表与STATE为准，原科学计算及1e-12保持。


历史3项C/IID FedSA增量（seed91002/04/07）终轮valid三视图评价：EXACT3_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED，新增严格离机接受3。固定CPU112–119/8线程、nice10/idleIO/CUDA隐藏；原125与Full不重推。来源入口tmp/celeba_mechanism_valid_C_after25_20261009/execution_candidate。该批次闭合时FedSA为8/10；最新覆盖以本页当前表与STATE为准，原科学计算及1e-12保持。

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
