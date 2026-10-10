# CURRENT: GuardFed返修实验 — 实测 2026-10-10T17:11:19.358149+00:00

当前服务器：ssh -p60350 root@89.22.197.55，实例52183675；repo /workspace/GuardFed-celeba-expanded。用户明确授权停止sglang，模型/文件保留。213.224.31.105:26712当前内部状态未知，不自动切换。先遵守/etc/vast-agents-guide.md，既有SHA为42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa。

## 当前执行与验收

| 阶段 | 实际状态与分母 | 接续入口 |
|---|---|---|
| CelebA机制消融 | 当前观测完成296、活动8、等待496、失败0；已独立严格验收并离机295/800新增，另100 Full显式复用 | server_reactivation_20261009/latest_formal_live.json；celeba_mechanism_v1/EXECUTION.md及dispatch receipt |
| FLGMM验证搜索 | 32/32已严格验收并离机；最新来源绑定终轮/活动读STATE对应快照，不把未验收完成项计作接受 | tmp/celeba_flgmm_screen_20261009_v2_dispatch/LATEST_BACKUP.json及accepted_delta_after6_20261009/ROOT_ADOPTION_REVIEW.json |
| FLGMM完整覆盖 | 96新+4复用；状态ROOT_ACTUAL_FLGMM96_VALID_COVERAGE_STARTUP_AND_ROUNDS_VERIFIED，短程5新+2参考已核；新增70轮离机接受59/96 | tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_SEVEN_CANARY_CLOSURE.json及ROOT_COVERAGE_STARTUP.json |
| 组合基线验证搜索 | 32/32项已严格验收、离机并经独立复核；冻结规则选λ20/τ0.1/lr0.001，n=1 | tmp/celeba_hybrid_screen_execution_20261009/LATEST_BACKUP.json |
| 九方法旧checkpoint三视图评价 | 900/900已严格验收并离机；原CPU872服务因native偏差failstop EXITED，不重启 | tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009/chunk_039/cumulative_900_accepted.json |
| 机制三视图评价 | 累计295份：minus_U100、minus_C100各十完整场景；minus_A95，已采用4个完整IID场景。raw/native/shared及10/9/6seed面板保留，其他控制未齐 | U表：celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md；C表：docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_full100_20261010/snapshot/TABLES.md；A表：docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_four_scenes40_20261010/TABLES.md |

准确11份C补集三视图已正常退出、0残留worker并严格验收离机，累计C12、U100。110个归档成员、99指标/264计数/33规则通过，native偏差全0；原101份及Full不重推。凭据tmp/celeba_mechanism_valid_C_after1_20261009/execution_candidate/backups/incremental_20261009T193419Z/ROOT_ADOPTION_REVIEW.json。C IID Benign十seed评价齐备，该单场景三视图表亦已独立验收；该历史11项增量只含F Flip两seed；后续8项已补齐F Flip十seed并完成两场景表验收，其余八个C场景仍未齐。 C IID Benign三视图表现已核验162统计标量、81展示单元和216计数指标，入口docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_Benign10_20261009/snapshot/TABLES.md。raw删除C后ACC−0.014pp、AEOD−0.00483、ASPD−0.00383；native/shared为−0.083pp、+0.00292、−0.00142。保留10/9/6面板及Full2CPU/8GPU对C10CPU、环境/选择历史；不作C必要性、因果或显著性主张。

历史8项C/IID/F Flip seed91003–91010终轮checkpoint评价状态EXACT8_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED，新增离机接受8。只重建valid三视图，CPU112–119/8线程/nice10/idleIO/CUDA隐藏；旧112与Full不重推。最新C表入口docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_full100_20261010/snapshot/TABLES.md，其余0个C场景仍待完成。

删除C的IID Benign native十seed论文表已独立核验54统计标量/27展示单元/10对checkpoint，保留9/6seed面板；入口celeba_mechanism_v1/native_C_Benign10_20261009/TABLES.md。十seed配对删除差ACC−0.083个百分点、AEOD+0.00292、ASPD−0.00142，9/6面板方向有变化，不作必要性/因果/显著性结论。该封存单场景快照中的F Flip只有两seed、不纳入其均值；最新完整场景评价与表以本页当前表记录为准。

主机制服务guardfed_celeba_mechanism_formal，固定70round/valid-only/8并发，IID(alpha5000)/non-IID(alpha5)×5场景×10共享seed；100 Full身份已复核，旧权重不重训/重复打包。FLGMM原搜索服务已正常EXITED、0worker，32/32严格离机，冻结规则选Tg20/L2/lr0.001，32评分与32候选均值标量经独立及root复核；前两分差0.00004978，仅n=1搜索不作SD或显著性结论。其后5新+2参考短程已严格离机，FLGMM完整覆盖启动状态True，新96项仍待逐项严格验收，不把短程计入论文样本。组合基线32项验证搜索已正常EXITED、0worker，完整严格离机并独立复核；冻结规则选λ20/τ0.1/lr0.001，同时为准确率冠军及唯一三指标Pareto候选。n=1，不报跨seed SD/显著性；100格已生成96新+4显式复用清单；七个真实三轮门检已全部严格验收及离机，254归档成员、两组同轮数新旧对照通过；96新增70轮valid队列已实际启动，另4旧匹配结果显式复用；首worker实测round1，启动时新增70轮接受0，最新严格离机根接受1/96。全程不运行test。

主机制最近实测CPU 12.67/122.88核，RAM 78.29GB，磁盘余1.060TB；GPU/温度/RecoveryAction与近期错误读同一实时JSON。只在真实轮次/日志、进程身份和资源证据支持时判断健康，低瞬时占用不重启。服务标签与完成文件不代替验收。

## 当前恢复与研究选择

原CPU失败为FairGuard/IID/F Flip/seed91009：native超原1e-12，65成员失败现场完整保留，原chunk036的10份strict partial当时未登记，后来通过显式审阅导入派生436账本。独立单模型GPU诊断已复现原三指标，差值全0；当前CPU/GPU native/raw只有image172599一处翻转，共享校准预测无翻转。三份归档与保存数组已独立验收；缺历史GPU逐图数组，不声称唯一历史根因，原424账本保持不变。凭据NATIVE_GPU_DIAGNOSTIC_ROOT_VERIFICATION.json。

旧436来源链及新GPU队列的前2批已严格验收、离机登记，累计460/900（CPU434、GPU26）；import11新增CNN推理0，旧424/425/436账本及原CPU失效现场不改。原464范围按43批次、每批至多11条、单GPU/单线程派发；实际CPU106协调、CPU105 worker/nice10已核。已登记与远端闭合分开统计，保存预测按原规则重建9指标、24混淆计数；root重拟合核验引用原远端strict，未声称本机重新拟合。第三批在worker资源预检、CNN之前触发Protected main800 health failed而自动停下。当前主训练正常，但失败瞬间的三个健康条件未保存原始截图，不能唯一归因为任务交接。原服务未重启，两条成功partial随后通过原strict部分验收与保存数组复核显式登记，新增CNN推理0；仍缺440条，修复需独立版本和不重复已完成项的补集。独立V2工程修复已通过root源码差异审阅、60个健康组合、11个资源拒收边界及两份实际快照schema核验，仅资源预检与输入保全变化，原科学body/strict及24个成员字节不变；新440精确补集已在独立目录启动，2026-10-09T15:24:25.802273+00:00实际CPU106协调、CPU105单GPU/单线程worker、nice10/idle及资源凭据通过；观测440条推理正常退出、远端闭合440条，新增离机接受0，累计仍为460。初次supervisor审批文件名不一致导致contract前退出、未创建输出或推理；原日志与配置保存后仅修正包外路径，现用配置SHA71826d102a628eb9fa0869dfa9f360a71527f2c595d8467d7464f6b720f429f5。V2队列随后已有440条通过原严格验收、全部archive/member SHA和独立保存数组复核，累计900/900（CPU434/GPU466），仍缺0；原460账本不改，离机验收新增CNN推理0。 native1e-12、原model/source/data/map/root/valid、同checkpoint全部视图与失败保留规则不变；混合CPU/GPU来源不能冒充统一设备的最终公平比较。入口tmp/celeba_valid_gpu_recovery_implementation_20261009/README.md。监控不自动启动准备包。

LoGoFair原映射提案保留：四条件共用固定image-ID哈希20组，80个root(label,Male)格最小116人。用户2026-10-10委托主代理决定，现已采用该虚拟cohort，明确不能称真实训练client公平性。实际真实分数3post-round接口门检通过40个Beta拟合、19,867验证样本及序列化重载预测exact，但恒定负预测保留，只作流程证据、科学结果0；原32草案不改，新搜索包独立冻结，尚不能称32项完成。门检及root审查入口tmp/celeba_logofair_real_gate_20261010/ROOT_REVIEW.json。

## 已完成证据与剩余交付

2454项历史新增训练、九方法900原始验证结果及旧备份保持原值；当前九方法三视图评价已验收900/900。重放为混合CPU/GPU来源的验证集评价，最终test仍未完成。旧TableII的480个原值已追溯实际重复数，缺乏依据的SD不补造。完整入口REBUTTAL_COMPLETION_20261009.md；英文rebuttal已对齐24块原意见、40处本地引用及209项SHA声明，尚不能把待补实验写成完成。

原20项cu130与另20项cu128真实图像三轮门检已严格接受、离机，两套Full与原worker短程张量/指标/诊断精确。Fed-NGA/Huber四条真实图像三轮门检含240梯度/攻击oracle已接受；FLGMM的CPU2/GPU4及组合基线的CPU4/CUDA4门检已接受。所有恒定负类和工程/数值失败保留。三轮门检不证明70轮跨环境等价或科学性能优势；门检服务已EXITED，不重启。

完整17行比较现有10方法native千格表已独立采用，仍缺7方法完整多seed：Fed-NGA、FedWA、Huber、FLGMM、SmartFL、FedDNA及组合控制。用户2026-10-10已接受Huber的R^p恒等投影CNN适配，仅报告经验结果、不继承原理论保证；LoGoFair人口亦已决定，梯度五个常规字段按原算法落实，不再把H/L写成待决。64搜索源包已冻结、独立审查并实际启动；此前宽CPU调度mask及日志tee误判均为训练前工程失败，已保留并修复，最新实测须读STATE新增凭据。最终评价主终点/测试边界仍待冻结；FedWA/SmartFL/FedDNA忠实规格仍缺，不能用简化旧分支冒充。主机制800、完整机制三视图、正文及最终回复仍未完成。Fig3原脚本/ForestDiffusion执行身份仍缺；已核数值与缺失来源明确区分。

已接受native场景的10/9/6种子中期论文表：celeba_mechanism_v1/native_interim100_20261009/rendered/TABLES.md。仅展示10个齐备的Full–minus_U配对场景，保留所有指标及取舍，不补造未完成场景，不以Full最佳seed对比消融均值。在IID Sp-DFA场景，Full准确率较高、去U的两个公平性差距更低，不能声称每项不可或缺。AEOD为绝对TPR差，不是完整equalized odds；Full98cu128+2cu130、多数旧driver570.211.01和当前driver595.84差异、seed91001选择历史均披露。native含各方法原校准，不能据此单独证明聚合机制。native100十场景表已单独核验540统计标量，旧九场景54展示行不变；所有十场景删除U的ACC/ASPD均更低，AEOD八场景更高、两场景更低。non-IID FedSA/S-DFA删除U后ACC分别下降0.601/0.453个百分点，公平性方向存在取舍。九场景三视图已由另一份root凭据独立闭合，数量与来源见下方；不从native表推断评价完成。

十个完整场景、100对Full–minus_U checkpoint的raw/native/shared三视图表已独立验收：1620均值/样本SD标量、1800组计数指标和810展示单元复算通过；旧九场景243统计行及184记录精确保持。IID/non-IID×五场景×十seed已齐，另外统一保留9/6seed面板，native/shared在200记录相同。新增non-IID Sp-DFA删除U的raw准确率下降0.151个百分点、两个差距均值上升；native准确率下降0.121个百分点但ASPD下降，保留取舍。Full5CPU/95GPU及98cu128/2cu130，control100CPU/cu128；混合设备、历史环境、验证选择和test暴露仍披露。此表只闭合minus_U，不是其余七variant完成或最终test，也不能证明每项不可或缺。入口celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md。

九方法三视图论文表已另行完成并通过root实际900份原receipt连接、8100个组计数指标重建及4860个均值/样本SD核验：outputs/guardfed_tables/celeba_nine_method_three_view_20261009/README.md。完整IID/non-IID、五场景、10/9/6共享种子和三视图均平行保留；旧native三指标及展示表值精确一致，94个旧JSON的SD最后bit差异最大2.78e-17单独保留。英文24意见完整草稿入口docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A90_reader_20261011/rebuttal_integrated_20261011.md；最新完整作者审阅稿已纳入U/C各十场景、A四IID场景、LoGoFair100及十方法native千格表；24原意见逐字、两旧全文可逆恢复、59均值SD单元/118数值pointer/69事实/108方向/88链接经root及独立语义复核通过。A40全部指标取舍和10/9/6面板保持；七方法完整覆盖、六机制变体、P1–P6、最终评价及正文仍未完成，保留COMPAS反例/真实n/环境/选择史及全部pending。旧封存稿保留。仍为作者审阅稿，正文源文件未应用，最终评价与其余方法/机制未写成完成。

准确11份机制三视图评价已严格验收并离机，累计71；non-IID F Flip十seed及FedSA seed91001，排除已闭合60。实际服务正常EXITED、11完成、0残留worker、无失败；120 archive成员、99指标、264混淆计数和33预测规则通过，native偏差全0。archive SHA cbbaab743385b5c3354a4d93538a66188d0fed5c96df6245e547c484c17e0e87；backups/incremental_20261009T152532Z/ROOT_ADOPTION_REVIEW.json单独核源码/数据/终轮及全部成员。历史六场景表保持不变；新七场景三视图表仍须实际Full配对和独立统计验收，不从71条数量推断表完成。

准确11份after71三视图valid评价已严格验收、离机备份及root登记，累计82份；仅覆盖此前native82减已闭合71，non-IID FedSA seed91002–91009及S-DFA seed91001–91003。实际服务正常EXITED、11完整、无残留/失败，110个archive成员、99指标/264计数/33规则均验证，native偏差全0。Full仅引用900已接受的原三视图身份，原71记录不变。现有七场景均值表保持原封存；新FedSA仍仅九seed，不能称八个完整场景。新批验收入口tmp/celeba_mechanism_valid_incremental_after71_20261009/execution_candidate/backups/incremental_20261009T163050Z/ROOT_ADOPTION_REVIEW.json。该历史评价范围不含其余variant；后续minus_C等native终轮单独验收，不从源准备推断评价完成。

新增准确10份after82模型评价在首worker审批检查停止：固定10项范围仍遇旧11项基数断言。实际EXITED、0worker、0完成、输出目录无结果文件，失败发生于运行时依赖绑定和CNN之前；新增离机接受0。原失败source/log/审批/启动检查完整保留，原82评价及主800训练不变；旧after82禁止重启。证据tmp/celeba_mechanism_valid_incremental_after82_20261009/execution_candidate/ROOT_FAILURE_REVIEW.json。独立工程修复须新namespace和有效10审批正向门检，准备不计启动或接受。 修复版仅改审批数量11→10和独立namespace/pin；有效10审批及拒收门检通过，原科学计算保持。实际新服务guardfed_celeba_mechanism_valid_after82_v2，状态COMPLETE_STRICT_OFFSERVER_NO_NEW_FULL_INFERENCE，新增离机接受10。Full与已接受82不重推，固定CPU112–119/8线程/nice10/idleIO/CUDA隐藏，后续观测和验收按实际凭据。

最后准确8项minus_U三视图评价已正常EXITED、0残留/失败，全部严格验收并离机，累计100份minus_U。89个归档成员、72指标/192计数/24规则验证通过，native偏差全0；原92及Full不重推、C4排除。独立验收凭据tmp/celeba_mechanism_valid_incremental_after92_20261009/execution_candidate/backups/incremental_20261009T183102Z/ROOT_ADOPTION_REVIEW.json；十场景三视图论文表须另有配对统计验收，不从评价数量推断表完成。

九方法校准解释已独立复算2052标量：先每seed平均十场景，再跨seed统计；原native下GuardFed的ACC/AEOD/ASPD均值分别优于6/8、8/8、7/8基线，shared下为7/8、4/8、1/8。这是均值方向，不是显著性或seed胜率。原生公平性优势不能全部归因于聚合。全部正负差、10/9/6面板保留，入口outputs/guardfed_tables/celeba_nine_method_view_attribution_20261009/REPORT.md；英文解释补稿rebuttal_validation900_addendum_20261009.md不替换原封存24意见稿，不改变主终点。

九方法三视图论文表已编译为9页A3横向PDF：outputs/guardfed_tables/celeba_nine_method_three_view_pdf_20261009/celeba_nine_method_three_view.pdf。raw/native/shared各3页10/9/6种子，2430个均值±SD单元及4860个展示数字与原TeX精确一致，九页视觉/页界检查通过。仅label唯一化及wrapper排版，原67成员封条保持；本次无统计重算、推理或test。

提交版源项目尚缺：paper.md虽是IEEEtran LaTeX，但属于不同标题、方法和表结构的历史稿；已核19输入SHA及三工作副本，未找到所查paper.bib/archi.png/inv.pdf，也未认证为提交PDF同版。正文未修改、完整构建未声称。有限检索记录manuscript_source_locator_20261009/REPORT.md；已询问提交版路径，其他训练继续。

Fig3终轮修正候选已交付outputs/guardfed_figures/synthetic_terminal_candidate_20261009/fig3_terminal_candidate.pdf：260条第70轮完整三指标、26设置及78均值独立核验，全部13设置/数据集保留；只一个历史seed，不算场景SD，不用旧逐列最优或争议FairScore。图中明确历史test可见、ForestDiffusion执行身份缺失/PCA标签局限及checkpoint二进制SHA未恢复。候选未采用，不替换原图，不宣称P4关闭。

## Git与巡检

最近已验证推送：0f67597e34f3b3454bbc1600708c649904976d7a，分支codex/revision-evidence-baselines-20260928，511份committed blob逐SHA及远端分支核验；后续本机变化未自动算作已推送。记录publication_closed_increment54_verified_20261011.json。

三小时聊天任务guardfed-training-health仍PAUSED；本会话没有原生automation_update工具，未编辑调度器或建立替代cron/Windows任务。supervisor持续运行训练不等于聊天巡检恢复。待原生接口可用时按server_reactivation_20261009/MONITOR_HANDOFF.md恢复同一任务；不从历史计划自动派发新队列。

下方为历史记录；当前事实以本入口、TRAINING_STATE.json及对应实际凭据为准。源/数据/冻结配置保持一致、无重复worker且已验收项严格跳过时才有限恢复外部中断；数值/逻辑错误保留证据，不循环重试、不改统计/seed/并发或driver/实例。


历史5项C/IID FedSA增量（seed91001/03/05/06/08）终轮valid三视图评价：EXACT5_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED，新增严格离机接受5。固定CPU112–119/8线程、nice10/idleIO/CUDA隐藏；原120与Full不重推。来源/实际预检与后续验收入口tmp/celeba_mechanism_valid_C_after20_20261009/execution_candidate。该批次闭合时FedSA仅5/10；最新覆盖以本页当前表与STATE为准，原科学计算及1e-12保持。


历史3项C/IID FedSA增量（seed91002/04/07）终轮valid三视图评价：EXACT3_COMPLETE_STRICT_OFFSERVER_ROOT_ADOPTED，新增严格离机接受3。固定CPU112–119/8线程、nice10/idleIO/CUDA隐藏；原125与Full不重推。来源入口tmp/celeba_mechanism_valid_C_after25_20261009/execution_candidate。该批次闭合时FedSA为8/10；最新覆盖以本页当前表与STATE为准，原科学计算及1e-12保持。


新增梯度搜索已实际启动：Fed-NGA32+Huber32，70round/valid-only/n=1；新服务guardfed_celeba_gradient_screen64_v2a，1 worker/CPU105/GPU1/nice10/idle，启动时round 18及physical GPU UUID已核。最新实测终轮70任务43/64，终轮严格离机根接受42，二者分开记录。首Fed-NGA候选constant-negative（ACC0.516686、AEOD/ASPD0）保留，不作冠军判断。前两次root预检把宽调度mask、日志tee误判资源/重复worker的工程错误均在训练前退出并保留；科学source/64jobs原字节不变，既有800队列未改。实际凭据入口tmp/celeba_gradient_screen64_v2_root_operations_20261010/ROOT_STARTUP_REVIEW.json。

LoGoFair32原30post-round搜索在F运行，4个原接受模型/cache已实际提取并核SHA；已原strict闭合32/32，root adopted32；不重训CNN、不评价test、不选未齐recipe，保留恒定预测与全部候选。输出F:\YananResearchStorage\GuardFed\logofair_screen32_20261010\attempt001。
完整32原strict及635744项保存预测复核通过，冻结规则选LoGoFair-DP_07（global/local delta0.06、post_lr0.005、30轮）；准确率冠军不同、四项Pareto和8项恒定预测保留。单模型seed91001/fit_seed1719，四条件不当独立seed；虚拟20cohort非真实client公平性。

LoGoFair固定配置100覆盖已完整root采用：96新30轮后处理+4显式复用，IID/non-IID×五场景×十模型seed、fit_seed1719固定。原strict与1,986,700保存预测/300指标复算误差0，独立1027哈希/198统计/99展示通过；1项恒定预测及10/9/6面板完整保留。虚拟20cohort非真实client公平性，不重训CNN或test。bulk留F，四compact报告及恢复索引在本机；入口docs/server_deployment_20260923/training_20260923/celeba_logofair100_accepted_20261010/TABLES.md。
十方法native论文表已独立采用：1000格、IID/non-IID各五场景，10/9/6seed面板；原九方法810统计对象精确保持，1800场景统计/900展示/540先seed内汇总统计/270汇总展示通过。LoGo为原fit_DP native，不能称1000三视图或final；入口outputs/guardfed_tables/celeba_ten_method_native_20261010/TABLES.md。

剩余620机制终轮valid三视图评价已实际启动guardfed_celeba_mechanism_remaining620_valid_v2a，排除原180与Full重复推理；CPU112–119单进程8计算线程/nice10/idleIO/CUDA隐藏，首条原strict闭合native差0且绑定原已接受checkpoint。最新remote闭合116，新离机根接受115；不把服务运行或服务器闭合计入论文表。首次taskset包装语法错误发生在Python执行前，失败字节保留。首条运输因原验证工具路径缺失停止后，已有归档未重建，原49成员与9指标/24计数/3规则离机核验并与原native188 checkpoint恢复链精确join；单次有限恢复凭据独立保存，不盲重试、不重复推理。入口tmp/celeba_mechanism_remaining620_root_operations_20261010/ROOT_STARTUP_REVIEW_V2.json。
C100完整IID/non-IID×五场景×十seed三视图表已独立采用；1620统计/810单元/1800指标/4800计数及全部配对、seed-first汇总通过，旧C80保持。入口celeba_mechanism_v1/three_view_C_full100_20261010/snapshot/TABLES.md；其他六变体及最终评价/正文未完成。
A单场景已核162统计/81展示单元/216计数指标，入口docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_Benign10_20261010/TABLES.md；native/shared删除差ACC−0.430个百分点、AEOD+0.002313、ASPD−0.003142。9/6面板方向变化与Full2CPU8GPU对A10CPU保留，不作必要性/因果/显著性主张。
A两完整IID场景表已独立采用：Benign/F Flip各十共享seed，324统计/162展示/360计数指标/960基础计数通过；旧24记录和Benign162统计/81展示保持。F Flip native/shared删除A差ACC约+0.001pp、AEOD约+0.00001、ASPD约−0.00088，9/6面板方向变化完整保留；不作每场景不可或缺主张。入口docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_two_scenes20_20261010/TABLES.md。其他八A场景及六变体余项未齐。
两项作者适配决定的英文回复/正文局部补丁已根核：24审稿原话保持、两个替换段可逆、原数字未变；正文未应用，入口docs/server_deployment_20260923/revision_20260923/rebuttal_adaptations_20261010/REBUTTAL_PATCH.md。
Fed-NGA/Huber完整200格的192新+8复用源准备已独立审查通过，仅为source-only：未选recipe、未生成实际jobs、未启动。须等待全部64搜索严格离机、冻结实际配置及新增攻击真实图像门检；不从准备文件推断实验完成。
新增攻击14项共同三轮门检源码已独立审查通过，实际图像门检仍0；仅source-only，不授权派发192。原正式70轮验收未放宽。
新增五方法三视图接线仅完成源码范围审查：四CNN标签须接各自原strict，LoGo原生必须保留DP后处理与虚拟映射。其cache的valid_native_prediction是FedAvg原始预测，不得冒充LoGo native；backbone raw/shared仅可标为诊断。没有新增评价/拟合，最终主终点未定。

组合100接线已在独立v3目录实际绑定147个metadata成员，原Python3.10汇总在服务器3.12用显式顺序求和精确重放，未放宽容差/修改选择。原v2绑定失败发生在任务生成前，证据保留。七门检CPU104/GPU0单worker，启动前主队列真实增长/GPU RecoveryNone/CPU配额通过；门检科学70轮样本0。实际入口tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010/ROOT_CANARY_STARTUP.json。
组合七门队列已正常EXITED，原strict/保存终轮张量/RNG摘要及全部成员哈希通过root验收；三轮不能证明70轮等价或每轮权重完整，论文样本0。入口tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010/ROOT_SEVEN_CANARY_CLOSURE.json。
组合完整覆盖CPU104/GPU0单worker/一计算线程，首worker PID85931的真实GPU UUID、源码/数据/配置身份及第一轮已核。旧canary授权原字节保留，新授权精确绑定七门离机root凭据；无自动重试、不运行test。入口tmp/celeba_hybrid_fullcoverage_root_binding_v3_20261010/ROOT_COVERAGE_STARTUP.json。

最新A40四完整IID场景表已独立采用：Benign/F Flip/FedSA/S-DFA各十共享seed，648均值SD/324展示/720计数指标/1920计数通过，旧A20的40对象字节/顺序、324统计/162展示精确保持。A40表采用时累计三视图240；S-DFA删除A的native/shared差ACC−0.361pp、AEOD+0.00542、ASPD−0.00804，FedSA为+0.134pp/−0.00065/+0.00141，各指标取舍和9/6方向变化均保留。Full3CPU37GPU/A40CPU、各40cu128；六其他A场景、剩余控制和最终评价/正文未齐，不作必要性/因果/显著性主张。入口docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_A_IID_four_scenes40_20261010/TABLES.md。

本机存储：2026-10-10用户指定F:/YananResearchStorage/GuardFed/；大文件写入前实核F卷标Yanan 2TB与容量，无内置盘回退。代码/配置/索引/精简报告保留E，服务器大文件优先原地保留；已存在E证据未删除或宣称全量迁移。记录LOCAL_STORAGE_20261010.json。

Hybrid首项70轮IID Benign seed91002已由原strict、188归档成员和8张量摘要/源码数据配置身份独审及root采用，新增1/96；4旧复用另计，7三轮门检不算正式样本。不生成单seed场景SD，不称100覆盖完成。入口tmp/celeba_hybrid_first1_root_adoption_20261010/ROOT_ADOPTION.json。

十方法native验证集表PDF已交付：三页分别为10/9/6共享seed，IID/non-IID各五场景，900组均值与sampleSD/1800数值逐项匹配源表，三页视觉检查通过。入口outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/celeba_ten_method_native.pdf；仍缺余七方法完整覆盖及冻结最终评价。
最新11条保存数组已独立核验101归档成员、99指标/264计数/33规则及22原native成员，累计三视图251=U100+C100+A51；native差0、原240保持。A的五IID场景各十seed齐备，non-IID Benign仅1/10；既有A40四场景表保持，新统计尚未汇总，不用部分seed填完整表。入口tmp/celeba_mechanism_remaining620_after240_root_adoption_20261010/ROOT_ADOPTION.json。
四新增CNN的三视图身份桥源码及64拒收/边界检查经root复跑，17原评价函数保持；当前仅注册旧FL6/Hybrid1/NGA8精确接受chunk，无Huber70轮proof则拒收。没有新增科学评价/fit/训练或test，不把准备当完成。
