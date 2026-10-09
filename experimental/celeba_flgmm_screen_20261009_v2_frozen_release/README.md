# FLGMM 32 项验证搜索：可审阅准备包

状态：**PREPARED_NOT_FROZEN；未上传、未安装服务、未启动训练。** 本包只准备已获授权的剩余基线扩展范围；最终启动由 root 审阅冻结 release、实测资源和重复进程后放行。

## 固定科学范围

Tg {10,20} × L {2,3} × local Adam LR {.0005,.001}，8 候选；IID alpha5000 / non-IID alpha5 × Benign / S-DFA，seed91001，共32项，每项70轮。train162770、clean train-root16277、valid19867，原 RGB64 CNN / Adam / batch64 / local epoch1 / 20客户端 / 4配置恶意客户端 / 重加权与攻击不变。全部原始负结果保留；不执行 final test 像素推理/评估、不进行 test 选择；完整缓存SHA读取包含test字节，共同旧 loader 仍物化全 split 标签元数据，不声称完全未触碰 test。

原 `prepare_jobs.py` 原样生成 `jobs/` 32任务及 manifest。原 `worker.py`、`accept_result.py`、adapter、pinned作者代码及许可证不改。原9成员中只有新副本 protocol 更新阶段事实，原 protocol 存 `evidence/original_protocol.json`；其余8文件逐字节保持不变，完整科学/数据21 SHA不变。`PROTOCOL_DIFF.patch` 展示完整协议差异，原封存 GroupB/CPU/GPU v2/v3 均不改。

原生成器的 manifest.next_action 固定文案仍提到 real-image gate review；保留此原始生成字节，不作为实时状态。已通过门检事实见新 protocol.passed_gate 与随附receipt，执行状态以外部授权/启动receipt为准。

评分 `frozen_score.py` 是已归档 `verify_rank.py` 的 `score` 函数原文提取（完整来源保留），没有重新实现权重。每个候选四条件分别评分再平均，全部8候选完整后选最高分；精确同分按 candidate ID 字典序。不把四条件当四seed，不报告sample SD/显著性。

## 两阶段冻结与启动

1. root核 `PACKAGE_SHA256.json`、协议diff、32 jobs与门检/备份凭据。
2. 在**新的、不存在的目录**执行 `python freeze_release.py --reviewed-sha <prepared seal SHA> --out <NEW_RELEASE>`。脚本只将新release协议状态改为 FROZEN，再调用原 prepare_jobs 重生成32任务，写 `FROZEN_PENDING_EXECUTION` 封条及来源链；不改此准备包，不安装或启动。
3. root核新release完整seal；发布到 `/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2`。使用已有isolated cu128 Python3.12.3 / torch2.11.0+cu128 / CUDA12.8 / RTX5090。再次核guide、CPU/GPU/内存/磁盘、RecoveryAction、主队列轮次和无重复FLGMMworker。只对本服务安装 `dispatch/` 的正常supervisor配置；原正式800服务不动。
4. root单独创建 `EXECUTION_AUTHORIZATION.json`，必须包含 `status:AUTHORIZED`、`scope:32_valid_only_screen`、**实际新release** `package_sha256`、`resource_review_utc`、`no_duplicate_workers_verified:true`。缺失此receipt时 runner拒绝运行。仅执行本服务的 `reread` / `update guardfed_celeba_flgmm_screen` / `start guardfed_celeba_flgmm_screen`。不使用全局update/restart。

服务最多两个worker，每GPU一任务，每worker一个CPU计算线程，nice10；autostart/autorestart=false/startretries0。线程库辅助线程数不等于计算核数；运行前以真实配额/已有任务预算为准。正常前台进程由supervisor管理；没有额外cron或聊天监督机制。

## 接受及恢复规则

`run_one.py` 只是原worker外层：训练前核release+21 repo/source/data SHA，原worker运行；终轮再核相同全部身份，原 `checked_result` 严格检查70轮/valid/same checkpoint/config/source/adapter/模型/诊断/controller后写独立 `screen_identity.json`，另外明确root16277。worker每轮原progress/state/diagnostics继续保留。原worker不提供完整优化器/RNG中轮恢复；v3的RNG记录专用于重复性门检，没有植入或改变原科学worker。

runner持OS排他锁防重复coordinator；跳过条件为原checked_result及外层before/after身份全部通过。结果/目录存在不等于通过。部分目录、任何failure或原队列失败记录阻止重启；不覆写、不自动重试、不自动改变attempt。异常立即停止派发并终止本coordinator仍持有的本队列worker，保留失败和部分产物。supervisor停止时kill整个本服务组；不触碰其他服务。

外部中断恢复必须先核无遗留worker、源码数据协议相同、原已完成任务严格接受、资源预算，并保留失败和partial目录。只有全为已接受或从未启动任务且无失败证据时，同一队列可严格skip后恢复；有partial/失败须root诊断并准备独立恢复attempt/显式映射，当前包不自动做此动作。日志存在而输出不存在也通过独占创建拒绝覆盖。

## 增量离机备份计划

以已严格接受job ID与上次已备份ID作差集。备份每个新增任务的全部job/config/provenance/model/result/原acceptance/外层identity/progress/state/diagnostics/已关闭日志，以及新release完整source/manifest/protocol/seal/授权/启动凭据。每份tar保留完整member路径、大小及SHA清单；归档SHA与逐member SHA在本机独立重算；已有模型不重复打包，显式保存恢复链。失败证据独立保留备份，不当科学样本。

32项完成后先全量原checker+包/数据终核，生成原指标、全部候选和冻结评分选择；summary仅标 `COMPLETE_ACCEPTED_BACKUP_PENDING`，离机核验后才可称阶段完成。备份工具复用主代理已有增量验收路径，本包不新增自动上传守护程序。100项确认与final test不在本包授权内。

## 已测证据与局限

CPU两条、GPU v3四条真实全数据三轮门检已通过并离机核验；GPU同条件跨卡模型/指标/控制历史/训练RNG逐位一致。原v2导入期RNG范围缺陷的失败保留，v3仅修正记录层，四模型/控制器/诊断SHA与v2相同。三轮均ACC0.516686、AEOD/ASPD0，退化负值不计论文性能记录；CPU/GPU模型与选择不同，不声称等价或70轮效果。

忠实度详见 `evidence/SCREEN32_PROPOSAL_AND_FIDELITY.md`：FLGMM author-code aggregation adaptation，作者largest-count GMM行为、零基Tg、bounds_2原行为保留；全模型/delta桥接有浮点舍入；显式零std扩展、空集合维持全局模型。不能称整套作者训练协议原封复现或承接论文全部理论保证。

## 独立 v2 路径守卫修订

原准备包与已上传release保留不变。旧source preflight在任何训练启动前错误拒绝已有CelebA共享数据软链；失败见evidence/PREFLIGHT_FAILURE.json。本修订仅为这6个已冻结数据入口声明精确解析目标与原SHA，逐项核二者；其他source仍强制留在repo解析根内。不新建软链、不普遍允许外部路径、不改原worker/数据/配置。REPO_SYMLINK_TARGETS.json列出实测映射；错误target、声明SHA或内容SHA均拒收。最终仍由root审阅新prepared seal后独立冻结。
