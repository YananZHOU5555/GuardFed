# LASA有界验证搜索worker（已准备，未训练）

独立于冻结 `lasa_20260928` pilot；adapter.py字节完全复用已通过真实图像pilot的版本，worker仅扩展任务配置验证/轮数/身份。仍仅包装aggregate_round，其余训练/攻击/root/valid/checkpoint由冻结core执行。方法标签为 `LASA-official`，真实含义是官方LASA聚合适配于冻结local Adam模型差分。原始pilot目录未修改。

## 候选与范围

8候选=`sparsity {.3,.5}` × `lambda_n=lambda_s {1,2}` × `local LR {.0005,.001}`。不加入lambda_n/lambda_s交叉不等的组合。完整列表在protocol.json和jobs/screen_manifest.json。

每候选4条件：IID alpha5000/non-IID alpha5 × Benign/S-DFA；共同seed91001，70轮，fulltrain162770/officialvalid19867，20客户端/4恶意槽、batch64、localepoch1、root10%和其余既有冻结参数。共32训练任务，raw native输出，不拟合额外阈值、不读test标签、不选中间best round。单seed验证搜索不能称最终test结果或统计显著性。

3轮preflight单独使用 `phase=preflight`、单独job ID及输出目录，限定以前通过的non-IID默认LASA(.3,1,1)、LR.001，Benign/S-DFA两份，不计入32任务。除experiment_suite/tag和run_id外，其训练配置与旧pilot一致，可对齐checkpoint tensor与全部逐轮metrics。正式screen只允许70轮，不接受把screen直接改3轮后混入汇总。

## 调用

准备文件已经生成，不需要在服务器重新生成。若要重新验证/复现manifest：

```text
python prepare_jobs.py --output jobs
python check_screen.py
python worker.py --repo /workspace/GuardFed-celeba-expanded --job /path/to/jobs/screen/LASA_s0.3_l1_lr0.0005_IID_Benign_seed91001_screen.json --out /path/to/new/result/dir
```

`--out`必须不存在；重复/部分失败结果目录直接拒绝，不覆盖、不重试。主队列如需跳过已验收任务，应检查结果身份后在外部跳过，不能让本worker重跑。失败保留failure.json/原始日志；不具备中轮恢复，不应宣传恢复能力。

输出job/provenance/progress/model/diagnostics/result/acceptance。job.source_hashes逐文件冻结完整core及图像资产；adapter_source_hashes冻结本目录adapter.py/worker.py/protocol.json/prepare_jobs.py。生成器拒绝覆盖内容不同的旧job；worker要求整个config与protocol派生值完全一致。改变其他配置须新协议版本。目录内代码/protocol在job冻结后不可再编辑，否则SHA检查拒绝。

结果round_summaries/trajectory必须1..70连续，config完全一致，result.metrics必须等于第70轮trajectory；model.pt固定终轮。preflight同样只验1..3。每层norm/sign/mask/fallback诊断完整保留，result继承全部攻击audit。acceptance含model/result/diagnostics SHA及phase/candidate。root数据只在冻结core既有流程使用，LASA聚合本身没有新增root信息。

## 父任务64队列接口

`jobs/screen_manifest.json` 的jobs是32个完整job字典，允许父队列另加output路径/外部调度元数据；不要改config/adapter/phase/tuning_candidate/identity。父队列负责跨方法评分与配置选择，worker不实现临时评分、更不按结果改候选。汇总键至少含method+tuning_candidate+distribution+attack+seed+phase，严禁把不同候选或3轮preflight混成重复seed均值。

选择规则由父任务在启动64队列前冻结；本worker不重定义该规则。全部失败/负结果保留。此处不声称已运行32任务，也不声称已证明基线性能。

## 本地验证与部署前检查

离线检查覆盖32任务/8候选/完整4条件、2个单独preflight、身份唯一、所有本地源码hash、候选/全配置约束、已通过pilot adapter字节不变、旧新包装器合成输出/诊断一致及其它方法透传。没有启动图像训练或访问服务器。

父任务部署后先运行两份preflight，对旧pilot逐tensor完全相等、三轮metrics逐项相等、所有选择索引/统计一致；checkpoint ZIP字节SHA可能受保存路径元数据影响，跨运行等价判据用tensor值而非只比较ZIP SHA，每份自身仍核SHA。通过后再按父任务资源安排启动32任务；源文件SHA核验与最终checkpoint验收仍必须做。
