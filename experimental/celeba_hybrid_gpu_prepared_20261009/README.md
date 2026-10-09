# Hybrid CUDA4 + 原32候选：仅准备

状态 **PREPARED_NOT_FROZEN**。没有部署、冻结、GPU调用、训练或推理。原group_a、CPU gate和writer repair9成员均不改。CPU4真实图像管线已另处严格闭环，不能据此称CUDA、70round或8候选性能通过。

本包准备四条独立三轮CUDA管线门检：IID/Benign与non-IID/S-DFA，每条件Hybrid和原core GuardFed，各seed91001；全部train162770/root16277/valid19867。另准备原8候选LR .0005/.001×lambda5/20×tau.1/.2×IID/non-IID×Benign/S-DFA，共32条70round valid-only，完全保留原四条件score均分、字典序精确并列规则。所有负结果、恒定预测和失败保留。n=1，不报sampleSD/显著性、不称完整17方法或final-test。

`scientific_snapshot/` 的原五个worker/adapter/validator/prepare/protocol及原seal逐字节复制。实际CNN/localAdam/攻击/clean-root评估和训练仍调用原冻结core；Hybrid调用原worker.aggregation_wrapper，legacy调用原core GuardFed。`body.py`是原CPU gate明确的设备与horizon仪器适配，提供逐轮同输入delta/trust/筛选exact oracle、完整独立训练对照、checkpoint终轮重新推理。与原CPU body差异写`diff_cpu_to_cuda.patch`；CUDA RNG也显式核与记录。不会要求CPU与CUDA模型逐位相同，也不把CPU结果填进GPU门检。

writer原递归visitor、NaN位模式、零方差原因及sidecar重建AST均不改；仅job域header显式扩为IID/non-IID、horizon3/70。仍只允许四个`attack_audit[0..3].fflip_label_corr_after`，须每项真实零方差overwrite机制与精确job/source绑定。Benign无任何undefined替换，全部finite；其它主指标、weight/control/未知NaN/Inf仍拒绝。旧worker通用nonfinite清洗不被本driver调用。新检查在原CPU完整8artifact约束上要求第9个sidecar及全部同checkpoint/source/job；70round是明确horizon扩展，不声称直接复用原screen诊断schema验收。

`driver.py`先拒绝PREPARED，再核外部SHA批准、所有源/job、CPU4实际strict/offserver前置证据、90秒内实时resource证明、独占新输出、无受限CPU重叠。screen另要求真实CUDA4 strict+offserver及相同core/worker/adapter/body/writer。旧PREPARED protocol/job不获运行权限。runtime_protocol是独立新协议：各job额外绑定其实际SHA，旧5项protocol仍是源快照身份，二者不混用。

资源只提议单GPU worker、计算线程1、nice10/idleIO、独占CPU104；具体物理GPU0/1与UUID由root实时核显存>=4GiB/正常状态后决定。最坏旧角色并存名义预算114+1=115<122.87999；这是预约线程数，不是利用率。CPU4及remaining7已退出后预算应按实时进程重算。本包不抢占、暂停、重启或修改正在运行的800/FL32/900队列，不安装包、开端口或新增巡检。

正常supervisor模板均autostart/autorestart=false、失败立停。没有中轮恢复承诺；run parent、failure sentinel或同包锁存在时拒绝隐式重试。先完成每个阶段原始结果/模型/配置/sidecar/log增量离机SHA，再进入下一阶段。

root下一步可审阅的最小流程：

1. 独立验收本准备seal和CPU4的56-member离机闭环。当前所有scope/status保持PREPARED；无需服务器执行来审查。
2. 在**新空执行副本**复制源码（不复制draft jobs/scopes/manifest/seal/任何runs）。保留scientific_snapshot五文件原字节；先按副本实际路径改service模板。gate的runtime_protocol可保持PREPARED，70round正式协议仍未冻结。
3. 在该副本运行`prepare.py --original-gate-root <originalCPUgate>`重新生成4/32 job与新的runtime_protocol SHA绑定。只有root明示GPU4授权后，把新gate_scope.status改`FROZEN_BOUNDED_HYBRID_CUDA_GATE_ONLY`，screen_scope仍PREPARED；重算新的scope/local文件与整体seal。填外部APPROVED_gate及64hex+LF .sha256，附真实CPU4证据/新鲜资源证明，安装该副本正常supervisor，**只运行gate4**。
4. GPU4全部严格接受、两对CUDA模型/指标/攻击/诊断/RNG exact且离机核SHA后，出具`FOUR_CUDA_CANARIES_STRICT_ACCEPTED_OFFSERVER`；列exact4IDs、source及all_pairs_exact。CPU与CUDA间无需/不得宣称exact，实际差异需单列披露。
5. root再审原32配置/四条件择优与资源，另建空screen副本，在其独立runtime_protocol把status改FROZEN后再materialize新jobs，随后把新screen_scope.status改`FROZEN_HYBRID_VALID_SCREEN_ONLY`并重封。源snapshot旧protocol不动。批准仅该32与真实CUDA4先验证据；当前包不代做这一步。
6. screen运行后用`summarize.py --approved ... --approved-sha256 ...`严格核全部32，输出全部8候选四条件均分、accuracy冠军、三指标Pareto和所有原结果；不分指标/条件择优、不删除退化候选、不自动开100格多seed或test。

freeze后运行入口（此处只是命令，不是授权）：

```text
ionice -c 3 nice -n 10 <cu128_python> -u <new_release>/driver.py run --kind gate --approved <externalAPPROVED> --approved-sha256 <SHA>
```

`resource_preflight`必需字段：at_unix、cpu_allocation、cuda_visible_device、gpu_uuid、gpu_free_memory_mib、no_duplicate_worker、no_restricted_CPU_overlap、existing_nominal_compute_threads、actual_quota_cores；其路径及SHA由外部root批准绑定。批准模板不是APPROVED。原loader可materialize全split属性元数据，本阶段不读test图像/推理/拟合/选择，不宣称untouched test。
