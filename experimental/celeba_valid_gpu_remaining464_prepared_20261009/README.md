本包仅本机准备，未部署、未推理、未启动服务。原425已接受不变；11项CPU partial/GPU诊断仍待独立审阅，不在本队列。

manifest.json精确引用原476提案中的UNEXECUTED465，减去已离机接受首1 `FairGuard_IID_FedSA_seed91003`，保留原顺序得到464。共43个chunk：42×11及末2；同时只运行1个GPU worker。首项 `FairGuard_IID_FedSA_seed91004`。prior/只保存原425 collector及首1 root离机证明的原字节；不重包旧模型、数组或归档。

remaining.py是无Torch的薄包装器。每chunk顺序调用已封recovery.py的run-chunk、accept、strict SHA绑定backup；原科学body、原1e-12、root/valid/split/70round、数据/模型/source/storage-map/三个视图守卫均由原入口执行。原实现包SHA为6ae15988b5d0b1ebe4371166afa99bceca015394cba8ecc6f987773621d55b56。包装器不改科学逻辑、pending主终点、既有425或原失败。

任一步非零、strict ID/指标不完整或archive身份不符立即failstop，保存queue_failure.json及原partial/logs，不启动下一chunk、不重试。输出必须是全新remaining464_attempt1，已存在即拒收；没有自动resume。恢复需root基于严格离机登记单独审阅新补集和新命名空间，不能原样再次start。错误前已闭合chunk保持REMOTE/PENDING_OFFSERVER，可逐差集离机核验。

成功chunk只写chunk_NNN.REMOTE_PENDING_OFFSERVER.json，绑定strict/archive/inventory SHA、前一收据SHA及新source/review；accepted_new_n始终0、cohort_registered=false。queue_exit只表示464远端闭合。每个archive包含queue_binding.json，引用已离机425和本wrapper seal；原archiver继续逐成员核SHA，不打包旧模型。离机SHA/member/保存数组严格验算及显式device/source登记仍由root独立完成，不把远端归档算接受。

orchestrator独占CPU106、nice10/idleIO，不导入Torch；child在Popen preexec设CPU105，原recovery再自绑105及1线程/interop，单GPU0实际UUID仍由原资源门检查。nice仅在外层不足10时补到10；supervisor一次nice启动，child不再调用nice -n10。CPU106不能直接继承给原worker，否则105可用门会拒收。Linux真实spawn和资源实测尚未验证，root部署后需核实际进程。

部署只需本完整sealed包到 `/workspace/guardfed_checks/celeba_valid_gpu_remaining464_prepared_20261009`；原implementation、476提案、v2/v3/v4、900 storage及cu128 Python仍用现有精确路径，不新增依赖。新输出parent为 `/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/remaining464_attempt1`，不得预建；wrapper创建。现有recovery执行根目录已存在，可承载supervisor日志。

root另存包外ROOT_REVIEW_REMAINING464.json：由ROOT_REVIEW_TEMPLATE.json生成，填实际queue seal/GPU UUID，status改为ROOT_APPROVED_GPU_VALID_RECOVERY_V1、execute_remaining464及execute_new465改true、approved_ids精确等于manifest.ids；两个import仍false，原科学边界不变。模板默认PREPARED与false拒收。review同时绑定425 collector、首1离机证明、旧implementation和新manifest；不能修改sealed模板。

未来审阅后先inspect（只核源码/身份，不推理）：

```bash
PY=/workspace/guardfed_envs/celeba-cu128-20261009/bin/python
PKG=/workspace/guardfed_checks/celeba_valid_gpu_remaining464_prepared_20261009
REVIEW=/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/ROOT_REVIEW_REMAINING464.json
"$PY" "$PKG/remaining.py" inspect --review "$REVIEW" --review-sha256 "$REVIEW_SHA" --package-sha256 "$QUEUE_PACKAGE_SHA"
```

guardfed_celeba_valid_gpu_remaining464_20261009.conf.template复用现有normal supervisor设置：autostart/autorestart=false、startretries=0、stopasgroup/killasgroup=true。root将两个SHA占位符替换到包外新conf，仅targeted reread/update/start该唯一program；本包不安装、不启动、不创建巡检。不得直接安装仍带占位符的template。

本地selfcheck使用真实464/425身份和临时合成文件，不运行子进程/CNN：12项拒收，6种流程包含两chunk成功仅REMOTE、三步分别失败、混ID和错archive全部立即停；AST核CPU继承、一次nice和无Torch导入。结果在selfcheck.json和LOCAL_CHECK.log。

既有CPU与新GPU逐IDdevice provenance必须保留；混合device的raw/shared汇总不称统一设备最终公平比较或final test。11项不导入时，即使这464全部接受也最多889/900；正式native/shared主终点仍pending。本准备包不改变任何正式选择。
