# Hybrid96正式启动helper v3：SOURCE ONLY

本包只准备源码，未SSH、绑定、启动、修改旧stage/STATE/Git或生成实际门检/root证明。新输出namespace为 `/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/fullcoverage_operations`；同一冻结stage仍为其`stage`子目录。

复用已审canary_operations/v3短argv+stdin运输、main_health原字节及resource()原核心。resource仅phase名称、canary服务EXITED与返回该实测字段变化；仍外层CPU107/nice10/idleIO/CUDA隐藏、正式coordinator和单child CPU104/GPU0/1线程。主8/FL102–103/gradient105/remaining112–119只观察。保留guide SHA、所有进程线程受限affinity冲突、主1..8无failure且真实增长、quota/RAM/GPU0显存及UUID/Recovery/OOM/disk门。failure receipt新增异常携带的resource_guard_inputs。

正式入口独立服务`guardfed_celeba_hybrid_fullcoverage`，调用原`stage/run_fullcoverage.py run --repo /workspace/GuardFed-celeba-expanded`。autostart/autorestart=false、startretries=0，无重试/无改recipe/seed/method/70轮valid协议。未改原25成员科学算法。

执行必须提供实际：GATE_ACCEPTANCE、其OFFSERVER_VERIFICATION、root closure、fresh主队列baseline和root批准，各自外部SHA及实际package/helper seal。closure须为`ROOT_SEVEN_HYBRID_CANARIES_OFFSERVER_ADOPTED`且绑定同package/gate；gate须为`SEVEN_HYBRID_CANARIES_STRICT_PASS_BACKUP_PENDING`，exact7 IDs/2原配对/0正式样本/test=false。原Hybrid离机验证schema按已有restore_verify.py：local.accepted_new=7、same_horizon_pairs=2、total_runs=7、rounds=3、formal_table_samples=0，包/gate SHA一致。未预填任何未来SHA。

远端核全部原package/source/data/job身份、gate最终文件/日志hash与exact manifest顺序，再调用原original_runner()['reused_records']核4条旧screen引用。要求旧screen及canary均EXITED、无同stage进程；96新runs/logs/queue/summary/failure均不存在。历史coordinator.lock须为原普通文件，用非阻塞flock确认无持有者并释放，绝不按文件存在误拒或删除锁。原canary授权SHA由root批准绑定，原字节先存入新helper证据PREVIOUS_CANARY_AUTHORIZATION.json；新授权及pending均exclusive-create，然后复核原SHA并os.replace。任何随后失败保留整个现场，禁止盲重跑。

新授权完全沿common.authorized实际接口：max_workers=1、cpu_threads_per_worker=1、allowed_cpus=[104]、gpu_index=0、automatic_retry=false及gate_root_closure_path/gate_root_closure_sha256；新root closure复制在helper独立目录，旧门检/授权证据留存。START只表示supervisor启动命令成功，不代表96/100科学接受；后续原strict与离机采用独立完成。

根后续实际CLI（当前未执行；ACTUAL_*必须真实文件）：

```powershell
python -B tmp/celeba_hybrid_fullcoverage_launch_operations_20261010/v3/launch.py --approval ACTUAL_APPROVAL --approval-sha256 SHA --gate-acceptance ACTUAL_GATE --gate-acceptance-sha256 SHA --gate-offserver ACTUAL_OFFSERVER --gate-offserver-sha256 SHA --root-closure ACTUAL_ROOT_CLOSURE --root-closure-sha256 SHA --baseline ACTUAL_BASELINE --baseline-sha256 SHA --package-sha256 ACTUAL_PACKAGE_SHA --helper-seal-sha256 ACTUAL_HELPER_SEAL_SHA
```

APPROVAL_TEMPLATE只给字段契约，不是批准。baseline格式沿原checked_utc/queue_completed/active实际progress，须≤1小时，资源receipt≤120秒。新fullcoverage_operations必须不存在；发生超时或任一失败须先实际检查服务和授权，不能再次运行来覆盖。

已本机一次`python -B .../check_source.py` exit0：13项原helper函数/原common授权内存检查，含实际授权字典正向、partial/错status/package/gate/CPU拒收、缺gate-root与自动retry拒收、模拟非阻塞flock路径、保留旧授权→原子替换→启动顺序。所有proof/host数据仅内存MOCK，无Torch/SSH/subprocess/真实锁/文件proof生成；SOURCE_CHECK不是Linux资源或图像门检证据。
