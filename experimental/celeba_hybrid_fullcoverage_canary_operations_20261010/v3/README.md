# Hybrid 七项真实三轮门检：启动源码候选

只准备，未SSH、绑定、运行科学代码或改变服务。复用已执行FL启动运输及逐字main_health；独立服务guardfed_celeba_hybrid_fullcoverage_canary，一次启动、七项按原run_canaries顺序执行。该服务绝不调用96正式队列。

固定stage /workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/stage；外层helper CPU107/nice10/idleIO/CUDA隐藏，科学coordinator及每个顺序child CPU104/GPU0/1计算线程。FL102/103、gradient105、主8和remaining112–119仅观察，不修改。supervisor autostart/autorestart=false、startretries=0。

执行前须root外部提供实际PACKAGE、BOUND_TRANSFER_VERIFICATION与ROOT采用、批准及真实主队列增长baseline的SHA。新root采用字段契约：status=ROOT_HYBRID100_BOUND_METADATA_ADOPTED、package_sha256、bound_offserver_sha256、new=96、reused=4、canaries=7、execution_authorized=false、final_test=false。此处未创建任何实际root证明。

审批字段：status=ROOT_AUTHORIZED_SEVEN_HYBRID_CANARIES；scope=seven_same_horizon_3round_canaries；service、cpus=[104]、cpu_threads=1、formal100_started=false、final_test=false，以及package_sha256/helper_seal_sha256/bound_offserver_sha256/bound_root_review_sha256/baseline_snapshot_sha256。缺字段或不同SHA拒收。

baseline沿原格式：checked_utc、queue_completed、active=[{id,progress:{round:实际值,...}}]；来源必须为实际主队列测量，审批绑定原字节，不能用预期轮次构造。启动要求其≤1小时且真实共同ID轮次或完成数增长，1..8实际worker和无失败/近期错误，容许70轮交接。resource≤120秒；cgroup CPU实测与声明预算/配额、RAM≥8GiB、GPU0≥4096MiB/绑定UUID/RecoveryNone、OOM、磁盘、旧Hybrid EXITED、sglang STOPPED和无同stageproducer均实查。扫描全部进程线程的≤16核受限affinity，禁止CPU104冲突；运输前另外核CPU107。宽mask只表示可调度，不声明CPU独占或空闲。源/data通过原common流式hash，不整读images.npy。

未来根执行CLI（此源码准备未执行）：
python -B launch.py --approval ACTUAL_APPROVAL --approval-sha256 SHA --bound-offserver ACTUAL_TRANSFER_VERIFICATION --bound-offserver-sha256 SHA --bound-root-review ACTUAL_ROOT --bound-root-review-sha256 SHA --baseline ACTUAL_BASELINE --baseline-sha256 SHA --package-sha256 ACTUAL_PACKAGE_SHA --helper-seal-sha256 ACTUAL_HELPER_SEAL

所有本地attempt和远端canary_operations均不覆盖。超时/失败保存原输出，不盲重试；一次start可能已发生，须先检查实况。START只表明启动，七项原strict/pair检查及离机采用另行完成。没有70轮性能或test/全返修完成主张。SOURCE_CHECK只有语法/契约与4个affinity谓词检查，不是真实服务器资源或图像门检证据。

## v3记录层绑定
原7成员launcher封条a7b694822e8ce2d06eeb594bc2f25e4863f1cb576f612064d98eac53d4557827保持。此子包只改stage/transport namespace及实现源身份外部绑定。ROOT_BOUND和APPROVAL必须同时给出相同实际implementation_source_seal_sha256；远端stage/PREPARED_SOURCE_SEAL逐SHA一致才启动。该实际v3源及绑定尚待root产生，当前无未来SHA、无执行。Python3.10/3.12原sum末bit差异修复属于另一个实现源，不在此launcher改科学代码；原失败由root保留。
