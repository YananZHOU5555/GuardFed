# NGA96 初始资源 bootstrap（只准备）

两执行源，现成 runner 不改：`initial_preflight.py` 从已冻结 `job_preflight.py` 派生；`root_bootstrap.py` 由根代理明确授权后通过 supervisor 运行。当前没有SSH/F访问、资源采样、门检、训练、模型/数组/科学执行，未写STATE/Git。whole14离机/root proof与实际运行批准仍为晚绑定；SOURCE_APPROVAL只是冻结源接纳，不等于启动批准。

初始probe在**同一真实 bootstrap PID**中通过 runpy 执行，Linux `/proc/.../cmdline`仍为bootstrap。它不要求/填写controller_pid，不伪装已有run_queue；proof明确 producer_role=ROOT_INITIAL_RESOURCE_PROBE_NOT_QUEUE_CONTROLLER、实际bootstrap_probe_pid及planned_first_job_id。sys.argv仅用于本次probe argparse，真实/proc argv未改。完成后 `os.execv` 以同一PID运行原 `run_queue.py`，不留下另一个CPU105受限bootstrap父进程；后续原runner每job原preflight、真实controller和继承common锁FD原样。

继承的四函数 sha/read/command/cpu_sample raw+AST完全相同；全部source/data完整SHA loop原字节相同，包括六真实共享数据、source及配置。12处完整可逆metadata/初始入口修改在INVERSE_EDITS/SOURCE_DIFF中，逆恢复到原job_preflight SHA2e4d8dd6…exact。未改冻结package任何文件、scientific函数/模型/recipe/seed/参数/并发/资源分母。

原guard保留：旧64自然EXITED/rc3、无旧worker/失败；main RUNNING或完整800自然终态、failed为空/active≤8；all项目compute线程窄mask≤16不重叠CPU105、无重复队列/worker；nominal+1≤quota、GPU1 UUID/free≥4096MiB及双GPU Recovery None、RAM≥8GiB、workspace≥40GiB；guide/source/package/作者决策/每job/full data SHA。新增初始门：实际affinity105、所有线程环境1、nice≥10、ionice真实idle；CPU105 SMT兄弟的受限项目线程冲突也拒绝；common lock非阻塞探针；main active progress原JSON（尚无首progress则null）和active failure.json检查。memory.events是累计原样记录，不冒称本次OOM增量；观察顺序非原子，不称host独占。

初始probe持有原common锁直到写完proof。wrapper保存真实command/stdout/stderr/exit、INITIAL_RESOURCE和RUNNER_HANDOFF后释放探针锁并exec；原runner即刻重新取得同一路径锁。短间隙若竞争者先取得，原runner会fail-stop，不能宣称无竞态或自动重试。后续原逐job门只保留原CPU105窄mask政策，未暗改冻结源为新增SMT规则。supervisor必须autorestart=false、startretries=0，原queue failure/部分结果全部保留；成功handoff后runner退出由supervisor记录，wrapper不会提前写runner exit0。

## 根代理后续唯一命令（未执行）

先将本包两源原字节部署至根指定独立server目录；用本FILES seal核原字节。whole14根必须已实际strict/offserver采用，满足原coverage_contract全部scope/comparison/14顺序/3round/42total字段。其scope/comparison路径要Linux可读，不能使用Windows/F路径。根明确批准本96后使用现有cu128解释器，不再gate/selection/freeze。

```sh
env CUDA_VISIBLE_DEVICES=1 GUARDFED_CPU_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 taskset -c 105 nice -n 10 ionice -c 3 \
 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B ROOT_DEPLOYED_DIR/root_bootstrap.py \
 --out /workspace/celeba_fednga_fullcoverage_results_20261011/attempt001 \
 --receipt-dir /workspace/guardfed_checks/celeba_fednga96_bootstrap_20261011/actual001 \
 --source-approval ACTUAL_SOURCE_APPROVAL_ON_LINUX \
 --source-approval-sha256 79edd0dde78c9dc23583a22953d3babebb357a4fcebb7197941cc5726b2ff961 \
 --gate14-adoption ACTUAL_WHOLE14_ROOT_ON_LINUX --gate14-adoption-sha256 ACTUAL_WHOLE14_ROOT_SHA
```

out与receipt-dir均必须从未存在，不允许将receipt-dir放入新out之内；任一失败保留停止。初始resource实际生成并在90秒内交原runner，模板resourceSHA为null，不预填结果。96新增+4显式screen引用，Huber96不随之启动。全部bulk只server；本包不含部署、SSH或F下载程序。

限制：只compile/原字节inverse/AST与CLI静态连接检查；Linux runpy/exec/descriptor/资源行为尚未实际验证，根需读source后决定执行。初始probe是真实测量候选，不是已运行PASS，不是新科学checker。
