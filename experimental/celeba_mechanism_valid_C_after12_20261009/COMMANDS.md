# 独立审阅与未来入口

本机无CNN门检（已运行通过）：

    python -B tmp/celeba_mechanism_valid_C_after12_20261009/check_prepared.py

root审阅：`FILES_SHA256.json`（science10）、`execution_candidate/EXECUTION_SOURCE_SHA256.json`（execution11）、`SOURCE_REUSE.json`、`MINIMAL_SOURCE_DIFF.patch`、`LINEAGE.json`、`INPUT_PINS.json`、`NATIVE_INPUTS.json`、`SELF_CHECK.json`及`PACKAGE_RECEIPT.json`。

只有root另行批准、部署全新namespace并核Linux事实后，才可执行原one-shot installer：

    cd /workspace/guardfed_checks/celeba_mechanism_valid_C_after12_20261009/execution_candidate
    taskset -c 112-119 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B install_once.py --draft-sha256 <actual-ROOT-EXECUTION_DRAFT-SHA>

root外层运输应从开始采用短SSH argv，例如`['ssh', '-p', '60350', 'root@89.22.197.55', 'python -B -']`，完整只读预检或安装code经`input=code.encode()`传stdin；不把大字典/code嵌入argv。此处没有执行运输或安装命令。

服务：`guardfed_celeba_mechanism_valid_C_after12`，autostart/autorestart=false，startretries0。旧C_after1必须EXITED/0worker。新服务独立失败保留、不在旧目录resume，不自动重试。

待producer闭合后，root按原工具一次快照差集备份（无CNN）：

    taskset -c 110 ionice -c 3 nice -n 10 env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B backup_completed.py

下载实际archive/receipt到独立`execution_candidate/backups/<actualtag>`后，本机调用：

    python -B tmp/celeba_mechanism_valid_C_after12_20261009/execution_candidate/verify_backup.py <actual-local-delta-directory>

原saved-array checker只重算已保存预测，无CNN。远端closed、备份存在与准备门检均不等于离机接受；root须实际核验/登记，旧112与Full模型不重算、不重包。
