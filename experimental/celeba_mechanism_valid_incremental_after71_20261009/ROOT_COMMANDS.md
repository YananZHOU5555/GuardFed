# ROOT最简审查、启动与离机备份入口

当前所有运行动作均为**PREPARED / NOT EXECUTED**。只操作新的`celeba_mechanism_valid_incremental_after71_20261009`，不重启旧next11服务。

1. 核`HANDOFF.json`和`DEPLOYMENT_MANIFEST.json`、science `FILES_SHA256.json`及execution `EXECUTION_SOURCE_SHA256.json`实际字节SHA；审准确11、排除71、原native82和Full100绑定，以及最小source diff。`deployment_source.tar.gz`仅含这两封条及其列出成员，无旧运行批准/输出、模型或数组。ROOT可按既有服务器连接方式部署到全新父目录，保持包内相对路径；本次没有传输或部署。
2. ROOT按现行guide读取服务器实际状态，确认隔离Python及原7依赖路径、原source/data/终轮SHA、实际配额/内存/disk/GPU恢复状态；CPU112–119不得有其他restricted线程占用，保留原8线程与额外3角色预算。不要把旧preflight或静态额度当当前实测。安装入口会再次执行原真实preflight门禁；此处尚无实际preflight结果。
3. ROOT独立从`execution_candidate/ROOT_REVIEW_TEMPLATE.json`创建新的`ROOT_APPROVED.json`，status设为`ROOT_REVIEW_PASS_BOUNDED_AFTER71_VALID_REPLAY`，execution_authorized_within_existing_user_request=true，并填实际execution seal。原source审查SHA仅说明科学复用；ROOT必须另行核当前新输入/输出边界。其余精确11、closed71、science/scope/inventory/bridge、CPU、native1e-12和禁止项保持。
4. ROOT独立从`APPROVED_TEMPLATE.json`创建新的`EXECUTION_DRAFT.json`，status设为`APPROVED_AFTER71_MECHANISM_VALID_REPLAY_ONLY`，绑定新ROOT_APPROVED实际SHA和新execution seal。外部记录draft真实SHA，不能从prepared模板直接启动。

审核/授权之后的原操作入口（**本次未执行**）：

```bash
cd /workspace/guardfed_checks/celeba_mechanism_valid_incremental_after71_20261009/execution_candidate
/workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B batch.py inspect
ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B install_once.py --draft-sha256 ACTUAL_EXTERNAL_DRAFT_SHA256
supervisorctl status guardfed_celeba_mechanism_valid_after71
```

原installer按真实preflight通过后安装单独`guardfed_celeba_mechanism_valid_after71`；autostart=false、autorestart=false、startretries=0。失败保留停止，不覆盖已有输出或隐式重试。

仅对已有completed且producer已关闭的新ID做原差集备份（**本次没有重放结果，未执行**）：

```bash
ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B backup_completed.py
```

原工具排除已备份ID，不复制Full或训练权重；首段保留science/execution源、实际批准和preflight/startup。ROOT沿原段顺序下载真实archive及backup_receipt至本新目录对应`execution_candidate/backups/incremental_<ACTUAL_UTC>/`，完成下载后使用原离机入口：

```powershell
python -B E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_valid_incremental_after71_20261009/execution_candidate/verify_backup.py E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_valid_incremental_after71_20261009/execution_candidate/backups/incremental_ACTUAL_UTC
```

离机入口仍核archive/每成员、保存数组三视图9指标/24计数/3规则、原native1e-12；root拟合由原server strict闭包复算。远端正常退出或archive产生均不构成科学接受。只有ROOT独立验收后才可登记；当前新11没有three-view接受记录。
