# 命令仅供根最终批准后的执行；本包未部署/运行

## 本机复核

核父 `FILES_SHA256.json` 的14项与封条 SHA `65f706e8c7c7d7e18e76c8a300dd845297c97b3bd5c6c182bbbb8aad0103b5ff`，再核本包 `EXECUTION_SOURCE_SHA256.json` 每项SHA/size。运行以下检查只读科学包并创建/清理本目录临时fixture，不访问服务器或导入torch/numpy：

```powershell
python -B E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_valid_incremental_next11_20261009/execution_candidate/selfcheck.py
python -B E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_valid_incremental_next11_20261009/execution_candidate/batch.py inspect
```

selfcheck比较对象为已审next37执行源；不把旧执行树重新部署，不重启旧37队列。

## 根实测及批准（本阶段未做）

部署保持父science14+原封条、execution子目录所列成员+执行封条的相对位置；不复制旧37/旧8的运行输出、批准文件或预测。远端父目录 `/workspace/guardfed_checks/celeba_mechanism_valid_incremental_next11_20261009`；本执行目录为其 `execution_candidate/`。不要预建 runs/approvals/logs，不覆盖已有失败或证据。

科学SCOPE本来不含outputs/dependency_paths，所以执行封存 `RUNTIME_BINDINGS.json` 只补这两个字段。7依赖路径与原37逐值一致；只允许当前11的独立父目录/runs namespace，实际批准映射到execution_candidate/runs。字段碰撞（包括试图覆盖科学scope/recipe/tolerance）、错ID/错路径/错依赖均拒收。这个映射是外层路径适配，不声称父SCOPE原本含此字段。

根先实际读取 `/etc/vast-agents-guide.md`，核现行guide42be4f…、isolatedcu128 Python、原v2/v3/evaluator/evidence/inventory/manifest/protocol和全source/data/终轮SHA可用；service/worker身份和真实主训练round推进、CPU配额/内存/disk/GPU recovery健康。准确CPU112–119必须无其他restricted进程或线程占用，保留原classifier预算（已含本8）再加保守3不超过实际配额。这是配额保留，不是实测CPU消耗或122.88的永久假设。

随后根独立把 `ROOT_REVIEW_TEMPLATE.json` 写到新的 `ROOT_APPROVED.json`：status=`ROOT_REVIEW_PASS_BOUNDED_NEXT11_VALID_REPLAY`，execution_authorized_within_existing_user_request=true，execution_seal_sha256=实际新封条SHA；source14/scope/inventory/bridge、selected11/closed60/source_only_root_review_sha256保持。科学review `b1ff1fad...` 本身没有执行效力。

根独立把 `APPROVED_TEMPLATE.json` 写到新的 `EXECUTION_DRAFT.json`：status=`APPROVED_NEXT11_MECHANISM_VALID_REPLAY_ONLY`，root_approval_sha256=实际ROOT_APPROVED原字节SHA，execution_seal_sha256=实际封条；其余输入/11输出/依赖/CPU/native1e-12/no test/no Full inference/no retry不变。必须在外部记录draft真实SHA，模板默认拒收。

```bash
cd /workspace/guardfed_checks/celeba_mechanism_valid_incremental_next11_20261009/execution_candidate
/workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B batch.py inspect
ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B install_once.py --draft-sha256 ACTUAL_EXTERNAL_DRAFT_SHA256
supervisorctl status guardfed_celeba_mechanism_valid_next11
```

installer只有在外部draft/root/source/seal/空输出/无重复/全线程CPU占用/预算/输入SHA等门禁全通过后才安装新独立服务。服务 autostart=false、autorestart=false、startretries=0；服务单次nice10/idleIO、单进程8线程，顺序fresh child；首次失败保留停止。installer自身nice10与supervisor独立启动的服务nice10不叠加。禁止调用旧37服务或修改健康队列。不能修改失败包后重新启动来绕过门禁。

## 新完成项差集备份与离机核验

只对写入completed标记且worker/log已关闭的新ID：

```bash
ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B backup_completed.py
```

沿原段序复制每段archive及backup_receipt到本机对应 `backups/incremental_<UTC>/`，下载命令必须真正完成后再验收。工具按差集排除以前备份项，不复制Full模型或原训练权重；首段保留当前source与实际批准/preflight/startup。远端archive本身不等于离机通过。

```powershell
python -B E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_valid_incremental_next11_20261009/execution_candidate/verify_backup.py E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_valid_incremental_next11_20261009/execution_candidate/backups/incremental_ACTUAL_UTC
```

本机工具核archive/member SHA与所有保存数组的三视图预测规则、计数和原native1e-12；root拟合仍由原server strict bridge复算。后续段要求以前段已存在本机OFFSERVER证明，不覆盖 verified_extract/OFFSERVER_VERIFICATION。根最终独立核后才登记；本候选尚无科学接受、无新资源实测。
