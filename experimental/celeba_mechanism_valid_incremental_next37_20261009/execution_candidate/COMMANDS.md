以下仅为根审阅后的命令，当前未执行。禁止从模板直接启动。

部署文件保持两层相对关系：父目录科学 `FILES_SHA256.json` 所列16项+该封条，以及 `execution_candidate/EXECUTION_SOURCE_SHA256.json` 所列候选项+该封条。不复制旧15模型/预测/失败包，不复制根 `root_source_review/`。远端父目录为 `/workspace/guardfed_checks/celeba_mechanism_valid_incremental_next37_20261009`，候选为其 `execution_candidate/`；不要预建 runs、approvals、logs 或运行凭据。父目录 inputs 中的原 ledger 快照属于科学封条输入。

根在 fresh Linux 当前 guide/服务/资源复核后，将 `ROOT_REVIEW_TEMPLATE.json` 独立审阅为 `ROOT_APPROVED.json`：status=`ROOT_REVIEW_PASS_BOUNDED_NEXT37_VALID_REPLAY`，execution_authorized_within_existing_user_request=true，execution_seal_sha256=本候选真实封条SHA；保留 source95978…/scope5eeb…/inventory0f837…/bridgea1eba…/精确37和closed23全部冻结字段。可增写根的 live budget/source review/批准事实；模板无批准效力。

根将 `APPROVED_TEMPLATE.json` 独立填写为 `EXECUTION_DRAFT.json`：status=`APPROVED_NEXT37_MECHANISM_VALID_REPLAY_ONLY`，root_approval_sha256=实际 ROOT_APPROVED.json 原始字节SHA，execution_seal_sha256=本候选真实封条SHA，其余精确 scope/IDs/outputs/dependencies/CPU/native1e-12/no retry 字段保持。外部保存 draft 原始SHA。installer核外部 draftSHA→rootSHA→source/science/execution seal；service核 installer 新APPROVED SHA。不能把 source-only review 当作执行批准。

已有只读依赖沿父SCOPE的 dependency_paths：cu128 Python、v2/v3 replay、原evaluator/evidence_v4、原baseline inventory、机制manifest/protocol、接受60的终轮文件与科学source/data。原资源工具 `.../celeba_mechanism_valid_replay_20261009/execution_attachments_v2/execute_one_v2.py` SHA5d537f…、supervisor utilities、guide42be4f… 必须存在。installer对37条输入执行原全SHA预检，不依赖本地模拟去声称远端存在。

```bash
cd /workspace/guardfed_checks/celeba_mechanism_valid_incremental_next37_20261009/execution_candidate
/workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B batch.py inspect
```

可在本机重新执行无CNN selfcheck；其比较对象为项目 `tmp/celeba_mechanism_valid_incremental_v2_execution_20261009` 原15执行层，不要求把这份历史执行树新增部署到服务器。根实际启动只用 `inspect` 和 installer。

根最终一次性安装并启动的最小命令如下；`ACTUAL_EXTERNAL_DRAFT_SHA256` 必须换成根外部核验字面值。不要先 nice10 再在同进程里追加 nice；此命令安装器nice10，而服务由原supervisor产生、其脚本一次nice10，CNN子进程继承nice10。

```bash
ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B install_once.py --draft-sha256 ACTUAL_EXTERNAL_DRAFT_SHA256
supervisorctl status guardfed_celeba_mechanism_valid_next37
```

只有根实际批准且 installer 全部 preflight PASS 时才会定向 reread/add/start。Linux fresh `/proc` 全线程 scan必须无其他restricted owner占112–119；旧classifier nominal包含本8，再加保守3≤actualquota；formal service必须RUNNING，disk>10GiB、Recovery可核None、所有source/data/terminal SHA须完全相同。任一拒收留失败、不自动重试或重新命名偷跑。

根可对已完成且 producer关闭的新ID做原增量备份（不会推理、不能将remote归档称离机）：

```bash
ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B backup_completed.py
```

下载每段 `backups/incremental_<UTC>/incremental_valid_three_views.tar.gz` 与 `backup_receipt.json` 到本机同候选路径。按段原序执行，prior链必须已有本地OFFSERVER证明；不覆盖已有 verified_extract 或OFFSERVER_VERIFICATION。

```powershell
python -B E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_valid_incremental_next37_20261009/execution_candidate/verify_backup.py E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_valid_incremental_next37_20261009/execution_candidate/backups/incremental_ACTUAL_UTC
```

离机原字节 archive/member 与保存数组规则验证成功后，根另行登记科学接受。当前本候选new accepted=0，Full新推理=0，test推理=0，所有执行待根审阅批准。
