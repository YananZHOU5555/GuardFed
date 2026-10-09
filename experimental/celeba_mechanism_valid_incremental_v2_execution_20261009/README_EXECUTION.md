# 已批准15项机制终轮valid三视图重放

只执行固定23真实终轮库存中，原已闭合8项之外的15个minus_U：IID Benign seed91009–91010、F Flip seed91001–91010、FedSA seed91001–91003。没有新训练、Full推理、test或自动重试。

## 不可变入口

- 科学准备19成员封条：`70d0d920c4c5351c42efc9968fe3c38eed431d208b94bc8af486ba49d869a42d`；原字节在`sealed_source/`。
- 根批准：`ROOT_APPROVED.json`，SHA `701f95342f1da3a7e0ba6247b85320a7f5043ccf92475c1c713105866636f8de`。
- 执行34成员封条：`EXECUTION_SOURCE_SHA256.json`，SHA `7f35b8c2e8f6c1dfe610f2750d721cb20650dc19bcd9b9e3d03fe244d52f9aa0`。
- 实际外部批准：`APPROVED.json`，SHA `ae5a289fffa9d89badfd152f1c0105a764f21e69afa4d011ee5886fe2e210313`。
- 原7协调器到本15的完整执行层diff在`COORDINATOR_DIFF.patch`；唯一输出迁移在`PATH_MAPPING.json`，不改原SCOPE字节。

服务`guardfed_celeba_mechanism_valid_incremental15`，远端根目录`/workspace/guardfed_checks/celeba_mechanism_valid_incremental_v2_execution_20261009`。每项fresh child，顺序单进程8线程、CPU112–119、nice10、idle IO、CUDAhidden。正常supervisor服务不自动启动、不自动重启、startretries0；失败保留停止。真正父runs目录由协调器在首child前创建。

启动前137个原source/data/artifact成员SHA与34执行源核验通过。10:32启动时资源预算包含本任务为106计算线程≤122.87999核；此前Hybrid已完成退出，所以实际不是旧快照的114。主GPU任务和FL服务均未改动。进程/实际CPU增长/已接受数量见独立`live_*.json`。

## 启动离机证据

`startup_delivery/startup_delivery.tar.gz` SHA `382fce3384d3772b2c7167ce5d9df738ecef4a1166efeb2d954c241e74cb9974`；43内容成员加1inventory已逐SHA/size核，见同目录`OFFSERVER_VERIFICATION.json`。其中包含全部执行源、批准、preflight、部署archive及两次实时观察。启动包不代表15项全部科学完成。

## 新结果增量闭合

每项worker调用原封存bridge的`replay_one`及`accept_saved_predictions`后，写strict_acceptance，父进程确认正常退出后再写completed_ID。后续备份只收completed且producer已关闭的新ID差集；checkpoint已在原5条科学备份链，此处不重复打包模型，也不重包旧8重放。

`backup_completed.py`只读取已完成数组/收据/源身份，保存逐成员清单、整包SHA与前序receipt SHA；初包保存来源，后续仅新增结果。`verify_backup.py`离机核archive及每成员，再用`verify_saved_increment.py`独立重算valid三视图ACC/AEOD/ASPD、分组混淆计数及阈值预测规则。该每ID科学检查loop与原7验收AST精确一致，证明见`saved_array_source_reuse.json`；源摘要见`BACKUP_TOOLS_SHA256.json`。最终闭合以所有15个ID严格接受、离机复算和非重叠备份链为准。

Full三视图保持MISSING，未与外部三视图receipt做身份绑定前不填入。777未完成项没有checkpoint或假SHA。本批重放仍是valid-only，不能称test确认或机制全表完成。

## 保留的工程观察

本地Windows精确远端路径比较最初因反斜杠拒收，统一为as_posix后10项执行层拒收检查通过；未产生科学运行。GPU观察命令最初`nvidia-smi -d RECOVERY_ACTION`不被驱动支持，改为只读完整`-q`，两卡实测Recovery Action=None；部署前旧source archive/seal留在`history_predeploy_v1/`。离机Python3.10不支持tarfile的filter参数，首次提取尚未写成员，改为显式校验所有成员均为相对普通文件再逐项安全写入。以上均未改科学函数、容差或已封证据。
