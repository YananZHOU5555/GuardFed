# FLGMM 首批2/32已离机闭合

首候选 `FLGMM_Tg10_L2.0_lr0.0005` 的 IID Benign、IID S-DFA 已各完成70轮，seed91001，valid19867/train162770/root16277。原worker退出后，单项原checked_result、外层before/after source/data21 SHA、job/config/adapter/checkpoint同终轮身份严格通过。保留全部原始输出，不提前排名。

| 条件 | ACC | AEOD | ASPD |
|---|---:|---:|---:|
| IID Benign | 0.8598681230 | 0.0552625270 | 0.0993795247 |
| IID S-DFA | 0.8676700055 | 0.0416437991 | 0.0905690108 |

这些是n=1的两个validation条件；当前候选未收齐四条件，其他候选也未完成，不代表最终选择或与其他方法的正式优劣。

22成员增量archive为 `backups/first_two_20261009/accepted_first_two.tar.gz`，SHA `638d7bba07fcc1f3a9953a8f71fb412516d8b5ffcc134eb3b3794cf82dedcbbf`。逐成员大小/SHA与archive整体SHA已在本机重核，恢复后再次调用原单项checker，`OFFSERVER_ACCEPTANCE.json` 状态为 `PARTIAL_ACCEPTED_OFFSERVER_VERIFIED`。本机Python/Torch仅用于身份检查，没有重跑训练或图像推理，不宣称CPU与GPU等价。

原producer PID18911/18912均退出；冻结coordinator可能仍持有闲置日志描述符，日志/产物在归档前后逐字节SHA及大小稳定。未改冻结runner以关闭描述符，也未将增长中的活动日志或模型当作完整结果。

新链为 `BACKUP_CHAIN_first_two_20261009.json`，入口 `LATEST_BACKUP.json`。原启动 `BACKUP_CHAIN.json` 仍原样保留；本批不重复打包67成员启动源码包或旧canary模型。后续增量以新链两个accepted IDs为排除集合，沿原 `screen_common.accepted` / `source.accept_result.checked_result` 做严格单项接受，再补新成员、核离机SHA后推进链。

当前原队列已自然运行同候选的non-IID Benign与non-IID S-DFA，仍max2/每GPU1/CPU1/nice10。保持原方法/配置/seed/评分，不自动retry，不中轮resume，不发起formal100/test。发现逻辑/数值/身份异常保留证据并报告；32项全部接受且备份后才运行完整候选选择与阶段交付。
