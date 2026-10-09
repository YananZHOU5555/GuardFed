# FL96：accepted5后的单次完整结果增量

本次严格接受并离机校验2项 IID Benign seed91007/91008，累计7/96新增70轮任务，另4项原70轮复用只引用。采单个collector live_snapshot，未追后到终轮。

collect_delta.py、verify_delta_offserver.py与原已批准源逐字相同；外层仅替换owned namespace及previous实际接受5链，CPU106单线程/nice10/idle/CUDA隐藏。原strict及源/数据/配置/job/checkpoint同终轮、真实alpha、70轮valid19867/root16277通过；archive27成员及全部SHA、原本机checker通过，负结果未删。原producer已退出，collector完成后CPU106无restricted线程。

服务器验收cu128、本机2.8 CPU只读保存张量和记录；无CNN/训练/test/阈值拟合，不声称runtime相同。未改LATEST_BACKUP、STATE、RUNNING、Git或队列。

根采用入口 ROOT_READY_HANDOFF.json、batch/OFFSERVER_ACCEPTANCE.json及DELIVERY_FILES_SHA256.json。采用后下一collector previous使用本次batch/OFFSERVER_ACCEPTANCE.json及其SHA；禁止initial-empty或重包旧模型。原验证命令见VERIFY_COMMAND.json，已exit0，不应覆写restored目录重复执行。
