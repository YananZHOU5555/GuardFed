# 后续900输入的独立存储绑定

当前两条CPU canary使用已验收Full100的原仓库路径。`replay.py`及本轮plan v2按实测字节封存，不会在同名版本中加入新路径行为。主代理正在为其他800条准备独立artifact store；本文件是后续接入约束，当前CLI尚未读取storage_map，不表示900已重放。

storage_map与远端恢复验收receipt必须共同由主代理核验，并显式把两文件SHA传给未来入口。不能只信map里的`verified=true`或从文件自身读取预期SHA。map绑定原始`model_inventory.json` SHA `3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd`，不改库存原字节。

每条record必须有唯一id和完整三kind（checkpoint/result/raw_job）；三者分别绑定库存原archive、archive SHA、member、member SHA、bytes及实际绝对存储路径。Full100使用原路径，其他800仅允许`/workspace/guardfed_checks/celeba_validation900_restore_20261009/artifact_store/<原member>`；不得指向其他历史partial/failure目录，也不得改变原历史output。900 id×3kind必须完整、唯一且与恢复验收成员逐项一致；不接受缺kind、重复id、交叉模型、漏seed、未验归档或未验成员。

未来读取顺序：先独立核map/restore receipt SHA和库存SHA，再逐项核实际文件大小/SHA/解析目标。原raw_job的id/config/source/adapter/output仍按原字节验收；`original_remote_output`仅保存历史来源。调用原`checked_result`时可构造独立runtime job副本，仅把其读取output绑定为map指定的result/model共用父目录，并记录该路径替代；不得把副本当原job SHA，不得写回raw_job或库存。若result/model实际不在同一目录，必须先改变受信恢复布局或提供等价只读文件解析接口，不能绕过原checkpoint校验。

所有source/data/adapter仍按原库存哈希读取。三个artifact与输入map/receipt的SHA和resolved path均在重放前后复核。变更map、成员、root/train/valid ID或预测规则立即拒收；不自动恢复、不静默回退到原历史目录、不放宽1e-12。

后续实现必须有实际map篡改、重复id/kind、错误成员路径、混用checkpoint/result、存储缺失和root-ID篡改拒收检查；在两条canary被审阅后，以显式新版本和新输入计划交付。它不能产生final dispatch receipt、读取test标签或默认启动900。
