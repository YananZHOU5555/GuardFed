# CelebA训练客户端 Smiling×Male 联合分区审计

**通过：20份分区、400个客户端四格联合计数已恢复并独立复算。** 这是对先前只恢复敏感边际的增量证据；未修改原审计、原900结果、写作包或实验协议。

## 已核实范围

- 来源为已验收GuardFed-AD2+的Benign Full对照，IID(实际α=5000) / non-IID(实际α=5) ×共享seed91001–91010，共20个原70轮result。原数值/模型未重算；这里只读取配置和数据契约。
- 原冻结loader首先选官方train，再按Male分层抽样10% root；余下146493图像按Male=1、0的顺序shuffle/Dirichlet切到20客户端。**Dirichlet作用于敏感Male组，并非目标Smiling标签。**
- 执行的是SHA固定原core的sample_server_dataframe、create_client_data_dict、apply_root_noise与其helper的原AST函数，未导入或执行整套loader。root/client步骤与原loader相同；配置中两种root噪声均为0，无synthetic。
- 当前服务器89.22.197.55:60350，只在`/workspace/guardfed_checks/celeba_joint_partition_20261009/`落派生文件；读取实例guide后用现有cu128 Python、CPU单线程、CUDA_VISIBLE_DEVICES空执行。没有安装包、改服务、训练、推理或读取像素。
- metadata.npz和cache manifest SHA与20个原result全部一致。只向NumPy解码Smiling/Male的162770行官方训练前缀；本目录train_only_audit_metadata.npz仅含训练标签/敏感数组及完整ID/split。验证集只用ID核hash，未解码或使用验证/测试标签数组。NPZ原文件整体SHA核验属于字节身份检查，不能表述为完全未读取该压缩文件。

## 精确门检

| 检查 | 实测通过数量 |
|---|---:|
| 已验收原Benign result归档member SHA/bytes及本次remote原文件SHA | 20/20 |
| train、valid、root ID SHA | 60/60 |
| 原完整root sampling/noise audit精确一致 | 20/20各一套 |
| 原client样本总数 | 400/400 |
| 前次Sp-DFA审计的Male敏感边际 | 800/800 |
| NumPy独立实现的有序client ID哈希 / 联合行哈希 | 各400/400 |
| NumPy独立实现的client四格数 / root四格数 | 1600/1600；80/80 |
| 独立root联合行hash | 20/20 |
| root/client互斥、client/client互斥、完整覆盖train且与valid互斥 | 20/20 |
| 跨seed摘要标量复算 | 88/88，最大绝对误差0 |
| 审计前后受保护源码、metadata、manifest及原result | 24/24路径SHA不变 |

独立校验器没有调用原函数：以NumPy RandomState单独重放每个Male组的root抽样，再以default_rng重放客户端分割；它与原函数重放产物的全部400个有序client-ID和联合行哈希吻合。root由10个独立seed确定，在两分布中相同，不能当20个独立root。新client-ID/联合行hash是本次重建收据，**原result没有保存这些client级ID hash**；原保存ID hash覆盖train/valid/root，另有client样本数。

## 实际联合分布

下表先在每个seed内对20客户端计算描述量，再跨10个共享seed报告mean±sampleSD(ddof=1)。TVD以该seed去root后的客户端池Male×Smiling四格比例为参照；均值为客户端等权。百分比SD单位为百分点。范围表是每分布200个client观测的描述性范围，不能解释为200个独立seed。

| 分区描述量 | IID | non-IID |
|---|---:|---:|
| 每seed最小Smiling=1比例（%） | 46.976±0.365 | 44.224±0.422 |
| 每seed最大Smiling=1比例（%） | 49.008±0.354 | 51.501±0.618 |
| 20client正标签比例populationSD（百分点） | 0.550±0.112 | 1.964±0.203 |
| client联合TVD均值 | 0.007366±0.000723 | 0.113408±0.017341 |
| client联合TVD最大值 | 0.013766±0.002466 | 0.286706±0.023435 |
| 每seed最小client四格支持 | 1175.2±6.5 | 431.5±91.4 |
| 全部client正标签比例范围（%） | 46.278–49.630 | 43.747–52.381 |
| 全部client联合TVD范围 | 0.000744–0.016917 | 0.002075–0.323654 |
| 空client / 缺敏感组 / 缺标签 / 缺任意联合格 | 0/0/0/0 | 0/0/0/0 |

**400客户端均有四格支持；全部IID的最小单格为1161，non-IID为278。** 这是本20分区的实测支持，不保证任意seed、强α、root策略或部署中都不缺组。Smiling标签异质性明显弱于Male边际异质性，与敏感分组切分的实际实现一致；不能称为任意label-Dirichlet或所有图像异质性的充分覆盖。

## 前4攻击位置的联合覆盖

Benign原对照中没有执行恶意攻击。以下报告其他相同分区场景所使用的前4客户端位置覆盖份额；分母为该seed客户端池相应四格样本量，不包含root。4/20客户端不意味着各格或所有样本恰有20%被覆盖。

| 前4客户端覆盖份额（%） | IID | non-IID |
|---|---:|---:|
| 全部样本 | 19.964±0.102 | 18.874±3.132 |
| Male=0,Smiling=0 | 19.872±0.236 | 18.214±3.430 |
| Male=0,Smiling=1 | 20.005±0.139 | 18.354±3.370 |
| Male=1,Smiling=0 | 19.969±0.184 | 19.666±5.567 |
| Male=1,Smiling=1 | 20.027±0.258 | 19.710±5.614 |

non-IID四格覆盖在全部seed/四格中范围14.138%–30.903%；潜在攻击影响的样本/联合组覆盖并不固定。相同seed/alpha/配置分区先于attack，load_bundle/loader没有attack参数；本次Benign与前次Sp-DFA 400个样本总数及800个敏感边际也逐一吻合。这不是对其他所有方法运行的逐文件再验收。

## Root和原20分区明细

训练总体Male0/Smiling0、0/1、1/0、1/1计数为43688/50821/41002/27259。root每seed为16277，敏感分层数9451/6826；四格全部支持，最小2666。10独立seed的root联合TVD为0.003486±0.002155；本次全量root标签/ID/计数和原audit精确一致。

[per_client_400.csv](per_client_400.csv)给400客户端的四格、双边际、联合TVD、缺组标记、ID/联合行hash和原result SHA；[per_root_20.csv](per_root_20.csv)给20份root；[per_partition_20.csv](per_partition_20.csv)给每seed分区和攻击位置覆盖。

| 分布 | seed | Smiling=1比例% min–max | client最小四格 | root四格 Male0/Smiling0、0/1、1/0、1/1 |
|---|---:|---:|---:|---|
| IID | 91001 | 46.991–49.142 | 1180 | 4318/5133/4107/2719 |
| IID | 91002 | 47.214–49.482 | 1170 | 4300/5151/4108/2718 |
| IID | 91003 | 46.577–48.752 | 1176 | 4315/5136/4067/2759 |
| IID | 91004 | 47.246–48.846 | 1185 | 4403/5048/4144/2682 |
| IID | 91005 | 47.411–48.552 | 1161 | 4356/5095/4138/2688 |
| IID | 91006 | 47.208–48.826 | 1178 | 4373/5078/4092/2734 |
| IID | 91007 | 46.785–49.630 | 1179 | 4364/5087/4095/2731 |
| IID | 91008 | 47.287–48.811 | 1174 | 4431/5020/4160/2666 |
| IID | 91009 | 46.758–49.272 | 1176 | 4395/5056/4094/2732 |
| IID | 91010 | 46.278–48.772 | 1173 | 4349/5102/4121/2705 |
| non-IID | 91001 | 43.758–51.446 | 419 | 4318/5133/4107/2719 |
| non-IID | 91002 | 44.250–50.926 | 534 | 4300/5151/4108/2718 |
| non-IID | 91003 | 44.184–52.265 | 328 | 4315/5136/4067/2759 |
| non-IID | 91004 | 43.915–52.198 | 278 | 4403/5048/4144/2682 |
| non-IID | 91005 | 44.027–51.143 | 449 | 4356/5095/4138/2688 |
| non-IID | 91006 | 44.014–51.135 | 458 | 4373/5078/4092/2734 |
| non-IID | 91007 | 44.621–50.551 | 501 | 4364/5087/4095/2731 |
| non-IID | 91008 | 44.824–51.262 | 330 | 4431/5020/4160/2666 |
| non-IID | 91009 | 44.904–52.381 | 477 | 4395/5056/4094/2732 |
| non-IID | 91010 | 43.747–51.705 | 541 | 4349/5102/4121/2705 |

## 证据文件和复现

- [reference_contracts.json](reference_contracts.json)：20个原配置/数据契约、完整原result/member SHA与bytes、恢复归档身份；[prepare_inputs.py](prepare_inputs.py)从已验收本地Full100归档安全选取这些记录，逐member核SHA。
- [audit_joint_partitions.py](audit_joint_partitions.py)：remote单线程原函数重放；[frozen_functions.txt](frozen_functions.txt)为执行AST的原源码摘录，[frozen_loader.txt](frozen_loader.txt)是未执行的原loader全文。完整两份源码字节保留于sources/，SHA均固定。
- [remote_metadata_receipt.json](remote_metadata_receipt.json)：缓存/官方源hash、训练前缀请求范围；[acceptance.json](acceptance.json)：原身份门检、20分区、环境和种子摘要；[remote_run.log](remote_run.log)保留原执行输出。
- [verify_delivery.py](verify_delivery.py)验证SSH取得的archive SHA和memberSHA，再进行离线NumPy独立重放、400有序ID/联合行及1600四格检查；[independent_verification.json](independent_verification.json)与[transfer_verification.json](transfer_verification.json)保存验收。
- [verification.json](verification.json)记录交付及外部来源hash；[FILES_SHA256.json](FILES_SHA256.json)覆盖除其自身以外的全部文件。

已执行复现：`python -B verify_delivery.py`。若要用原AST再离线重放，可在已有numpy/pandas/torch环境执行`python -B audit_joint_partitions.py --offline`；它只产生offline_前缀结果，不能代替已经保存的remote原身份收据。无需联网/安装包/读取原图/非训练标签。

## 可供审阅的英文补充段落

We additionally recovered the clean label-by-sensitive joint allocation for all 20 CelebA partitions (two distributions and ten shared seeds). Replaying the frozen training-only root sampler and client splitter exactly matched all archived train/validation/root ID hashes, 400 client sample counts and 800 previously recovered sensitive margins. An independent index-based implementation matched all 400 ordered client-ID and joint-row hashes and all 1,600 client joint counts. The split applies Dirichlet allocation to Male groups rather than to Smiling labels. Across the 200 client observations per distribution, the Smiling=1 fraction ranged from 46.278% to 49.630% for IID and from 43.747% to 52.381% for non-IID. Every client contained all four Male-by-Smiling cells; the smallest cell had 1,161 and 278 examples, respectively. The within-seed mean client joint total-variation distance from its client-held population was 0.007366±0.000723 versus 0.113408±0.017341 across ten seeds. The 16,277-example root also retained all four cells, with minimum support 2,666. These are descriptive diagnostics of the realized sensitive-group allocation; they do not establish arbitrary label-skew robustness, predictive fairness or method superiority.

本段仅为有证据的补充候选；未写入原正文/当前rebuttal，也不代表最终作者主张获批。本子项关闭“这20个CelebA clean client联合计数缺失”，不关闭剩余方法集成、机制科学实验、冻结final评价、Fig3或FD执行来源缺口。
