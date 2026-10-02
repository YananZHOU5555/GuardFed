# Table II 历史 shards 扩展审查

## 后续Windows选轮脚本来源补全

后续另一归档的 `three_plans/*Last10Selected` 与生成脚本解释了原表 F-Flip/FedSA 来源；此前“未发现解释新配方”的结论仅适用于本README下面限定的5090 raw归档。已经直接用原始完整轨迹独立复算：两计划各880次选轮、5280个逐seed指标值及2640个summary标量全部通过，最大数值误差2.23e-16；AdaAggRL/FedWA候选三种子joint规则12条/60逐记录值/40汇总标量也通过，最大1.12e-16。

**F-Flip与FedSA选轮规则因方法组而异，不是统一终轮协议。** F-Flip受保护组=min(AEOD+ASPD)，其他=max；FedSA FedAvg/FairFed=minACC，其他=maxACC；61..70并列取较后轮。AdaAggRL三种子则采用最大ACC-.5*(AEOD+ASPD)，并列保留先轮。已有十种子均值/SD可恢复，但须披露这些历史规则，不能说成所有方法统一round70。`selection_verification.json` 保留规则、数量、误差、source hashes及AdaAggRL逐seed轮次/指标。

## 5090 raw归档限定范围的检查

范围：`tmp/git_sync_assets/guardfed-5090-20260923.tar.gz` 内全部 11 个 paper_tables/attack_strength JSONL 文件。原文件逐字节 SHA-256，候选记录完整 canonical JSON SHA-256；不改原归档、不计算第二份既有主队列统计、不启动训练。

结果：**新增同配方 cohort=0；新增可用于种子统计的 baseline 记录=0。**

| 文件类别 | 核验 |
|---|---|
| paper_tables/raw_results.jsonl | 690 个过滤后基线记录全部与已读副本精确重复；归档无另一个 paper_tables 原始备份 JSONL |
| attack_strength/raw_results_before_dedup 与 before_ratio_dedup | 两个整个文件字节 SHA 相同；各 1200 条过滤后基线记录均是既有主队列原记录 |
| ratio_shard0/1 | 每个 1200 条 main 基线记录均重复；ratio 本身是不同恶意比例 AD2+，排除 |
| method_shard0/1 | 1120/1122 个基线记录均为既有主队列子集 |
| shard3 | 360 条基线记录与既有记录精确重复 |
| shard2 | 360 条完整记录 SHA 不同，但逐条核实唯一差异为 duration_sec；同 run_id 的配置、三终轮指标、全部 trajectory 及其他元数据精确相同，不计新增 seed |
| calibration_results | 72 条为单独 calibration 输出，不能当原 Table II 同协议训练种子 |

未发现之前遗漏的基线 S-DFA/Sp-DFA 十种子，也未发现能解释原投稿 F-Flip/FedSA 数字的新历史配置或生成规则。本结论仅覆盖这一个 2026-09-23 归档，不声称其他机器或未归档位置不存在其他文件。

原先结论保持：后期攻强主队列 Benign/F Flip/FedSA 有十种子，但与旧表攻击配方不同；旧 paper_tables FedSA 有 123/456/789 三种子（不是十种子均值），其他旧表 baseline 在所核同配方中单种子。旧导出 `build_clean10_true_tables.mjs` 明确 baseline seed123、last10 ACC=max/AEOD-ASPD=min 逐指标选择，不能把其点值称十种子均值。投稿表与导出不一致的 F-Flip/FedSA 来源仍应逐格标识未追溯，而不凭新 shard 重复补 SD。

详细逐成员 hash、记录重复数及 metadata 差异在 `summary.json`，全文件构成在 `inventory.json`。`novel_records.jsonl` 仅保存 360 个时长不同的完整记录及其原成员/行号，供核验；不是新实验结果。统计源代码/原数据内容/checkpoint 身份 hash 仍无独立证据。
