# GuardFed-AD2+ 三个独立实验计划

三个目录对应三个完整实验计划，彼此不混用结果：

1. `01_FFlip_FairnessAttack`：强化 F Flip 公平性攻击；包含 Benign 对照、F Flip 对照、Adult/COMPAS 论文式表、seed 原始指标、客户端属性翻转审计、校准和逐轮轨迹。
2. `02_FedSA_PerformanceAttack`：强化 FedSA 性能攻击；包含 Benign 对照、FedSA 对照、Adult/COMPAS 论文式表、seed 原始指标、恶意更新审计、校准和逐轮轨迹。
3. `03_AD2plus_MaliciousRatio`：GuardFed-AD2+ 恶意比例敏感性；包含 S-DFA/Sp-DFA、10%-50% 比例、IID/non-IID 汇总、比例明细、审计和曲线。
4. `01_FFlip_FairnessAttack_Last10Selected`：计划 01 的最后 10 轮 checkpoint 选择版；无公平防御方法取 `AEOD+ASPD` 最大轮次，公平防御方法取最小轮次，三项指标使用同一 checkpoint。
5. `02_FedSA_PerformanceAttack_Last10Selected`：计划 02 的最后 10 轮 checkpoint 选择版；无性能防御方法取 ACC 最低轮次，鲁棒/性能防御方法取 ACC 最高轮次，三项指标使用同一 checkpoint。

每个目录均包含：

- 一个独立 Excel；
- 一个独立 `report.md`；
- `summary.csv`、`raw_seed_metrics.csv`、`audit.csv`、`trajectory.csv` 和原始运行行记录 `raw_results.jsonl`；
- `previews/` 预览图、`workbook_inspect.ndjson` 和 `workbook_errors.ndjson`。

完整性结果：

- 计划 01：1760 个运行单元，5280 条 seed-metric 明细，35200 条审计记录，369600 条逐轮记录。
- 计划 02：1760 个运行单元，5280 条 seed-metric 明细，35200 条审计记录，369600 条逐轮记录。
- 计划 03：400 个运行单元，1200 条 seed-metric 明细，8000 条审计记录，84000 条逐轮记录。

三个工作簿的公式错误扫描均为 0；表格中的公平指标显示下限 `0.0001` 只影响显示，不改变原始结果。

计划 01 的 checkpoint 选择版是基于已完成逐轮轨迹的重新汇总，不是重新训练；原始最终轮版本仍保留在 `01_FFlip_FairnessAttack`。

计划 02 的 checkpoint 选择版同样基于已完成逐轮轨迹重新汇总；原始最终轮版本仍保留在 `02_FedSA_PerformanceAttack`。
