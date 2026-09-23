# Plan 02 — 强化 FedSA 性能攻击

比较 Benign 与强化 FedSA，在固定范数约束下审计恶意更新偏移和 ACC 变化。

## 固定协议

- Adult、COMPAS；20 clients；主实验 4 个恶意客户端；70 rounds；local epoch=1；batch size=256；learning rate=0.005。
- 10% clean server/root data；无 synthetic root data；IID alpha=5000；non-IID alpha=5。
- 10 个 seeds：123、456、789、1001、2024、3141、4242、5050、6060、7070。

## 真实结果摘要

本计划包含 1760 个运行单元、5280 条 seed-metric 明细、35200 条客户端审计记录和 369600 条逐轮指标记录。

下表给出各数据集和分布下，攻击相对 Benign 的 10-seed 均值变化。ACC 的 delta 为攻击值减 Benign 值；AEOD/ASPD 的 delta 同样为攻击值减 Benign 值。

| Dataset | Distribution | Metric | Benign | Attack | Attack - Benign |
|---|---|---:|---:|---:|---:|
| adult | IID | ACC | 81.4583 | 80.9562 | -0.5020 |
| adult | IID | AEOD | 0.0366 | 0.0373 | 0.0007 |
| adult | IID | ASPD | 0.0683 | 0.0629 | -0.0055 |
| adult | non-IID | ACC | 81.6303 | 81.4423 | -0.1879 |
| adult | non-IID | AEOD | 0.0410 | 0.0310 | -0.0101 |
| adult | non-IID | ASPD | 0.0706 | 0.0667 | -0.0040 |
| compas | IID | ACC | 65.8801 | 66.1933 | 0.3132 |
| compas | IID | AEOD | 0.0513 | 0.0518 | 0.0005 |
| compas | IID | ASPD | 0.0403 | 0.0342 | -0.0060 |
| compas | non-IID | ACC | 65.7559 | 65.8693 | 0.1134 |
| compas | non-IID | AEOD | 0.0423 | 0.0492 | 0.0069 |
| compas | non-IID | ASPD | 0.0450 | 0.0336 | -0.0114 |

## 解释

FedSA 的预期作用是通过恶意更新偏移影响模型性能。应重点观察 ACC 的攻击前后变化，同时检查 Audit 中的更新范数和方向审计，避免把常数预测或未训练状态当作公平性优势。

本报告只描述实际运行结果；如果不同 seed 或分布出现局部波动，不将其强行解释为严格单调趋势。 

## 文件说明

- Excel：Adult_Table、COMPAS_Table、Summary、Raw_Seed_Metrics、Audit、Calibration、Trajectory_Mean 和 Charts。
- raw_results.jsonl、raw_seed_metrics.csv、audit.csv、trajectory.csv：本计划的筛选原始记录，保留完整精度。
- Fairness 显示下限 0.0001 只用于展示，不改变原始值。
