# Plan 01 — 强化 F Flip 公平性攻击

比较 Benign 与强化 F Flip，在不修改真实标签的前提下审计敏感属性翻转和公平风险变化。

## 固定协议

- Adult、COMPAS；20 clients；主实验 4 个恶意客户端；70 rounds；local epoch=1；batch size=256；learning rate=0.005。
- 10% clean server/root data；无 synthetic root data；IID alpha=5000；non-IID alpha=5。
- 10 个 seeds：123、456、789、1001、2024、3141、4242、5050、6060、7070。

## 真实结果摘要

本计划包含 1760 个运行单元、5280 条 seed-metric 明细、35200 条客户端审计记录和 369600 条逐轮指标记录。

下表给出各数据集和分布下，攻击相对 Benign 的 10-seed 均值变化。ACC 的 delta 为攻击值减 Benign 值；AEOD/ASPD 的 delta 同样为攻击值减 Benign 值。

| Dataset | Distribution | Metric | Benign | Attack | Attack - Benign |
|---|---|---:|---:|---:|---:|
| adult | IID | ACC | 81.4583 | 81.0970 | -0.3612 |
| adult | IID | AEOD | 0.0366 | 0.0350 | -0.0016 |
| adult | IID | ASPD | 0.0683 | 0.0632 | -0.0051 |
| adult | non-IID | ACC | 81.6303 | 81.3739 | -0.2563 |
| adult | non-IID | AEOD | 0.0410 | 0.0255 | -0.0156 |
| adult | non-IID | ASPD | 0.0706 | 0.0610 | -0.0096 |
| compas | IID | ACC | 65.8801 | 66.2689 | 0.3888 |
| compas | IID | AEOD | 0.0513 | 0.0538 | 0.0025 |
| compas | IID | ASPD | 0.0403 | 0.0439 | 0.0036 |
| compas | non-IID | ACC | 65.7559 | 66.2689 | 0.5130 |
| compas | non-IID | AEOD | 0.0423 | 0.0398 | -0.0025 |
| compas | non-IID | ASPD | 0.0450 | 0.0265 | -0.0185 |

## 解释

F Flip 的预期作用是破坏敏感属性与标签之间的统计关系，而不是修改真实标签。表格和 Audit 工作表应共同阅读：公平指标的变化需要在 ACC 仍通过有效性门槛时解释。

本报告只描述实际运行结果；如果不同 seed 或分布出现局部波动，不将其强行解释为严格单调趋势。 

## 文件说明

- Excel：Adult_Table、COMPAS_Table、Summary、Raw_Seed_Metrics、Audit、Calibration、Trajectory_Mean 和 Charts。
- raw_results.jsonl、raw_seed_metrics.csv、audit.csv、trajectory.csv：本计划的筛选原始记录，保留完整精度。
- Fairness 显示下限 0.0001 只用于展示，不改变原始值。
