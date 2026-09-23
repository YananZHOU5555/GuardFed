# Plan 02 revised — FedSA last-10 performance checkpoint selection

在最后 10 轮内统一选择性能 checkpoint：无性能防御的方法取 ACC 最低轮次；鲁棒/性能防御方法和 AD2 系列取 ACC 最高轮次。

## 固定协议

- Adult、COMPAS；20 clients；主实验 4 个恶意客户端；70 rounds；local epoch=1；batch size=256；learning rate=0.005。
- 10% clean server/root data；无 synthetic root data；IID alpha=5000；non-IID alpha=5。
- 10 个 seeds：123、456、789、1001、2024、3141、4242、5050、6060、7070。

## Checkpoint 选择规则

For each method/dataset/distribution/seed, select one checkpoint from rounds 61-70. Unprotected FedAvg and FairFed use the minimum ACC; Median, FLTrust, FairGuard, FLTrust+FairGuard, GuardFed, FLGMM, FLAURA, LayerGuard, SmartFL, FLTG, FedDNA, LASA, Fed-NGA, Huber-BRFL, LoGoFair, AdaAggRL, FedAMM, FedAA, GuardFed-AD2 and GuardFed-AD2+ use the maximum ACC. ACC, AEOD and ASPD come from the same checkpoint.
性能选择分数定义为 ACC；三个指标均使用同一个被选 checkpoint。

## 真实结果摘要

本计划包含 1760 个运行单元、5280 条 seed-metric 明细、35200 条客户端审计记录和 369600 条逐轮指标记录。

下表给出各数据集和分布下，攻击相对 Benign 的 10-seed 均值变化。ACC 的 delta 为攻击值减 Benign 值；AEOD/ASPD 的 delta 同样为攻击值减 Benign 值。

| Dataset | Distribution | Metric | Benign | Attack | Attack - Benign |
|---|---|---:|---:|---:|---:|
| adult | IID | ACC | 81.4583 | 81.8295 | 0.3712 |
| adult | IID | AEOD | 0.0366 | 0.0331 | -0.0035 |
| adult | IID | ASPD | 0.0683 | 0.0730 | 0.0047 |
| adult | non-IID | ACC | 81.6303 | 82.8329 | 1.2026 |
| adult | non-IID | AEOD | 0.0410 | 0.0362 | -0.0048 |
| adult | non-IID | ASPD | 0.0706 | 0.0867 | 0.0161 |
| compas | IID | ACC | 65.8801 | 66.5281 | 0.6479 |
| compas | IID | AEOD | 0.0513 | 0.0424 | -0.0089 |
| compas | IID | ASPD | 0.0403 | 0.0202 | -0.0200 |
| compas | non-IID | ACC | 65.7559 | 66.7873 | 1.0313 |
| compas | non-IID | AEOD | 0.0423 | 0.0410 | -0.0013 |
| compas | non-IID | ASPD | 0.0450 | 0.0299 | -0.0150 |

## 解释

FedSA 的预期作用是通过恶意更新偏移影响模型性能。应重点观察 ACC 的攻击前后变化，同时检查 Audit 中的更新范数和方向审计，避免把常数预测或未训练状态当作公平性优势。

本报告只描述实际运行结果；如果不同 seed 或分布出现局部波动，不将其强行解释为严格单调趋势。 

## 文件说明

- Excel：Adult_Table、COMPAS_Table、Summary、Raw_Seed_Metrics、Audit、Calibration、Trajectory_Mean 和 Charts。
- raw_results.jsonl、raw_seed_metrics.csv、audit.csv、trajectory.csv、checkpoint_selection.csv：本计划的筛选原始记录，保留完整精度。
- Fairness 显示下限 0.0001 只用于展示，不改变原始值。
