# Plan 03 — GuardFed-AD2+ 恶意比例敏感性

只运行 GuardFed-AD2+，考察恶意客户端比例 10%-50% 下 S-DFA 和 Sp-DFA 的 ACC、AEOD、ASPD。

## 固定协议

- Adult、COMPAS；20 clients；恶意客户端数为 2/4/6/8/10（10%-50%）；70 rounds；local epoch=1；batch size=256；learning rate=0.005。
- 10% clean server/root data；无 synthetic root data；IID alpha=5000；non-IID alpha=5。
- 10 个 seeds：123、456、789、1001、2024、3141、4242、5050、6060、7070。

## 真实结果摘要

本计划包含 400 个运行单元、1200 条 seed-metric 明细、8000 条客户端审计记录和 84000 条逐轮指标记录。

比例序列按 10%、20%、30%、40%、50% 展示，分别对应 2、4、6、8、10 个恶意客户端。以下为 AD2+ 的最终轮 10-seed 均值序列：

- adult / S-DFA / IID / ACC: 81.2737, 80.6793, 81.6170, 81.1853, 79.2264
- adult / S-DFA / non-IID / ACC: 82.0393, 81.5844, 81.5658, 81.2212, 79.7277
- adult / S-DFA / IID / AEOD: 0.0344, 0.0376, 0.0322, 0.0417, 0.0274
- adult / S-DFA / non-IID / AEOD: 0.0445, 0.0520, 0.0240, 0.0335, 0.0257
- adult / S-DFA / IID / ASPD: 0.0577, 0.0531, 0.0663, 0.0611, 0.0396
- adult / S-DFA / non-IID / ASPD: 0.0738, 0.0679, 0.0671, 0.0605, 0.0466
- adult / Sp-DFA / IID / ACC: 81.5121, 81.3248, 80.9151, 81.1296, 80.8294
- adult / Sp-DFA / non-IID / ACC: 81.5041, 81.1694, 80.9224, 80.6116, 80.2264
- adult / Sp-DFA / IID / AEOD: 0.0375, 0.0286, 0.0274, 0.0395, 0.0231
- adult / Sp-DFA / non-IID / AEOD: 0.0300, 0.0366, 0.0362, 0.0380, 0.0363
- adult / Sp-DFA / IID / ASPD: 0.0662, 0.0644, 0.0567, 0.0611, 0.0554
- adult / Sp-DFA / non-IID / ASPD: 0.0650, 0.0588, 0.0572, 0.0546, 0.0494
- compas / S-DFA / IID / ACC: 66.0799, 65.4860, 65.1458, 65.8531, 65.4374
- compas / S-DFA / non-IID / ACC: 66.5605, 65.6911, 65.3726, 65.9773, 64.7570
- compas / S-DFA / IID / AEOD: 0.0517, 0.0485, 0.0541, 0.0486, 0.0464
- compas / S-DFA / non-IID / AEOD: 0.0485, 0.0476, 0.0642, 0.0515, 0.0296
- compas / S-DFA / IID / ASPD: 0.0327, 0.0340, 0.0426, 0.0346, 0.0343
- compas / S-DFA / non-IID / ASPD: 0.0323, 0.0322, 0.0345, 0.0298, 0.0173
- compas / Sp-DFA / IID / ACC: 66.1447, 65.8045, 65.5994, 66.0097, 66.1447
- compas / Sp-DFA / non-IID / ACC: 65.4428, 65.7721, 66.0691, 65.4644, 65.6102
- compas / Sp-DFA / IID / AEOD: 0.0585, 0.0607, 0.0674, 0.0344, 0.0512
- compas / Sp-DFA / non-IID / AEOD: 0.0622, 0.0398, 0.0570, 0.0461, 0.0573
- compas / Sp-DFA / IID / ASPD: 0.0350, 0.0347, 0.0569, 0.0260, 0.0349
- compas / Sp-DFA / non-IID / ASPD: 0.0372, 0.0302, 0.0280, 0.0271, 0.0454

## 解释

比例实验用于观察攻击者比例变化下的实际退化和公平风险变化。由于不同数据集、分布和攻击组合存在随机性，曲线可能局部波动；结论应同时报告端点、整体方向和异常拐点，而不能把每一列强行解释成严格单调。

S-DFA 将公平性和性能攻击施加到同一恶意客户端；Sp-DFA 将两类攻击分配到不同恶意客户端。Ratio_Summary、Ratio_Detail、Audit 和 Charts 应联合阅读。

## 文件说明

- Excel：Ratio_Summary、Ratio_Detail、Summary、Raw_Seed_Metrics、Audit、Calibration、Trajectory_Mean 和 Charts。
- raw_results.jsonl、raw_seed_metrics.csv、audit.csv、trajectory.csv：本计划的筛选原始记录，保留完整精度。
- Fairness 显示下限 0.0001 只用于展示，不改变原始值。
