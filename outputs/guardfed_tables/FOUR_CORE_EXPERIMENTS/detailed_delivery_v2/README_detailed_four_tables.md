# GuardFed-AD2+ Detailed Four Experiment Tables

这个版本不是 compact 摘要，而是把之前做过的四组大表重新整理成可读的完整表。Excel 文件名：

`GuardFed_AD2plus_Four_Experiments_Detailed_Tables.xlsx`

## 工作簿结构

- `01_Ablation_FULL`：完整消融表，包含 dataset、distribution、attack、profile、五类参数、ACC、AEOD、ASPD、FairAvg、Score。共 96 行明细，顶部另有 8 行关键效应摘要。
- `02_ServerRoot_FULL`：完整 server/root 分布表，包含 Adult 和 COMPAS、IID/non-IID、Benign/FedSA、三种 server/root 分布层级，以及 client alpha、root skew、Group/Sensitive/Label TVD、ACC、AEOD、ASPD、FairAvg、Score、相对 IID root 的 ΔACC/ΔFairAvg。共 24 行明细，底部附相关性/差值摘要。
- `03_Synthetic_FULL`：完整 10% clean 与 synthetic generation 表，包含 1%-10% clean ratio 的 dataset 汇总、distribution/attack slice 明细，以及所有 generator 的汇总。三块分别有 20、80、26 行。
- `04_FedSA_FULL`：完整 FedSA 新性能攻击表，包含 paper-style baseline 大表、所有方法汇总、AD2+ 参数候选 ranking。三块分别有 66、88、184 行。

## 看表方法

- ACC 越高越好。
- AEOD/ASPD 越低越好。
- FairAvg 不是算法，是辅助指标，等于 `(AEOD + ASPD) / 2`。
- 如果某个方法 ACC 已经明显塌陷，那么它的低 AEOD/ASPD 不能作为有效公平性胜出。
- 这版表保留完整维度；论文正文可从这些大表中再抽主表或 appendix 表。

## 四组实验的预期结论

1. 消融：去掉性能项后，性能攻击下 ACC 应下降；去掉公平项后，公平攻击或双重攻击下 AEOD/ASPD/FairAvg 应上升。
2. Server/root 分布：server/root 越接近 IID，root reference 越可靠；偏移越大时，Adult 上 ACC 下降、公平风险上升更明显。COMPAS 保留为完整结果，但不建议强行声称严格单调。
   - Client IID = Dirichlet alpha=5000；Client non-IID = Dirichlet alpha=5。
   - IID server/root = 10% clean stratified root, controlled skew=0.00。
   - Mild non-IID server/root = controlled skew=0.05。
   - Moderate non-IID server/root = controlled skew=0.10。
3. 10% clean 与 synthetic：10% real clean 稳定、透明，适合作为主实验设定；少量 real + synthetic 在部分 generator 下能降低公平风险，但 generator-dependent。
4. FedSA 新性能攻击：AD2+ 在 joint score 上表现最强，重点是兼顾 ACC、AEOD、ASPD，而不是只追求某一个单指标第一。
