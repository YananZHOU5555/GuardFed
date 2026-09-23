# GuardFed-AD2+ 强化攻击与恶意比例实验说明

## 实验范围

本目录包含 Adult 和 COMPAS 两个数据集上的强化攻击实验，以及 GuardFed-AD2+ 在恶意客户端比例 `10%/20%/30%/40%/50%` 下的双向攻击敏感性实验。Class-B FL 和 GuardFed-ACT 均未纳入。

固定协议为：20 个客户端，主表每次 4 个恶意客户端，70 轮，1 个本地 epoch，batch size 256，learning rate 0.005，10% clean server/root data，不使用 synthetic root data；IID 使用 `alpha=5000`，non-IID 使用 `alpha=5`。每个主实验使用 10 个 seeds，表格数值为第 70 轮的 10-seed 均值。

## 攻击配置

- F Flip：通过校准选出统一的 `all_unprivileged` 模式，保持真实标签不变，并在 Audit 工作表记录敏感属性修改覆盖率和标签变化数。
- FedSA：统一使用校准得到的 `gain=4.5`、`norm_ratio=3.0`，保留范数约束，Audit 工作表记录恶意更新范数、范数比和方向变化。
- 比例实验：仅运行 GuardFed-AD2+；S-DFA 中所有恶意客户端同时执行两类攻击，Sp-DFA 中公平攻击组和性能攻击组分开执行。恶意客户端数分别为 `2/4/6/8/10`。

## 结果如何阅读

四个论文式工作表分别给出 Adult/COMPAS 在 F Flip/FedSA 下的完整方法对比，并同时保留 Benign、F Flip 和 FedSA 三列。ACC 越高越好；AEOD、ASPD 越低越好。绿色加粗表示当前列最优，蓝色下划线表示第二优。公平指标只在运行通过有效性门槛时参与排名；无效运行保留真实值但灰显，不能用模型未训练或常数预测造成的接近零公平指标获得排名优势。

`Ratio_Summary` 给出每个数据集、攻击、恶意比例和指标的 IID/non-IID 均值与标准差；`Ratio_Detail` 保存每个 seed 的比例实验；`Ratio_Charts` 给出 12 个曲线图。比例实验的实测曲线可能出现局部波动，因此报告时应描述总体变化和数据集差异，不应把每个指标强行解释为严格单调。

## 数据完整性

- 主实验：22 个方法 × 2 个数据集 × 2 种分布 × 3 类攻击 × 10 个 seeds = `2640` 个运行单元，对应 CSV 中的 `7920` 条三指标明细。
- 比例实验：2 个数据集 × 2 种双向攻击 × 5 个恶意比例 × 2 种分布 × 10 个 seeds = `400` 个运行单元，对应 CSV 中的 `1200` 条三指标明细。
- 另有 `80` 条 smoke 记录保留在原始 JSONL 中，但不进入主表和比例汇总。
- 完整逐轮轨迹保存在 `results/attack_strength/trajectory.csv`，原始运行记录保存在 `results/attack_strength/raw_results.jsonl`。

## 交付文件

`GuardFed_AD2plus_AttackStrength_10Seeds.xlsx` 是最终工作簿，包含 README、四个论文式主表、比例汇总、比例明细、10-seed 原始指标、攻击审计、攻击校准和比例图表工作表。Fairness 展示值低于 `0.0001` 时只在显示层显示为 `0.0001`；原始 CSV/JSONL 保留完整精度。

本批结果来自实际运行记录，没有为了强化趋势而手工改写表格数值。
