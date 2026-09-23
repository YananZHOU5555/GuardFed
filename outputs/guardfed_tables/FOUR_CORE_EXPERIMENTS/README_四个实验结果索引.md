# GuardFed-AD2+ 四个核心实验结果索引

这个目录把当前要给导师/论文使用的四类实验结果集中放在一起，避免和旧版本结果混在一起。

总目录：

`E:\OneDrive\文档\GuardFed\outputs\guardfed_tables\FOUR_CORE_EXPERIMENTS`

总 Excel：

`GuardFed_AD2plus_Expanded_FedSA_TrueResults_FINAL.xlsx`

总说明：

`FINAL_experiment_summary.md`

## 1. 消融实验

目录：

`01_ablation`

主要文件：

- `adult_ad2plus_ablation.md`: Markdown 版 AD2+ score/component ablation 明细。
- `adult_ad2plus_ablation.csv`: 原始逐实验单元结果。
- `adult_ablation_by_tag_summary.csv`: 按 ablation tag 汇总后的表。
- `GuardFed_AD2plus_advisor_experiment_report.pdf`: PDF 报告。
- `GuardFed_AD2plus_advisor_experiments_true_results.xlsx`: Excel 汇总。

说明：

这组是 AD2+ 的 score/component ablation，覆盖 Adult 的 IID/non-IID 和多种攻击。核心看 `full_score` 与 `no_utility_U`, `no_centrality_C`, `no_alignment_A`, `no_fairness_risk_F`, `no_violation_V`, `reward_only_R`, `penalty_only_P`, `macro_R*_P*` 等配置的对比。

## 2. Server/root distribution 同分布/分布敏感性消融

目录：

`02_server_distribution`

主要文件：

- `server_dist30_strongfloor_v4_summary.md`: Markdown 总结。
- `server_dist30_strongfloor_v4_by_alpha_dataset.csv`: 按 dataset 和 alpha 汇总。
- `server_dist30_strongfloor_v4_by_alpha_slice.csv`: 按 dataset/distribution/attack/alpha 汇总。
- `server_dist30_strongfloor_v4_curve_diagnostics.csv`: ACC range 和 adjacent jump 诊断。
- `server_dist30_strongfloor_v4_joint_raw.csv`: 原始汇总行。
- `server_dist30_strongfloor_v4_acc_curve.png`: ACC 曲线。
- `server_dist30_strongfloor_v4_score_curve.png`: joint score 曲线。
- `server_dist30_strongfloor_v4_fairness_curve.png`: AEOD/ASPD 曲线。
- `preview_ServerDist30_Strong.png`: Excel sheet 预览。

Excel 中对应 sheet：

- `ServerDist30_Strong`
- `CurveDiag_Strong`
- `Figures`

说明：

这是最终采用的 30 个 Dirichlet alpha、3 seeds 的版本。相比旧版 label-preserve，strong-floor 版本把 ACC 波动压到了更合理范围。

## 3. 数据生成 / 10% clean server data 消融

目录：

`03_synthetic_generation_10pct`

主要文件：

- `synthetic_stratified_clean_cap10_dense_v4_summary.md`: Markdown 总结。
- `synthetic_stratified_clean_cap10_dense_v4_by_dataset.csv`: Adult/COMPAS 分开汇总。
- `synthetic_stratified_clean_cap10_dense_v4_by_slice.csv`: 按 dataset/distribution/attack 汇总。
- `synthetic_stratified_clean_cap10_dense_v4_joint_raw.csv`: 原始汇总行。
- `preview_Synthetic_v4_Dataset.png`: dataset 汇总预览。
- `preview_Synthetic_v4_Slices.png`: slice 汇总预览。

Excel 中对应 sheet：

- `Synthetic_v4_Dataset`
- `Synthetic_v4_Slices`

说明：

这组实验比较 1%-10% real clean root data。最终关键结论是：

- Adult: `10% real clean` 的 joint score rank = 2。
- COMPAS: `10% real clean` 的 joint score rank = 1。

因此 10% clean server/root data 是当前最终表中采用的合理配置。

## 4. FOE 替换为新性能攻击 FedSA

目录：

`04_new_performance_attack_FedSA`

主要文件：

- `fedsa_paper_style_selected_ad2plus.md`: 论文风格 Markdown 表，第一加粗、第二下划线。
- `fedsa_paper_style_selected_ad2plus_display.csv`: 带 Markdown 标记的展示表。
- `fedsa_paper_style_selected_ad2plus_raw.csv`: 数值原始表。
- `fedsa_ad2plus_candidate_compact_ranking.csv`: baseline + AD2+ candidate ranking。
- `fedsa_ad2plus_candidate_joint_summary.csv`: 详细 ranking 汇总。
- `fedsa_ad2plus_candidate_summary.md`: 候选总结。
- `preview_FedSA_Table.png`: paper-style 表预览。
- `preview_FedSA_Ranking.png`: ranking 表预览。

Excel 中对应 sheet：

- `FedSA_Table`
- `FedSA_Raw`
- `FedSA_Ranking`

说明：

这里是 FOE 替换成 FedSA 后的新性能攻击实验。所有方法都在 FedSA 下重新跑过。主表中 GuardFed-AD2+ 在四个 slice 的 joint score 均为第 1：

- Adult IID
- Adult non-IID
- COMPAS IID
- COMPAS non-IID

注意：

AD2+ 不是每一列 ACC 都最高。部分 ACC 第一来自 FedAA/FedAMM/FedDNA，但这些方法的 AEOD/ASPD 明显更差。因此最终结论应写为：GuardFed-AD2+ 在性能与公平性的 joint defense score 下最优，并且在 FedSA 下公平性指标最突出。

