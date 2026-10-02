# Adult Table II 完整恢复交付

2026-10-02。输出覆盖16个历史展示方法、IID/non-IID、5场景、3指标，480格均有底层数值。不是最终可提交的忠实十seed基线比较。

- `table2_complete_review.pdf`：4页，前两页同终轮70修正候选，后两页历史选轮复原。
- `table2_complete_review.tex`：独立审阅文档入口；另有4个表体`.tex`。正文移植需要适配IEEE栏宽并保留统计/方法脚注。
- `table2_round70_revision.md` / `_cells.json`：每run三指标同终轮70；保留真实不同配方与n。
- `table2_historical_reconstruction.md` / `_cells.json`：复原旧选择规则、全部隐藏值及有重复记录的sampleSD；不作为无偏统一排名。

294格n=1、180格n=10、6格n=3；n=1的上标1表示单次值，SD不可计算，不是±0。FairGuard IID FedSA ACC在历史源中为54.13%，原投稿为59.13%；同终轮候选为53.94%。FedWA源为AdaAggRL；Cosine行混合GuardFed-AD2/GuardFed实现，均已脚注说明。

[证据与更正说明](../../../docs/server_deployment_20260923/revision_20260923/table2_trace_20261002/追溯报告.md)、[完整逐格来源](../../../docs/server_deployment_20260923/revision_20260923/table2_trace_20261002/source_map_complete.json)。无新训练，未修改原投稿PDF。数值重建完成不等于补齐缺失seed或认证原法实现；组分母支持和历史模型二进制身份仍有保留证据的局限。
