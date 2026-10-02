# Table II追溯入口

当前结论以[追溯报告](追溯报告.md)、[完整源映射](source_map_complete.json)、[重建验收](complete_reconstruction_verification.json)和[交付收据](delivery_receipt.json)为准。480格数值链已对齐、44隐藏值已恢复；不支持整表10seed或忠实基线身份的声明。

`初步追溯报告_历史快照.md`、`trace_summary.json`、`condition_candidates.json`、`hidden_value_candidates.json`、`隐藏值候选清单.md`和`独立种子审查.md`保留早期有限搜索结果，其“未匹配”不代表当前最终结论。扩展检索见`extended_*`和`lineage/`，新发现已在主报告解释。

复现入口为`build_complete_table2.py`。大型原子JSONL和两份完整归档保存在本机`tmp/`，没有纳入Git；逐格映射保留相对路径、记录行号/seed/run身份，收据保留输入内容哈希。仅克隆Git可以读表与审计记录，但重建需同时具备匹配哈希的保留归档。发布小报告不代表发布全部原数据/模型。
