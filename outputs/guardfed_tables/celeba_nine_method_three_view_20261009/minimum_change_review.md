# 最小差异核对

原表脚本SHA256为`4ae22c85c3fcc1f8e01ac9b3fc4dbf0ce45ea676f39a677e337782cb0d2b3529`。直接读取并执行原`values_for`函数和原`render`函数；原函数中的`statistics.mean`/`statistics.stdev`、ACC百分数缩放、2/4位精度、方法/类别/场景排列均保持不变。原脚本保持原字节。

仅有两处render展示字符串差异：标题加raw/native/shared calibration视图名；把单分布表不准确的“All 90 … cells”改成“All displayed cells”。可审阅[renderer_presentation_only.diff](renderer_presentation_only.diff)。全局notes替换为当前三视图及实际混合设备边界；每面板环境计数仍由原render代码从实际rows计算。原PDF sink改为无写入sink，因此交付PNG/Markdown/TeX片段，不产生PDF。

schema adapter读取原900库存/collector/proof/strict/receipt/array，沿原`merge_sources.py`的`source_method`别名映射连接FedAA-DDPG及LASA显示名。冻结evaluator自己的`METHODS`仍为原值，receipt的`FedAA`、fit payload和fit_sha256保留原字节，没有把显示别名回写科学输入。

原evaluator的`predict_views`与`evaluate_frozen_predictions`、原replay的`check_native`均直接由AST读取执行。仅把JSON阈值键恢复为原接口要求的整数0/1；没有拟合函数、模型载入、optimizer、CNN或test调用。

新增producer的其余代码负责证据身份/哈希验证、数据形状适配、保存provenance与调用原函数。独立audit不导入producer，重新读取已保存JSON/CSV/Markdown/TeX，用NumPy mean/std(ddof=1)核对2430个汇总对及3240个展示单元格；其计算仅为核验，不替换生产值。

没有新增显著性检验、聚合score、最佳view选择或论文胜负结论；原负结果、方差和常量预测完整保留。native与旧900展示表完全相同；旧snapshot全精度SD的94个末位差完整列出，不声称旧JSON的SD为bitwise相等。
