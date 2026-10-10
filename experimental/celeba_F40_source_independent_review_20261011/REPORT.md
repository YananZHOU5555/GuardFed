# F40 独立源码审阅

结论：**PASS（仅源码与晚绑定契约）**。未发现阻塞性问题；没有执行构表、统计验证、模型、数组、SSH或科学采用。

实读 F30→F40 五份 Python 源码的完整差分，以及候选 binding/build/verify_saved/panels/verify_numeric、输入白名单、未绑定模板和说明。候选封条为 `f8f4f48bcc4cc278b6968f8d81f298af1de7e12205fad6e7104a0fa9ebdabd5b`；11个成员的当前 SHA/大小逐项核对。

- **晚绑定有效。** 当前没有 ROOT_BINDING.json；模板根状态为 false、未来输入为空。build 与 verify_saved 均在数字工作前调用 load；要求明确 binding SHA、被审源码封条、实际 root/index/native/inspection SHA 串联。父330 root/index固定，新增只允许 IID S-DFA 91001–91010，接受数严格330+10=340。更大的 native 接受集合不能扩大表范围。
- **旧表保留门完整。** F30 records/tables/TABLES.md 及旧根在输入 SHA 白名单；旧60个记录从原表直接拼接在前，序列化片段逐条比较原字节和顺序。九面板中原 Benign/F Flip/FedSA 行与旧表对象完全相等，覆盖486统计数及243显示格。独立 saved verifier 再核旧记录、表行和所有 Markdown 数值；这保证旧结果与显示数值保持，不声称扩展后的整份 Markdown 与旧文件字节相同。
- **身份和统计未改。** 新十条先经原 receipt_identity，绑定同一个70轮 checkpoint 的 raw/native/shared 三视图；Full 原 full_record 检查 checkpoint/reference，同 seed、IID、S-DFA、alpha5000及完整数据划分配对。统计仍原 statistic/summarize；固定10/9/6 seed面板、sample SD(ddof1)、逐seed minus_F−Full，负值不筛除。独立 fsum/样本SD和confusion-count算术体保持；只增加场景和相应基数。没有跨场景伪样本量或显著性/必要性推断。
- **边界清楚。** 预期80记录、40对、648 scalars/324 cells及720 metrics/1920 counts均只是未来输出要求，不是已测结果。保留valid暴露、seed91001选择史、设备来源和历史test暴露说明；不执行拟合/推理/训练/test，不称F100或所有控制完成。

后续仍需 root 实际采用340输入、独立绑定真实hash、单次build及原saved verifier通过才能使用F40结果。本审阅没有证明尚不存在的统计输出正确，也没有复核旧全量科学数据。
