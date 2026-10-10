# A20 + LoGoFair100 完整英文稿独立文字审阅

结论：**PASS，适用于作者审阅稿采用；零实质性阻断，一项非阻断编辑建议。** 此结论不表示返修完成、正文已应用或可提交。

审阅固定目录：`docs/server_deployment_20260923/revision_20260923/rebuttal_integrated_A20_LoGo100_20261010/`。两份英文稿均完整阅读；被审11成员封条 SHA256 为 `713cfb3d32c05ea0a61d419f344cda823b5fc36c33e80233961cf86366a207d8`。

独立核对结果：24条审稿原话与原 comment_source_map 逐字、顺序一致；11成员文件SHA/大小、34来源pins均匹配；两稿81个本地链接均存在，1个继承GitHub URL仅核文本，未联网核远端。完整回复覆盖AE、R1、R2、R3及P1–P6；正文插入稿有10个完整候选章节，仍明确未应用到投稿正文。

新增段落保留了以下实质边界：

- A20仅覆盖IID Benign和F Flip的20配对模型；其余八个A场景、其他未完成控制及完整目标方法比较没有被写成完成。原COMPAS删除组件反例、恒定预测、历史Table II混合实际n和缺失SD仍在。
- LoGoFair100明确96新结果+4复用、固定recipe07和fit seed1719；20个虚拟cohort不是真实训练客户端。DP明确指demographic parity，不是differential privacy。拟合后的LoGoFair prediction与FedAvg缓存的valid_native_prediction明确区分。
- 十方法1000条native没有被写成1000条三视图或完整17方法。原九方法900三视图、共有校准归因和新native比较保持分开。AD2+的ACC/AEOD较好、LoGoFair的ASPD较低，两边取舍保留，没有所有指标胜出、显著性或每个组件必需的结论。
- AEOD定义为absolute TPR gap；native/shared相同的指标与计数没有被当成独立校准获益。保留选择seed91001、验证集曝光、初始240条官方test结果已看过、混合CPU/GPU/cu128/cu130和driver限制；未称从未查看的holdout或冻结final test。正式primary endpoint仍pending。

非阻断编辑建议（不改变任何结果）：完整回复第302行在新增A20段后回到C的IID Sp-DFA；正文稿第197行在A20段后回到C的五IID seed-first summary。原文后句/链接已明确C，故没有错误数值或范围结论；但快速阅读可能暂时错认比较对象。后续作者编辑时，可将前者首句改为“Returning to C-deletion, for IID Sp-DFA, …”，后者标题改为“C-deletion cross-scene interpretation.”。本审查未修改两稿。

这是独立文字及有限来源核对。没有重新计算统计、读取模型/预测数组、拟合、推理、运行作者检查器、联网、修改canonical/STATE/Git或作科学采用。原链接指向可访问的本机证据；对外投稿仍需作者整合正文、编译定位并转换证据链接。
