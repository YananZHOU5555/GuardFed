# 六场景三视图机制证据：回复与正文候选

**DO_NOT_SUBMIT_BEFORE_FULL_COHORT。** 本目录只提供已接受中期数据的写作增补；最终论文主张、native/shared 主口径和完整队列提交均未代作者决定。未改原 rebuttal、训练/评价源码、状态、账本或 Git；无网络、CNN、训练、test 或新科学指标计算。

交付入口：

- [英文回复候选](candidate_replies.en.md)：AE 新增段、R3.2 末段替换、R3.7 新增段、P2 pending 行及必需披露。
- [中文释义](candidate_replies.zh.md)与[正文插入段](MANUSCRIPT_INSERT.md)。
- [n=10 三视图证据摘录](evidence_excerpt.md)：直接复制已接受 JSON 中的均值、sampleSD 和同 seed 配对差，不重新估计指标；全部精度保存在 [numeric_references.json](numeric_references.json)。9/6 面板直接链接原已接受表。
- [评论与接入定位](comment_alignment.json)：原 24 意见 source_map 的 AE/R3.2/R3.7；P2 明确为内部 pending 项；路径、原文、行号、SHA 均保留。
- [来源映射](source_map.json)、[有限内容验证](verification.json)及 FILES_SHA256.json 封条。

锁定 root 接受的 three_view_interim_20261009T145900Z：ROOT_REVIEW SHA `d69255d2059ff6a7449b036ff1ef34225dcd2546c1a027699ba6bd8b35dea55c`。仅六场景 60 minus_U+60 Full，n=10/9/6 双方同规则。native 七场景表 interim_tables_20261009T144558Z 仅用于核对重叠 native 单元和区分 scope，额外 non-IID F Flip 不纳入三视图。

原 R3.2 的 280 条 tabular 矩阵（260 新+20 历史 Full）、原 R3.7 的 COMPAS 全12条件准确率反例、六条件三指标更优和校准逆序都保留。英文段落直接回答评论，并说明已接受六场景 U 中期比较不等于完整 800 新 controls 或整个机制900。方向限定于 n=10；9/6 子面板不被当作必然保持方向或未曝光确认集。

自动核对只读原 SHA、JSON 行和保存的 checkpoint 三视图记录；数字替换与表格摘录不引入新的拟合、指标、均值或 SD 算法。`build_and_check.py` 为一次性本机生成入口，拒绝覆盖已有交付；封存后请读取既有产物，不在本目录重跑生成。模板中英文数字使用具名 token，token → 原 JSON pointer/完整值可追溯。所有英文表格/回复链接仅指向存在的本机证据，不查询文献或网络。

关键限制保留：valid-only、AEOD 为绝对 TPR gap、91001 择方与 valid 曝光、Full5CPU55GPU 对 controls60CPU、Full59cu1281cu130 对 controls 当前 driver595/cu128，历史 driver 未一致。native/shared120 保存指标和组混淆计数相同不是独立校准增益证据，也不是重新核验 prediction vector 相等。没有显著性、必胜、所有机制必要或统一设备 final 比较的宣称。
