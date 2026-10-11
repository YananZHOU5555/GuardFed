# 24点回复实质提交审查（限定历史截止点）

结论：**未发现新的数值错误、softmax 推导错误或实验结论方向错误；发现1处需要收窄的统计口径表述。** 当前稿仍是已明确标注未完成的作者审阅稿。既有待办不重复列为新错误。

审阅对象：`docs/server_deployment_20260923/revision_20260923/rebuttal_working_F80_FL90_Hybrid20_clarified_20261011.md`，SHA256 `ad6f962dc12b7e584cb1bcb75936adcd3efdd09106e3310b2ca336aa2bae7b28`。按 F80 / FL90 / Hybrid20 固定截止点理解；后来的 NGA 接受结果及覆盖扩展不追溯判为本稿错误。只读原文、现有紧凑证据和相关源码；未运行原始大验收、重新拟合或统计，未 SSH/F/Git/STATE 操作。

## 新发现：统计范围表述过宽（P2）

**位置：当前稿第447行。** “All repeated-model tables use sample SD (ddof=1), matched seed sets and one terminal checkpoint for all metrics.”

“All” 与稿件保留的历史重建范围不一致。历史 Adult Table II 的180个 n=10 指标格、6个 n=3 指标格来自末十轮选轮，而非均取终轮；十种子历史协议还按方法改变选择方向。当前稿第157行明确同时保留历史选轮重建和统一round70修正候选，第167–171行也已区分新增表与历史统计。因此问题是末尾总括句重新抹平了此前正确保留的区别，不是新增表计算错误。

证据：`docs/server_deployment_20260923/revision_20260923/table2_trace_20261002/追溯报告.md` §3、§5；同目录 `complete_reconstruction_verification.json`；`outputs/guardfed_tables/table2_recovered_20261002/table2_historical_reconstruction.md`。

最小建议（未修改现稿）：将首句限定为 “The new repeated-training tables and the round-70 correction candidate use one terminal checkpoint for all metrics; sample SD (ddof=1) is reported where repeated seeds are available. Historical reconstructions retain their explicitly stated selection rules and actual repeat counts.” 配对种子规则只用于适用的新增配对比较。该修订无需重跑实验。

## 关键核查通过

| 核查 | 实质判断及证据 |
|---|---|
| 原话与覆盖 | 原附件 SHA 与 `rebuttal_20261009/comment_source_map.json` 一致；24个引用块逐字对应唯一原评论，38条引用行，顺序 AE→R1→R2→R3。未发现漏掉某条评论。 |
| 条件 softmax | 第45行的恶意质量上界 `m/(h exp(μ/τ)+m)` 推导成立；明确实际保留集、h/m边界、分离假设、范数条件及不保证整个算法。未把空恶意集算作非空分离成功。`score_analysis/coverage_summary.json` 与 README 保留完整/稀疏日志区别；伴随插入稿§4同样限定命题。 |
| AEOD 与公平范围 | 第227行和伴随插入稿明确为绝对TPR差（equal opportunity），不是完整equalized odds。已检查本地冻结 `tmp/fl_SpDFA10_final100_source_prepare_20261011/originals/core.py:155` 的 TPR/ASPD 实现；多群体及交叉群体只作为未验证扩展。 |
| 信息预算/攻击知识 | 根数据、训练总体重加权信息、合成器总体support/marginals、推理时敏感组、已知恶意数量及攻击侧root-reference均披露；未宣称各算法信息预算相同。既有 `method_information_access_paper_appendix_20261011/APPENDIX.md` 进一步区分平台与聚合器输入、局部Adam与同点梯度。 |
| COMPAS 反例 | 对现有 `calibration_final_acceptance/formal_paired_summary.csv` 的已保存均值作方向核对：12/12删除条件ACC高于Full，6/12三指标同向改善；IID Full与删A数值吻合。non-IID删F的raw AEOD 0.239597→0.250984、calibrated 0.049143→0.044831吻合。未以校准掩盖反例或虚构因果。 |
| 不同群体收益 | 同CSV的Full/non-IID group0 TPR 0.417638→0.575338、FPR 0.161773→0.297088，与稿中反例吻合；未把较小差异直接解释为所有群体受益。Adult缺失Full checkpoint的边界明确。 |
| Table II及不确定性 | 完整重建回执支持480格、44隐藏值、435匹配与1差异；n=1/10/3分别294/180/6。当前具体回复正确说明单种子SD不可用、不能用轮间变化补足，不把混合checkpoint三元组当同模型。未找到新数值错误。 |
| 图像新增表结论 | 有限核对FL90 non-IID FedSA/S-DFA、Hybrid20 F Flip及F80 non-IID FedSA保存展示值和符号，均与当前回复吻合。固定10/9/6子集、校准视图非独立、混合环境和负结果均保留。未重新计算表统计。 |
| 验证与最终测试 | 当前稿开头、R3.10及P3明确验证开发、已暴露官方test、未选primary endpoint；未将去掉一个选择seed或六seed子集包装为未触碰确认。 |
| 适配忠实性 | 明确FairFed/FairGuard/组合/FedAA/LASA为适配；FLGMM保留作者代码与论文描述差别；Hybrid是自定义控制；LoGoFair虚拟cohort及DP含义、Huber投影/理论边界均说明。缺少FedWA/SmartFL/FedDNA定义没有用简化分支替代完成。 |

以上通过是针对本次有限核对范围，不能替代完整数据重验或最终稿逐表验收；文献链接本轮未重新联网核验。

## 作者决策与已诚实披露的未完成项（不是新错误）

- **待作者决定**：最终primary prediction/evaluation endpoint及主张范围；统一round70候选是否替换历史表、无法认证的方法名称如何删除或限定、synthetic/Fig.3保留或撤回哪些主张。这些决策不能用更多已暴露验证结果自动代替。
- **确实尚未完成回应闭环，但稿件已经说明**：R1.4的历史生成器/图来源，R2的最终表替换及真实重复数呈现，R3.5/6/11与匹配投稿源的实际整合，R3.8的完整方法覆盖/忠实定义，R3.9最终可用复现发布。P1–P6已有明确对应；本次不将它们重复包装成新审查发现。
- **本轮不要求新增实验**：评论已有直接回应或诚实边界。现有条件理论、COMPAS反例、校准归因及统计修复可继续用于修订；不能据此声称全返修已完成。

未改现稿或任何canonical文件；唯一新增产物为本审查文件。
