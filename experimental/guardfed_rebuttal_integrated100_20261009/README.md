# 完整24意见英文作者审阅稿：U100固定快照

当前可连续阅读的两个交付文件：

- `rebuttal_integrated_20261009.md`：完整AE、R1、R2、R3共24块原评论及对应英文回复。
- `manuscript_insertions_integrated_20261009.md`：完整正文插入候选副本，保留方法、理论、审计、负结果、限制和集成清单。

**DO_NOT_SUBMIT_BEFORE_FULL_COHORT。** 本任务仅更新作者审阅副本，没有修改submitted manuscript、STATE、Git或旧封条。U100三视图十场景和native100已由根独立接受；该事实不等于其他七机制、剩余八方法、正式主终点或冻结final test完成。正文原source project仍需取得；发现的历史paper.md不视作submitted-version source。

基底为sealed `guardfed_rebuttal_integrated71_v2_20261009`，并逐字核其canonical副本及原comment_source_map。新增依据是根接受的U100三视图/独立native100、after92精确8采用凭据、九方法900三视图PDF与2052标量校准解释，以及英文addendum中已验证的terminal Fig.3候选与正文源边界。37个实际文件路径/SHA/字节数在`INPUTS.json`、段落位置和before/after在`SOURCE_MAP.json`；最小文本差异在`UPDATE_DIFF.patch`。

R1.3直接解释逐模型clean-root阈值、原生/共享规则、校准配对变化和不利比较；R3.2/R3.7把U删除和校准控制合并解释。十场景native删除U的ACC均低0.121–1.384pp、ASPD均低、AEOD八高两低；raw AEOD六低四高，新non-IID Sp-DFA raw ASPD反而更高。数字只读已接受统计并格式化，没有重新计算科学均值、SD、阈值或指标。

原280 tabular/260同checkpoint对、COMPAS反例、旧Table II真实n与n=1 SD unavailable、FairFed/FairGuard等adaptation身份、AEOD绝对TPRgap、mixed device/cu128/cu130/driver、seed91001选择和valid/test曝光均保留。10/9/6面板使用相同seed规则；native/shared主口径仍pending，没有显著性、保证胜出或every-term-necessary主张。

检查：24原评论逐字/逐hash一致，37处新增数值JSON pointer与格式一致，37个Markdown链接本地存在/外链仅语法检查，21段必要更新之外206段hash保持；反向撤销全部21个替换可精确恢复两份原v2稿。命令`python -B tmp/guardfed_rebuttal_integrated100_20261009/verify_delivery.py`只读复核。

两次生成前本地检查失败单独保存在PREWRITE_FAILURE_1/2：先误按files mapping读取旧members seal，后误以comment_map字典顺序配对正文顺序；均在稿件写入前拒绝。现读取实际members与封存comment identity顺序，未改原评论/原证据、未放宽数值或身份守卫。修正后检查通过，失败记录保留。

本包无需新PDF、LaTeX、网络、CNN、训练或科学统计；最后采用、正文集成与提交决定归作者。
