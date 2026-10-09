# 2026-10-09 rebuttal writing package

已完成：24条行动性意见的原话/完整英文回应、中文证据对齐、可直接插入的正文候选、8/8推荐文献一手身份和相关性核查及BibTeX。当前状态为**有真实证据支持的工作稿**，不是已提交/可直接宣称所有实验完成的版本。

- [完整英文逐条回复](rebuttal_20261009.md)：AE1、R1四条、R2八主题、R3十一条；原编号完整保留。
- [中文证据对齐表](审稿意见证据对齐表.md)：每条已有证据、边界、下一交付。
- [英文正文插入候选](manuscript_insertions_20261009.md)：DFA定位、真实AD2+、root/合成资源、条件定理、符号、模型、结果统计和局限。
- [推荐文献核查](recommended_literature_audit.md)、[BibTeX](recommended_references.bib)、[DOI元数据](verified_doi_metadata.json)。
- [审稿原话附件](reviewer_comments_verbatim.md)、[逐块源映射](comment_source_map.json)、[交付核验](verification.json)。
- [Adult score独立验收报告](../../training_20260923/score_analysis/restoration_and_score_report_20261009.md)：已安全恢复原4文件、140原始文件/8440轮/14条件全量重算；20历史Full与原JSONL行精确接回。门禁/最终集合、空恶意类和条件上界分别报告。
- [CelebA实际分区/root审计](../../training_20260923/celeba_partition_audit_20261009/README.md)：20原结果、400client总数精确匹配；按冻结RNG恢复敏感组数量，40个原Male计数额外核验。原审计只恢复敏感边际；后续联合扩展见E13，不改写原封存审计的范围。
- [E13 CelebA训练联合分区审计](../../training_20260923/celeba_joint_partition_20261009/README.md)：20分区的1600个client四格数及80个root四格数已重建并独立复核；400原总数/800敏感边际、train/valid/root ID hash精确匹配，无空端/缺组/缺标签/缺联合格，最小client四格IID1161、non-IID278，root2666。真实Dirichlet按Male组而非Smiling标签分配；新client-ID/联合hash只是重建收据，不能冒充原运行日志。未读取评价标签/图像或进行推理训练。
- [Adult/COMPAS强α实际分区审计](../../training_20260923/tabular_partition_audit_20261009/README.md)：60分区、1200client总数、2400敏感边际、240翻转计数全部匹配；原tabular标签四格恢复并做一致性核验。α=.1的空端/单组及1.3%–66.7%恶意样本覆盖完整披露。
- [旧synthetic840数值与来源审计](../../training_20260923/synthetic_lineage_audit_20261009/README.md)：840条原记录及同轮/终轮导出全追溯；旧逐列最优797条无共同checkpoint，已排除同模型解释。210种子内记录/70三seed设置汇总可用。
- [Fig.3/PCA另一suite恢复审计](../../training_20260923/synthetic_figure_recovery_20261009/README.md)：260原记录/PCA40条/26设置汇总全追溯；250/260旧triplet及PCA40/40无同轮，n10是seed123的十场景而非十seed。Fig.3强兼容全十场景+1−半和，正文S-DFA+1−全和及TVAE1+9概括存在具体冲突；仍缺原绘图脚本/输入与FD执行身份，不把兼容关系当来源SHA。归档PCA为完整协方差Gaussian控制，无显式PCA分解/降维。

已将旧草稿中七方法/缺十方法1000格/FedAA-LASA只有pilot的过时文字改为**九方法900/余八方法800格、FedAA-LASA200已完成**。七方法700仅在共享校准控制范围内保留。统一说明valid选择历史、14cu130+886cu128、旧test暴露、true-n和校准贡献；没有虚写未跑结论。

仍需：P1剩八基线、P2图像机制、P3最终冻结评价、P4原Fig.3脚本/输入及FD执行和per-run生成器/模型/cache/旧checkpoint身份、P5主稿集成与位置、P6最终release/作者主张。PCA另一suite原记录的数值子项已完成，不再列为未找回；Fig.3最终修正范围、公式、场景和同轮/真n展示待作者选择，未声称已重绘或获最终批准。Adult严格score/mass、两类实际分区（含E13 CelebA clean client联合计数扩展）和两套synthetic数值链已完成。116轮gate外再纳入及负margin不能包装成全AD2+保证；强α空端/单组、797/840及250/260旧跨checkpoint数据和合成负例完整披露。旧TableII数值追溯已完成，但单seed不能产生真实SD，方法身份和历史选轮问题仍需正式处理。

机制执行状态只在[中文内部边界](审稿意见证据对齐表.md)记为截至交接“20项三轮预检计划、6项完成”；没有机制800正式结果，不把管线预检或服务器恢复身份当科学结论。英文回复不包含动态服务器状态。

原审稿信与原稿txt、旧回复、总览、900表及统计审计、TableII追溯、shared控制、source snapshots、frozen protocols及封存的260/PCA和E13联合分区包是来源。详见verification.json中的路径和SHA。原投稿及既有记录未覆盖；本次写作任务未启动训练或发表外部消息，E13为单独验收的只读CPU数据重建。Markdown内容/引用/数值和原话已核验；本次未生成排版PDF，不声称完成主稿编译/页码检查。
