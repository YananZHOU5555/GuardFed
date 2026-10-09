已定位可编辑IEEE主稿候选：[仓库paper.md](E:/OneDrive/文档/GuardFed/paper.md)。它虽然扩展名是.md，内容却是完整`\documentclass[lettersize,journal]{IEEEtran}` LaTeX正文，含abstract、主文、附加理论节和bibliography入口。此前仅按.tex/.bib的负证据不覆盖此文件；补查项目文档后已纠正。**它是历史主稿候选，尚不能认证为给定提交PDF的同版源；完整可构建项目仍缺资产。**

`paper.md`实际字节SHA `2676fa0f6e09d4a1a11714251cfa456572faaf11e5fcf7d2dc9c630621349073`，79,618字节；两个worktree副本均为78,614字节、SHA `b56825949dbff0ab2e2425093c8a455a455bfb8bc5424f972f049049621b3c8a`。差别仅CRLF/LF，规范化为LF后的正文与三处HEAD blob均相同，Git blob SHA1 `0fb73ce00732f722a8b746441a6a0148738a54fa`。未改动这些原文件。

| 源码入口或依赖 | 实际判定 |
| --- | --- |
| 主入口 | `paper.md`，IEEEtran journal；标题在第26行，S2–S7为Related Work/Problem Setup/DFA/System Design/Experiments/Conclusion。 |
| 宏和包 | theorem/lemma/definition三个newtheorem；未声明自定义newcommand或外部input/include。主要包含amsmath/amsthm、graphicx、cite、multirow、subcaption、booktabs、algorithm2e。 |
| Bib | 第1000–1001行要求IEEEtran样式和`paper.bib`；没有内嵌thebibliography。**paper.bib未找到。** |
| 图 | 第209行要求`archi.png`，第488行要求`inv.pdf`；**两者均未找到。** |
| 查找边界 | 上述三资产在repo及两个worktree根、给定PDF同级目录均不存在；仓库含worktree的精确文件名扫描（含hidden/ignored、排除.git）也无匹配。未搜索其他用户盘。 |
| 构建 | MiKTeX的latexmk/pdflatex/xelatex/lualatex/biber/bibtex已在PATH；仅检查路径。现有表格/导师报告生成器不是该主稿的确认build入口；未编译，不能宣称完整项目可构建。 |

给定正式PDF：[IEEE_TDSC__二次版___GuardFed____.pdf](E:/Edge下载/IEEE_TDSC__二次版___GuardFed____.pdf)，12页，SHA `549a4191b1ac560bdaba79d9dce3b11693ab69c4a8cc0f7f52be31afa1110071`。其题名为“To Kill Two Birds with One Stone: Defending Both Utility and Fairness in Federated Learning Systems”，与历史源标题“GuardFed: A Trustworthy Federated Learning Framework Against Dual-Facet Attacks”明显不同。PDF八个主节为Introduction、Preliminaries and Research Gap、Dual-Facet Attack、Mini-Benchmark and Empirical Insights、GuardFed System Design、Experiments、Root Data: Vulnerability and Opportunity、Conclusion；历史源独立保留Problem Setup，且没有PDF新增的Mini-Benchmark和Root Data主节。历史源第509行起为“Convergence and Fidelity Analysis under Hard-Threshold Aggregation”，PDF第8页却有“Theoretical Analysis of Soft Aggregation”。因此不能把两者当同版，仅据paper.md生成稿也不能冒充给定PDF的可编辑原项目。

有界定位证据：当前repo HEAD `64d86504531b671112540a011aabab484669b8d2`含102个跟踪文件，0份.tex/.bib但含上述paper.md；revision-publish HEAD `d9130d3987ce38ed5a7a3a2083221b8f1bee95da`含52份.tex、1份.bib，其中8份是独立报告/表格包装、44份是表格片段；publish-5090另有5份补充报告TeX。`E:/Edge下载`仅按GuardFed/IEEE/TDSC/Overleaf命名筛选同级文件，只有给定PDF，未匹配源码包或同名项目目录。

既有项目与返修文档未发现Overleaf/ShareLaTeX地址。明确Git地址是`https://github.com/YananZHOU5555/GuardFed`，返修稿引用其固定commit `51629c2df5e56259a0d750744b7dc9299349efd0`；三worktree本地origin也指向该实验/返修证据仓库。没有找到另一个被明确标注为论文源码的远端项目；未访问这些URL或搜索远端历史。

`877cab.tex`与主稿必须分开：[windows恢复副本](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/training_20260923/synthetic_figure_recovery_20261009/additional_sources/windows__877cab5288192c1c.tex)的SHA `877cab5288192c1c54e920c0f0a194c3f11d6803c3922308e06c72a80ac9deef`，标题明确为“GuardFed-AD2+ 实验补充报告”，article类；与当前repo同名实验补充报告逐字节一致，不能代替IEEE主稿。唯一跟踪的[recommended_references.bib](E:/OneDrive/文档/GuardFed/docs/server_deployment_20260923/revision_20260923/rebuttal_20261009/recommended_references.bib)是审稿推荐文献核查材料，不能猜作缺失paper.bib。[三视图表片段](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_three_view_20261009/native/celeba_iid_noniid_ten_seed.tex)也只是table*环境，依赖multirow/graphicx，并非主论文。

交root的实际P5入口是上述**历史paper.md候选**、[v2回复/正文插入候选](E:/OneDrive/文档/GuardFed/tmp/guardfed_rebuttal_integrated71_v2_20261009/README.md)及[已接受900表](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_nine_method_three_view_20261009/README.md)。正式同版工程、paper.bib和两图尚缺；可支持的后续修订必须保留该版本边界，不能用假图、猜Bib或补充报告冒充。此结论不意味着用户在其他位置或Overleaf上没有源码。

本轮仅写本REPORT和[EVIDENCE.json](E:/OneDrive/文档/GuardFed/tmp/guardfed_manuscript_source_locator_20261009/EVIDENCE.json)，后者SHA `bc9914c19063dbde9e6e81bfa2f1af8c30029b5e587dbdf73ddb91b3c78dd6d4`，含全部实际来源pins、章节差异和有界负证据。未编辑/编译原稿、未下载或SSH、未修改Git/canonical，未重做631源Fig.3审计或建立新框架；交付后停止。
