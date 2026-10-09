# Reviewer 2 推荐文献：8/8 身份与相关性核查

核查日期：2026-10-09（Australia/Sydney）。八条标识均已找到可信身份和一手支持。下表仅写由实际访问的原文摘要、论文全文或出版记录支持的内容；不是新增基线实验、DP实现或复现认证。完整标题、作者、venue和已确认DOI见 [BibTeX](recommended_references.bib)。四条正式DOI的注册元数据另存 [verified_doi_metadata.json](verified_doi_metadata.json)。

| 审稿信标识 | 已核对身份/主来源 | 支持讨论与本研究关系 | 证据粒度及边界 |
|---|---|---|---|
| arXiv:2012.02447 | Abay等，2020；[作者预印本](https://arxiv.org/abs/2012.02447) | 讨论FL偏差来源、预处理和训练内干预及异质数据；帮助把本工作放入已有联邦偏差缓解研究，而不是把群体公平首次引入FL | 已访问作者摘要。正文可引用方法类别；不据摘要断言具体密码学或DP保证，不冒称本次实现了其三种方法 |
| 10.3233/FAIA240671 | Corbucci等，PUFFLE，ECAI2024；[出版页](https://journals.sagepub.com/doi/abs/10.3233/FAIA240671)、[全文](https://journals.sagepub.com/doi/pdf/10.3233/FAIA240671) | 参数化探索隐私、效用、公平的折中。用于说明加入隐私约束后问题更复杂；与GuardFed的恶意更新防护目标不同 | 出版页及9页论文已访问；本稿不复制其最大改善率作为GuardFed证据。隐私保证属于该论文设置，不转移给GuardFed |
| arXiv:2503.15163 | Rychener/Kuhn/Hu，AISTATS2025；[正式PMLR记录](https://proceedings.mlr.press/v258/rychener25a.html)、[作者预印本](https://arxiv.org/abs/2503.15163) | 对不可按客户端自然分离的全局MMD公平正则做函数跟踪，并分析收敛及DP情形。用于区分全局公平优化与GuardFed的root风险评分 | 已核对正式venue、卷258、页865–873及摘要；不声称该工作已有DFA鲁棒性或本稿继承其收敛证明 |
| IEEE document9378043 | Zhang/Kou/Wang，FairFL，IEEE BigData2020；[IEEE页](https://ieeexplore.ieee.org/document/9378043/)、[作者机构记录](https://experts.illinois.edu/en/publications/fairfl-a-fair-federated-learning-approach-to-reducing-demographic/) | 多智能体强化学习与安全信息聚合用于公平/准确率权衡，说明本研究前已有联合目标优化 | IEEE页有机器人验证；作者机构原始研究记录和DOI注册共同确认作者/标题/页码/摘要。**FairFL不是FairFed**；未读完整IEEE算法细节，不能据此写忠实复现规格 |
| arXiv:2108.08435 | Cui等，FCFL，NeurIPS2021；[正式论文页](https://proceedings.neurips.cc/paper_files/paper/2021/hash/db8e1af0cb3aca1ae2d0018624204529-Abstract.html)、[作者HTML全文](https://arxiv.org/html/2108.08435v3) | 将客户端预测损失与本地公平约束写作多目标优化，分析Pareto/性能一致性。用于区分客户端一致性与群体公平，也避免把加权多目标思想说成新发明 | 摘要及全文已访问。其受约束优化的理论不证明本稿候选选择/硬筛选鲁棒性 |
| arXiv:2109.08604 | Rodríguez-Gálvez等，FPFL，PriML/NeurIPS2021；[作者原文](https://arxiv.org/html/2109.08604v2)、[作者机构页](https://machinelearning.apple.com/research/enforcing-fairness) | 在private FL下用修改的微分乘子法处理群体公平约束，展示DP噪声可能使欠代表组更不利 | 作者全文和机构摘要已访问；v2为2022修订，不误写成NeurIPS主会论文。GuardFed没有DP预算或同等保证 |
| AIES article36730 | Taik/Chehbouni/Farnadi，AIES2025；[正式页](https://ojs.aaai.org/index.php/AIES/article/view/36730) | 从生命周期、利益相关者和具体伤害审视过窄系统级公平定义，支持限制“差距更小=所有人受益”的表述 | 正式页摘要/卷期/页码/DOI已访问；它是分析与框架论文，不应写成又一个实测攻击聚合器 |
| 10.1145/3715275.3732152 | Corbucci/Heilmann/Cerrato，FAccT2025；[ACM原文](https://doi.org/10.1145/3715275.3732152)、[会议论文](https://facctconference.org/static/docs/facct2025-206archivalpdfs/facct2025-final1129-acmpaginated.pdf) | 分析客户端公平需求不一致时参与联邦的实际收益，并比较本地/联邦/聚类方式。直接支持补充客户端或群体层面的收益分析 | ACM原文已访问；PDF后续重复打开超时不取消已取得记录。作者/页2232–2248由DOI注册确认。不将其结论外推成每个公平干预都会伤害某群体 |

所有八条对相关工作或局限均有实质用途，当前无需以“无关”拒绝推荐。它们不全是恶意客户端防御算法，也不是审稿人要求逐一额外实现的八条baseline。是否扩展实测baseline应由威胁模型、接口与资源决定，不能仅因被推荐就把未实现方法填入结果表。

## 可直接插入相关工作的英文候选

Federated group-fairness research predates our threat setting. Abay et al. study bias mitigation through preprocessing and training interventions, whereas FairFL uses reinforcement learning and secure information aggregation to balance fairness and prediction performance. FCFL formulates client-wise performance and fairness constraints as a multi-objective problem, and function tracking addresses global MMD-based fairness regularization that is not naturally separable across clients. These objectives should be distinguished from assessing potentially poisoned updates against a trusted reference. \cite{abay2020mitigating,zhang2020fairfl,cui2021addressing,rychener2025global}

Privacy is an additional requirement rather than a consequence of reporting smaller disparity. PUFFLE studies privacy–utility–fairness trade-offs, and FPFL enforces fairness constraints in private federated learning. GuardFed does not provide a differential-privacy guarantee; its trusted-root and population-statistic assumptions must be stated independently. \cite{corbucci2024puffle,rodriguez2021fpfl}

Aggregate group disparities also leave important benefit questions unresolved. Taik et al. advocate connecting fairness definitions with concrete harms and stakeholder needs; Corbucci et al. study client-level benefits when fairness objectives differ. Accordingly, lower AEOD/ASPD is not interpreted here as an improvement in every group's error rates or every client's outcome. We report utility jointly with disparity, retain degenerate predictions, and use group-specific TPR/FPR where available. \cite{taik2025fairness,corbucci2025benefits}

## 引用精度

- BibTeX中的arXiv DOI是仓库DOI，不伪装为正式会议DOI。Rychener及Cui优先用正式venue记录，arXiv标识用于对应审稿链接。
- PUFFLE的SAGE出版页日期与Crossref登记相差一天，不影响2024年份；未据此杜撰卷/页。本条目保留已确认会议/系列/出版者/DOI。
- 不凭引用增加实验主张，也不声称完整相关工作已自动合入原稿；正文整体cite-key碰撞、参考编号和编译由主稿集成检查完成。
