# 三个剩余基线：新来源恢复增量（2026-10-10）

本轮新增恢复 **FedWA 的 AAMAS 2022 作者海报**，但海报仍未给出可执行的 DRL 规格；FedWA、SmartFL、FedDNA 均不能据此标为忠实实现完成。新增忠实组件、训练和评价均为 **0**。这是本轮检索结果，不是作者没有实现或全文永久不可得的证明。

只用了10次新请求：5个定向搜索、3次web打开/点击（含一次PDF工具读取失败）、2次普通公开GET。没有重跑旧51+31响应或42作者路径；搜索结果中再次出现的旧论文/预览仅保存在搜索回执，没有再打开。请求逐项见 [REQUESTS.json](REQUESTS.json)，旧回执提取的109条确切URL见 [PRIOR_PATHS_EXCLUDED.json](PRIOR_PATHS_EXCLUDED.json)。

## 本次真正新增的材料

从新发现的 [AAMAS 2022报告入口](https://underline.io/lecture/49565-adaptive-aggregation-weight-assignment-for-federated-learning-a-deep-reinforcement-learning-approach) 获取原HTML。网页正文没有展示海报下载，但其 `__NEXT_DATA__.props.pageProps.fallback` 的 `thin_lectures/49565` 对象明确绑定本题名、原AAMAS论文和 `posterDocument`，文件名 `744_poster.pdf`、声明大小269457字节。[官方海报PDF](https://assets.underline.io/lecture/49565/poster_document/ab82b43fd5d98fdc68d8b39eb0b6269d.pdf) 经普通公开GET成功下载，实际大小相同，一页，SHA256：

`d337f8c9c0199ed404e382b816036d90fd2bc1a9be09db1746b0b8b366cc97f0`

本地 [PDF](FEDWA_AAMAS2022_OFFICIAL_POSTER.pdf)、[提取文本](FEDWA_AAMAS2022_OFFICIAL_POSTER.txt)、[页面渲染](FEDWA_POSTER_PAGE1.png) 与 [网页目标记录](FEDWA_UNDERLINE_NEXT_DATA.json) 均保存。文本SHA `870b5e2b29f97274cdb3874e9bfd66031465c25980394249a661d7fbed58c6f1`。web工具无法直接打开此PDF，随后普通GET返回真实 `%PDF-`，没有登录或绕过挑战。原网页字节和两个HTTP回执保留。

海报中央给加权全局目标及全局/本地参数概念，右下流程图只标示 “RL Agent determines weight”。人工查看整页和流程图后，没有发现具体状态、奖励、RL更新式或默认配置。页面嵌入的本讲字幕结果为0，playlist/slideshow为空；这只界定该公开页面，不能推断其他载体不存在。已有三页原论文SHA仍为 `3766b6d06c2568370e7be52c7f8ab4ae9417a373b7e66afb462296e0e1e29e48`。

## 不重复的旧失败路线

| 方法 | 已试且本轮不重跑的路线 | 不能解决的原因 |
|---|---|---|
| FedWA | Semantic Scholar/arXiv/GitHub精确题名；OpenAlex别名DOI；SCUT作者页 | 无算法完整附件；别名回到原摘要；同名作者混入无关领域 |
| SmartFL | Dong/Gao机构与固定主页源码；UWA/NJUST题录；ORCID、Semantic Scholar、arXiv；Elsevier XML/FULL/PDF | 作者条目无全文/代码链接；XML仅题录、FULL401、PDF入口为challenge HTML；同名ICLR方法不对应 |
| FedDNA | 作者Elsevier分享链、机构/ORCID/RShare；Semantic Scholar/arXiv/GitHub/Zenodo/DataCite；已核作者两个仓库与42路径 | 分享链曾ATP-3错误，不能说永久失效；仓库是压缩/异步或Hyperledger/FedAvg，无目标指纹算法；同名其他论文排除 |
| 共用 | 三条Unpaywall请求、本地5份候选PDF检查 | 422缺真实邮箱，未伪造联系身份；旧本地候选未命中，不再扩大私人文件扫描 |

完整既有证据仍在三份输入报告及其回执中。本轮只固定输入SHA，不重新审计无关作者代码。

## 仍必须补齐的最小规格

| 方法 | 具体阻塞项 | 最小可继续输入 |
|---|---|---|
| FedWA | state维度/编码/缩放/客户端次序；action到权重的约束和参数/差分语义；reward公式、数据权限与时序；RL算法、网络、优化器/学习率、探索/回放/target/初始化和更新频率 | 对应该论文的完整MDP/策略实现与默认配置，或算法完整版/附录。原三页摘要与新海报均不足 |
| SmartFL | pairwise divergence与删除顺序/并列/停止/f依赖；pseudo-global公式及模型/差分语义；召回和90度/零范数边界；最终权重/norm/空集与Appendix C默认值 | DOI `10.1016/j.inffus.2025.103555` 完整算法章节及Appendix C，或明确对应的原聚合源码/配置 |
| FedDNA | probe来源/权限/规模/预处理/固定性；指纹层、跨probe归约/归一/距离；历史初始化/身份/更新/衰减/缺席和一致性式；MAD中心/倍数/零值/等号；权重/空集与模型默认值 | DOI `10.1016/j.jisa.2025.104358` Sections3/4及附录，或对应Flower strategy/probe/model/default代码 |

这些缺项改变算法输出和可访问数据，不能自行选常见DDPG、cosine多数、参数距离MAD或任意默认值而仍声称原法。没有为加权目标这种已知片段另写“忠实组件”，因为它不能补齐决策机制。论文给定的概念与未定义的实现、图像模型/校准权限适配待决，在 [DECISION.json](DECISION.json) 分开列出。

有上述材料后，最小恢复路线是逐式固定输入输出与默认值，先做组件公式oracle，再讨论共同图像管线的明确适配。当前没有解锁GPU/训练、正式freeze或论文基线完成行。没有联系作者、SSH、改共享文件/状态或Git。
