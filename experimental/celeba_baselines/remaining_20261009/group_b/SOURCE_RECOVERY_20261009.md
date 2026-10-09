# 三项基线的原始资料恢复记录（2026-10-09）

结论：本轮有界恢复没有取得足够的新全文/代码，FedWA、SmartFL、FedDNA 仍不能标为忠实实现完成。FLGMM 已通过的组件与合成图像管线验收继续有效。本轮没有修改 FLGMM adapter、worker、冻结 core、搜索草案或旧结果，没有启动任何服务器/GPU任务，也没有联系作者。

本轮从已有 `sources/download_receipt.json`、`repository_search_receipt.json` 和 OpenAlex 原始记录继续查证。新增原始响应及 URL、UTC 时间、HTTP 状态、类型、大小、SHA256 保存在 `sources/recovery_20261009/`；各 `*_receipts.json` 和 `*_receipt.json` 是逐次记录。`recovery_summary.json` 合并这些记录。HTTP 200、论文题录、出版社预览和完整论文是不同证据层级，本轮没有把它们合并计数。

## FedWA：现有三页全文可读，但算法没有给全

精确身份：Enwei Guo, Xiumin Wang, Weiwei Wu，**Adaptive Aggregation Weight Assignment for Federated Learning: A Deep Reinforcement Learning Approach: Extended Abstract**，AAMAS 2022，pp. 1610–1612，DOI [10.5555/3535850.3536051](https://doi.org/10.5555/3535850.3536051)。已有 [IFAAMAS 三页原始 PDF](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf) 的文本和下载哈希继续复用，本轮重新核对第 2 节公式 (1)–(3) 与第 3 节框架。

这份完整 extended abstract 给出局部/全局目标、聚合权重和 DRL 框架，但没有以下可执行细节：

- 状态向量如何由全局/局部参数编码、维数和缩放。
- 奖励公式、奖励所需的评估数据、数据访问权限与评估时点。
- 使用的策略/值函数算法、actor/critic 架构、动作如何映射为非负且和为 1 的权重。
- 优化器、学习率、探索噪声、回放/目标网络与更新日程。

新增路线：Semantic Scholar DOI 查询成功，但 `openAccessPdf.url` 为空；arXiv 精确标题查询返回 0 项；GitHub 精确标题仓库搜索返回 0 项。OpenAlex 标题搜索出现同名、同作者的别名 DOI [10.65109/xyks7287](https://doi.org/10.65109/xyks7287)，正常解析重定向到原 ACM DOI 页面，未发现扩展版全文。OpenAlex 的作者作品列表混入化学论文，不能把同名作者作品当成本方法的后续版本。SCUT 作者入口正常请求只返回很短的页面，未取得算法资料。

可供后续恢复的具体材料：**上述论文的算法完整版本/附录，或作者 FedWA 源码与默认配置**。只补同一份三页 PDF 不能补齐这些缺项。不能用 FedAA 的 DDPG 或项目旧的 cosine 权重启发式代替 FedWA。

## SmartFL：已找到作者官方入口，仍只有题录/预览

精确身份：Qihao Dong, Yansong Gao, Chunyi Zhou, Shengyuan Yang, Boyu Kuang, Anmin Fu，**SmartFL: Simple majority rule based Byzantine-robust federated learning**，Information Fusion 126 Part A，103555，2026，DOI [10.1016/j.inffus.2025.103555](https://doi.org/10.1016/j.inffus.2025.103555)。

新增且实际核查的 primary 路线：

- [第一作者高校主页](https://person.zufe.edu.cn/dongqihao/zh_CN/index.htm) 明确链接 [作者个人主页](https://cobbd.github.io/)。检查作者公开仓库列表、主页源码树及 [SmartFL 论文条目](https://github.com/CobbD/CobbD.github.io/blob/ec919ea0a0c522900c3d54150b3517964d6b3aad/_publications/2026-01-15-smartfl.md)：条目只有题录，没有 `paperurl`、PDF 或源码链接。相关源码树固定到 commit `ec919ea0a0c522900c3d54150b3517964d6b3aad`。
- [共同作者 Yansong Gao 的论文目录](https://garrisongys.github.io/garrison/publications/)、公开仓库列表与主页源码树均核查；树固定到 `b97b6bfbd41c56c10e85a39291a94f02ed75fb74`，未定位本篇全文。早先对 `GarrisonYS/garrison` 的请求拼写少了一个 `g`，404 已保留，随后使用正确账号 `garrisongys` 核查成功，不把该 404 当成不存在的证据。
- [UWA 机构记录](https://research-repository.uwa.edu.au/en/publications/smartfl-simple-majority-rule-based-byzantine-robust-federated-lea/) 是摘要与题录；实际 HTML 没有本篇附件 PDF 链接。
- Semantic Scholar DOI 查询成功但无公开 PDF；arXiv 标题精确查询返回 0 项。ORCID 对应记录只指向 DOI。
- Crossref 给出 Elsevier XML 检索链接。普通 XML 请求虽然 HTTP 200，但只有 `coredata`，没有正文、公式或 algorithm；`view=FULL` 返回 401。普通 PDF 入口虽然 HTTP 200，实际为浏览器 challenge/download preparation HTML，不是 `%PDF`，未计作全文，也没有尝试规避 challenge。

出版社公开预览仅足以确认两阶段方向：先利用成对归一化模型形成参照，再取与该参照正相关的贡献。无法据此确定：成对删除对象和次序、停止条件与恶意数要求、完整模型/更新量语义、参照向量公式、最终权重与范数处理、90° 等号/零向量边界、完整模型定义及 Appendix C 默认超参数。

可供后续恢复的具体材料：**该 DOI 的完整正文及 Appendix C/补充材料，或作者对应源码**。同名的“server-side aggregation via subspace training”SmartFL 是另一篇方法，不能替代本篇。

## FedDNA：作者分享链接已找出，正常访问暂时错误

精确身份：Aditya Garg, Naman Bansal, Sumit Yadav, Nisha Kandhoul, Sanjay K. Dhurandher, Isaac Woungang，**FedDNA: Behavioural based approach for byzantine defense in federated learning via model fingerprinting and adaptive thresholding**，Journal of Information Security and Applications 97，104358，2026，DOI [10.1016/j.jisa.2025.104358](https://doi.org/10.1016/j.jisa.2025.104358)。

新增且实际核查的 primary 路线：

- 第一作者[公开发表说明](https://in.linkedin.com/in/aditya-garg-60186a226)在搜索索引中给出两条短链接：`https://lnkd.in/gbnvR83J` 与 `https://lnkd.in/gvQtK_sZ`。正常访问短链接页面，分别解析为作者官方 share link [authors.elsevier.com/c/1mNTC7tT2C~7Mb](https://authors.elsevier.com/c/1mNTC7tT2C~7Mb) 与本篇 DOI。share link 本轮访问重定向到 `https://www.sciencedirect.com/user/error/ATP-3?pii=S2214212625003941`，HTTP 400、页面为 `Client Request Error`，未取得论文。**不把暂时错误说成过期、永久失效或全文永不可得**。直接 LinkedIn 页面返回 999，短链接原始 HTML 与目标保存在 receipts 中。
- [Isaac Woungang 的机构论文目录](https://www.cs.ryerson.ca/iwoungan/publications.html) 包含精确本篇题录，但该条目没有附件/代码链接。第一作者、第二作者和 Nisha Kandhoul 的公开 ORCID person 响应没有额外研究网址；Isaac 的作品记录未提供本篇全文。
- Semantic Scholar DOI 查询无公开 PDF；arXiv `FedDNA` 查询未取得本篇；GitHub 精确“model fingerprinting + federated”仓库搜索无结果。作者作品索引混入不同领域同名作者，未将其当作身份确认。
- RShare 机构搜索入口本轮返回 202 空正文，不能当作搜到了全文，也不能当作完整机构检索的阴性证据。
- Crossref 链接的普通 Elsevier XML 只有题录；FULL 请求 401；PDF 入口为浏览器 challenge HTML，均未计作全文。

公开预览可确认模型内部激活指纹、行为历史与 MAD 阈值。仍缺：probe 来源/样本量/预处理及数据权限、激活层/指纹聚合和归一化、历史初始化/更新与一致性公式、MAD 公式与乘子/零 MAD 边界、最终聚合权重和空集合处理、论文模型与完整默认超参数。

可供后续恢复的具体材料：**该 DOI 的完整 PDF（含 Section 3、Section 4、Appendix）、作者 accepted manuscript，或对应 Flower 策略/模型/probe 代码及默认配置**。动态节点对齐、DNA 序列重建等同名 FedDNA 不是本篇；参数距离 MAD 也不能充作模型激活指纹算法。

## 全文获取边界与后续行动

本地补查只针对 `E:/Edge下载` 中 5 份可能学术、标题不明的 PDF，使用已安装 `pdftotext` 提取前两页并匹配三法名称/标题，结果为 0 命中。个人 Receipt/Invoice/Transcript/Enrolment/application/guide 文件以及标题已知的其他稿件均排除；未保存无关正文。机器回执为 `sources/recovery_20261009/local_pdf_scan.json`。文件名搜索阴性本身没有被当作内容搜索完成。

Unpaywall 的三条 DOI 请求本轮均返回 422（要求真实 contact email）。没有伪造邮箱或使用作者邮箱冒充联系信息；因此不能声称完成了 Unpaywall 位置查询。已保存的 OpenAlex、Semantic Scholar 和作者官方资料继续作为检索线索，不是算法正文。

本轮新增忠实 adapter 数量为 0，新增训练数量为 0。上述两篇完整期刊 PDF与 FedWA 完整算法规格到位后，可先逐条映射 algorithm/公式/默认值，写固定输入输出组件回归，再进入共同图像管线；在材料到位前保留三行“未复现”，不替换为旧简化分支，也不称全部 17 方法已完成。
