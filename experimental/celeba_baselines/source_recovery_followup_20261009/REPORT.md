# FedDNA / SmartFL 原始来源恢复补查（2026-10-09）

**结果：新增核验了一作的 GitHub 身份链和两个联邦学习项目，但仍未取得目标 FedDNA 正文或可对应论文的实现；SmartFL 新增通讯作者机构条目，仍没有算法全文或代码。没有据此制作猜测性的 adapter，也没有训练或评价。**

本包只记录本轮新增路线。此前 51 个响应及作者分享链接 ATP-3 HTTP 400 的历史证据不重复请求、不重新审计。这个错误不能证明链接永久失效。`group_b`、最终评价准备包、旧结果和全局状态均未修改。

## 本轮新增证据

### 1. FedDNA 一作账号恢复成功，目标实现没有恢复

作者 [Aditya Garg 的公开个人资料](https://in.linkedin.com/in/aditya-garg-60186a226) 的搜索索引同时出现精确 FedDNA 论文公告与此前 DRDO/Hyperledger 项目说明。该项目短链 [g4Vp5Va5](https://lnkd.in/g4Vp5Va5) 经普通公开 GET 返回外链确认页，明确指向 [Adityagarg8384/Federated-Learning-Using-Hyperledger-fabric](https://github.com/Adityagarg8384/Federated-Learning-Using-Hyperledger-fabric)。论文公告单独可定位为 [作者公开帖子](https://www.linkedin.com/posts/aditya-garg-60186a226_feddna-behavioural-based-approach-for-byzantine-activity-7413207197771476992-E3ky)。个人资料身份链有公开来源支持，但**作者名下其他代码不会自动成为 FedDNA 官方实现**。

GitHub 公开接口一次返回该账号 22 个公开仓库；本轮沿两个最相关项目核查：

| 项目 | 固定 commit | 实际核查范围 | 判断 |
|---|---|---|---|
| `Adityagarg8384/btp` | `bc763e184471839cf7e55302790851dab7395f55` | 完整、未截断的 715 项路径树；全部 17 个非空 Python 文件；逐文件 Git blob SHA-1 与下载 SHA-256 | 包含压缩更新、Flower 客户端、同步平均及按时间衰减的异步聚合，没有论文标识或目标指纹/MAD机制，不能当 FedDNA |
| `Adityagarg8384/Federated-Learning-Using-Hyperledger-fabric` | `04bc33446cd912ce91593a5806394873bc84ea89` | 完整、未截断的 728 项路径树；README、`flower.py`、`server.py`；逐文件 Git blob SHA-1 与下载 SHA-256 | Hyperledger 集成与普通 FedAvg 管线，所查文件不能提供 FedDNA 算法规格 |

例如 btp 的 [`server3.py`](https://github.com/Adityagarg8384/btp/blob/bc763e184471839cf7e55302790851dab7395f55/server/server3.py) 定义按秒衰减的 staleness 权重及同步 FedAvg 阶段；这是与目标论文不同的聚合逻辑。`main.py` 的 `history` 是训练返回记录，不是已确认的 FedDNA 行为指纹历史。全部 17 个非空 Python 源文件中，目标方法名、指纹、probe 与常见 MAD 函数标识的辅助检索均未命中；同时人工检查了聚合分支。**这些检查仅界定所查 commit，不证明作者没有其他公开或未公开实现。**

没有执行下载的任何代码。仅做静态解析；btp 的 `server/server2.py:112` 现有缩进错误被原样记录，其余所查 Python 可解析。此错误属于作者其他项目，不能当作 FedDNA 实现故障或我们训练故障。

### 2. 仓储精确查询没有提供新全文

下列普通 API GET 已保存原响应和哈希，结果都为空：

- [GitHub repositories: FedDNA](https://api.github.com/search/repositories?q=FedDNA&per_page=50)：`total_count=0`，结果不包含代码搜索或所有可能标题。
- [Zenodo: FedDNA](https://zenodo.org/api/records?q=FedDNA&size=10)：记录数 0。
- [DataCite: FedDNA](https://api.datacite.org/dois?query=FedDNA&page%5Bsize%5D=20)：记录数 0。

这些是查询范围内的负结果，不是永久无全文证明。检索还遇到两个**不同方法**：AAAI 2026 的 DNA 序列重建 FedDNA（Lin 等，DOI `10.1609/aaai.v40i28.39524`）和 ECML 的 normalization-layer aggregation FedDNA（Duan 等）。作者、标题和机制均不同，未用来填补目标方法。

### 3. SmartFL 新机构来源只确认身份

[南京理工大学况博裕（Boyu Kuang）主页](https://teacher.njust.edu.cn/wlkjaq/2025/0711/c14082a10107/page8.htm) 包含精确作者组、题名及 `Information Fusion 126:103555`，可确认目标为 Dong 等的 simple-majority Byzantine-robust SmartFL。保存的 HTML 中没有 PDF、GitHub 或 SmartFL 目标链接。

软件贡献者 Shengyuan Yang 的公开搜索未取得与该论文直接相连的账号。检索命中的 Wisconsin 指针分析主页不能凭同名认定为该共同作者。`toledosakasa/SMARTFL` 是程序错误定位方法，也不是本论文。没有重新读取之前已核过的 CobbD / Yansong Gao 仓库。

## 尚不足以写忠实 adapter 的具体信息

目标 FedDNA 是：Aditya Garg, Naman Bansal, Sumit Yadav, Nisha Kandhoul, Sanjay K. Dhurandher, Isaac Woungang，*FedDNA: Behavioural based approach for byzantine defense in federated learning via model fingerprinting and adaptive thresholding*，JISA 97:104358，DOI [10.1016/j.jisa.2025.104358](https://doi.org/10.1016/j.jisa.2025.104358)。本轮没有获得 Sections 3/4，因此以下仍无可核验精确定义：

| 组件 | 需要的原始定义 | 目前可确认程度 |
|---|---|---|
| Probe | 来源/访问权限、数量、输入构造、预处理与固定或重采样规则 | 未获得 |
| Fingerprint | 取哪层激活、跨样本/神经元如何组合、归一化和相似度 | 仅作者公告确认利用内部激活行为；不足实现 |
| History | 初始化、按客户端 ID 持久化、更新顺序/衰减、缺席和新客户端处理、一致性公式 | 未获得 |
| Threshold | MAD 所施加分数、中心、乘数、零 MAD 与并列规则 | 仅确认 MAD-based adaptive thresholding 的概念；不足实现 |
| Aggregation | 保留集权重、空集和全异常行为、完整模型或差分、默认值 | 未获得 |
| Model/defaults | 层定义、数据和训练协议、所有常数及 ablation 对应 | 未获得 |

SmartFL 目标 DOI 为 [10.1016/j.inffus.2025.103555](https://doi.org/10.1016/j.inffus.2025.103555)。仍需论文算法正文及 Appendix C，才能确认成对删除顺序/终止准则、伪向量公式、原模型或差分、重新纳入规则、权重与零向量边界。新的机构条目没有补齐这些信息。FedWA 不在本轮追加检索范围；此前扩展摘要缺少 DRL state/action/reward/network/defaults 的结论保持原状。

## 停止边界与下一步

本轮已完成新的作者项目路线与 SmartFL 通讯作者机构路线；没有循环分享链接、没有绕过访问限制、没有伪造邮箱或发送联系消息。31 个新增 HTTP 响应全部 SHA-256 复核，19 个下载 Python 文件逐项 Git blob 身份核对；搜索索引线索与下载全文严格分开。原始代码虽保存为服务器返回的 `.html` 后缀，`classified_receipts.json` 已明确标作 `source_code`，并非网页全文。

有新材料后才继续恢复：目标 FedDNA 的合法论文全文（尤其 Sections 3/4 与附录），或作者明确对应此 DOI 的 strategy/probe/model 代码；SmartFL 的全文/Appendix C 或作者直接关联的实现。取得材料后按表中定义提取可执行规格，并先通过组件回归再接图像管线。**当前不解锁训练，不把猜测分支写入论文基线表。**

可重放本地核验：`python tmp/celeba_baselines/source_recovery_followup_20261009/analyze_sources.py`。输出见 `source_inspection.json`；查询和网页证据见 `search_notes.json`，响应身份见各 `*_receipts.json`，产物身份见 `FILES_SHA256.json`。
