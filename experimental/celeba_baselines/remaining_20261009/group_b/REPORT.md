# 四个剩余基线：实际实现与来源缺口（2026-10-09）

已实现 **FLGMM 作者代码聚合适配器**，并通过直接执行冻结作者代码 AST 的跨轮回归。FedWA、SmartFL、FedDNA 已准确定位投稿引用，但现有可得材料不足以忠实重建，不用旧启发式替换。没有接触服务器或改动历史结果。

| 投稿方法 | 本次交付 | 可进入下一步的程度 |
|---|---|---|
| FLGMM [32] | `flgmm_adapter.py`、作者冻结源码和许可证、`check_flgmm.py`、验收 JSON | CPU 组件通过；还需主 worker 接入及真实图像门检 |
| FedWA [28] | 原会议三页 PDF/提取文本、准确规格缺口 | 缺 DRL 策略实现规格，不可认证为原法 |
| SmartFL [33] | 原法身份、主来源入口及检索回执 | 缺两阶段算法完整定义，不可认证为原法 |
| FedDNA [35] | 原法身份、主来源入口及检索回执 | 缺激活指纹/历史/阈值完整定义，不可认证为原法 |

## FLGMM 的实际实现

[作者仓库](https://github.com/HantaoZhu/FLGMM) 的当前 tree 与此前冻结 commit 均为 `a064c82a4bc460e168ce09e0654924a39d258bcc`。本次重新下载的 `flgmm.py` SHA256 为 `a3f8ff07cffcf451d7aae83a3dd08949495da598fb2e688f8278d057ca6b0aff`。仓库为 MIT，许可证随附。`models/Fed.py` 和 `utils/options.py` 也已固定。论文 DOI 为 [10.1016/j.inffus.2025.103569](https://doi.org/10.1016/j.inffus.2025.103569)。

保留作者代码的完整轮间行为：每轮先取所有 local models 的等权中心；按逐参数 Euclidean 距离拟合两分量 GMM；选**人数最多分量**，以其均值和总体标准差标准化全部客户端距离并积累历史。零基轮次 `< Tg` 用当轮 GMM 筛选；`== Tg` 用累计标准化历史确定控制限并按历史均值筛选；`> Tg` 用当轮标准化距离与固定 UCL 比较。最终在入选客户端上等权聚合，空集合维持旧全局模型。

调用接口：

```python
adapter = FLGMMAdapter(client_ids=range(20), warmup_rounds=20, control_width=3)
aggregate_state, diagnostics = adapter.step(attacked_local_model_states, client_ids=range(20))
if aggregate_state is not None:
    global_model.load_state_dict(aggregate_state)
fullstate["flgmm"] = adapter.state_dict()
```

必须传**攻击处理后的完整 local model state**，保持全部客户端身份与顺序。此接口拒绝部分参与、非浮点模型 buffer、NaN/Inf、异构 state 及不同 recipe 恢复；现 CelebA CNN 的浮点 state 可对接。无需 root/valid/test 数据做聚合。不要先 flatten 再改变距离求和顺序，也不要将模型状态误当更新量直接加到全局模型上。

`warmup_rounds` 保留原代码 `ccepochs` 的零基边界：Tg=20 时前 20 次调用为 GMM，第21次调用确定 UCL。`state_dict/load_state_dict` 保存历史、轮次、阈值、稳定客户端身份和 recipe；调用方仍需保存模型、优化器及 RNG，单独保存此 state 不能宣称整个训练可恢复。

### 原论文/作者代码差异与旧报告纠正

1. 出版社论文介绍说按**较小均值分量**识别良性客户端；作者函数40–48行实际选**人数最多分量**。本适配器忠实执行后者，必须标为 *FLGMM author-code aggregation adaptation*，不能称完整论文公式逐项复现或直接承接理论保证。
2. 作者在控制限拟合时计算 `bounds_2`，后续却使用 `bounds`。适配器保留可观察行为，没有静默修正，也没有另加 n−f 筛选。
3. 2026-09-28 的旧 fidelity 报告认为后期新检测未用于聚合。这一点不成立：原源码444行先执行 `excluded_clients = excluded`，463行实际按该列表过滤。本次跨轮 gate 直接证明一个前期正常的客户端变恶意后退出聚合集。
4. 原代码在分量标准差为0时会除零；适配器仅在该情形使用 float64 epsilon，诊断明确记录。全同模型时 z=UCL=0，监测期遵循作者严格 `z < UCL` 边界而跳过更新。该数值扩展单列测试，不称上游逐位一致。

### 已完成的组件门检

运行 `python tmp/celeba_baselines/remaining_20261009/group_b/check_flgmm.py`。`component_acceptance.json` 为真实运行输出：CPU torch2.8.0、numpy2.2.6。独立 oracle 直接 AST 提取并执行冻结作者 `decompose_normal_distributions`、`plot_control_chart`、`euclidean_distance`、`FedAvg_0` 及 FLGMM 主分支，绘图与无关性能测试使用无副作用 stub。

7轮横跨 Tg=2 的 GMM、控制限拟合和监测，**聚合 tensor 逐位相同，全部历史/UCL/选择索引精确相同**；额外验证第4轮新恶意客户端影响实际聚合、跨阶段 JSON 恢复、最大人数不等于最小均值分量、零方差/空集、输入和恢复身份拒绝、输入不被修改。没有真实 CelebA/GPU gate，没有性能结果。

## 另外三方法的准确缺口

**FedWA [28]**：[AAMAS2022原始PDF](https://ifaamas.org/Proceedings/aamas2022/pdfs/p1610.pdf)，Guo/Wang/Wu，页1610–1612，文件SHA `3766b6d06c2568370e7be52c7f8ab4ae9417a373b7e66afb462296e0e1e29e48`。已全文提取阅读：公式仅到 local/global objective 及加权 objective；第3节给 DRL 选权重的流程描述，没有 state 编码、reward、actor/critic/network、更新规则、探索/训练优化器或候选超参。没有可确定的 DRL 算法可忠实编码。旧 AdaAggRL 固定系数组合没有学习过程，不能放原法行。需要作者实现或扩展稿的这些定义；同名 FedWAM 机器人项目不对应本引文。

**SmartFL [33]**：[出版社](https://www.sciencedirect.com/science/article/pii/S156625352500627X)、[作者所在 UWA 的书目页](https://research-repository.uwa.edu.au/en/publications/smartfl-simple-majority-rule-based-byzantine-robust-federated-lea/)，Dong/Gao/Zhou/Yang/Kuang/Fu，DOI `10.1016/j.inffus.2025.103555`。公开预览足以核实按多数方向构建 pseudo-global 再召回正相关模型，但不足以固定 pairwise 删除次序、停止规则/需要的恶意数、原参数或 delta 语义、归一后 pseudo 的定义、最终 norm 处理与权重和90度等号规则。需要第4节完整算法/公式和实现细节。不要换成 ICLR2023 的同名 server-side subspace training SmartFL，也不要把旧 row-mean cosine top-n−f 分支认证为原法。

**FedDNA [35]**：[出版社](https://www.sciencedirect.com/science/article/pii/S2214212625003941)，Garg/Bansal/Yadav/Kandhoul/Dhurandher/Woungang，DOI `10.1016/j.jisa.2025.104358`。公开预览证实使用固定探针的激活行为指纹、历史一致性和 MAD 自适应阈值。仍需第3节具体 fingerprint 层/跨probe归约、probe 构造与数据访问、历史参考/更新/初始化、consistency公式、MAD multiplier/零MAD边界、最终权重/空集处理及超参。参数距离/范数/root-cosine 三标量的旧 MAD 分支没有这些核心机制，不可代替。

2026-10-09 实测 OpenAlex 对 SmartFL/FedDNA 返回 closed 且未列 repository fulltext（回执已存）；出版社直接抓取返回403，公开预览可读。GitHub repository search 未找到对应实现。此证据只说明**本次未获得足够来源**，不证明作者未公开代码或其他合法渠道不存在。未发送邮件、未购买论文或访问未经授权凭据。

随后完成的有界来源恢复见 [SOURCE_RECOVERY_20261009.md](SOURCE_RECOVERY_20261009.md)：SmartFL 已查到第一作者官方机构主页和固定版本的论文条目，但条目没有全文/代码链接；FedDNA 查到作者公开 Elsevier share link，正常访问暂时返回 `ATP-3 / Client Request Error`，不能据此断言失效或永久不可得。Elsevier XML 的 HTTP 200 仅为题录，PDF 入口的 HTTP 200 实为 challenge HTML，均未当作全文。另对本机下载目录 5 份可能学术且标题不明 PDF 仅检查前两页，三法均未命中，未读取个人票据/成绩单/申请材料，未保存无关正文。具体 URL、状态、原响应 SHA 与缺失定义见该报告及 `sources/recovery_20261009/recovery_summary.json`。新增可认证 adapter 和训练均为 0；现有 FLGMM 交付保持不变。

## 接管后的最小搜索建议

`candidate_configs.json` 给 FLGMM 的 **8个建议候选**：Tg∈{10,20}、L∈{2,3}、localAdamLR∈{0.0005,0.001}。先按已授权共同图像协议做 real-image gate，再由主代理冻结 valid-only 四条件搜索清单；这些是建议，不是已授权/已启动 manifest。70轮足够覆盖两个阶段；作者示例200轮用Tg50、L3，不能把其默认500轮/Tg100原样搬来导致70轮完全没有监测期。

调参比较继续保持所有候选和负结果、同一冻结评分规则、seed91001及IID/non-IID×Benign/S-DFA；选好 recipe 再接双分布五场景十种子，旧900条不重跑。另三方法未定义完整算法，故没有编造候选。相应文件都在本目录，`FILES_SHA256.json` 记录字节哈希，便于接入与验收。

## 已接入冻结图像 worker 的追加交付

`worker.py` 只在单独加载的冻结 core 中包装 `aggregate_round`；其他方法仍传回原函数。每次 `aggregation_wrapper` 创建新控制器，固定20个 cid 顺序；输入 bridge 为 `local_i=global+attacked_delta_i`，输出 bridge 为 `delta=aggregate_full-global`，随后沿用 core `apply_update`。空集给零 delta。每轮记录完整 `state.json` 和 `diagnostics.json`；终轮保留 `model.pt`、原始结果、来源和验收哈希。state JSON 是聚合器证据，**不是模型/优化器/RNG完备的续训 checkpoint**；worker 拒绝已有输出目录，不声称中轮恢复。

`check_worker.py` 已真实运行通过：

- 非零全局模型和两个 dyadic 客户端簇下，桥接后的 `apply_update` 与直接完整模型聚合**逐位相同**，并显式验证“直接把完整state当delta”会给错误结果。普通float32下桥接可能有舍入差异，逐轮记录 `model_delta_reconstruction_max_abs`，不泛称逐位等价。
- 直接调用字节冻结 `cdd55865…` core 的 `run_experiment`：20客户端，各4张**合成 uint8 RGB64**图像，clean root8张、评价16张，CPU三轮，实际 CelebACNN/localAdam/重加权/S-DFA metadata flip和FedSA攻击/root评价/native指标路径；没有用 toy 函数替代这些训练和攻击步骤。仅用合成 bundle 替代数据读取。Tg=1让三轮覆盖 GMM→UCL→monitor；这是测试参数，不是候选配置。
- 每轮state/诊断和终轮model/result已保存，独立新run控制器仍从0开始；未冻结协议的启动被拒绝。真实 CPU 管线耗时约6.57秒，仅是这次小门检记录。

真实输出见 `integration_acceptance.json` 和 `integration_toy/`，它们**不是CelebA性能结果**。初读的 `integration_20260928/source_snapshot` 是CRLF（原始字节SHA `cd216da8…`，LF归一才为冻结 `cdd55865…`），所以门检使用 `tmp/revision-publish-20260928` 中字节精确的 `cdd55865…` core；没有修改旧快照或放宽正式hash。

`protocol.json` 与 `screen_jobs_draft/` 已生成完整32份**未冻结**70轮valid-only搜索模板：8候选×2分布×2场景，seed91001，原 `.0005/.001` LR范围。worker 在 protocol.status 非 `FROZEN` 时拒绝启动。主代理审阅、做真实图像门检并冻结后，可用 `prepare_jobs.py --out <new_directory>` 重新生成包含该冻结协议hash的正式jobs；此脚本本身既不冻结也不开训。所有冻结源/数据hash继承已验收的共同协议，并逐job校验。`accept_result.py.checked_result` 提供后续runner严格复用验收入口，核对70轮、valid19867/train162770、终轮指标、状态历史/阈值、源码/配置和所有产物hash；目前无70轮真实结果可验收。
