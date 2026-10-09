# LoGoFair-DP 评分与后处理桥交付（2026-10-09）

**实际管线已接通并通过合成图像验收；正式状态仍 PREPARED_NOT_FROZEN，中心 CelebA 的 client-ID 映射未裁定，没有新增真实 CelebA LoGoFair 性能结果、GPU或test运行。** 新文件仅在本目录；已发布 group_a/gradient 快照及旧结果没有修改。

交付包括 `bridge.py`、32条后处理草案、100个既有FedAvg终轮模型/评分缓存复用清单、一个已验收FedAvg真实checkpoint与margin cache的核SHA副本、合成RGB64端到端检查、严格验收检查和哈希清单。

## 已落实的忠实机制

复用[作者官方LoGoFair仓库](https://github.com/liizhang/LoGofair) commit `7044815cf813cdad53fca7bab426c4d2194ab505`，通过只读既有 `../logofair/adapter.py` 执行固定官方DP优化器方法。每client保留两项local_mu、每轮等权平均global_lambda、执行官方true_H本地细化、保留每client/group阈值；40个BetaCalibration MLE由官方calibration方法调用。没有调用旧LoGoFair-style全局两阈值替代分支，没有开发或宣称EO分支。

沿用已披露的适配：group priors只来自校准root，修正上游train+valid+test计数除valid分母的问题。官方浮点溢出、缺敏感组、缺两类校准label及精确阈值tie都明确失败，不自动池化、不改beta、不换随机tie策略。因此这是官方DP机制及明确适配，不是unmodified main.py全轨迹复刻。

## FedAvg与评分的真实身份

`reuse_manifest.json` 引用已验收共享校准700中全部**100个FedAvg**终轮checkpoint与原始root/valid margin cache，对应2分布×5场景×10seed。每条保存原source_job完整config/source hashes、checkpoint SHA、original result SHA、margin cache SHA、已有accepted record及原服务器路径。它是复用清单，不代表本次重新验收了100条完整模型，更不代表100条LoGoFair已经完成。

本次完整核验并复制的例子为 `FedAvg_IID_Benign_seed91001`：

- 真实模型SHA：`2bd941918aa225a41de864c78d65620c86c5851b39487bcaff4fe97d9e9dff4b`。
- 原结果SHA：`72292ed8be4bc2be5f4a4ee95f6a88f6ad98f890cc2da748393dd60d1d53e82a`。
- 既有margin cache SHA：`39e5fe16ea5f06c0a7594fd6b1c1cbc27cf1a71a261c1b4f0db7228cce380f64`。
- model/result/source_job从已核SHA的StageA增量归档提取，逐成员核inventory SHA；margin cache对照原已验收record核SHA。来源记录见 `accepted_fedavg_reference/reference_receipt.json`。

`verify_external` 再核70轮完整轨迹、终轮指标、FedAvg/method/condition/seed、完整config/source身份、clean root无noise、train162770/valid19867、不相交约定、native及既有margin-cache原始指标。checkpoint仅以 `weights_only=True`载入。没有重新训练或修改该CNN；也没有因该例原生恒定负预测而删除或换取更好例子。

两种明确评分入口：

1. `model_scores(model, uint8_RGB64, batch_size)` 用真实CNN `softmax(logits)[:,1]`，同时保留native argmax，检查state与模式不变。合成图像门检实际执行了此入口。
2. `scores_from_accepted_margins(cache)` 用二分类恒等式 `sigmoid(logit1-logit0)` 转换既有accepted margin cache，避免重复CNN推理。float32 sigmoid与重新softmax可差1ULP，实际路径明确记录；原margin/native决定另留存，不把浮点转换称为逐位等同fresh softmax。

真实root和valid的标签、敏感元数据及顺序只能来自同一已验收cache。正式验收重新核这批external artifacts，并逐数组比对score/label/attribute/native缓存，不能换模型、root、样本或把其他模型指标拼进一行。

## client-ID：输入显式绑定，尚未代替用户决定

接口要求用户批准后提供两份文件：

- NPZ四列：`root_image_id`、`root_client_id`、`valid_image_id`、`valid_client_id`，均为一维int64。
- metadata JSON：FROZEN/approved、client人口语义与生成规则、root/valid行顺序的image-ID数组SHA。

每job分别绑定两个文件SHA，正式输出原样保存这两份文件。root/valid image-ID数组SHA必须与原FedAvg结果一致，样本不能重叠，client IDs必须覆盖0..19；实际预测使用的映射再逐数组对照原输入。metadata/source hash、阈值、校准state、同模型全部指标都进入验收链。

可审阅的协议选择已经写进 `protocol.json.decisions.mapping_population`：

| 选择 | 可以保留什么 | 必须披露/改变什么 |
|---|---|---|
| 对既有clean root与中心valid预声明20个虚拟cohort | 既有100个CNN、同样root/valid样本及评分cache、统一全局评价 | 固定规则应独立于评价label/score；例如image-ID的预声明哈希分桶。local约束针对声明的虚拟人口，不是真实训练client。当前worker只支持这种明确适配；本次没有生成它 |
| 保留真实训练client的独立holdout与同源评价 | 可以定义真实local人口 | 原root在client划分前已留出，中心valid也没有原client-ID，不能事后追认；需新增样本/映射协议及新的身份验收，可能需要新FedAvg基模型。当前central-cache worker不支持把它改名接入 |

**不能用一个dummy ID、把全局阈值冒充local/global机制，或声称中心CelebA天然具有真实训练client身份。** 当前32草案的两个mapping SHA都是null，因此完全不可执行。另3项待批准事项为后处理搜索范围、tie/缺人口/数值失败策略、真实映射门检；4项均UNRESOLVED。

## 草案与可执行入口

32草案为已有8个LoGoFair-DP候选 × seed91001下 IID/non-IID × Benign/S-DFA。每候选 global/local delta∈{0.02,0.06}、post_lr∈{0.001,0.005}；post_rounds30、local/global steps20、beta1000、calibration=True，复用各条件FedAvg第70轮同一模型。现有3轮gate不等于这些30轮拟合完成。

选择规则沿用同一个冻结score四场景均值、精确并列candidate字典序；保留所有候选、accuracy冠军与三指标Pareto。n=1，无sampleSD/显著性，四场景不是四个独立seed。评价label不进入BetaCalibration、local/global优化或mapping选择；fit仅接收root label。

```powershell
# 生成新目录的草案；不会改变现有协议或生成真实ID映射。
& <隔离netcal环境的python> .../prepare_reuse.py --jobs-only --out <全新job目录>

# 仅在协议及mapping已经独立批准冻结、job哈希重生成后可运行。
& <隔离netcal环境的python> .../bridge.py `
  --repo <冻结core仓库> --reference <已核验FedAvg参考包> --job <单个job.json> `
  --mapping <显式mapping.npz> --mapping-meta <映射metadata.json> --out <全新输出目录>
```

冻结后重新生成job需向 `prepare_reuse.py --jobs-only` 同时提供 `--mapping` 与 `--mapping-meta`；预处理器不批准或冻结任何东西。未批准状态在创建输出前拒绝；既有输出目录拒绝覆盖。首次元数据生成入口也拒绝覆盖现存protocol/reuse清单，避免重置批准记录。

验收接口是 `bridge.checked_output(Path(job), Path(output), loaded_core, Path(reference))`。它重新核外部checkpoint/root/cache/source/config身份、当前job/source/protocol哈希、映射两文件与样本行身份、完整post_rounds与有限lambda、同一序列化分类器的threshold/history/预测、全部三指标及artifact SHA。任何failure证据都阻止复用；缺result返回None。

状态文件只序列化自有结果中的netcal calibrators与数值状态，不pickle动态官方client类或训练数据；SHA检查通过后再恢复，并用同一官方predict方法重放。预测是确定阈值规则，本接口没有随机预测算法；随机状态扰动与随机输入置换用于核验复现性。

## 已执行的检查与范围

`check_pipeline.py`：PASS。一个实际已验收FedAvg CNN，为**1600张合成root RGB64 +320张合成评价RGB64**提取softmax评分；20个合法合成client ID、40个官方BetaCalibration、3轮官方local/global后处理。为了稳定覆盖每client/group的两类beta拟合，synthetic root label按其cohort内score秩构造，明确仅为门检，不估计真实accuracy。

- 实际CNN score与直接softmax一致、模型不变、重复提取逐位相同；接受的真实CNN仍只有一个，同样权重用于所有synthetic评分。
- 重新完整拟合时threshold和预测逐位相同；序列化/重载预测及同一分类器三指标一致；RNG扰动、随机输入置换后还原顺序的预测仍逐位一致。
- 错误checkpoint/config身份、错误mapping行顺序、未知client、缺group/两类label、NaN score、错误state SHA与精确阈值tie均拒绝；PREPARED正式入口在创建输出前拒绝。

`check_acceptance.py`：PASS，共14项结构检查。包含未冻结/未批准拒收、外部checkpoint身份、拼接指标、缺post round、threshold与序列化state不符、源码组件变更、预测mapping/score/label/prediction改变、损坏state SHA及保存failure拒收。

完整结构路径只在临时目录使用构造的19867行**合成fixture**，其外部数据验证被明确mock；不是对真实CelebA的评价，也没有赋予真实client身份。临时FROZEN metadata与fixture已删除，实际协议哈希保持不变。真正external reference字节身份另由上面的实际核验完成。旧prepared/gate版本保存在本目录历史目录，当前入口为 `screen_jobs_draft/manifest.json`。

运行复用了 group_a 已验证的隔离环境：Python3.10 / Torch2.8.0+cpu / netcal1.3.6 / NumPy1.26.4，主训练环境未改变。服务器后处理需在相容、已验证的隔离环境运行；不复制该venv冒充CUDA训练环境。官方source与既有adapter保持外部只读，部署时保留 `../logofair/adapter.py` 与固定upstream路径。`FILES_SHA256.json`核交付文件及外部源码。

剩余事项具体为：裁定local人口/映射协议，提供和冻结两份映射输入，真实score-only小轮门检，然后冻结32候选搜索与多seed覆盖。本次没有自动裁定或开始这些科学实验。
