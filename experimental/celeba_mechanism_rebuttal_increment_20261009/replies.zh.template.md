# CelebA 机制回复增补：中文释义与接入位置

**DO_NOT_SUBMIT_BEFORE_FULL_COHORT。** 这里只提供当前中期证据的可审候选，不代表完整机制实验、最终评价或论文已经完成。native/shared 正式主口径仍待作者决定。原回复和正文文件一律未改。

## AE 增补的意思

新增图像证据只有六个齐备场景的 60 个 minus_U 与 60 个历史 Full checkpoint，每场景十个相同 seed 配对。n=10 native 面板里，删除效用评分项 U 使六场景平均 ACC 均下降 {{ACC_MIN}}–{{ACC_MAX}} 个百分点，但 ASPD 六场景均降低，AEOD 五场景升高。原始 raw 视图则有五场景 AEOD 降低。这支持准确率与不同差异指标之间的权衡及预测口径依赖，不支持“每项机制都必要”。保留原 tabular 证据和 COMPAS 不利结果；其余图像干预仍待完整证据。

## R3.2 接入方式与意思

只替换原回复最后一段，保留前两段。原 280 条 tabular 分析矩阵不撤回：260 条新训练及 20 条历史 Adult Full；COMPAS 的 12 个删除条件平均 ACC 都高于 Full，其中六个三指标均更优。新增的是独立删除 U 的图像中期证据：IID 五场景和 non-IID Benign，每个程序 60 个终轮模型，同场景同 seed 配对；每模型的三个指标和三种预测视图都来自它自己的同一个 round-70 checkpoint。IID Benign native 三指标从 Full 的 {{FULL_ACC}}%/{{FULL_AEOD}}/{{FULL_ASPD}} 变为 minus_U 的 {{DELETE_ACC}}%/{{DELETE_AEOD}}/{{DELETE_ASPD}}。[主表与配对差]({{EVIDENCE_TABLE}}) 保留 n=10 均值和样本标准差 ddof=1；[原接受表]({{SOURCE_TABLE}}) 同时保留双方一致的 9/6 seed 面板。不能将单一 U 干预写成 C/A/F/V/N、hard filtering 和 candidate selection 均已完成或均必要。

## R3.7 接入方式与意思

在原两段之后新增一段，不删 COMPAS 的准确率反例或 F 删除前后校准逆序。新增图像反例是：n=10 中删除 U 的 ACC 六场景下降、ASPD 六场景降低，native/shared AEOD 五场景升高，仅 IID Sp-DFA 降低；raw AEOD 的方向相反，五场景降低，仅 IID Sp-DFA 升高。120 个展示模型的 native/shared 三指标及保存的组混淆计数字典相同，不能从这份相同结果独立归因“共同校准带来了增益”，也不因此决定正式主口径。候选补偿、冗余、root 估计波动只列为待证假设。以上方向限定于 n=10，不保证 9/6 子面板或所有数据集一致。

## P2 的意思

P2 仍为 pending；只把已完成部分具体写明为六场景 60 对、三视图、同规则 10/9/6 seed 及配对差。其余 variant/scene、环境可比性和完整机制结论仍待完成，不把队列运行或已训练数量当作完成三视图。

## 必须随表保留的可比性边界

- valid-only；AEOD 是绝对 TPR 差，不是完整 equalized odds。三个视图不是三组独立训练。
- n=10 用 91001–91010；n=9 排除择方 seed91001；n=6 保留 91005–91010。双方使用相同规则；这些 valid seed 均已曝光，不是未触碰确认集。
- Full 展示模型推理为 5CPU+55GPU，minus_U 为 60CPU。历史 Full 训练为 59cu128+1cu130，后者是 non-IID Benign seed91001；minus_U 为当前 driver595/cu128。历史与当前 driver 未保持一致，不是统一设备的最终公平比较。
- native 含各程序原 root-only 校准；raw 为 margin>0、tie0；shared 为相同冻结 root-only 拟合规则，组阈值预测使用 >=。不能把程序之间差异全部归于聚合，也不能从 native/shared 相同结果声称独立校准收益。
- 不宣称显著性、必胜、所有组件必要、完整 800 新 controls 完成或最终 test。[七场景 native 表]({{NATIVE_TABLE}}) 的额外 non-IID F Flip 不进入本六场景三视图推论。
