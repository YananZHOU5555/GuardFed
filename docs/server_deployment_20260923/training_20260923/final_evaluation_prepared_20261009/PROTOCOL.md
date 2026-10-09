# CelebA 最终评价准备稿（2026-10-09）

**状态：PREPARED_NOT_FROZEN。没有测试集调用、标签读取、模型推理、新训练或派发。** 本目录仅覆盖已经严格验收的九方法900个终轮模型，不代表17方法1700格，也不表示整个返修完成。协议重大选择尚未决定；清单保存模型身份，不是一份获准运行的队列。

## 要回答的问题与历史边界

在保持方法、配方、模型权重和终轮规则固定后，比较不同分布/攻击下准确率与群体差异，并区分聚合本身和预测校准的贡献。最强替代解释是：验证集/开发种子的选择、预测阈值权限差异或运行环境造成表面优势。最终评价必须在读取新的评价结果前冻结比较方式，不能让结果决定主表视图、seed、checkpoint或指标。

1. 最初240次CelebA补充实验已完成并查看官方test结果。`../celeba_tuning_v1/PROTOCOL.md` 明确后续工作是“post-initial-test development”；`../celeba_final_acceptance/verification.json` 为240条旧结果验收。旧test暴露是数据集层面的历史，不能因换了seed、checkpoint或扩大方法表而抹去。
2. 七方法的recipe在91001、non-IID Benign/S-DFA验证条件筛选，再转移到完整网格；FedAA/LASA在91001的IID/non-IID×Benign/S-DFA四条件筛选。保持原选中recipe，不用900格或未来test重新选recipe。91001各方法不能冒充未经选择的独立确认种子。
3. 91002–91004和更后续validation结果也已经看过。91005–91010是当时预先固定的六seed子集，到了今天已经是已观察历史，不能重新称为“当前未见过”。其用途是保留原先预声明的敏感性分析。
4. 900模型中14条由torch2.11.0+cu130训练，其余886条cu128。四项迁移首轮逐位匹配及共享校准700 native replay通过，不证明所有70轮训练环境等价。后续评价必须记录实际评价环境，并区分“同权重重评一致”和“训练环境等价”。

若后续选择官方test作为终点，正确表述是**在已暴露数据集上、冻结后进行的最终评价**，不能称pristine/never-seen test。完全独立外部确认需要另一个事先未参与开发的数据来源/留出方案，其样本与训练影响需另立协议；本稿不自行启动该路线。

## 可复用的固定模型网格

九方法：FedAvg、FairFed项目适配、Median、FLTrust、FairGuard项目适配、FLTrust+FairGuard项目适配、GuardFed-AD2+、FedAA-DDPG适配、LASA官方聚合的local-Adam差分适配。分别包含IID（真实client_alpha=5000）和non-IID（5）、Benign/F Flip/FedSA/S-DFA/Sp-DFA、共享seed91001–91010，9×2×5×10=900个模型。

`model_inventory.json` / `.csv` 逐条列出终轮70 checkpoint、result、raw job的存储成员与SHA、原配置、源码/adapter/data哈希、图像/clean root身份、split与样本数、实际alpha、攻击、seed、原训练环境和验证集原指标。模型从备份重新定位并按字节哈希；未用torch反序列化模型，也未加载数据标签。`prepared_acceptance.json` 和 `identity_issues.json` 区分本地已核身份与仍需服务器检查的门禁。

原训练样本162770、valid19867，Smiling预测/Male敏感属性，RGB64无增强，CNN `Conv32/64/128_3x3_ReLU_MaxPool_GAP_Linear2`，deterministic FP32。官方partition2预计19962张。评价样本ID及顺序SHA目前留空，必须在冻结前仅用split元数据核定，随后与实际预测逐行对应；不能只用样本数代替身份检查。旧config中的`test_rows`/`X_test`命名在900结果中实际指valid，必须按`image_data_contract.evaluation_split`判断。

本轮不改变训练配置的`celeba_evaluation_split`字段，也不把原valid result改成test。未来仅在独立evaluation job中记录新的target split，同时保留原training config。`evaluation_jobs_draft.json`有900条模型引用，但不含运行入口、GPU分配或有效dispatch授权。

## 预测视图与待选主表规则

以下规则作为可审阅选项，不在准备阶段指定新的主视图。不能按test结果从中择优；若决定采集多个视图，必须对全部方法/模型统一采集并完整报告。

| 视图 | 固定行为 | 回答的问题与限制 |
|---|---|---|
| native | 保持每个方法既有预测接口；当前九方法仅AD2+使用其原recipe的clean-root群体阈值，其他八方法为logits argmax | 方法端到端表现；校准权限差异须披露，不能据此单独归因聚合机制 |
| raw | 所有模型使用logits argmax，margin=`logit1-logit0`，预测为`margin>0`，精确0判class0 | 同预测接口下聚合/训练产生的权重效果；没有推理时群体阈值 |
| shared calibration | 所有方法在各自同一clean train-root上使用同一已实现的阈值拟合规则，再应用至目标集 | 校准权限/预算匹配后的方法比较；依赖推理时Male，需明确权限与成本 |

Shared候选配方直接继承已验收 `../celeba_shared_calibration_v1/manifest.json`：`acc_floor`、max_acc_drop=.005、budget=.06、base_weight=1、temperature=.03、41quantiles、enabled=true。不新增搜索范围，不为某个方法换预算。群体候选由root的.02..98分位数和0/min−1e−6/max+1e−6生成，8位小数去重排序；准确objective、扫描顺序及`math.isclose`分支继承字节冻结core，不另写近似公式。

**边界必须保留：**raw argmax对应`margin>0`；冻结`metrics_from_group_thresholds`对应`margin>=threshold[group]`。因此零阈值校准与raw在精确零margin时也不能被悄悄当作同一规则。记录零margin数量，不在评价后改tie处理。Native calibration按模型自己的原配置；shared按共同配置，各视图内部ACC/AEOD/ASPD必须来自同一权重和同一预测向量，不能跨视图拼最佳指标。

只用clean train-root标签和敏感组拟合阈值；不使用valid或test标签拟合、选择、停机、调参或修正阈值。同一seed/distribution的root ID应与原contract逐字节匹配；若根样本因loader版本变动不同，停止并记录，不以“同seed”代替身份。

当前共享校准700的结果与margin缓存已经完成，但**只覆盖前七方法的valid**；不能直接把这些结果当900 shared，也不能把valid缓存改名test。FedAA/LASA200若要共享校准，需冻结后补充同模型评价，不需重训。

## 冻结前须明确的重大选择

`protocol.json.decisions`全部为null；本目录不代用户做决定。

- 最终比较范围：等17方法忠实集成/真实图像门检和完整覆盖后统一评价，或先明确标注九方法阶段性交付。当前900准备稿只服务已有模型，不能关闭八方法缺口。
- 评价终点与用语：是否在披露旧test开发历史后，对官方partition2做冻结终点评价；或另立独立外部确认方案。不能宣称新seed使旧test重新独立。
- 主表视图与配套视图：native或shared哪个回答论文主张，raw作为机制控制如何呈现；采集范围需预先固定，不按未来数值选择。
- 正式配对推断：是否报告置信区间/显著性，预先固定检验、双侧/单侧、多重比较和缺失处理。未决定前只准备原始三指标和均值/sampleSD，不自行指定新显著性门槛。

沿用此前已冻结的描述性统计：每方法×分布×攻击按十个共享seed均值±sampleSD(ddof1)，附排除91001的九seed和91005–91010六seed。跨场景先在每个seed内平均，再跨seed计算，场景不能充独立重复。所有负结果、恒定预测、零公平性数值、失败和缺失保留；不使用AD2+最好seed对基线均值，不改四处看到的阈值选择历史。

主表保留ACC、AEOD、ASPD；当前AEOD实现是绝对TPR gap，不是完整equalized odds，正文须解释/更正术语。是否增加FPR/full-EO诊断另作预声明，不能默默替换旧指标或旧结论。自定score仅保留开发历史，不能替代最终三指标或决定test后主张。

## 实施门禁与验收要求

1. 用户/主代理完成上述决定后另写冻结版本并记录哈希；不能仅把本稿status改为FROZEN就忽略未决项。先恢复服务器可观测性，读`/etc/vast-agents-guide.md`，核有效数据/源码/模型和无重复进程；当前SSH拒绝不能用历史GPU空闲替代实时证据。
2. 在目标环境安全加载900个`weights_only=True` state_dict并查keys/shapes/有限值。模型SHA、原result/job/config/recipe/source/data必须与本清单一致。旧结果和备份只读，新的evaluation输出单独目录；不训练、不重写权重。
3. 首先在valid上重放900 native。前700有共享校准replay证据，但不能据此自动省略新增200及新评价代码的门检。每条与原终轮三指标差异≤1e−12；必须核root/valid IDs。遇不一致停dispatch、留证据、定位首个差异，不自动接受“环境近似”。这一门禁只能重放旧valid，不得查看test择优。
4. 目标split ID SHA、target count=19962、train/root disjoint和图像顺序全部冻结；模型/阈值/预测规则先定，后一次生成target预测，再由独立验收计算指标。保留完整目标预测和group confusion counts；标签仅用于已冻结评价，不能反馈选配置或排除结果。
5. `acceptance_contract.json`规定所有视图从相同checkpoint、同一目标样本与顺序产生；校准阈值及fit-input/root-hash落盘。核样本完整、两组正/负分母、有限margin、所有round70/seed/alpha/attack身份，复算ACC/绝对TPR gap/ASPD与prediction rate。分母为0时不能伪置0公平性；按冻结缺失规则报告。
6. 每模型保存原身份引用、评价环境/源码、root/目标ID、thresholds、margins/predictions哈希、指标和验收。按新接受ID差集备份，核archiveSHA及成员哈希；不可重复搬运未变权重。失败/负结果不可删除；partial输出拒绝覆盖，只有外部中断且完整身份一致时按既定skip-accepted机制恢复。

本目录未提供测试集执行器，防止准备稿被误当可运行阶段。主代理接管时复用现有shared calibration图像推理与strict acceptance实现，先扩展必要身份适配并验证，冻结后再派发。最终完成需900对应的全部冻结视图、独立验收、统计和备份；而整个返修还需八方法、机制消融、正文及逐条回复。
