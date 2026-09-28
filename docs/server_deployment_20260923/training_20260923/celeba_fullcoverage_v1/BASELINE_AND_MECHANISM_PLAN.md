# CelebA 完整覆盖：基线与机制补充范围

日期：2026-09-26。状态：本地证据审计与后续计划；本文件不表示对应方法已经集成、实验已启动或结果已获得。仅核查本地提交稿、实现和既有审计，不新增外部检索。

## 结论与计数边界

用户要求“足够全”，应覆盖正文的分布、攻击、方法和机制，而不能以七方法或三个 adapter 的完成代替全 benchmark。主代理拟推进的第一阶段是固定当前七方法配置、IID/non-IID × Benign/F-Flip/FedSA/S-DFA/Sp-DFA × 十共享种子，即 **700 个模型记录，复用 56 个已有 valid 记录、新增 644 个训练**。这是待实时核验的执行目标，不是本文件测量到的完成数。

正文提交稿实际列出 **16 个基线**，加 GuardFed-AD2+ 是 **17 行**。当前七方法包含六个基线和 Ours，因此全文方法覆盖还差十行。每行上述矩阵为 100 个记录；全文对齐是 **1700 个方法×分布×场景×种子记录**，相对第一阶段还缺 **1000 个记录**。这里“记录”不等于新增 CNN 训练：LoGoFair 后处理和共享校准应复用匹配的终轮 checkpoint；全部计数也不包含调参探索、集成 canary 和机制消融。

七方法的配置是此前 non-IID Benign/S-DFA 上选择的配方。固定它们铺满场景能回答跨分布/攻击迁移，不等于每个分布都已充分调参；两种说法必须区分。所有新增表仍是 valid-only。十种子并不能自动把已用于调参的验证集变成独立测试证据。

## 正文方法逐项对齐

本地依据：`tmp/pdfs/tdsc_submission/submission.txt` 第 292–309 行与 Table I/II 第 319–396、440–495 行；不要用较旧的 `GuardFed_AD2plus_final_paper_tables.md` 代替当前投稿正文方法名单。

| 正文方法 | 当前 CelebA 状态 | 完整比较需要补齐的工作 |
|---|---|---|
| FedAvg、Median、FLTrust | 当前七方法内，有已验收结果 | 扩齐双分布、五场景、十种子；保持匹配数据/训练协议 |
| FairFed | 当前七方法内，项目适配 | 明确名称与适配边界；如声称原法复现，另做源码/公式忠实度验收 |
| FairGuard | 当前七方法内，root 适配 | 同上，不能仅靠名称认定忠实复现 |
| FairGuard → FLTrust | 当前 `FLTrust+FairGuard` 项目组合 | 查清串联顺序、root 公平筛选和余下客户端重加权；不能仅凭名称认定等同正文串联基线 |
| LoGoFair | 官方 DP 后处理组件及合成 CPU 检查已存在，未接入完整 CelebA | 校准/评价客户端身份映射、BetaCalibration 实测、真实图像概率缓存、完整后处理验收 |
| FedAA | 官方 DDPG 组件 CPU 对齐/恢复检查已存在，未完成图像端到端 | 跨轮状态/动作映射/replay、根集 reward、70 次有效聚合时序、完整恢复检查 |
| Fed-NGA | 公式聚合组件已存在，尚无完整梯度训练路径 | 同点梯度、服务器步长、攻击梯度语义、batch 累积和一轮公式验收 |
| FedWA | 当前正文有该行；旧实现只有 `AdaAggRL` inspired 分支 | 先核对正文引用和映射，再取得/核对完整机制；不可把启发式 AdaAggRL 直接改名为 FedWA |
| LASA | 旧分支标 `LASA-core` | 逐层稀疏化、筛选、参数与原法核对；尚无新 CelebA 忠实度验收 |
| Huber-BRFL | 旧分支为迭代 Huber M-estimator 核心 | 验证目标、阈值、迭代停止与聚合权重；尚无新 CelebA 忠实度验收 |
| FLGMM | 旧分支明确 `FLGMM-lite distance GMM` | lite 版本不能无披露代表原法；补全原法机制及匹配验收 |
| SmartFL | 旧分支明确 majority pseudo-direction 的 core | 核对完整原法和数据访问；尚无新 CelebA 忠实度验收 |
| FedDNA | 旧分支明确 fingerprint MAD 的 core | 核对原法指纹/异常评分/筛选机制；尚无新 CelebA 忠实度验收 |
| Cosine Similarity + Fairness Deviation | 正文第二种 hybrid；当前七方法缺失 | 查明旧 `GuardFed` 与该正文行的真实映射，固定准确实现后加入；这是项目控制，不需冒充外部原法 |
| GuardFed-AD2+ | 当前配方与原 5090 实际实现一致路线 | 保留候选选择、硬筛选和组阈值校准的真实说明；机制归因另见下文 |

仅在本地资料中未见完整验收，不代表某外部原法不支持图像或无法实现。上述待查方法不可用旧简化分支“先跑完再称原法”。旧论文结果保留，并单独披露其实现状态。

**代表性充分比较档：** 当前七方法 + LoGoFair/FedAA/Fed-NGA，形成十行，覆盖常规、公平、鲁棒、自适应和组合路线；至少还需三个可信 adapter 的 300 个匹配记录及其有界调参。可以称“代表性扩展比较”，不能称全文全部基线齐全。

**正文全 benchmark 档：** 17 行全部保留和说明，除三个 adapter 外，再补 FedWA/LASA/Huber-BRFL/FLGMM/SmartFL/FedDNA/cosine hybrid 七行，即另 700 个记录。官方算法的训练目标/预算不兼容统一 Adam 时，单独披露合理适配和原法机制，不能通过统一接口抹掉算法本体。若部分原法最终无法可靠复现，应显示缺口/原因，而不是用未经核验的替身填表。

## 三个现有 adapter：有边界的参数搜索

以下是**待冻结的具体搜索建议**，不是已授权协议的隐式修改。每个方法在两分布、Benign/S-DFA 上用固定探索种子 91001 评价；四个环境等权平均预先约定的 score 选一个跨场景配方，不按场景分别取最漂亮参数。保留全部候选、ACC 冠军和 Pareto 集；十种子确认中另列不含 91001 的新种子汇总。其他新原法同样需要有限、机制相关的搜索，不能只调 GuardFed。

| 方法 | 可执行候选范围 | 搜索量和固定项 |
|---|---|---|
| LoGoFair-DP | `(global_delta,local_delta)` 为 (.02,.02)、(.02,.05)、(.05,.02)、(.05,.05)；`post_lr` 为 .001、.005 | 8 候选 × 2 分布 × 2 场景 = **32 次后处理**，复用 4 个 FedAvg 模型；固定官方 beta=1000、30 post rounds、mu/lambda 各20步、BetaCalibration=True。数值故障保留，不静默换 beta。EO 上游问题未修复前不进入该搜索 |
| FedAA | actor/critic 同步 LR 为 .001/.01；保留客户端数 k=10/16；本地 LR=.0005/.001 | 8 候选 × 4 环境 = **32 次70轮训练**。含官方 actor/critic 实际默认 .01 与50%保留率；固定噪声 .1、reward=干净 root ACC、tau=.001、gamma=.99、replay16。保存全部策略状态，不引入 GuardFed fairness reward |
| Fed-NGA-gradient | `server_eta` 为 .001/.003/.01/.03/.1/.3，先一轮有限性/步长量级门检 | 6 候选 × 4 环境 = **24 次70轮训练上限**；固定同点客户端均值梯度、count weighting、无聚合后重归一。此 eta 是显式服务器步长，不是旧 Adam LR 的改名。若全部量级失效，记录搜索边界并做版本化修订，不能静默无限扫参 |

LoGoFair 的 32 次后处理不需要再训练 32 个 CNN。十种子全矩阵的 LoGoFair 行可复用匹配 FedAvg 的 100 个模型。FedAA/Fed-NGA 终轮必须保存各自算法状态；Fed-NGA 的计算预算差异和攻击注入点必须披露。以上 56 个完整训练 + 32 后处理探索不计入固定配方的 300 个确认记录。

另外七行先各完成机制映射清单与确定性小例子，才能冻结具体专属参数范围；仅凭本地 lite/core 分支推测原法调参范围会造成虚假忠实度。本阶段不虚构其已完成状态或精确算力需求。

## 同校准对照：优先复用 checkpoint

核心问题是“公平性提升来自聚合，还是来自组阈值校准”。对第一阶段全部 **700 个终轮 checkpoint**，保存未经校准输出和**同一冻结 root 校准规则**下的输出，共 **1400 个评价记录、0 次新增训练**。其中已有同口径记录直接复用；新增数量是差集，不机械地重算 1400 次。

- 校准拟合只使用相同 train-derived root；valid/test 标签不得参与阈值拟合。冻结准确率容忍损失、预算、候选阈值数、并列规则；建议采用当前 GuardFed 的 .005 容忍损失作为一套共享控制。
- 固定同一模型 SHA，所有 ACC/AEOD/ASPD 来自同一 checkpoint；记录阈值、正预测比例、群组/标签分母。后处理改变会影响 ACC，不能只摘选改进的公平指标。
- 主比较保留各方法原生输出；共享校准表是机制控制，名称加“+ shared calibration”。LoGoFair 自有的客户端双层校准另列，不能被替换成两组全局阈值后仍称 LoGoFair。
- 先对 GF/FLTrust/FedAvg 的完整矩阵做最小归因是 **300 checkpoint ×2=600 评价记录**；但用户要求全面，最终扩至所有七方法。保存概率缓存可重复计算阈值和指标，不重复 CNN 推断。

## CelebA 机制消融：最小有解释力的十种子矩阵

原有 Adult/COMPAS 六项评分消融已验收，不重跑。要把机制结论延伸到图像数据，在 CelebA 补 **IID/non-IID × Benign/S-DFA/Sp-DFA ×10种子**。Full 的 60 个模型从第一阶段复用。

新增八个训练变体：去 U（utility）、去 C（centrality）、去 A（alignment）、去 F（fairness risk）、去 V（violation）、去 N（norm scaling/clipping）、去硬筛选、去内部候选选择。**8×2×3×10=480 个新增训练**，加60 Full形成540个模型；所有模型再做原始/同校准双输出，后处理不增加训练。

这是最小有解释力的图像机制矩阵，不是必要性结论的保证。如果要求机制消融也与五场景主表完全一致，八变体需 **800 个新增训练**，比最小矩阵多320；先把两种 DFA 的因果问题回答清楚，比无边界添加所有交互项更可控。

实现前的必要核查：

1. F/V/U/C/A 消融必须作用到**每个内部候选**，否则候选 override 可把被删评分重新加回来。日志逐轮验证被消融项恒为0。
2. “无硬筛选”若既有实现只是排序/保留率，而非绝对门禁，名称和变体应按真实执行路径定义；只改 keep=1 不保证关闭所有 gate。
3. “无候选选择”冻结一个预先声明的固定配方（例如当前候选家族 balanced），而不是事后选择最强候选代表无选择版；完整记录此控制同时改变了自适应能力和固定参数。
4. 无 N 的精确定义（取消 root norm scaling、clip或二者）沿用既有 U/C/A/F/V/N 协议，不随 CelebA 结果更换语义。
5. 终轮相同种子配对差值/均值±sampleSD；不能 Full 最佳 seed 对其他变体均值。保留不支持“每项不可或缺”的 COMPAS 和 CelebA 负结果。

## 已完成证据的复用与交付边界

状态文件记录的1390已验收正式训练包括：Adult消融120、异质性180、root噪声120；COMPAS消融140、异质性180、root噪声130；root-share160；ACSIncome120；原 CelebA四方法双分布三场景240。合计1150表格+240图像。历史控制、260个同checkpoint raw/calibrated归因记录也保留。它们支持已对应的配置/数据问题，不自动替代新CelebA配置的归因。

当前不优先重复加一个数据集或一整套架构。优先次序：完成七方法双分布五场景覆盖 → 并行复用checkpoint做同校准归因、落实三个可信adapter → 补全文另外七行的机制核验与运行 → CelebA最小机制矩阵 → 冻结最终比较后的独立测试评价。独立测试是否可直接复用这些终轮模型，需要另有冻结评价协议；测试集已有历史暴露须披露，不能称从未接触的测试集。

交付应包括完整方法×分布×攻击×种子覆盖清单、算法真实性状态、每个记录的配置/源码/数据/checkpoint身份、全部失败与负结果、均值±sampleSD、成对seed差值和训练/后处理时间。第一阶段700格完成时，明确标记后续基线与机制未完成；不得报告“足够全实验已全部完成”。

## 本次只读核查材料

- `tmp/pdfs/tdsc_submission/submission.txt`：正文实验名单、IID/non-IID、五场景、十种子协议。
- `tmp/publish-5090/scripts/reproduce_paper_tables.py`：旧baseline分支及其lite/core/inspired注释，AD2+候选覆盖实现。
- `docs/server_deployment_20260923/training_20260923/celeba_expanded_v2/BASELINE_FIDELITY.md`。
- `tmp/celeba_baselines/logofair/AUDIT.md`、`adapter.py`、`upstream/options.py`。
- `tmp/celeba_baselines/fedaa/ADAPTER_REPORT.md`、`fedaa_official_adapter.py`。
- `tmp/celeba_baselines/fednga/README.md`、`fednga.py`。
- `docs/server_deployment_20260923/training_20260923/TRAINING_STATE.json`、`celeba_seedcheck_v1/selected_recipes.json`。

本子任务未修改算法、未操作服务器、未启动训练。所有后续数量为协议设计值，实际复用与新增量以去重后的 manifest 和验收记录为准。
