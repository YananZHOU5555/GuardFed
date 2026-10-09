# Fed-NGA / Huber：独立梯度桥准备交付（2026-10-09）

**已完成独立 draft worker、64条验证搜索草案、18轮合成RGB64真实CNN管线门检与22项验收结构检查。状态仍为 PREPARED_NOT_FROZEN；5项协议决策尚未批准，无真实CelebA/GPU/70轮科学结果，未SSH、未启动队列。** 新文件仅在本目录，已发布 group_a 快照与旧实验未修改。

## 原方法与旧 worker 的实质差别

| 项目 | 原算法要求与本接口 | 旧冻结 worker / 仍须明确的适配 |
|---|---|---|
| 客户端上传 | 同一个全局参数点的正经验风险梯度；完整客户端样本按小batch累加梯度和，最后统一除分母；客户端 optimizer 步数为0 | 旧实现 localAdam 多步后的 model delta 不是该梯度，不能改名复用 |
| Fed-NGA | `w_next = w - eta * sum_i (n_i/sum n) * g_i/||g_i||`，保留方向抵消 | 无 median norm恢复、无对合并方向再次归一化；精确零梯度贡献零方向但保留n权重，是明确扩展 |
| Huber | 固定 `Ti=T0+M/sqrt(ni)`，最小化样本数加权向量 Huber目标，再做 `w_next=w-eta*center` | 不在求解内重算 MAD阈值；不收敛立即拒收并保留诊断，不做模型更新 |
| 全局步长 | 独立 `adapter.server_eta`；不从Adam LR、local epoch或gradient norm推导 | `config.learning_rate` 仅在可选 rootAdam威胁参考中使用，不作为客户端梯度或全局步长 |
| 样本权重 | 聚合始终使用真实原始client样本数ni | 经验风险可显式选原始CE或共享重加权CE；二者含义不同，需在冻结前决定 |
| root / 公平推理 | 两种防御聚合都不读取root方向或client候选模型root AEOD；报告仍用原native argmax | 只有指定上传攻击可读取干净训练root威胁参照，不能把这一访问隐称原论文防御输入 |

Fed-NGA依据[NeurIPS2025最终论文](https://proceedings.nips.cc/paper_files/paper/2025/file/29c71a71cf3b354036d9b413cd049cbf-Paper-Conference.pdf)式8–10与Algorithm1；完整客户端是同点子集梯度取全集的情形。官方最终PDF已保存于 `sources/fednga_final.pdf`，文本仅供核查。Huber依据[AAAI2024正式论文](https://ojs.aaai.org/index.php/AAAI/article/download/30181/32095)式3–6与Algorithm1。它的投影 `Pi_W` 在当前draft中显式提出 `W=R^p` 的恒等适配，尚未批准，也不据此声称其理论假设在CNN上成立。

真实梯度、式9与固定Huber solver直接调用只读已验收组件 `../remaining_20261009/group_a/adapters.py`；Fed-NGA底层为 `../fednga/fednga.py`。两者哈希锁定。冻结数据加载/CNN/敏感元数据翻转/native评价来自旧core，不调用旧同名Adam-delta聚合分支。

## 上传攻击的符号桥与不可推断项

旧core的FedSA-inspired攻击作用于下降差分 `d` 与clean descent参照 `r`：

`A(d,r) = d - gain * ||d|| * unit(r)`，随后执行原norm cap；零范数按原分支回退。

论文上传的是正梯度，所以新桥明确采用 **`g_attack = -A(-g, r)`**。若 `r=-g_root`，则未截断前为 `g - gain*||g||*unit(g_root)`；这是攻击下降方向后的符号共轭。没有把梯度塞进真实大模型state再相减，因此不存在借由Adam delta或state相减抵消来伪造梯度。服务器eta只在聚合后应用一次。

实现支持两种明确命名的威胁参照，但未替用户批准：

- `frozen_root_localadam_delta`：保留旧core的clean rootAdam更新算法和攻击者访问；改变梯度方法后，完整轨迹并不与旧训练相同。当前64草案暂取此提案。
- `same_point_unweighted_gradient`：用当前全局点干净训练root的完整未加权梯度负方向，机制较直接，但改变原攻击参考方向。

delta缩放及zero攻击可显式映射。**`foe_mode=state` 被拒绝**：旧式 `-0.5*local_state-global_state` 依赖已训练的绝对local模型状态，对纯梯度上传没有唯一对应。当前草案只覆盖Benign/S-DFA，S-DFA使用既有明确FedSA模式；不能静默把state模式替换成新攻击。

F-Flip继续调用冻结 `client_runtime_data`：仅改Male元数据与由此派生的重加权，图像/分类y不变。实测原始未加权CE下 F-Flip-only梯度逐位不变；共享重加权CE下梯度发生变化。这个null效果应保留。若选共享重加权，目标为 `sum(weight*CE)/sum(weight)`，需要标明为公平预处理适配，不能同时声称是未改原始经验风险。

## 需批准的5项决策与草案

`protocol.json.protocol_decisions` 全部标 `UNRESOLVED`：经验风险是否重加权、攻击root参照、server eta与Ti搜索范围、Huber恒等投影、真实图像门检。当前的取值是可审阅提案，不是已批准正式协议。

64草案为2方法各8候选 × IID(alpha5000)/non-IID(alpha5) × Benign/S-DFA × seed91001，均70轮、完整train162770/valid19867。Fed-NGA提出8个独立eta：0.001、0.003、0.01、0.03、0.1、0.3、1、3；Huber提出 eta∈{0.03,0.3}、T0∈{0.01,0.1}、M∈{0,1}。这些量纲不能从Adam LR直接推得，尚未证明覆盖真实CNN最佳区间；冻结前可据仅训练/root的真实pilot裁定。这里没有读取测试集或为某法胜出调整统计规则。

后续搜索沿用既有四条件综合score均值择一recipe、精确并列candidate字典序，保留全部候选、准确率冠军和三指标Pareto。n=1，无sample SD/显著性主张；四场景不是四个独立seed。

## 可执行入口与部署约束

```powershell
python tmp/celeba_baselines/gradient_bridge_20261009/prepare.py --out <全新job目录>
python tmp/celeba_baselines/gradient_bridge_20261009/worker.py `
  --repo <冻结core仓库根目录> --job <单个job.json> --out <全新输出目录>
```

当前第二条必在创建输出前拒绝。`prepare.py` 只读协议生成job/manifest，不更改状态、批准决策或启动队列；独立审阅完成后，只有FROZEN且所有决策APPROVED的协议才可生成匹配的新冻结job。冻结或任何源码/候选变更后须重新生成清单和哈希，不沿用旧草案。

部署需保留本目录与 `../remaining_20261009/group_a/adapters.py`、`../fednga/fednga.py` 的相对结构。`--repo`指向有固定 `scripts/reproduce_paper_tables.py`、`src/celeba_data.py` 和全量cache的core仓库，core/data全部哈希在每job中。训练依赖沿用服务器Torch/NumPy/SciPy等；不依赖LoGoFair的netcal隔离环境。

验收调用端将本目录加入import路径后使用 `accept_result.checked_result(Path(job), Path(out))`。它检查完整1..70轮、同终轮三指标、valid和样本身份、source/data/job/component/protocol/checkpoint哈希、真实ni与root分区总数、客户端optimizer=0、root只用于已声明威胁、Huber固定Ti/收敛/目标/权重及攻击审计。存在failure即拒收；缺result返回None。worker不覆盖已有目录，不承诺中轮恢复；失败保留traceback与solver诊断，不能盲重试。

## 已执行检查和边界

`python .../check_pipeline.py`：PASS，证据 `pipeline_gate.json`。

- 108个随机/零梯度/零root参照/三种支持消息攻击，与冻结core攻击的符号共轭结果逐位相同，另有二维符号手算；state型攻击拒绝。
- Huber独立SciPy BFGS凸目标/梯度oracle一致；故意限制求解次数的不收敛任务拒收。
- 两法 × 两种root参照各3轮，再加共享重加权路径3轮及首条确定性重放3轮，共**18轮合成uint8 RGB64、实际CelebACNN**；20个client每轮4/8张不等样本量、batch3，root8张、eval16张。
- 共360次同点client经验梯度与攻击上传核验；369次独立整cohort autograd梯度oracle（包括9次root同点梯度），允许batch浮点误差。每次全局参数保持不变，客户端optimizer步数始终0；Fed-NGA聚合向量与独立式9逐位一致，Huber检查真实向量目标驻点与固定Ti。
- 每轮模型更新等于原点加已验证带符号step；保存checkpoint重新载入native评价完全一致；确定性重放最终全部张量与轨迹逐位一致。client root公平评价、客户端Adam及AD2+阈值校准均设失败钩子，确认没有调用。

`python .../check_acceptance.py`：PASS，证据 `acceptance_gate.json`。22项结构检查含PREPARED启动拒绝、伪FROZEN而决策未批准拒绝、seed/alpha/终轮/split改变、缺轮、终轮指标混合、客户端Adam替代、ni改变、root进防御、错误root参照、checkpoint SHA损坏、失败证据与Huber不收敛/阈值/目标/权重错误拒收。

验收的完整70行路径只用临时构造的**MOCK结构fixture**，不是70轮训练；临时FROZEN元数据已删除，实际协议内容保持不变。首次fixture将non-IID来源元数据配到IID草案，被验收器正确拒收；修复记录保存在 `failed_gates/acceptance_mock_distribution.json`。准备清单随着验收源补齐而旧版保留在 `prepared_history_*`，当前以 `screen_jobs_draft/manifest.json` 为准。没有科学结果受到影响。

`FILES_SHA256.json` 记录稳定源文件、当前草案、合成证据、历史草案与外部组件哈希；不包含自身和pycache。**这些PASS均是组件/结构证据；真实CelebA、GPU执行、70轮收敛、多seed性能、正式test评价均未完成。** 主任务可在服务器恢复后据5项具体决策审阅协议，完成真实小轮门检，再冻结启动。
