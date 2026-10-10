# Fed-NGA + Huber：64项验证搜索源包

**FROZEN 科学配置；PREPARED_NOT_DISPATCHED。** Fed-NGA32 + Huber32，各原8候选 × IID/non-IID × Benign/S-DFA × seed91001；70轮、train162770/valid19867。未SSH、部署、训练或推理，实际新接受0。本包不含LoGoFair任务。

原 `worker.py`、`accept_result.py`、group_a `adapters.py`、`fednga.py` 全部逐字复制；`prepare.py` 仅修改 manifest 的过期 limitation 文案。原64 prepared协议/封条/结果保持不变，新协议与64 jobs独立冻结。原作者记录 `AUTHOR_DECISIONS.json` SHA aee89e8210b5aa83d8ee814d5afde4bb655f3ba559bb53ae6ebcd1ca0d46851d 已绑定：H接受R^p恒等投影的CNN实用适配，只作经验结论，不继承凸性/覆盖数保证；五routine字段按既有原法/共用设置落实，H/L已非待决。

原科学路径不变：同点正经验梯度、unweighted CE、0客户端优化器步、原始ni权重；原攻击参照 `frozen_root_localadam_delta` 仅供攻击，上传 `-A(-g,r)`，S-DFA保留原元数据与fedsa路由，不改标签。Fed-NGA无范数恢复/合并后再归一化；Huber固定Ti=T0+M/sqrt(ni)、原加权向量目标与不收敛拒收。保留全部候选、失败和负结果，无自动重试/中轮恢复。

新命名空间是唯一科学metadata变化；seed、阶段validation_screen、任务接口、候选、原数据hash和科学body均不变。四个真实3轮CPU门检已root严格离机：12轮、240同点检查、86成员、科学表记录0；实际snapshot的worker/checker与本包逐字相同。无需仅为命名空间重做四项门检。该证据不证明GPU数值等价、全部eta稳定或70轮收敛；首次GPU工作需root实际Linux/源数据/环境/资源预检与启动观察，不能把旧CPU门检写成GPU结果。

唯一额外运行适配在 `run_queue.bind_shared_inputs`：复用原门检6个注册共享数据目标的逻辑名称、确切物理路径、大小、SHA；原guard拒绝越出repo的其他路径，未放宽到任意symlink。worker/checker科学函数不改。`run_queue.py`只串行调度一个子worker，每项使用独立进程，避免重复设置Torch interop线程的生命周期问题；原checked通过才记录server完成。离机接受仍为0，后续原严格备份/异机验收由root安排。

建议最多1 worker：物理GPU1/CPU105/1thread/nice10/idleIO。FL102–103、Hybrid104、CPU评价112–119及现有main8均受保护。RESOURCE_PLAN只是计划；root须读当前guide、扫描所有线程，核无≤16核限制性affinity reservation与CPU105重叠/cgroup容量、GPU1 UUID/可用显存、无重复worker和actual source/data hash，再提供真实外部预检文件及SHA。没有生成实际预检PASS或派发审批。

选择口径仍为**每候选每条件先算原frozen_score，再平均四条件；每方法最高，精确并列candidate字典序**。原frozen_score.py原字节复用；保留全候选/ACC冠军/三指标Pareto。n=1，无SD/显著性；四条件不是独立seed，不保证某法胜出，不自动采用recipe或启动100。现有loader materialize过test属性/划分元数据，不能称untouched test；没有test图像评价/拟合/选参。

本地只生成代码/config/小证据；模型、数组、日志、归档仅服务器或先验卷标Yanan 2TB/容量后的F:/YananResearchStorage/GuardFed。本包源入口拒绝Windows模型输出；E不存新模型或原始大产物。

`SELF_CHECK.json`实际通过：64身份/32+32/16候选×四条件、原worker/checker/component字节、旧源hash、原checker missing-result/失败拒收、原metadata变异拒收和少量资源MOCK正反边界；没有CNN、模型写入或真实资源检查。原完整structural gate继续作为未变checker回归依据，不重复整套科学门检。

## V2仅部署资源语义修复

V1首次实际root资源预检把主800的大affinity mask误作CPU105独占冲突，训练worker0、结果0；原失败封条不改，具体原观察及SHA见V2_LINEAGE.json。大mask为可调度集合，不表示CPU独占或空闲。V2只改部署predicate字段/status与对应fixture/CLI/资源说明，全部64job及科学snapshot原字节。外部root预检仍必须实测并记录所有受限线程、broad-mask线程、CPU测量和预算；不凭本地MOCK允许运行。SELF_CHECK.json是V1历史科学metadata证据，V2实际有限回归见V2_CHECK.json。
