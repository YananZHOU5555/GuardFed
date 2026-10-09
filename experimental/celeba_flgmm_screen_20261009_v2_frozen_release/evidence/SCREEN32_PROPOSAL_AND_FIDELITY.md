# FLGMM：下一步32项验证集搜索提案（尚未冻结/启动）

v3四项真实GPU三轮CANARY已全部通过个体验收、同seed跨卡逐位重复和离机复验。新增归档SHA256为 `1b514349e3d49f3c26eff9cd9f39d80c0ad0a66a5fe1f9627192e9c2b30bc02a`；139个新增成员及4个复用模型的恢复映射均已核验。原v2的`REPEAT_MISMATCH`完整保留：它错误纳入了SciPy导入期文档示例的熵状态，v3仅修复记录范围；四项模型、控制器与诊断文件的SHA均与v2完全相同。

这证明当前环境下已检查管线及短程重复性，**不证明70轮效果、跨CPU/GPU等价或优于其他方法**。四项三轮终值均为ACC 0.516686、AEOD 0、ASPD 0，退化值完整保留且不进入论文性能表。CPU/GPU模型最大绝对差分别为0.0142328343和0.0375047773，选择/控制历史也不同。

## 建议冻结的最小科学搜索

| 项目 | 固定范围 |
|---|---|
| 8个候选 | Tg∈{10,20} × L∈{2,3} × local Adam LR∈{0.0005,0.001} |
| 4个条件 | IID(alpha5000)/non-IID(alpha5) × Benign/S-DFA |
| 每项 | seed91001、70轮、完整训练集162770、clean root16277、valid19867 |
| 共享协议 | 原RGB64 CNN、20客户端、配置4恶意客户端、local epoch1、batch64、原重加权/攻击/数据划分；不用final test |
| 选择规则 | 直接复用已冻结的综合分函数，先四条件平均；最高分recipe，严格同分按candidate ID字典序；不重写评分/权重 |
| 报告 | 保存全部32项和负结果、原始ACC/AEOD/ASPD、逐轮诊断及checkpoint；n=1，不报sample SD或显著性 |

Tg1只用于CANARY阶段覆盖，不是此搜索候选。Tg10/20保证70轮实际经过GMM、UCL拟合和监测阶段；不能将上游长程默认Tg直接搬入70轮而没有监测期。

具体冻结步骤供root审阅后执行：

1. 在新的screen source/stage目录复制`sealed_group_b`，保留本目录与原GroupB协议字节不变。原`protocol.json`仍是`PREPARED_NOT_FROZEN`。
2. 新协议记录本v3门检及离机receipt、现有21项科学源/数据身份、实际cu128环境与共同评分实现的SHA。只在新副本更新过时的“No real-image gate yet”说明；候选/种子/数据/选择规则保持上述既定值。
3. root核对并冻结新副本后，调用原`prepare_jobs.py --out <new_screen_jobs>`生成32个含新协议hash的job；不能用旧draft job配新protocol。冻结清单须覆盖source、adapter、protocol、全部job和数据身份。
4. 新的正常supervisor队列按实时资源预算分配GPU；可沿用本次每GPU1worker、每worker CPU1的上限。不得追加到正在运行的正式800队列。每job开始/结束都核源码数据身份，收齐终轮70及原`accept_result.checked_result(job_path, output)`全部检查后才接受。
5. 每个输出目录只对应一次明确attempt；失败保留、不自动循环重试。原worker不支持完整中轮恢复，controller JSON不能替代模型/优化器/RNG。已验收任务通过严格checker跳过，不能只看文件存在。
6. 32项完成、严格验收并增量离机备份后再选recipe。双分布×五场景×十seed的100项确认及最终test评价属于后续独立协议，当前没有启动授权。

## 忠实度及必须披露的适配

| 层面 | 实际实现与边界 |
|---|---|
| 原始来源 | HantaoZhu/FLGMM，commit `a064c82a4bc460e168ce09e0654924a39d258bcc`；冻结`flgmm.py` SHA `a3f8ff07cffcf451d7aae83a3dd08949495da598fb2e688f8278d057ca6b0aff`；原MIT许可证随附。 |
| 方法名称 | 使用 **FLGMM author-code aggregation adaptation**。作者代码按人数最多GMM分量选择；论文介绍的较小均值说法不同，不能称逐公式完整论文复现或承接理论保证。 |
| 聚合输入 | 20个稳定cid的攻击后完整local model。桥接为`global + attacked_delta`，作者聚合后再转`aggregate_full - global`；不把完整模型误当delta。 |
| 核心行为 | 全客户端等权中心、逐参数Euclidean距离、两分量GMM、标准化距离历史、零基Tg边界、固定UCL监测及入选模型等权平均均沿用作者可观察行为。 |
| 计数/数据访问 | FLGMM不按样本量改写作者的等权聚合；counts只作身份诊断。聚合评分不读root/valid/test。共同core仍为攻击与诊断计算clean-root模型，这一点要披露。 |
| 数值扩展 | 原`bounds_2`计算后仍用`bounds`的行为保留。零标准差仅以float64 epsilon扩展并记录；空集合维持旧全局模型。不能声称该边界是上游逐位等价。 |
| 本项目训练适配 | CNN、Adam、batch64、重加权、攻击、数据划分使用共同CelebA协议；这是聚合算法接入共同训练框架，不是作者原训练项目原封不动的复现。 |
| 实测精度 | v3两个场景各两次跨卡重复的模型8个张量、全部指标、attack/selection/controller及训练RNG均精确相同；full-model/delta桥接最大舍入差实测1.862645149e-9。不能泛称浮点桥接数学上逐位无差。 |
| 记录与恢复 | 导入期RNG单独保留，训练期显式创建/重新使用的生成器及观察到推进的导入实例都纳入严格比较。仅在记录边界观察推进，不拦截每次方法调用；也不宣称完整训练可续跑。 |

原始方法差异、组件oracle与来源记录见封存`sealed_group_b/REPORT.md`。本提案未修改算法、选择规则、种子或旧结果，也没有冻结或启动32项搜索。
