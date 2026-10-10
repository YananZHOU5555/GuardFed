# 新增五方法三视图：最小接线范围审查

状态：**SOURCE COMPATIBILITY REVIEW ONLY**。只读实际源码和小型接受元数据；没有载入 900 条记录/逐图数组，没有 Torch、推理、阈值拟合、SSH、派发或共享状态修改。完整 source SHA 和函数行号见 `SOURCE_PINS.json`。下面是接线方案，不是新增结果验收或正式评价批准。

## 结论

FLGMM、CosineFairnessHybrid、Fed-NGA-gradient、Huber-BRFL-gradient 的终轮均是原 CelebA CNN `state_dict`，原生报告均走 logits argmax。因此可以复用原 `rebuild_root`、margin 提取、root-only 共享阈值和计数/指标科学函数；**不能直接把新记录塞进旧 900 runner**。需要小型、方法专属的身份桥接及明确方法注册，不能以旧 `run_revision_ablation.checked_result` 代替这些方法各自的严格验收器。

LoGoFair-DP-official-adapted 的原生模型是“FedAvg checkpoint + 已拟合的 BetaCalibration/局部与全局阈值 + 固定虚拟 cohort 映射”。普通 CNN checkpoint 不足以代表它。其 native 已有严格 saved-prediction 验收路径，可直接复用；raw/shared 的语义须明确为该底座的诊断控制，或另经作者确定新的 LoGoFair 后处理复合规则。**不能把两组共享阈值替换其双层后处理后仍无注释地称 LoGoFair 原生结果。**

本次五方法加旧九方法是 **14 行、最多 1400 个不同方法×场景×seed 记录**，不是 17 行。原正文另外三行 FedWA、SmartFL、FedDNA 不在本次审查范围；不据历史规划表推断其当前开发状态。三视图不是三倍独立样本，也不代替作者决定最终主终点。

## 实际旧入口的兼容边界

- 接受事实入口 `accepted900_root`：900 unique records、900 saved receipt join、原 native 三指标 2700 个精确对应；推理设备 CPU434/GPU466，训练 cu128886/cu13014，主终点 pending。这是已有证据入口，不是本次重新审计。
- `replay_cpu.validate_inventory`（103）硬编码九方法/900；`validate_original`（275）要求原 raw-job/output/source 字段和旧 runner 单参数 `checked_result(job)`；`replay_schema_v4.normalized_views`（37）只特判 FedAA/LASA，其余分支还要求原 `revision_job.checkpoint_sha256/torch_version`。新增结果普遍不符合这些旧字段约定，不能伪造这些字段写回原结果。
- `evaluator.fit_views`（156）和 `predict_views`（177）都有九方法白名单；注册四个新 CNN 标签后，native/raw 才能沿原非 AD2+ 分支 `margin > 0`。共享分支仍为 root 拟合后 `margin >= group_threshold`。0 margin 的 tie 规则必须保持，不能用 sigmoid>=0.5 偷换 argmax。
- 最小实现可沿 v4 的 private-function/global-binding 方法：独立新 namespace 中明确扩展方法集合、把方法专属 strict 接口包装为旧科学函数所需的私有 canonical view。原 artifact 字节、实际 source_method、runtime 和 SHA 保留；每一派生字段注明来自哪个实际 receipt。旧 900 inventory/源码/接受链不动。全局将旧库存数字改成 1400 或静默合并新 cohort 均不合适。
- 原 `require_dispatch`（231）另外绑定 final protocol 决策/900 native receipt；这不是当前 validation 扩展的自动批准接口。继续沿已有 valid-only implementation replay，独立冻结每个已接受完整100 cohort，主终点/test 决策继续保留。

## 每方法的确切接线

| 新方法及原生输出 | 原 strict / 实际产物 | 复用和最小新增 |
|---|---|---|
| `FLGMM-author-code`；native=raw argmax | 实际 bound stage `source/accept_result.py.checked_result(job_path, output)`，外层 `screen_common.accepted`；`model.pt` + `state.json` + diagnostics/result/provenance + acceptance | 96 新使用实际 fullcoverage checker，4 复用使用对应旧 screen checker。先保留 GMM/UCL state70 与 source/recipe 接受链，再从接受 receipt 取 checkpoint SHA；只把其来源规范化给原 replay。预测不需要重放 GMM controller。原 `revision_job` 没有旧900要求的 checkpoint/runtime 字段，不能借旧900 checker通关。 |
| `CosineFairnessHybrid`；native=raw argmax | `body.checked(entry, scope)`，实际 coverage 接口 `common.accepted` / `runtime.functions`；model/native_replay/diagnostics/provenance/RNG/acceptance | GPU runtime 断言和 writer sidecar 是原接受契约，不能在 CPU 上伪装 torch/device 元数据。可复用已接受的外部 strict proof + SHA-bound record-layer translation，或在其原实际环境执行原检查；不重复训练。原 NaN Pearson→null 的受限 sidecar语义保持，不把 null 扩展到指标。 |
| `Fed-NGA-gradient`；native=raw argmax | 原 `accept_result.checked_result(job_path, output)`；model/diagnostics/result/provenance/acceptance；future coverage derived checker保持70轮 | 用其同点梯度 recipe/source/component/local hashes及原终轮接受器，不改成旧 core 的同名简化聚合。未来100须真实96+4严格接受后再建库存；同点梯度训练与 local-Adam 基线预算差异仍披露。预测可复用同 CNN evaluator。 |
| `Huber-BRFL-gradient`；native=raw argmax | 同 gradient checker，额外固定阈值/收敛/原 objective trace | 同上；保留 identity R^p 适配标签、solver失败与原停止准则。后续共享阈值是输出校准诊断，不构成 Huber 理论保证或聚合机制因果证据。 |
| `LoGoFair-DP-official-adapted`；native=官方 DP 后处理预测 | `bridge.checked_output(job_path, output, core, reference)`；`post_state.pkl`、`scores_predictions.npz`、mapping/metadata/result/acceptance；`summary100.run`已连接新96+旧4 | 复用已有 `summary100.record_identity`、原 strict 和保存的 `prediction` 重算；不要取 `valid_native_prediction` 当 LoGo native，该字段明确是 **FedAvg 原始 argmax**。native身份必须同时绑定 backbone checkpoint、原margin cache、post_state、settings、fitseed1719、mapping/metadata和三个源码SHA。 |

表中 actual FL fullcoverage 来源已另 pin；historical group_b 仅帮助辨认原 schema，不能代替正式 bound stage checker。Hybrid/gradient 的未绑定覆盖实现仍是 source 准备，表述不构成已完成100。

## 共用科学部分原样保留

1. 先逐方法读取实际 strict/offserver/root-adopted inventory，唯一100=`2 distributions × 5 attacks × seeds91001..91010`，每方法实际选择 recipe + 96新/4引用并集。不混入 canary、搜索非选中候选、旧 simplified 同名方法或 pending SHA。未齐100可以保留实际接受差集，但本方案不据此生成完整100评价 jobs/均值表。
2. `rebuild_root`（203）按原 config/seed/alpha重建 root，严格匹配 root IDs/顺序SHA、client counts、162770 train联合及不相交；保持16277 root和19867 valid，同 checkpoint raw/native/shared 全指标。原训练 device/environment 字段与实际评价环境分别保存，不假称一致。
3. `thresholds_from_root`（109）仅拿 root margins/y/Male；复制原 `fit_group_thresholds` code object，将 `model_margins`替换为 root-only缓存，不能触及 valid标签。共享 recipe固定 base_weight1、budget.06、temperature.03、quantiles41、max_acc_drop.005、objective acc_floor。先 fit、再用 valid margins/Male预测、最后评分；不可按valid结果重选阈值/seed/round。
4. `group_metrics`（69）继续一次预测向量生成 ACC、AEOD=绝对TPR差、ASPD=正预测率差与所有分母/混淆计数。保留常量预测与零公平差，不能把它们写为“公平方法获胜”。`check_native`（292）仍是三指标1e-12，权重前后逐张量身份保持。
5. 对四个新 CNN方法优先复用已存在且身份完整的 root/valid margin缓存；没有则在后续实际授权时仅做一次必要 margin提取，而非重新训练。原900曾出现同checkpoint CPU native超1e-12而GPU符合的边界案例，因此不能直接宣称CPU可替代GPU。复用已有 GPU replay的计算路径及环境绑定；失配保留停止，不提高容差、不反复挑运行设备/结果。

## LoGoFair：可以立刻复用什么，什么仍需明确

`bridge.verify_external`（79）已经核 FedAvg70终轮/checkpoint/config/root/valid/cache；`scores_from_accepted_margins`（144）采用已接受float32 margins的 sigmoid，明确与新 softmax可能差ULP。`fit_predict`（174）只向官方Beta/局部全局优化提供 root labels、fixed20虚拟cohort和fitseed1719；`checked_output`（299）重新加载受SHA约束的已拟合state并重现保存预测，不重新fit。现在新summary已经保留constant/negative记录和环境史，这一条 native 分支无需再写一个evaluator。

**最小、不增推理/拟合的三列对照候选**：LoGo原生列用其已接受 `prediction`；raw列引用同一个 FedAvg checkpoint的原accepted900 raw receipt；shared列引用同一个 FedAvg checkpoint的accepted900 shared receipt。三个来源必须逐ID/seed/scenario/config/rootIDs/checkpoint/cache join，并标注后两列是“LoGoFair backbone / postprocessing-replacement diagnostic”，不是 LoGoFair 在共享阈值后仍完整保留其方法。它们将有意与 FedAvg对应两列相同，不是独立新增CNN方法证据。是否用这种诊断控制进入总表需作者确定，不能在本次报告中选定最终口径。

如果作者要“LoGo自身Beta/双层后处理 **再加** 共享校准”，现有三视图定义没有可直接接入的单一margin：预测器依赖cohort与Male的局部校准及阈值，不能从最终binary prediction反造continuous score。必须另行定义完整连续score、root/target变换和组合顺序，并冻结验证；本次不建议借完成行数静默实现这一科研改动。

另一个必须避免的旧接口陷阱：core `evaluate_for_reporting`（276）把字符串 `LoGoFair` 路由到AD2式group calibration；这不是当前official DP adapter。不可把新真实method重命名为该旧字符串或简单追加到普通 `fit_views`白名单，否则native可能成为错误的CNN/两组阈值输出。

## 后续最小交付与阻塞

- 先等每方法实际100原strict/offserver闭合，产一个小型来源索引：ID、method/source_method、seed/distribution/alpha/attack、完整原config、root/valid ID和模型SHA、raw job/result/source/component/adapter/hash、strict/rootproof、实际training/eval环境；LoGo另加post_state/mapping/cache/fitseed。全部从实际产物读取，不建立未来值。
- 在旧v4私有schema桥接方式上只实现四CNN标签的严格identity适配，科学margin/root-fit/metrics函数不改。先在各方法一个已接受代表性checkpoint核native1e-12与预测规则；通过再绑定精确100 valid-only范围。该小门检属于未来实现验证，不是本次执行授权。
- LoGo native继续现有summary100；raw/shared引用或组合口径先明确。之后统一输出原统计读取器需要的三view metrics/counts/receipt pointers即可，不复制科学evaluator或900旧数组。
- 10/9/6seed均值±sampleSD(ddof1)，跨10场景必须先每seed平均；三view平行保留负差与全部负结果。选择seed91001/历史validation及test元数据暴露、mixed CPU/GPU/cu128/cu130、方法适配各自披露；不新增显著性/因果主张，不决定主终点，不称final test。

本次仅完成上述来源兼容性审查。没有新增 actual job、预测、fit、接受记录或完整17行声明。
