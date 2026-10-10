# Gradient64 final18 — source preparation only

实际已采用仍为46/64（NGA32、Huber14），父root `e5875e4d…`；根代理提供的21:44:05观察是59终轮，**不是59已验收**。其中NGA32/19恒定、Huber27/24恒定仅是保存元数据观察。本包没有读取模型、计算分数、选recipe、执行SSH或任何新训练。

`PENDING_BINDING.json` 从原64冻结manifest减已采用46，固定剩余18个Huber ID及原job SHA。旧46顺序精确保留；本批不能缩为当前已终轮13项，也不接受未来其他队列。当前actual64快照、选参和root接受字段为null；没有生成实际 `AUTHORIZED_SNAPSHOT.json` 或运行收据。

## 最小源码变化

复用 `tmp/gradient_native_after42_20261011` 的原collect→download→verify→close→finalize及薄root adopter。仅重绑定父46/目标64/精确18、新namespace、归档名与结果计数，另明确修正两处最终批次门：

1. 原工具只接收RUNNING，在64正常结束后会拒绝。现在要求正确服务 `guardfed_celeba_gradient_screen64_v2a EXITED`、supervisor rc3、原queue worker为零、原64 manifest按序全部strict completed、没有QUEUE/job failure；收集前、释放时均检查。不是允许任意EXITED。CPU110所有restricted thread owner检查、nice10/idle I/O/CUDA隐藏、源码/数据、quota/RAM、F卷标签/容量、fresh namespace、失败留存均原样。
2. 原after42 root adopter的“新4全部恒负”仅适用于那一批，已移除该集合级断言。每条仍按原ACC/AEOD/ASPD完整比较报告恒负性，所有负结果保留；不能预设剩余18条恒负，也不能因此改候选、Huber迭代/容差或选参规则。

两份逐ID科学验收循环及保存tensor布局/dtype/finite循环逐字和AST相同。原 `checked_result` SHA `2c5d7699…`、archive verifier `3d78db7e…`、pinned helper `985f04d6…` 保持。最终18模型应各8个tensor、93506元素；此为旧CelebACNN保存结构，不是新性能断言。旧46模型不会重新归档，只有原共享source和本批18模型/结果/诊断。

`SOURCE_DIFF.patch` 是上述五份源码的完整差异；`SOURCE_CHECK.json` 记录7个Python文件、3段内嵌远端源码compile，以及1个正/10个拒收的纯元数据fixture。没有执行原scientific checker或读取arrays。

## 实际64到齐后的现有入口

以下命令**尚未执行**。先由root取得原full-five完整64且自然EXITED的新快照，记录实际路径/SHA。元数据binder只绑定该快照，不启动或验收：

```text
python -B tmp/gradient64_final_acceptance_prepared_20261011/bind_snapshot.py --snapshot ACTUAL64_RAW.json --snapshot-sha256 ACTUAL_SHA
```

binder检查完整64原顺序、每条70轮/非活动/原acceptance/identity、无failure/source drift、guide与共享数据注册；只输出compact SNAPSHOT、AUTHORIZED_SNAPSHOT及SOURCE_FILES封条。大raw仍留原保存位置，不复制。该观察不能代替执行时的原CPU110与source/data/producer门。任何绑定失败保留现场，不能伪填64或循环采样刷门。

root随后审实际绑定并给原review合同：`status=PARENT_AUTHORIZED_FIXED_GRADIENT64_DELTA_AFTER46_CLOSED64`、`root_adoption_performed=false`、实际 `source_files_sha256`、`authorization_sha256` 和精确18 `exact_new_ids`。仅该实际root授权后依次执行各一次：

```text
python -B tmp/gradient64_final_acceptance_prepared_20261011/run_once.py collect --review ACTUAL_ROOT_REVIEW.json --review-sha256 SHA
python -B tmp/gradient64_final_acceptance_prepared_20261011/run_once.py download --review ACTUAL_ROOT_REVIEW.json --review-sha256 SHA
python -B tmp/gradient64_final_acceptance_prepared_20261011/run_once.py verify --review ACTUAL_ROOT_REVIEW.json --review-sha256 SHA
python -B tmp/gradient64_final_acceptance_prepared_20261011/run_once.py close --review ACTUAL_ROOT_REVIEW.json --review-sha256 SHA
python -B tmp/gradient64_final_acceptance_prepared_20261011/finalize.py
```

remote namespace `/workspace/guardfed_checks/gradient64_final_acceptance_prepared_20261011`；本地bulk只 `F:/YananResearchStorage/GuardFed/gradient64_final_acceptance_prepared_20261011/batch`，原Yanan 2TB/Healthy/容量+reserve检查不变。原run入口保留每次命令/exit/stdout/stderr及失败；不自动retry。新root adopter仅由root在真实交付封条/成员核验后运行：

```text
python -B tmp/gradient64_final_acceptance_prepared_20261011/adopt_root_prepared.py --delivery-sha256 ACTUAL_DELIVERY_SHA --handoff-sha256 ACTUAL_HANDOFF_SHA
```

它只写本目录root proof，不修改STATE、LATEST或Git。`screen64_complete=true`仅在原64 strict/offserver全部采用后出现，不意味着选参、14门检或192覆盖完成。

## 完整64后才选择两recipe

使用原 `tmp/celeba_gradient_fullcoverage_prepare_20261010/prepare.py::selected` 与原 `frozen_score.py`，不复制新公式实现。每种方法8候选，各有seed91001的IID/non-IID × Benign/S-DFA四条件。原分数为：

`accuracy − .35×(.45×aeod + .45×aspd + .10×max(aeod,aspd)) − .10×max(0,max(aeod,aspd)−.06)`。

四条件按原排序、原Python算术平均；每方法最大值胜出，完全相等时candidate ID字典序。保留全部64/所有候选，另外展示accuracy冠军/Pareto；这仍n=1，四条件不算四独立seed，不造SD或显著性。Huber恒负不是公平性成功，不排除也不通过改score逼其胜出。

现有 `tmp/gradient64_to_fullcoverage_readiness_20261011/wiring.py summary64` 已处理首1 legacy加后续增量链、原job/result/model/acceptance及F恢复元数据身份。它及freeze/runner仍是原交接明确标注的未独立采用草稿；root应只审这段既有接线，不另造转换框架。实际64采用后才能调用：

```text
python -B tmp/gradient64_to_fullcoverage_readiness_20261011/wiring.py summary64 --native-root ACTUAL64_ROOT.json --native-root-sha256 SHA --out NEW_COMPLETE64_SUMMARY.json
```

该入口只读小JSON、保留原模型SHA，不重新加载模型或跑strict，最后调用原selected。root另采用选参收据 `ROOT_GRADIENT64_COMPLETE_STRICT_OFFSERVER_ADOPTED`，绑定summary、screen seal、64全offserver和两赢家；不能把native64 proof改status冒充。然后原prepare可直接用，无需本包新wrapper：

```text
python -B tmp/celeba_gradient_fullcoverage_prepare_20261010/prepare.py --summary ACTUAL64_SUMMARY.json --summary-sha256 SHA --root-adoption ACTUAL_SELECTED64_ROOT.json --root-sha256 SHA --out NEW_PREPARED_DIR
```

## 14真实三轮门检 → 192新覆盖的依赖

现有source review已通过的包与精确SHA见 `DEPENDENCIES.json`，均不构成实际dispatch：

1. complete64原strict/offserver/root + root选参；原coverage source安装exact组件，协议从PREPARED经独立source审阅冻结，重新计算job/manifest SHA，实际 `ROOT_GRADIENT200_FROZEN_SOURCE_ADOPTED`。
2. 原 `celeba_gradient_fullcoverage_gates_prepare_20261010/metadata.py --bound ACTUAL_FROZEN --stage-approval ACTUAL_SOURCE_ROOT.json --approval-sha256 SHA --out NEW_GATE_STAGE` 生成14独立三轮job。两方法各7：固定non-IID alpha5/seed91002/fulltrain162770/root16277/valid19867；F Flip/FedSA/Sp-DFA的screen与coverage配对共6，加screen Benign1。原旧四个Benign/S-DFA图像门检不重跑。每job原adapter需要单独fresh≤120秒、8CPU、FP32、CUDA隐藏、nice10/idleIO实际许可；原64必须EXITED/零worker/无failure，保护主队列。首次失败即停，不改solver。
3. 原 `adapter.py --stage STAGE --repo /workspace/GuardFed-celeba-expanded --job EXACT_ID --approval FRESH_ROOT_JOB_APPROVAL.json --approval-sha256 SHA` 顺序执行14；原 `compare_saved.py --stage STAGE --repo /workspace/GuardFed-celeba-expanded` 比较六对tensor/指标/梯度/RNG及两组F Flip metadata-null。F离机成员检查/root采用后方有 `ROOT_GRADIENT14_SAME_HORIZON_STRICT_OFFSERVER_ADOPTED`。门检不是论文表记录，也不证明GPU/CPU数值等价。
4. 原 `gradient64_to_fullcoverage_readiness_20261011/run96_queue.py` 与 `coverage_contract.py` 需要额外局部source/runtime审阅：两个96新＋4引用，共192新/8引用；单worker/公共锁，不能用旧64硬编码runner。实际source approval、14root及原resource proof的三个SHA入口均已预留。当前没有冻结bound source、14实际job/结果、资源凭据或192运行许可。本任务不启动它们。

原unweighted CE下Male-only F Flip不改标签/输入，null不是输入攻击鲁棒性；root reference仅供攻击者。Huber恒等投影为已获作者同意的CNN经验适配，不继承原理论。全部70轮完成并严格验收后再用原summary输出10/9/6面板、ddof1、先seed内场景平均；当前不生成这些未来统计。最终test始终未授权/未执行。
