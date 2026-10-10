# FL native67 → fixed FedSA10: source preparation only

固定范围仅原96任务manifest中的 non-IID / FedSA / seed91001–91010，十个完整ID及原job SHA见 `FROZEN_SCOPE.json`。父root为 `a11cd9ed94ab136b49dd41975d45f684d88236b6918c295461089a98adbc7db0`，实际离机父链为 `21f9b5d3cb07405d4100d28c83b1ce578e27fdb24382f9631d28eb3c7029f637`。父67顺序及链原字节保留。若未来全部验收并由root采用，新增native累计为77/96，另4份screen复用单列；这不是三视图81或完整80场景表。

本包未连接服务器、未绑定终轮快照、未收集/下载/运行strict或tensor checker、未加载模型/数组，也未采用任何结果。`PENDING_BINDING.json` 中未来快照及执行review仍为空；**不存在 `AUTHORIZED_SNAPSHOT.json`**，当前不能执行流程。不要将源码门检PASS写成科学接受。

复用父after63的直接流程，只有固定IDs、父链/计数、独立namespace和CPU109→110资源元数据变化。原collector逐ID科学loop、strict/archive尾部、原离机checker及根adopter逐record循环保持原字节/AST。CPU110单线程、nice10、idleIO、CUDA隐藏；原all-thread窄mask占用检查、guide/source/data/package校验、F卷label/Healthy/容量及只写新F目录守卫保留。原preflight/close仍要求原service RUNNING、1–2 worker，因此若未来service已经EXITED，应停止并交root审查实际工程差异，不在此预放宽。负结果不筛选，不选recipe、不推理/拟合/训练/test，不重包旧67或screen4。

未来由root一次实际观察并晚绑定：十ID均70轮终态、不在active、无failure且旧67仍在terminal集合；source/data/package一致。root创建真实 `AUTHORIZED_SNAPSHOT.json`（status为 `ROOT_AUTHORIZED_FIXED_EXACT10_AFTER_NATIVE67_NOT_ACCEPTANCE`），填写实际 `snapshot_path/sha256/utc/terminal_count`，保留 `PENDING_BINDING.json` 中的prior/IDs字段。终轮总数允许≥77，但collector只收固定十项，后续其他terminal排除。root执行review必须绑定 `source_adoptable=true`、`exact_selected_ids`、本包 `PREPARED_FILES_SHA256.json` SHA及真实 `authorization_sha256`。共同LATEST若已前进，原守卫拒绝，需要重新审查绑定，不能补写旧链。

根审核实际绑定后可顺序执行以下原入口（此处只是命令模板，未执行）：

```powershell
python -B tmp/fl_native_after67_FedSA10_prepared_20261011/run_once.py collect --review ROOT_EXECUTION_REVIEW.json --review-sha256 ACTUAL_REVIEW_SHA256
python -B tmp/fl_native_after67_FedSA10_prepared_20261011/run_once.py verify --review ROOT_EXECUTION_REVIEW.json --review-sha256 ACTUAL_REVIEW_SHA256
python -B tmp/fl_native_after67_FedSA10_prepared_20261011/run_once.py finalize --review ROOT_EXECUTION_REVIEW.json --review-sha256 ACTUAL_REVIEW_SHA256
```

`verify` 沿原顺序下载新archive/receipt至 `F:/YananResearchStorage/GuardFed/fl_native_after67_FedSA10_prepared_20261011/batch`，执行原离机member/record和保存tensor检查，随后close；不重跑成功server strict。远端独立目录为 `/workspace/guardfed_checks/fl_native_after67_FedSA10_prepared_20261011`。任何错误保留、停下，无自动重试。未来finalize写 `ACTUAL_README.md`，避免覆盖本准备说明。

实际strict/离机交付完成后，只有root可运行准备的根接纳脚本：

```powershell
python -B tmp/fl_native_after67_FedSA10_prepared_20261011/root_adopt.py --handoff-sha256 ACTUAL_HANDOFF_SHA256 --delivery-seal-sha256 ACTUAL_DELIVERY_SEAL_SHA256 --delivery-files ACTUAL_MEMBER_COUNT
```

根adopter沿原语义核archive/member/同checkpoint/70轮/valid162770及19867、原接受器来源和父67，再原子更新共享LATEST。源码中107archive members、80tensor/935060elements是按已接受原布局缩放的**拒收守卫**，并非本次实测。新输出hash完全待实测。原server cu128与本机CPU Torch验证环境差异仍披露；没有预测数组，不声称数组复算。首个本地生成器的重叠替换逆恢复断言失败保留于 `PREPARATION_FAILURE.json`，发生在新源写入及任何科学执行之前。
