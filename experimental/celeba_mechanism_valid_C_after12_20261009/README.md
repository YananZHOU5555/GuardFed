# C_after12：exact8 valid三视图准备包

本包仅准备`minus_C_IID_F Flip_seed91003`至`seed91010`八项。输入固定于native120的`root_delta_20261009T195854Z`严格/离机快照，减去已root接受的旧三视图112（U100+C12）。不纳入后续观测或尚未接受的C FedSA。当前包状态始终`PREPARED_NOT_APPROVED`，新三视图接受数0；旧112及Full100引用保持不变。

`NATIVE_INPUTS.json`绑定actual inspection、94成员native归档/receipt/offserver proof、17段/120唯一ID ledger以及两份root审查；`INPUT_PINS.json`保存实际读取来源SHA。新增八项checkpoint/result/rawjob/config/source/data/root/client分区来自真实原归档，不填未来模型。`inventory_actual120_Full100refs.json`保留旧112记录逐JSON一致、Full100引用逐JSON一致；新增记录均为原70-round完整valid19867、train162770/root16277，同seed/scene Full配对，native容差固定`1e-12`。

复用已执行C_after1科学桥。11个未改函数逐源码相同，其中`bind_runtime`包含原`replay_one`和`accept_saved_predictions`科学体；`validate_inventory`只调整120/680、精确集合和112/8边界，`require_approval`唯一计算无关改动是审批基数11→8。原冻结core/evaluator/v2/v3/evidence及variant检查、三视图预测/阈值/混淆矩阵/同checkpoint三指标规则不变，Full只引用九方法900实际接受来源，不推理或打包Full权重。

执行候选沿用原batch/install/backup/verify入口及新独占命名空间`/workspace/guardfed_checks/celeba_mechanism_valid_C_after12_20261009`。服务名`guardfed_celeba_mechanism_valid_C_after12`；CPU112–119、单进程八Torch线程、nice10/idleIO、CUDA隐藏；序列fresh子进程、失败保留且无自动重试。installer须实际确认旧`guardfed_celeba_mechanism_valid_C_after1`为EXITED且无旧worker，当前CPU无restricted owner、guide/资源/所有科学来源和八项artifact SHA通过，才按root外部审批安装。这里没有测量Linux资源可用事实。原installer的显示字段`PASS_EXACT37_PREFLIGHT`沿袭旧实现；实际选择/审批/计数门均为exact8，不把该历史字段解释为37项范围。

`ROOT_REVIEW_TEMPLATE.json`和`APPROVED_TEMPLATE.json`都未批准。root须独立审阅当前science/execution seal，并新建`ROOT_APPROVED.json`和外部SHA绑定的`EXECUTION_DRAFT.json`；后者必须绑定新的root批准SHA、当前execution seal和八项精确ID。继承的`SOURCE_REVIEW_SHA=b1ff1…`仅指已批准科学谱系，不替代当前外部执行批准。禁止复用旧服务/输出或将模板视为权限。

`SELF_CHECK.json`是本机metadata门检：库存120正向通过，八项有效approval和八个实际worker调用链到达科学bind前的sentinel；67项错误计数、duplicate/foreign/旧112/Full、variant/mask/source/model/split/seed/alpha/tolerance、审批/运行绑定漂移拒收。外部审批只存在测试内存，Linux runtime、文件路径和锁被明确模拟；没有Torch/NumPy导入、CNN、训练、SSH、实际派发或接受登记。旧完整函数与resource模块逐源码/字节相同。

首次prepare在父库存SHA门前因本地文本适配误改哈希数字停止，原stdout/stderr与`PREPARE_ATTEMPT1_FAILURE.json`保留；精确恢复已接受父SHA后第二次构建及门检通过，没有放宽守卫、修改旧包或产生评价结果。

将来原backup只归档producer已闭合的新数组/receipt/approval/log，模型不重包；原verify用固定valid cache重算每项9指标/24计数/3预测规则。全八项预期72指标/192计数/24规则，实际结果须按原strict接受、离机复核再由root登记。本包不宣称完成这些检查。评价仍valid-only；原冻结loader可materialize所有split标签元数据，不进行test图像推理/拟合/选择。混合CPU/GPU重放历史、训练环境/selection91001与valid曝光限制继续保留，正式primary endpoint仍pending，C其他8场景在该快照未完整。
