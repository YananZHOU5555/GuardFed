# 14项梯度新攻击门检：执行依赖独立只读判断

**结论：沿用当前封存包、且不改既定协议时，必须先完成64项严格离机/root采用及每方法选参，再冻结选中coverage源；原64队列还必须真实退出。当前不能把源码fixture中的“代表配置”直接作为这14项实际门检运行。** 这属于现有包明确的身份与执行合同，不是“科学上任何短程检查都必须等64”的普遍命题。

本轮只读源码/协议/既有review并核紧凑源封条，未SSH、未导入科学模块、未执行check_prepared/门检/拟合/推理/训练，未改原封存文件。父代理提供的17:25观测是gradient44终轮，不是本次独立实测，也不等于64项严格采用；本轮没有增加科学结果。

## 决定性依赖

1. `tmp/celeba_gradient_fullcoverage_gates_prepare_20261010/README.md:7,15` 明定使用“actual complete-64, root-adopted selected recipe”，而非任意第一候选。`metadata.py:51–55` 要求root64状态、accepted_count=64、all64_offserver_verified、summary SHA，调用原coverage选择器重算，并要求selected/root64.selected_candidates/BOUND_INPUTS.winners完全一致。
2. `tmp/celeba_gradient_fullcoverage_prepare_20261010/prepare.py:24–52` 只接受全部原64个ID、job/source/component/checkpoint身份与strict/offserver flags；每候选四条件冻结分数平均，按每方法最高分、完全并列按候选ID词典序选取。`bind:96–105` 再核独立complete64根采用；不能用best-so-far、当前44或已完整Fed-NGA32替代该二方法合同。
3. `metadata.py:46–47,57–73` 还要求实际`ROOT_GRADIENT200_FROZEN_SOURCE_ADOPTED`、bound_inputs SHA、两份精确96新+4复用manifest，且协议严格等于选中候选的`FROZEN`版本。coverage binder产出的`PREPARED_NOT_FROZEN`（prepare.py:115,126）不能直接执行；冻结后须刷新协议与job-local hashes并核组件原SHA。
4. `adapter.py:60–68` 要求每child新鲜<=120秒实测凭据、8个无restricted-affinity冲突CPU、quota/RAM、主机制健康与增长、无重复worker，且`old_gradient64_exited=True`。这项是运行硬门，与是否CPU看起来空闲独立。既有review明确将`old64_alive`列为拒收。
5. `check_prepared.py:25–26` 取每方法第一候选生成的14条只是metadata fixtures；其HANDOFF保留selected_recipe=null、real_jobs_bound=0、real_image_gates_run=0、execution_authorized=false。`jobs_for()`可构造配置不等于有实际可执行scope/selected配置。

## 已有门检究竟验证什么

两方法各7项：F Flip/FedSA/Sp-DFA各screen与coverage配对，再加Benign screen参考；总14任务、各3轮、合计42轮，non-IID alpha5、seed91002、CPU严格FP32/8线程/单fresh进程。这里“共同三轮”指相同短程比较条件，不指事先已有独立于选参的可运行代表配置。

比较6对screen/coverage和2对F Flip/Benign null：终轮张量、三轮指标/诊断、梯度audit、native replay与声明的Python/NumPy-global/Torch-CPU RNG。F Flip是unweighted CE下Male元数据改变而Smiling和像素不变的null，不能称真实输入/标签攻击鲁棒性。FedSA/Sp-DFA沿原FOE攻击语义，Huber恒等投影及收敛规则不变。成功也仅有接口证据、正式表记录0；不证明CPU/GPU等价、70轮行为或授权192项训练。

coverage README中“短程adapter尚未在此包实现”是该早期coverage包自身范围；后续gates包已补齐源码并通过源码审查。因此当前阻碍不是缺这份adapter源码，而是实际64/选参、冻结绑定与运行凭据未满足。两个review都只采用源码准备，未授权或记录实际执行。

## 最小可执行下一步

现在继续原64搜索及有价值增量严格离机采用，保留恒负与失败；不要提前停队列、从部分结果选recipe或为跑门检移除old64退出门。可并行核对最终64输出到既有summary输入schema的最小身份映射，但不填造未完成字段，也不提前跑选择器。

原64全部完成并采用后，沿现有入口依次执行：

1. 形成实际64 summary及独立根采用（含原冻结规则选中的每方法候选），运行原`prepare.py --summary <actual64> --summary-sha256 <sha> --root-adoption <actual-root64> --root-sha256 <sha> --out <new-prepared-dir>`，仅生成两份96+4准备源。
2. 核原组件部署、选中协议独立冻结、刷新job/manifest SHA，并生成真实`ROOT_GRADIENT200_FROZEN_SOURCE_ADOPTED`。这一source approval不允许启动192。
3. 运行原`metadata.py --bound <actual-frozen-dir> --stage-approval <actual-approval> --approval-sha256 <sha> --out <fresh14-stage>`；root核实际14 ID/顺序/source。不能把本报告占位命令当实际绑定。
4. 确认原64服务真实退出；每项实际测量资源，给原adapter新鲜精确job/scope凭据，按绑定顺序单fresh子进程、8CPU、nice10/idleIO/CUDA隐藏执行；任一失败保留并停，不循环重试。
5. 全14完成后原`compare_saved.py`只读保存产物比较，严格离机/member SHA并root采用；本机bulk仅在fresh F=Yanan 2TB/Healthy/容量检查后写。通过后仍需单独完成192运行层绑定与实际批准，不自动派发。

若要在64结束前先验证攻击接口，必须明确另设有限探索scope/代表候选、独立namespace和review，并调整当前selected/frozen192/old64_exit绑定；那是新的门检合同，不是“现有14包不改协议即可运行”。本次没有准备或执行该替代方案。

## 本轮只读核验身份

- gates封条 `f068ad7f51fd6981b2211725d39009a5cb5ebbecfa1de680cf0d46aa52ef7760`：8成员SHA/字节匹配。
- coverage封条 `ea5a565d35c4f8df7d108679d1010baec92993da8f4b181a14ccde359334ed39`：10成员SHA/字节匹配。
- gates review：`tmp/celeba_gradient_fullcoverage_gates_review_20261010/REVIEW.json`，状态`INDEPENDENT_SOURCE_REVIEW_PASS_NOT_RUNTIME_OR_DISPATCH_APPROVAL`。
- coverage review：`tmp/celeba_gradient_fullcoverage_prepare_review_20261010/REVIEW.json`，状态`PASS_SOURCE_PREPARATION_ONLY_NO_EXECUTION_APPROVAL`；STATE所引SHA `792da228c102a7044628aa30f2fc6310bb486441c822d321ab62a2ee78843d09`。

实际读review SHA：`tmp/celeba_gradient_fullcoverage_gates_review_20261010/REVIEW.json` = `47291c77f49777d64a1949ce09fef0e57def822ddecf741be883860f4a8abb68`。

实际读review SHA：`tmp/celeba_gradient_fullcoverage_prepare_review_20261010/REVIEW.json` = `792da228c102a7044628aa30f2fc6310bb486441c822d321ab62a2ee78843d09`。

记录时间：2026-10-10T17:28:10.397154+00:00
