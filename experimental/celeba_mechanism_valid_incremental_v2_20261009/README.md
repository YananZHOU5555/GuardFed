# 机制终轮三视图增量 v2：23条真实库存，准备剩余15条

**PREPARED_NOT_APPROVED / 未部署、未启动。** 本目录只准备下一批15个已接受且离机的minus_U终轮checkpoint的valid三视图重放，没有创建服务、派发器或监督机制，没有训练、图像推理、Full权重复制。原8库存、原bridge、single/remaining7、原科学v2/v3/evaluator和所有封条均不改。

## 实际分母与身份

最新固定inspection是`mechanism_inspection_v4_20261009T092000Z/inspection.json`，SHA `68309b3375f89d03f5699e089ad338c72d8f9b56422a9b76f2c20f3f9657394f`。五份科学终轮增量链共334归档成员已重核SHA/大小/前向receipt链，形成23个真实终轮库存；100Full只引用原baseline库存，不重包。其余777项只保留待完成ID，没有checkpoint字段或假SHA。

原single1与remaining7的两个离机归档共88成员已再次核验，8项strict acceptance与上述真实checkpoint逐项一致；因此23减8，精确剩余15项，顺序沿原manifest：

| 条件 | seed | 数量 |
|---|---|---:|
| minus_U / IID / Benign | 91009–91010 | 2 |
| minus_U / IID / F Flip | 91001–91010 | 10 |
| minus_U / IID / FedSA | 91001–91003 | 3 |

完整ID在`SELECTED_15.txt`，完整23库存及100引用在`inventory_actual23_Full100refs.json`；下一15个checkpoint/result/raw-job SHA、原始路径、同终轮config/source/data/adapter、配对Full身份与输出提案在`SCOPE.json`。15个配对Full三视图显示为MISSING，表示本包没有加入其外部SHA绑定的三视图验收；不以旧校准结果替代，不为本任务推理或打包Full。以后仅能join严格接受的baseline三视图receipt。

`inputs/verified_ledger_23.json`是实际5链ledger的固定副本；主目录的滚动ledger后续增长不改变本次23条证据。所有归档、inspection和原科学源均由SHA固定。准备只验证已接受JSON/归档身份，不把此过程称为新的科学推理验收。

## 最小改动与复用证明

`MINIMAL_SOURCE_DIFF.patch`直接对比原8的bridge/prepare。新bridge只变scope、23库存/777待完成身份、已闭合8与待重放15白名单、CPU112–119批准范围；当前明确只接受实际minus_U，不实现或推断其他variant语义。

原`bind_runtime`完整函数的源文本与AST逐位未改，其中包含原`replay`和`accept_saved_predictions`闭包；`full_reference`、`reference_baseline_full`也未改。16个原科学函数的source/AST SHA见`reuse_function_hashes.json`，原13成员准备封条已重核，见`source_reuse_proof.json`。

仍使用原v2 `replay_one/metadata/rebuild_root/check_native`、原v3 private/hash/token工具及原evaluator的root-only阈值拟合、raw/native/shared预测和全部指标。native容差固定1e-12；same checkpoint、70轮valid19867、train162770/root16277、原划分及配置、mask/adapter/源/数据身份、有限数值与保存数组复核不放宽。科学body不重新编写。原v4 `accept_new`与原机制worker的严格接受仍在运行时再次执行。

`closed_eight.py`仅核原single/7两条离机恢复链及strict checkpoint身份，确保不会重放已闭合8项。`selfcheck.json`记录36项拒收边界：旧8重入、pending/Full、重复/混checkpoint、源或root/valid/config/seed变化、未批准/GPU/test/非112–119核、乱序、错误variant与native容差等。实际封存check_native函数体的1e-13/1e-11边界通过；这些是身份/标量测试，不是新图像证据，无Torch/Numpy导入。

## 可执行接口与下一次派发建议

没有新增runner框架。root审阅本包后，可在已有前台协调器中沿`SCOPE.selected_ids`依次开一个fresh child，每次仅一个8线程CPU进程；从原服务模式派发，nice10、idle IO，CPU112–119。每个child用单独、全新的`SCOPE.outputs[id]`，不复用单进程跑15项，不隐式加载partial。所有依赖路径沿用已验证的七项路径并逐SHA核；`SCOPE.dependency_paths`明确列出。

`APPROVAL_TEMPLATE.json`状态故意为未批准，直接调用会拒绝。root审查后须为每个child提供外部SHA绑定批准收据：status改为`APPROVED_BOUNDED_MECHANISM_VALID_REPLAY_ONLY`，scope、inventory SHA、bridge SHA、完整15 ID顺序、CPU112–119/max_processes1、valid/native1e-12均固定；output设为该child的唯一SCOPE路径。再调用：

```python
runtime = bridge.bind_runtime(inventory_path, inventory_sha, dependency_paths,
                              repo, model_id, approval_path, approval_sha)
runtime['replay_one'](approved_output, wall_seconds=1800)
accepted = runtime['accept_saved_predictions'](approved_output)
```

原bridge在调用时核Linux/cu128、CPU affinity、授权及完整source/data/artifact前后身份；shared/raw/native校准只使用train-root。当前没有产生有效批准收据，也没有调用bind_runtime。失败保存独立bridge_failure和partial，不循环重试、不改native容差。完成项分别严格接受并增量离机SHA闭合，原8只引用不重跑。

## 资源提案来自实测，但不是资源预留

2026-10-09T10:19:26Z只读快照：baseline CPU重放11×8=88计算线程（CPU16–103），Hybrid1×8=8（CPU8–15），主GPU训练8×1=8，FLGMM2×1=2，共106；原remaining7服务已EXITED。拟新增本批单进程8线程后为114，低于cgroup122.87999核配额。两秒实际CPU68.66核；控制器/辅助线程数量不按计算核重复计数。明细见`resource_snapshot.json`、`RESOURCE_PROPOSAL.json`。

CPU112–119没有被这些显式CPU重放worker占用；GPU worker保留其原affinity，不宣称该范围对所有宿主线程完全独占。root正式派发前仍须重新核实际任务、配额/affinity/内存及无重复输出，不能把此快照当未来资源保证。首次只读观察因`supervisorctl status`对已有STOPPED服务返回3而未写快照，修正观察器后成功；错误记录单独保留，训练未受影响。

## 解释边界

这是下一15个已训练checkpoint的valid-only实现重放，不是未来全部8种variant的执行授权，不涉及正式最终test，也不完成机制900全表。Full100原训练有98cu128+2cu130历史混合；本23个机制终轮cu128，仍须实际重放native一致，不假设CPU/GPU等价。原训练loader曾物化全split标签元数据；重放沿原prefix读取只解释train+valid，不做test像素/标签推理、拟合或选参，不声称项目从未触碰test。
