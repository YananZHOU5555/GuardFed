# Gradient fullcoverage preparation: independent source review

结论：**PASS，仅限准备源码可采用，不授权启动或统计接受。** 未发现需要修改被审包的实质错误。

输入仅限 `tmp/celeba_gradient_fullcoverage_prepare_20261010` 的10个封存成员、其 SOURCE_PINS 指向的原64准备工具与作者决策，以及原 protocol 绑定的 core 中攻击路由段落。逐成员核实际字节、大小和SHA；候选封条为 `ea5a565d35c4f8df7d108679d1010baec92993da8f4b181a14ccde359334ed39`。

使用纯stdlib在内存中展开 `prepare.render()`，按AST函数源码区间（包含decorator）对照原worker：14个科学/辅助函数逐字一致，只有 `validate_job` 的覆盖身份/seed/reuse边界和 `run` 的stage/status标签改变。展开差异与封存 SOURCE_DIFF 一致。checker新增五场景身份与client audit守卫；对照原core的 `attack_types_for_client` / `client_runtime_data`，FedSA的foe_mode仍由原core设为fedsa，未改变攻击或梯度科学定义。

选参检查入口为 `prepare.py:24` 的 `selected()`：必须exact64与原job/source身份及strict/offserver接受证据；逐候选采用原hash-pinned score对四条件求均值，再按负分数/候选名字排序。`grid()` 在 `prepare.py:81` 生成每方法96新任务，并显式引用四个原seed91001 Benign/S-DFA任务；两方法共192新+8复用。未从未齐64结果选择recipe。

统计范围检查入口为 `summary.py:14` 的 `stats()`：10/9/6固定seed面板，全指标同seed；先在每seed内平均十场景，再对seed求mean和sampleSD（ddof=1）。只读源码核对，未执行真实统计。后续summarize还须单独的实际冻结源批准和192新+8旧身份闭合；它本身不能替代独立离机接受链。

仅额外执行一项纯metadata正向（FedSA seed91010，显式关闭冻结要求作身份fixture）和三项拒收（默认PREPARED不能launch、原seed91001复用项不能重跑、seed91011越界）。未重跑作者的192任务/16拒收/64 tie/200 synthetic统计覆盖；这些作者fixture只按封存来源读取。审查进程未导入Torch或NumPy，未读取数组、运行CNN、SSH、生成任务/模型/结果或修改共享文件。

实际 HANDOFF 仍为PREPARED、0 jobs/0 results/recipe null；INPUT_TEMPLATE未来hash均null。训练前还需完整64实际root采用、冻结选定源/协议、原component部署、新seed/stage和三种新增攻击接口的真实图像门检，以及root资源核验与派发。作者H/L已批准的方向不重新提问。负结果、null攻击效应、validation选择、Huber经验CNN适配和未来独立test边界均保留。

具体检查结果、原文件pins与限制见 REVIEW.json。此次无被审源码修改。
