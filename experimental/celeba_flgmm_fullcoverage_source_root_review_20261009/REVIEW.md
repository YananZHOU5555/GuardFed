# FLGMM100源码候选：两项P2需修正

候选将原32搜索所得的一个外部批准recipe扩展为96个新70-round valid任务，并复用4个原接受记录；先执行两个原worker参考与五个新worker三轮canary。本次仅审查源码/metadata/AST，未选择recipe、绑定stage、执行canary或接触服务器。

1. **P2：失败退出被计入completed_new。** [run_fullcoverage.py:93](E:/OneDrive/文档/GuardFed/tmp/celeba_flgmm_fullcoverage_source_20261009/run_fullcoverage.py:93)用`96−pending−active`计数；失败child在82行同样从active移除。执行原`run`函数的惰性控制流fixture后，一项失败、一项严格成功被报告为completed_new=2，实际成功只有1。失败仍停止派发、等待peer退出并阻止summary，因此没有把失败纳入科学表；缺陷影响进度与交接语义。最小修复是初始化独立成功计数为已严格skip的数量，仅在exit0且原inspect接受后递增，失败不增。保留failed/pending及原failstop/drain规则。

2. **P2：优化模式会关闭同horizon比较，现有拒收也会消失。** [run_canaries.py:52](E:/OneDrive/文档/GuardFed/tmp/celeba_flgmm_fullcoverage_source_20261009/run_canaries.py:52)及后续model/RNG比较均为assert；`python -O`或`PYTHONOPTIMIZE=1`会移除这些表达式，仍可到达GATE_ACCEPTANCE的PASS写入。[bind_stage.py:61](E:/OneDrive/文档/GuardFed/tmp/celeba_flgmm_fullcoverage_source_20261009/bind_stage.py:61)试图以`assert not sys.flags.optimize`拒收，但该assert自身也会消失。对实际assert节点的纯AST探针表明：同一故意不等metrics fixture在optimize0拒收、optimize1不拒收。没有观察服务器启用优化模式；默认`python -B`路径不受此条件影响。最小修复是在bind_stage与新screen_common入口以显式`if sys.flags.optimize: raise ...`拒收，避免改动科学计算或原旧release。

除这两点外，本次未发现其他实质源码缺陷。该结论只覆盖已读调用链及以下有限检查，不证明GPU数值等价或实际运行成功：

- 23项候选封条全部通过且before/after不变；原`digest/write_json/aggregation_wrapper/load_core`逐函数源码一致。runtime/RNG设置至原core调用结束的整块字节一致；adapter、三份作者源/许可及已接受v3 RNG recorder字节一致。
- 以一个明确标为惰性metadata fixture的原声明recipe核96新任务+4复用cell=100唯一cell，五个non-IID/seed91001三轮canary。fixture不代表已选择recipe。四个实际旧job的SHA、source/adapter与70-round身份核对；未来实际选中四项仍由原legacy_check独立进程逐结果/model SHA复核。
- 原参考worker只改四个声明的horizon常量；实际旧validator在三轮job上通过。与新三轮Benign配置仅experiment_suite/tag不同；core仅将这两项用于run_id，没有改seed、数据或训练体。该源码比较不能代替真实同horizon model/metrics/controller/RNG比较。Tg10/20三轮尚未覆盖后续UCL/monitor，候选README已准确披露。
- 严格接受器保留完整五artifact SHA集合、原job/config/source、valid162770/19867、完整70或3轮、同checkpoint三指标、controller/history/diagnostics身份检查。`inspect`对existing partial且无terminal结果拒收；`run_one`对任意existing输出在runtime/worker调用前拒收。两条原函数AST探针通过，没有执行接受器或CNN。
- 失败child后不再派发；peer正常结束后仍failstop且不调用summary。另一个真实`run`控制流fixture在两peer活动时触发身份错误，两个peer均被wait/drain，无后续派发。以上均为惰性process/path对象，没有创建OS子进程。
- 原4 legacy在独立Python进程导入原release checker，避免新旧模块共享；同三轮比较使用相同core/horizon、同边界v3 recorder。三项新攻击只构成原terminal checker下的pipeline canary，候选没有把有限值当作强攻击证据。
- 资源入口要求外部package/scope授权、初始≤120秒资源收据、main8保护、CPU预算/RAM与两卡健康；每child使用一张可见GPU/单线程/nice10，coverage最多两child。后续worker绑定原启动资源收据而非重新实时测量，这是候选README已披露的边界；本次未进行Linux资源或guide实测。

证据见[CHECKS.json](CHECKS.json)、[OPTIMIZATION_PROBE.json](OPTIMIZATION_PROBE.json)、[PARTIAL_REFUSAL_PROBE.json](PARTIAL_REFUSAL_PROBE.json)。`audit_source.py`只抽取原函数并注入惰性依赖，不能用作运行/科学接受工具。优化探针第一次遗漏assert消息中的item fixture而NameError，已保留PROBE_ATTEMPT1_FAILURE.json；补齐惰性上下文后原assert探针通过，没有修改候选源码。

结论：先以独立v2修正这两项工程行为，再由root绑定实际32来源/外部recipe批准并进行真实canary。原23成员包保持封存。本次没有SSH、Torch/NumPy科学导入、训练、推理、实际子进程、STATE/canonical/Git写入，也未触及C表目录。
