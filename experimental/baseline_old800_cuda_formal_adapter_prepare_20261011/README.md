# Original800 formal CUDA API — source-only

`adapter.bind_old800(v4, args, exactcohortpin, actualapprovalpin, resource_gate)` 返回 `replay_one(wall_seconds)` 和 `accept_saved_predictions()`。本包没有运行队列、收集器或实际授权；不改变原15成员基础包，也不接触独立的500阶段。

范围严格为原900 inventory按原顺序排除GuardFed-AD2+的800个完整对象，八方法各100。Full100复用机制阶段同checkpoint的新CUDA结果，不能重复评或双计。FL100和LoGo100不在此API。FedAA显示别名不能传给科学函数：原`record.method='FedAA'`/`source_method='FedAA-DDPG-adapted-v1'`和LASA/LASA-official的转换继续完全使用原`v4.mapped_functions/normalized_views`。

GPU body逐字复用已封old800基础版本，仅原四个receipt标签与诊断版不同；原CUDA FP32提取、root-only原拟合、margin>0/native/raw与shared>=规则、native1e-12、保存数组及原saved科学块不变。`accept_saved_predictions`会按原规则重新拟合root校验保存结果，不能叫零拟合。原训练70轮、train162770/valid19867、788 cu128+12 cu130身份保留。此统一重评是独立新证据，不覆盖既有CPU或混合路径表，不追认单例GPU诊断，不读取final test。

## 根必须晚绑定的真实输入

`SOURCE_PINS.json`中的本地路径只是来源索引，不能作为Linux路径使用。根需冻结`EXACT_COHORT.template.json`的完整800对象和顺序，填真实逐record `ROOT_APPROVAL.template.json`；当前模板status无效、execution_authorized=false、所有运行路径/SHA为null。actual author B已经存在，仍须绑定其Linux实际副本。

`runtime_dependencies`必须给原v2/v3/v4/evaluator/semantic900的实际Linux路径与SHA，API同时校验模块`__file__`和原布局，不只校验同内容。`args.command='worker'`，不能用inspect跳过semantic门。API用原`v3.private`给原`v4.read_inputs`及`source_pins`私有EXTRA；原代码对象和共享v4.EXTRA保持不动。原900 storage/restore/semantic完整门均执行，semantic900历史`source_before`的路径/stat token仍须当前真实一致；本地旧receipt存在不代表服务器门已通过。

args字段为`repo, command, inventory, storage_map, storage_map_sha256, restore_receipt, restore_receipt_sha256, restore_acceptance, restore_acceptance_sha256`。真实路径必须保留原v4 BASE/v3和BASE/replay.py布局、原storage targets与原repo源码/cache；协议及所有record.source_hashes、adapter hashes仍由原source_pins核验。API调用时会读原checkpoint/result/job、加载模型/图像并做科学计算；当前准备未调用。

根另提供经审阅的`resource_gate(snapshot, approval)`实际源路径/SHA和90秒内真实Linux preflight：guide/source/data fullhash、原semantic tokens、无producer/重复helper/诊断/common-lock owner、单CPU全线程与SMT策略、主队列真实进度/errors/OOM/protected services、cgroup-v2 quota和nominal+1预算、RAM/disk、GPU UUID/free memory/Recovery。API调用前检查preflight，原body的before/after snapshot再次调用该原资源门；全hash耗时后必须仍取得fresh事实。500现有guard硬编码CPU107/其namespace，不能直接假装适用于old800；此处仅晚绑定根审后的实际guard，不另写资源框架。

## 最小单worker示例（未执行）

`worker_call_example.py`照原500 bootstrap顺序只加载一次`replay`，原导入设置interop1/CUDAhidden后，再绑定根给的GPU UUID；然后依次注册同一`replay_v3`和v4。1CPU/threads1、FP32、deterministic与TF32关闭，不重复设置interop。根应由原supervisor/taskset/nice10/idleIO启动一次，模板不是命令授权：

```text
taskset -c ROOT_BOUND_CPU ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B ROOT_DEPLOY_DIR/worker_call_example.py --approval ROOT_ACTUAL_PER_RECORD_APPROVAL.json --approval-sha256 ROOT_ACTUAL_SHA --wall-seconds 1800
```

根审源后先固定三个canary：FedAvg/FedAA/LASA的IID Benign seed91001（精确原IDs见SOURCE_CHECK的positive_fixture_ids，仅metadata fixture）；各自真实逐项授权/cohort/资源门需根冻结。仅原native与saved whole实际通过才可作为新正式证据；根决定是否顺序推进冻结800。此例只跑一条、无自动重试/设备fallback/自动发现，任何失败保留输出和原trace。科学产物只服务器或经fresh F盘Yanan2TB容量检查后最小离机；当前未写F。

## 已完成静态检查与限制

SOURCE_CHECK仅compile、父Full适配器23跨度raw逆恢复、GPU raw+AST/保存科学块逐字同基础包、三条纯metadata正例和五条拒收。没有import Torch、bind/replay/accept、forward/fit或模型/数组读取。首轮源码派生顺序错误保存在PREPARE_FAILURE与原preparer副本，发生于写adapter/科学输出之前；只修metadata替换顺序，无科学失败。

旧900语义门本身仍验证全部900身份，不等于重跑900推理。FL需要自己的原strict/whole route；LoGo新CUDA root/valid之后仍原CPU netcal1.3.6、fitseed1719和20虚拟cohort，其旧native是原CPU postfit cache，不能套训练GPU native0diff保证。二者明确尚未接通。不存在新的完整800/1000结果、论文表或正式cohort启动。
