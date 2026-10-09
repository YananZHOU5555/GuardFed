# 单模型CPU native失配：独立GPU诊断准备

**PREPARED_NOT_AUTHORIZED。没有部署、GPU/CNN执行、训练或派发。** 本包仅针对`FairGuard_IID_F-Flip_seed91009`。原chunk036失败和1e-12门限保留，不将任何chunk036 partial结果混入424已接受集合，不恢复原872队列。

## 已测事实与问题

| 来源 | ACC | AEOD | ASPD |
|---|---:|---:|---:|
| 原终轮记录 | 0.9040116776564152 | 0.04282574048724552 | 0.11859776792562177 |
| 已保存CPU重放 | 0.9039613429304878 | 0.04311601624921935 | 0.11871599918596698 |

CPU准确率净少1个正确预测，但不能仅凭三个汇总数确定哪些预测变了。原checkpoint/result/job/valid-ID及权重前后已核一致；原CPU数组仍属于失败诊断，native不满足1e-12。CPU最小绝对margin约2.61e-7只是近边界候选，**没有原GPU逐图数组，不能唯一确认该样本发生历史翻转，也不能将CPU/GPU浮点差异当成已证明根因**。

本次问题：同一checkpoint在当前原cu128环境的单次CUDA执行，是否重现原汇总数；其逐图margin/预测与已保存CPU数组有哪些实测差异？原GPU数组缺失意味着即使新GPU汇总精确match，也不能证明所有历史逐图预测相同或唯一归因。

## 固定输入与科学范围

`INPUTS.json`固定原900库存中的单条`MODEL_RECORD.json`、原model/result/job SHA、全部source/data/cache SHA、CPU receipt/arrays/worker失败、v2/v3/v4/evaluator/core/CNN源、原batch与路径绑定。`FAILURE_BINDINGS.json`逐字引用已离机65成员失败档案的15项输入，旧包和原模型不改也不重包。

- 模型终轮70、seed91009、IID真实alpha5000、F Flip、FairGuard原lr0.001，配置字节不变。
- valid19867/root16277，原train162770仅用于相同root/客户端元数据重建；不读取test像素或标签作推理/拟合/选择。
- batch64、原ID顺序、uint8 RGB64/255、FP32及原strict deterministic设置不变，不加AMP/TF32/编译。
- 原`metadata/rebuild_root/model_margins/extract_and_predict/fit_views/predict_views/evaluate_frozen_predictions/check_native`复用；native/raw仍margin严格大于0。shared阈值仍由该次GPU的clean train-root margins按原冻结规则拟合，不能用valid标签拟合，也不手工搬用CPU阈值以追求match。
- 所有视图同一checkpoint；保留全部margin、预测向量、root拟合/阈值、分组混淆计数和三指标。权重前后严格一致、无优化器和梯度。

## 最小执行适配

`SCIENCE_BODY_DIFF.patch`完整显示由原v2 `replay_one`派生的5处替换：模型`.cpu()`改`.to('cuda:0')`，以及scope/status/runtime-device/claim_limit诊断标签。逆向替换后源文本逐字恢复原函数。最后原`comparison['accepted']`拒收语句仍在，容差完全不变。

`adapter.py`复用v4的原路径及checked_result绑定，只把原CPU专用资源门换成独立GPU资源门；它没有CLI dispatcher。原CNN forward本就将每批uint8数据转到classifier设备并转FP32，因此不用改DataLoader/模型forward/归一化。

将来root审阅并另行批准后，外部fresh-child可以加载已固定源并调用`adapter.diagnose(...)`。它按原v2 metadata重建输入，原始source/data/三artifact前后hash相同；只在全新输出目录保存。原函数若native失配会保存原数组与receipt然后抛原异常；外层只把这个已保存的科学失配标为diagnostic结果，**不接受为正式样本**。其他异常保留并抛出，禁止循环重试。

`compare_saved.py`为未来数组离线比较接口：先核CPU与GPU receipt/checkpoint/权重/ID/数组SHA，然后复用原evaluator重算三视图指标、分组混淆计数和阈值预测规则，报告全部CPU→新GPU预测翻转的image_id/标签/敏感组/margin，以及root和valid全部margin差异摘要。没有生成或假造GPU数组，尚未运行该比较。

## 唯一资源提案与批准边界

**仅提案：host CUDA0、CPU105、单进程单线程、nice10、idle IO。** 实际GPU用hostGPU0当时的UUID绑定`CUDA_VISIBLE_DEVICES`，进程内只有logical cuda:0。执行前须实时核主训练/其他任务、CPU配额与CPU105无冲突、GPURecoveryNone和至少2GiB GPU空闲、内存至少8GiB余量、无重复诊断与全新输出。

root的精确外部批准须SHA绑定本包、INPUTS、唯一ID/输出、实际GPU UUID及120秒内资源收据。子进程导入科学模块前设置OMP/MKL/OpenBLAS=1，torch threads1/interop1、CPU105与CUDA UUID；该设置工作留给将来已批准的root执行层，当前不创建服务或监督机制。`APPROVAL_TEMPLATE.json`保持未批准且缺少实测UUID/资源/封包批准值，会被拒收。

严格限制：一次GPU诊断，无新增seed或参数、无训练/test/容差放宽/自动重试。无论match还是mismatch，旧CPU失败继续保留；不得自动把GPU诊断加入391/424或重启旧872。后续如何处理正式多设备重放由root另审。

## 准备验证与未知项

`selfcheck.json`只证明执行合同拒收及源码差异；不导入Torch/NumPy，不创建CNN，不算GPU科学通过。已核15个只读失败输入SHA，CPU副本与65成员已保全输入一致。来源证明见`SOURCE_REUSE.json`。

尚未知：当前GPU单次是否匹配原汇总、真实CPU→新GPU逐图翻转和root阈值变化、历史GPU逐图预测、历史CUDA kernel/driver/backend是否与当前完全相同。torch版本同为cu128不能消除这些未知。运行时资源和GPU UUID尚未批准。当前交付止于准备，不等待其余方法或批次。
