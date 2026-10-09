# 三视图minus_U100十场景：仅准备

当前状态 **PREPARED，等待after92实际8条的ROOT_ADOPTION_REVIEW及其外部SHA**。本包没有生成100已接受表，没有读取运行中预测或把RUNNING计为科学结果。既有已接受三视图边界仍为92个minus_U。仅在根提供实际采用凭据后运行下述单次离线builder。

固定增量为non-IID Sp-DFA seed91003–91010八条，来自after92封存库存；原92记录、Full100引用逐对象相等。原native104含minus_U100与minus_C4；C4只披露为native部分结果，绝不进入本三视图表。后续新的terminal不自动并入。

```powershell
python -B tmp/celeba_mechanism_three_view100_tables_20261009/build.py --root-review "tmp/celeba_mechanism_valid_incremental_after92_20261009/execution_candidate/backups/<ACTUAL_TAG>/ROOT_ADOPTION_REVIEW.json" --root-review-sha <ROOT_ACTUAL_SHA256> --output tmp/celeba_mechanism_three_view100_tables_20261009/snapshot100
```

输出路径必须是本目录下尚不存在的直接子目录；实际采用文件须在after92 backups内、文件名及外部SHA准确，并绑定92→100、exact8、原92未改、原92采用凭据、science/execution seals、archive/member与offserver身份。没有有效凭据时拒绝生成结果，不创建科学输出。

## 复用和校验边界

- `build.py`直接调用已接受final_builder_v2的`accepted_increment`，原`receipt_identity`/`normalized`及bridge `canonical`函数按AST原文提取。新8条保留原model/result/rawjob/config/source/root/valid/split/终轮70/native≤1e-12/三视图/权重不变/无optimizer与梯度守卫；archive/member逐SHA连接已有原strict/offserver，不重新推理或拟合。
- 原九场景184 records JSON对象不变；原243行/729显示单元精确保持。新8个配对Full仅引用已接受900的原record/source/checkpoint/config/views/fits，不复制模型、重推理或重拟合阈值。
- 生产统计直接复用封存evidence_v4的`summarize`/`statistic`、原10/9/6共享seed集合。`panels.py`相比原源只把九场景覆盖改为十场景，具体diff见`STATISTIC_SCOPE_DIFF.patch`。没有新科学metric定义。
- 实际输入通过后，目标为200模型记录、10完整scene、9 panels、270 JSON行/810显示单元、1620 mean/sampleSD标量；独立math.fsum/sqrt复核，同时从保存混淆计数复算1800指标并检查4800基础tp/fp/tn/fn字段。新增8的原离机证明还必须含72指标/192计数/24预测规则。以上是预期验收数量，当前未宣称通过。
- native表另对已接受native100全部90行原统计核≤1e-12；本地准备阶段仅做真实库存/来源正向与11项拒收、AST/CLI入口检查，未运行100统计。

输出为TABLES.md、tables.json、records.json、paired_per_seed.json、coverage.json、verification.json、SOURCE_BINDINGS.json。实际结果须再由根独立审阅/登记；本包不自动promote。

## 论文范围

valid-only，同终轮checkpoint三视图平行报告ACC百分比、AEOD绝对TPR gap、ASPD；sampleSD ddof=1，配对差为minus_U−Full。native含各自原校准，shared是冻结公共root-only规则；二者若数值相同也不能作为独立校准增益。保留全部负结果，不做显著性、组件必要性或择优seed主张。

Full与control的CPU/GPU重放、cu128/cu130训练及driver差异按逐记录保留；混合device不能称统一设备最终公平比较。seed91001选择及validation曝光历史仍适用，9/6面板不成为未曝光test。native/shared正式主终点仍pending；Full/minus_U十场景不代表另外七组件或机制900完成。

没有SSH、CNN、训练、test、阈值拟合、模型重包、STATE/Git或旧封条修改。
