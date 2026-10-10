# LoGoFair100 独立root审查器 — 源码准备，尚未执行

原summary已实际执行成功，本包只准备其独立只读审查器。运行前root外部核对本包 `FILES_SHA256.json`，再单次执行（标准Python，无Torch/netcal/numpy依赖）：

```powershell
python -B tmp/celeba_logofair100_root_review_20261010/review.py `
  --source-seal <本包FILES_SHA256.json实际SHA> `
  --summary-dir F:/YananResearchStorage/GuardFed/logofair_fullcoverage100_20261010/summary001 `
  --stage F:/YananResearchStorage/GuardFed/logofair_fullcoverage100_20261010/stage001 `
  --index F:/YananResearchStorage/GuardFed/logofair_fullcoverage100_20261010/attempt001/STRICT100_INDEX.json `
  --inputs F:/YananResearchStorage/GuardFed/logofair_fullcoverage100_20261010/root_approved_inputs001/ROOT_INPUTS.json
```

成功只写本目录 `ROOT_REVIEW.json`，不执行采用；首次异常保留 `REVIEW_FAILURE.json` 并停止。任一实际输出已存在即拒绝覆盖/自动重试。`-O/-OO/PYTHONOPTIMIZE` 被拒绝。

审查固定实际summary四文件与index SHA，原7成员summary source、14成员fullcoverage source、原100库存、32 root采用、stage manifest/source、ROOT_INPUTS/BIND_APPROVAL链。100科学cell严格为2分布×5场景×10seed，96新结果+4原screen引用；原recipe07/fitseed1719/30postround/70轮checkpoint-cache-root-valid来源和固定20虚拟cohort身份保留。4复用项与原32 strict index逐键一致，唯一明确新增metadata字段为已固定的cell_id，不改变原记录。

重新构造完整artifact映射并核对ACCEPTANCE100的1027项实际SHA；模型/NPZ只用1MiB流式哈希，不反序列化、不加载数组、不再次调用Torch原strict、不fit/CNN/test。100原result/acceptance小JSON分别连接原index；原保存预测300指标检查引用已SHA核验的ACCEPTANCE100，本入口不声称重新计算预测指标。

独立算术仅使用 `math.fsum` 和sample SD(ddof1)，不导入作者describe/statistic/summarize。重新核33行/198均值与SD/99展示单元；10场景先在seed内平均，再跨预定10/9/6seed统计，额外核75个seed内指标。ACC为百分比，AEOD为absolute TPR gap，ASPD为absolute positive-rate gap，原1e-12容差固定。唯一恒定预测及负结果保留，无重选recipe、显著性或最优seed主张。

十model seeds对应预训练FedAvg/分区；post-fit seed1719固定。20 image-ID cohort不是实际训练客户端。DP校准适配不能解释为EO或纯聚合因果。seed91001验证选择、其它valid已曝光、预训练混合cu128/cu130/设备、float32 margin sigmoid限制及历史test属性/split元数据曝光均仍适用。此审查不是final test，不选论文主终点，也不宣称整个返修完成。

源码准备仅做语法、独立算术AST与实际小JSON schema核对，不执行1027项哈希或真实33行算术。初始只读schema探针的三次字段假设错误保留在 `SCHEMA_NOTES.json`；均未触发实际review、未修改F/源/科学输出。
