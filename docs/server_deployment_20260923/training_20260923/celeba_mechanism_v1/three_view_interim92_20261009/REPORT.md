# 九完整场景三视图92：根复核交接

**已从实际严格离机链生成，待根复核/promote；没有选定论文主终点。** 原prepared包保持不变，最终产物位于本子目录snapshot92。

原71已采用表记录 + after71追加11的严格/离机/root采用链 + after82_v2追加10的严格/离机/root采用链，合为92个minus_U和92个配对Full，184条记录。Full三视图连接已接受900记录的实际来源，不重新推理Full。原after82失败scope不参与：其CNN前失败与空现场继续保留。

九场景完整90对/180记录；non-IID Sp-DFA seed91001/91002两对四条记录存incomplete_pairs.json，不入均值。三视图×10/9/6固定seed共9面板、243行、729显示单元格；全部负结果保留。

## 新增两场景的十seed配对差

方向为minus_U−Full；ACC单位百分点，两个gap为原尺度；均值±配对差样本SD。

| view | non-IID场景 | ΔACC(pp) | ΔAEOD | ΔASPD |
|---|---|---:|---:|---:|
| native | FedSA | -0.600997 ± 1.388572 | 0.003340 ± 0.009878 | -0.007468 ± 0.019000 |
| native | S-DFA | -0.453013 ± 0.975349 | -0.003744 ± 0.010162 | -0.002458 ± 0.026124 |
| raw | FedSA | -0.491770 ± 1.218369 | 0.000489 ± 0.010072 | -0.002184 ± 0.012251 |
| raw | S-DFA | -0.453013 ± 1.070315 | -0.014203 ± 0.022132 | -0.014302 ± 0.017303 |
| shared_calibration | FedSA | -0.600997 ± 1.388572 | 0.003340 ± 0.009878 | -0.007468 ± 0.019000 |
| shared_calibration | S-DFA | -0.453013 ± 0.975349 | -0.003744 ± 0.010162 | -0.002458 ± 0.026124 |

native/shared下FedSA删除U损失0.601pp ACC、AEOD更高、ASPD更低；S-DFA损失0.453pp ACC但两gap都更低。raw下同样没有“三指标删除后普遍恶化”的规律。184条native/shared完整view字典一致，不能把它们当独立重复或凭此选择主口径。仅作已观察validation的描述，不作组件不可或缺或显著性结论。

## 检查

- 独立math.fsum/sampleSD复算1458标量，max abs=1.4210854715202004e-14；全部729显示单元格通过。
- 原七场景189行数值JSON精确相同，Markdown显示行逐字相同；native92的81行统计在原1e-12内一致，486个格式化均值/SD显示一致。
- 增量21条实际archive/member重新逐SHA连接（旧11 archive110成员，新10 archive103成员），原receipt_identity/normalized函数AST提取执行，绑定原inventory/model/result/job/root/valid/source/三视图/strict/native1e-12与root采用。旧71和Full900复用其已接受封存记录，不重复旧88归档审计。
- 新10 root采用SHA：`b9e40d1ca565c0bcf146058433ff3e037ab4e824aa6972d1a3f3f47a088e8683`；archive：`479dff38cb1350916e6c07ad22f582a02ad2688f972052e5160289048e5d6a31`；库存v2：`bedb867ae9bdf965ef7143a5b4f077620911ec26d93346337307046e64b5c9da`。
- 输入、原源码函数SHA、增量链、builder SHA保存在INPUTS及snapshot92/SOURCE_BINDINGS/verification。原7、native92、prepared封条均未改。

## 披露和交付范围

显示Full90个模型：推理CPU5/GPU85，训练cu12888/cu1302；minus_U90个均CPU重放/cu128训练。mixed driver/runtime保留，不能声称统一设备实验或纯聚合因果。AEOD是绝对TPR差，native含各自原校准；raw/native/shared为同模型三个视图，root-only拟合规则未变。seed91001参与选择，9/6面板同样是已观察validation；无final test、无新显著性/CI或最佳seed选择。

没有SSH、CNN、训练、阈值拟合、权重复制、Git/canonical写入；本任务只接受已离机结果的连接及表格。九场景仅Full/minus_U，不代表其余七variant或全部900机制比较完成。

文件：snapshot92/TABLES.md、tables.json、records.json、paired_per_seed.json、coverage.json、incomplete_pairs.json、verification.json、SOURCE_BINDINGS.json。原prepared函数仅修改明确七→九场景列表和报错文字，statistics/summarize和种子集合仍用原源。
