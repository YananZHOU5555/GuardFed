已生成一个固定的真实数据快照：原已接受 minus_U60 与实际 Full100 身份交付配对，表中只取六个齐备场景的 Full60+minus_U60，共120个 checkpoint。六场景为 IID Benign/F Flip/FedSA/S-DFA/Sp-DFA 与 non-IID Benign；native/raw/shared_calibration 三视图各按全10seed、排除91001的9seed、仅91005–91010的6seed平行报告。每格是原 ACC百分比、AEOD、ASPD 的均值±sample SD(ddof=1)，同时保留逐seed的 minus_U−Full 配对差。九个面板、162行（含配对差）、486个指标单元；不按胜负选场景或seed。

主入口为 `snapshot_724_full100_mechanism60/TABLES.md`、`tables.json`、`records.json` 与 `paired_per_seed.json`。`coverage.json` 显示四个尚无完整minus_U的non-IID场景，未填缺；其他机制variant没有进入表。Full100身份交付确已从702时99条推进至724时100条；此表的source snapshot钉住724与实际native60，不读取之后的native71或后续collector。

`inputs.py` 仅读取已有严格接受/离机源：复用 next37 原科学桥 `validate_inventory`（配方、源、70轮、seed、alpha、root/train/valid及client分区）和原 Full reference/canonical 身份；先核原23的source/proof/archive pin，再接已被根采纳的5+32增量。七段原机制archive仅读取strict→bridge→scientific receipt，且与其各自原8/23/60版本inventory逐ID核对，未把历史wrapper canonical差异伪装成同一版本。Full100按实际身份交付的一对一映射读取原strict/offserver/scientific receipt/数组SHA与完整collector链。没有解包、复制或加载模型，也没有把旧native值替代raw/shared。所有receipt/schema/来源SHA在INPUTS与逐IDprovenance中。

`build.py` 复用冻结 `evidence_v4.py` SHA `3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef` 的原 `statistic/summarize`，未复制或修改数学函数；ACC只作原fraction×100的单位转换。原 `summarize` 决定完整共享10seed场景，原 `statistic` 计算三个预定面板；配对差来自原 `summarize.paired_per_seed`。`SOURCE_DIFF.patch` 对照既有native中期renderer，`SOURCE_REUSE.json` 保存原函数源码/AST身份。新的代码用于来源读取、三视图覆盖与输出，不形成新接受器或dispatch框架。

`verify.py` 已完成有限独立检查：120份原receipt的ID/checkpoint/三视图、972个均值/SD数值（独立fsum/二阶中心矩算术）、486个Markdown指标单元、540个逐seed配对指标差；最大独立统计舍入差 `1.4210854715202004e-14`，原native接受容差保持 `1e-12`。14项拒收覆盖漏seed/重复cell、root ID/alpha/容差/未来control、混checkpoint/seed/缺视图/混shared、外部join SHA漂移及Full缺失/重复/receipt SHA漂移。检查不加载标签数组或Torch，不重新算科研指标、拟合阈值或推理。

实测负结果按原样保留：全10seed下，删除U在六场景三个视图均使平均ACC下降；native/shared下六场景ASPD均更低，AEOD只有IID Sp-DFA更低，其余五场景更高。raw下AEOD五场景更低、IID Sp-DFA更高。本120记录native/shared的三指标及混淆计数完全相同，不能把二者当独立校准增益证据，更不能据此主张每个组件对每项指标都必要。不做显著性检验、择优score/Pareto或主终点决定。

表中Full重放实际5CPU+55GPU，minus_U60均CPU；完整Full100身份来源是5CPU+95GPU。表中Full训练59cu128+1cu130，cu130为non-IID Benign seed91001；完整Full100历史为98cu128+2cu130，另一个cu130是未进入本六场景的non-IID S-DFA seed91001。minus_U训练为cu128/driver595。逐ID运行来源保留；混合重放设备、历史driver/runtime及训练执行差异限制因果解释，不能称统一设备的最终公平比较。

本任务valid-only，只提取已闭合证据，没有网络/服务器命令、新CNN/训练/test、阈值拟合、队列启动、登记、canonical状态或Git修改。原loader曾物化全split的Smiling/Male元数据，因此只声称无test图像推理/拟合/择优；本次本机提取不读取标签数组。AEOD是绝对TPRgap而非fullEO。91001参与过recipe选择，另外9seed同样已曝光valid，不是前瞻未触碰确认集；三个面板对双方采用相同seed规则。native包含各自原校准，归因对象是整套程序；正式native/shared主口径仍待用户决定。最终由根审阅表/data/source后决定登记，不能把六场景扩写为机制900完成。

重现命令（只写新的owned输出目录；不覆盖本快照）：

```powershell
python -B E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_three_view_paired_interim_20261009/build.py --output E:/OneDrive/文档/GuardFed/tmp/celeba_mechanism_three_view_paired_interim_20261009/another_fresh_snapshot
```

独立验证入口 `verify.py` 固定核当前snapshot，产物用独占创建保护。原NUMERIC_CHECKS已存在时重读即可；如根需要独立复跑，可在独立复制的owned包中执行，不能覆盖当前证据。所有外部路径是原输入位置，未额外打包模型或标签。
