# C50 五个 IID 场景三视图表：仅准备

当前不生成表、不推理、不训练、不重新拟合阈值。C40 原采用80条记录、648均值/SD标量、324展示单元逐JSON字节、顺序及显示行精确保留。新增 Sp-DFA10 由已采用 after40 七项与未来 after47 三项真实归档连接，Full50 全部仅引用已接受900，不能以native代替raw/shared。

唯一待满足门槛：after47 的真实 `ROOT_ADOPTION_REVIEW.json` 和外部精确SHA；其实际147+3=150、选定 Sp-DFA seed91008/09/10、原147不变、science/execution封条、原strict/offserver/archive/member/receipt/checkpoint全部由原已验证join核验。当前封存native150库存只证明输入身份，不是三视图接受。22项静态输入已绑定，after47科学源/库存已实际封存，不猜未来proof。

根采用且授权构建后执行（这里的路径和SHA必须用实际值）：

```
python -B build.py --C3-adoption <actual after47 ROOT_ADOPTION_REVIEW.json> --C3-adoption-sha256 <actual SHA> --output <this-directory>/snapshot
python -B verify_numeric.py --snapshot <this-directory>/snapshot
```

预期100 unique records（Full50+C50）、810场景均值/SD标量、405展示单元、900 receipt指标、2400混淆计数结构检查。三视图 native/raw/shared，各统一10、9、6seed；sample SD ddof=1，差值为minus_C−Full，ACC原表百分比/差值百分点，gap维持原0–1单位。所有反向、零、负值保留。

`cross_scene_seed_first.json` 单独提供五场景等权平均的描述统计：每seed先平均五场景，再跨seed统计，另162均值/SD标量，不将50个场景当50个独立seed。没有新score、CI、显著性或主终点决定。场景统计继续精确复用原statistic/summarize；新增汇总只调用同statistic，独立math.fsum/sampleSD复算。

原root-only校准、同checkpoint、native1e-12门不变。披露混合CPU/GPU评价、训练CUDA/driver环境、seed91001筛选、历史validation/test暴露；AEOD是绝对TPR gap。五个non-IID C场景及其余机制仍未完成，不声称必要性、纯聚合因果或返修全部完成。

准备检查：27拒收及实际native150−147范围/旧C40字节检查通过；未来proof正向仅内存schema fixture，无落盘审批。切换到实际库存时首次拒收测试用records[-1]错误指向旧minus_U，导致wrong_variant测试无操作，已保存原异常，改为精确ID定位后通过；无表/统计/科研源受影响。`SOURCE_DIFF.patch`给出相对C40 V2已采用源的范围/连接/汇总差异。
