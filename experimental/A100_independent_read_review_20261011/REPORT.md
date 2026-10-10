# A100 独立语义与记录审查

结论：PASS（限定来源、记录等值与解释审查）；未发现科学采用阻塞。没有运行统计/checker/build/fit/CNN/train/test/SSH；数值重算与最终表采用由根代理完成。以下均读取作者实际 build/finish 产物，非本审查重新估计。

来源与保存：实际 replay300 根采用 e44122480c85e08461687bd58332aa9f30cc0ed807157966c3fd1769deb74a1e、index dfa3b4745d5cd2a681e0a5e9a38fdff215a7ab9bb6e5994cfb62e2786cf396c8，native300 b52484c71ea0e602e13f38f6505bc446d2ca2a80d95a859cf0236b078e087d1d。最后场景十个A seed中91001–05来自此前295链，91006–10来自新300链；不是把单次新增5误当完整10。已读20新增场景记录、配对checkpoint与相同data contract；同一record保留三视图、单一checkpoint。原逐记录来源校验体和固定70轮身份门沿用旧实现。

直接等值检查：200唯一记录、10场景×2方法×10seed完整。旧A90的180对象及顺序、JSON对象原文片段均exact；过滤新增场景后的全部旧tables对象相等，IID_SEED_FIRST.json原bytes相同。所有面板均固定10/9/6 seed，Full与minus_A的seed集合一致。此处未重算旧1458统计/729格；对象相等支持它们未变化。

差值统一为minus_A−Full：ACC负值支持Full，gap正值支持Full；gap负值表示删除A后该公平性指标改善。AEOD是绝对TPR差，非完整equalized odds。下表直接抄取已保存均值；ACC单位pp，两gap为分数。

| 范围 | n | native/shared ΔACC, ΔAEOD, ΔASPD | raw ΔACC, ΔAEOD, ΔASPD |
|---|---:|---|---|
| 新增non-IID Sp-DFA | 10 | -0.158, +0.00492, -0.00131 | -0.238, +0.00423, +0.00306 |
| 新增non-IID Sp-DFA | 9 | -0.295, +0.00597, -0.00420 | -0.362, +0.00257, +0.00116 |
| 新增non-IID Sp-DFA | 6 | +0.115, -0.00037, +0.00187 | -0.014, -0.00036, -0.00018 |
| 五non-IID seed-first | 10 | -0.217, +0.00115, -0.00032 | -0.215, -0.00019, -0.00109 |
| 五non-IID seed-first | 9 | -0.238, +0.00126, -0.00099 | -0.239, -0.00087, -0.00172 |
| 五non-IID seed-first | 6 | -0.149, +0.00047, +0.00132 | -0.193, -0.00093, -0.00098 |
| 十场景balanced seed-first | 10 | -0.197, +0.00033, -0.00111 | -0.175, +0.00145, +0.00029 |
| 十场景balanced seed-first | 9 | -0.178, +0.00028, -0.00125 | -0.160, +0.00104, -0.00020 |
| 十场景balanced seed-first | 6 | -0.168, -0.00092, -0.00070 | -0.155, +0.00052, -0.00060 |

解释：新增Sp-DFA校准视图10/9 seed是准确率与AEOD支持Full、ASPD支持删除A；6seed三指标方向均反转。raw10/9 seed均支持Full，但6seed仍有公平性取舍。五non-IID校准10/9 seed也是取舍，6seed三项方向支持Full；raw各面板均准确率支持Full而两gap支持删除A。balanced校准10/9 seed存在ASPD取舍、6seed两gap支持删除A；raw仅10seed三项支持Full，9/6seed的ASPD反向。保留这些不利或反转结果，不能概括为全面必要性、所有面板一致、显著性或纯A因果效应。

汇总函数来源已读：沿用原C100 aggregate_panels/aggregate_balanced_panels与sampleSD(ddof1)函数。non-IID只在临时lookup做分布标签投影，原records不改；每seed先等权平均5或10场景，再跨固定seed统计，因此n不是50/100。读取的全部200记录native/shared指标及计数对象完全相同；两列是同checkpoint不同描述，不能当成两次独立确认。

披露与设备：TABLES明确IID alpha5000/non-IID alpha5、valid19867、round70、91001选择史、开发期validation暴露、历史test暴露、9/6为描述性子集、未开启最终test/未选主终点。当前记录计数与正文一致：Full replay5CPU/95GPU、training98cu128/2cu130；minus_A replay100CPU/training100cu128。原Windows exact-refit/whole失败边界保留。仅A100完整；其余5控制变体、其余基线/整体返修未宣称完成。

非阻塞阅读建议：TABLES的顶层“Five IID scenes: seed-first aggregate”还统摄后面的non-IID与balanced子节，虽各子标题与段落已正确区分，最终编辑时可以改成“Seed-first aggregates”以减少误读；无需触碰科学表值。README/SOURCE_COMPILE等仍是准备历史，应以最终ROOT_READY_SUMMARY/交付说明区分当前实际输出，不能把历史pending当现时状态。

本次读取固定产物SHA：
- `records.json`: `9f5d5e92b68a6edf5586f045e9bc60d9e8b1a15f8acdc905fd68ef45ff7010c6`
- `tables.json`: `4323cc2e62cff05a1bffcaba804bb1bce2b17f95a9884f8d0ec38152c2bd6830`
- `TABLES.md`: `1d57badbc799c356ee4ff3b15fc06d52f682f19b73b6ac1c31ab30d74a65992a`
- `CROSS_SCENE_ADDITIONAL.json`: `e5676c1b74b70bc3a92e338d08123a14c381685e6c334656dd9cd94b6cc0da7e`
- `IID_SEED_FIRST.json`: `ce6088b23dd0e2181e83393510075a919dc52719d4b654f70a29205d398b5baf`
- `FOCUSED_CHECKS.json`: `cbcdf901644c7f59b1bb047b838633a1da4a8a5f5dbbead08eed3bfe17ae773f`
- `SOURCE_BINDINGS.json`: `d77b7e28a3b0c8eae1c57ce9af146272acfc56715c7a4d227b595003edf210f7`
- `DIRECTIONS.json`: `cabe040d6b108cc6c9a9f537589ed5c2264480c1bcb93e34783bc78b48f71875`
