# F20两场景表：独立有限源码审阅

**PASS，source_adoptable=true；未发现阻塞。** 仅允许root在真实views320/native≥320采用并绑定后，沿原入口一次build及saved verifier。本报告不授权派发，不代表表已生成或数值通过。审阅时ROOT_BINDING及实际输出均不存在。

- 封条 `aef78ed959b26e614cff5186b1f1178da49ef086b31701bbd96b72cf87608f19` 的10成员逐SHA/bytes一致；23个小型source/input pin一致。原900的27MB记录未重新读审，仅对照已接受封条中的同一SHA/bytes引用。
- [binding.py:19](E:/OneDrive/文档/GuardFed/tmp/celeba_F_IID_two_scenes20_table_20261011/binding.py:19) 要求310+10=320、原310有序prefix、准确IID F Flip91001–91010、完整new-record/binding/artifact集合、真实native≥320、相同inspection身份及无新增CNN/train/test/native偏差。`load()`核真实外部SHA及source seal；template不能直接通过。
- [build.py:75](E:/OneDrive/文档/GuardFed/tmp/celeba_F_IID_two_scenes20_table_20261011/build.py:75) 的逐ID循环，与F10仅场景字面值Benign→F Flip及对应错误说明字符串改变，AST其余相同。receipt identity、checkpoint/result/job、原Full900引用、root/train/valid数据及同seed/alpha5000配对仍用原来源；全部三view来自同receipt/checkpoint。没有按性能或差值筛选。
- 旧Benign20 objects直接前置；序列化object bytes/order、九panel中的旧162标量投影有守卫。旧81展示值由原样统计与同一formatter保持，saved verifier再核完整162cells。它们是未来实际检查，当前没有运行或宣称已经通过。
- 原C2 `verify()`函数直接raw bytes一致；C2 panels仅精确variant常量C→F重绑定。原statistic/summarize、math.fsum及ddof1不变，10/9/6采用相同预定seed配对；40records/20pairs/2scenes对应324标量、162cells、360count-derived metrics、960count checks。无跨场景总均值或把场景当seed。
- AEOD定义、ACC/pp单位、raw/native/shared规则、设备/训练环境动态计数、selection seed与历史test暴露均保留；native/shared相同不作独立确认。其余八个F场景未齐，不声称F100、显著性、必要性、因果隔离或final test。

审阅者独立于F20源码作者；此前参与A/C表源码审阅。未导入Torch、未运行fixture/build/verify/numeric/统计/科学/SSH，未改候选或共享入口。两次辅助静态断言分别漏映射诊断字符串、误把旧seal的dict成员当SHA字符串；已保存原trace并按实际字节/schema修正比较，均不是候选科学失败。
