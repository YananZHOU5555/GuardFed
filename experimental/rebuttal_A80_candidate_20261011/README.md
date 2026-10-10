# A60 → A80 作者审阅稿：source-only

当前只准备源码；没有绑定或读取尚未采用的 A80 表，没有生成新稿，没有重算科学统计。固定入口是已采用的 A60 全24条英文 reply 和 manuscript insertions。原稿 SHA、原检查器 SHA 保存在 `FIXED_INPUT_PINS.json`。

`build_and_check.py` 直接复用 A60 的四个 helper 函数（逐字/AST相同），并继续由原 A50 文档不变量循环验证所有24条原话、旧数字/链接/科学表、P1/P3–P6、原句和编辑正逆回放。没有新建文档验收框架。`SOURCE_DIFF.patch` 展示相对 A60 builder 的 source 差异。

增量只加入 non-IID F Flip/FedSA：原表的12组十seed paired mean±sampleSD（两个场景×native/raw×三指标，native/shared值需相等），以及18个场景×视图×固定10/9/6面板方向。方向直接读取已采用JSON中的均值符号，不预设 GuardFed 胜出，不重算均值/SD，不删负结果。所有面板原值仍通过已封存表链接可查。A80的覆盖、设备、Torch构建、原IID aggregate字节一致与评估边界全部绑定root proof。

A60完整旧内容保留并标历史；当前 AE、阅读概览、R3.2/R3.7 和 P2 的覆盖说明同步更新。旧段落移动和新段落插入均由 `EDIT_MANIFEST.json` 提供精确正逆回放。只有 P2 待办行可以改，其他旧表必须逐字相同。不得据此声称17方法齐全、所有机制完成、组件不可或缺、显著胜出或最终test完成。

收到实际 root A80 表路径与确切SHA后，才可按顺序执行：

```powershell
python -B tmp/rebuttal_A80_candidate_20261011/bind_inputs.py --A80-root <actual_ROOT_VERIFICATION.json> --A80-root-sha256 <actual_root_SHA256>
python -B tmp/rebuttal_A80_candidate_20261011/build_and_check.py
```

`bind_inputs.py` 只读取实际root及其5个文件pins、验证80对/160记录/8场景/非IID Benign,F Flip,FedSA/固定10,9,6面板/边界，输出新 `SOURCE_PINS.json`，不生成稿件。真正build后仍须审查新段落、12组数值、18个方向及scope，复跑既有 `--check`，再封存实际交付；root负责canonical采用。现阶段 `SOURCE_CHECK.json` 只证明compile和helper复用，不能冒称文档检查已通过。

全部新增文件仅在此私有目录。共享生成器、canonical、Git、SSH、旧结果均未操作。
