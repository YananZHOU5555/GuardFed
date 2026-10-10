# 原单场景独立审查器 → 两场景独立审查器

- 原 math.fsum 均值、ddof1 SD、confusion metrics 三表达式 AST 完全一致，hash 见 SOURCE_CHECK.json。统计索引仅由固定 Benign 改为当前 row.attack。
- 24记录/1完整场景扩展为40记录/2完整场景，每panel由3行扩展为6行；增加旧24规范JSON及Benign27行逐字回归。
- 原A12 index连接扩展为原A12+A20 index前缀及精确8新ID；绑定原receipt/fit/checkpoint/source，Full仍仅接受900 JSON引用。
- 交付目录由外部CLI给出，seal SHA外部必填；输出改到独占review目录，既有输出禁止覆盖，失败保存后停止。
- 不调用作者统计、原科学接受器、CNN或fit；不更改任何已接受来源、协议或主口径。
