# CelebA 机制消融：七个完整场景三视图配对表

主表：[TABLES.md](snapshot_724_full100_mechanism71/TABLES.md)。精确7场景 = IID五场景 + non-IID Benign/F Flip；每场景Full与minus_U共享10seed，另列去选择seed的9seed及91005–91010的6seed面板。3视图×3种seed面板=9面板、189行（各含Full、minus_U和配对差值）。ACC为百分数，差值为百分点；所有差值minus_U−Full。

全量个体证据保留71对/142 checkpoint记录，其中只有70对/140模型进入均值±sample SD(ddof1)表。non-IID FedSA仅seed91001的一对保存在incomplete_pairs.json，明确不进入任何均值/SD表。Full100沿已有724实际identity mapping连接，没有新推理；其他native后续结果不纳入此冻结快照。

新增11逐条核原scientific/bridge/strict receipt及inventory_actual71身份，原接收11归档完整120成员重新核SHA；不以native训练指标替代raw/shared。原六场景120条records及162行统计逐JSON完全相等，旧包未修改。科学14文件保持原SHA。

`NUMERIC_CHECKS.json` 独立核142份原receipt、1134均值/SD数值、567展示数值格、639逐seed配对差值，最大独立算术差1.42e-14（验算容差1e-12，未改变native严格门限）。独立7场景×3方法行×9面板的完整grid检查通过，14项身份/混用/缺seed等拒收通过。所有140展示模型（含另存2个不完整模型时为142）native/shared保存指标和混淆计数完全相同；此相等不能作为独立校准收益证据。

新增non-IID F Flip的10seed native/shared差值：ACC −0.996124百分点，AEOD +0.002269，ASPD −0.011180；raw分别−0.957366、+0.002009、−0.006084。删除U会降低准确率，却改善ASPD；全部改善/退化均保留，不声称各组件都不可或缺。

当前展示Full重放CPU5/GPU65，minus_U CPU70；Full训练cu12869/cu1301，minus_U cu12870。历史驱动、构建和设备差异保留在表注和每ID来源中，不能视为统一设备因果比较。Native含每个过程各自root拟合校准；raw/共享校准使用原规则与root-only。AEOD明确为绝对TPR差，不是完整equalized-odds。seed91001参与配置筛选，三面板均属于已暴露valid，不是未触碰test；不做显著性、择优或主口径选择。

复核入口：对FILES_SHA256逐项SHA/size核验；读SOURCE_DIFF及SOURCE_REUSE；verify.py从实际封存receipt独立核数。它用新建NUMERIC_CHECKS.json拒绝覆盖既有结果，复核可在独立复制的本包中删去该生成报告再运行（不要改变封存交付包）。build.py只允许本目录新空输出，用原statistic/summarize。全部操作只读已有JSON和归档字节，不载入模型或标签数组，不访问网络/服务器，不训练、不拟合新阈值、不推理、不登记全局状态。
