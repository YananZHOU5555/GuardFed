# A50 五 IID 三视图：独立审查

结论：**PASS，无科学阻断项**。本审查针对已固定候选的来源、选样、统计单位和表述；没有运行 builder、整套 verify_saved、CNN、校准拟合或训练，也未修改候选、canonical、STATE 或 Git。完整数值验收由 root 独立负责。

候选：`tmp/celeba_mechanism_A50_IID_candidate_20261010/`。入口 FILES_SHA256 SHA256 为 `0600c7ac5d04f8be45c9acf20cc00bb6162f618b46340fd11b6a3619dbf52f70`；HANDOFF SHA256 为 `3c2abd173199ef15fae7fb16c11ed84825ddb62f162432e438b9f1af38ed256c`，均现场读取核对。

## 关键证据

- 100 个唯一对象精确覆盖 Full/minus_A × IID 五场景（Benign、F Flip、FedSA、S-DFA、Sp-DFA）× seed91001–91010，即 50 对。没有择优 seed、丢弃负结果或混入 non-IID。`minus_A_non-IID_Benign_seed91001` 存在于已接受 251 索引，未进入任何候选记录或汇总。
- 与已接受 A40 原文件独立比较：旧 80 个对象内容、对象顺序及逐对象原始序列化字节全部 exact；旧 108 行对象（648 个均值/SD 标量）及 Markdown 324 个格式化单元格 exact。
- 实测 251 索引 SHA 与实际 ROOT_ADOPTION 的接受值相同。新增 10 个 Sp-DFA minus_A 的 checkpoint/三视图与该索引逐个匹配；10 个 Full 对象与原 900 记录的 checkpoint/config/三视图/fit 匹配，原 900 文件 SHA 相同。新增绑定均为 valid、终轮 70、19,867 张图。另独立读取 seed91001 的原 scientific receipt，核 SHA、checkpoint、config、三视图、fit 和预测数组身份一致，未访问 test。
- 全部三视图均保留。10/9/6 面板准确使用 91001–91010、91002–91010、91005–91010，相同集合用于两变体；不是以 Full 最佳 seed 对比消融均值。
- 使用独立 Python `statistics.mean/stdev` 从 records 计算全部 Sp-DFA 配对差，以及五 IID 场景先 seed 内等权均值、再 seed 间配对差的均值和样本 SD，共抽核 108 个标量，最大误差 `3.0531133177191805e-15`。没有把 50 个场景-seed 单元当作 50 个独立 seed。
- 逐对象统计训练均为 `2.11.0+cu128`（100）；Full 回放 CPU 3/GPU 47，minus_A 回放 CPU 50，与正文一致。选择 seed91001、开发期间 validation 暴露及历史 test 暴露均保留披露；9/6 面板明确为描述性敏感性分析。

## 结果方向与主张边界

Sp-DFA 的 Δ 定义为 minus_A−Full。10-seed raw：ΔACC −0.266±0.612 pp、ΔAEOD +0.00731±0.02104、ΔASPD +0.00145±0.01756；native/shared：−0.230±0.746 pp、−0.00950±0.00897、+0.00117±0.01870。10/9/6 面板 ACC 均值均下降；raw AEOD 均值上升而 native/shared 下降。raw ASPD 在 6-seed 面板方向反转，也已保留。

正文没有推导显著性、每项必不可缺、普遍公平性改善、因果隔离、A100 或非 IID 完整覆盖，表述可接受。五 IID 汇总也保留 raw 与校准视图不同方向；AEOD 明确是绝对 TPR gap，不是完整 equalized odds。

非阻断提示：

- “all 10/9/6 panels”指面板均值，不能转述为每个 seed 都下降。Sp-DFA native 10-seed ΔACC 中 4 个 seed 为正、6 个为负。
- 100 个对象的 native/shared 指标逐对象完全一致，是并列视图，不构成两份独立支持证据。
- 索引中的历史 PENDING 标签不是当前接受依据；依据是实际 ROOT_ADOPTION 对该索引精确 SHA 的接受。
- LaTeX 是未编译片段；正式排版仍需编译审阅。本次独立抽核不替代 root 的全 972 标量/486 单元格验收。

具体计算值、来源 SHA、检查范围及限制见同目录 `REVIEW.json`；本目录仅保存精简审查证据。
