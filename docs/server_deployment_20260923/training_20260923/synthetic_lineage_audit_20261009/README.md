# 旧合成数据实验：数值追溯通过，实现与图来源仍有缺口

本审计不训练模型、不改旧结果。原 5090 归档 SHA、其中的原始 JSONL member SHA 和现有 JSONL SHA 三者已核对。840 份 `expanded_synthetic_ratios` 记录覆盖 Adult/COMPAS、IID/non-IID、Benign/FedSA、三个种子 123/456/789；每份都保留完整最后十轮的指标。

## 已核验的结果

- `synthetic_joint_raw.csv` 的 840 行全部精确对应原始记录中按 `ACC - 0.5*(AEOD+ASPD)` 选出的同一轮指标，误差不超过 1e-12。终轮三指标也逐项匹配。
- 另一个较早导出 `expanded_synthetic_ratios_raw.csv` 的 840 行全部能追溯，但规则是分别取最后十轮的最高 ACC、最低 AEOD、最低 ASPD，再计算 score。其中 **797 行不存在同时取得三个值的 checkpoint**。这份旧导出不能直接作为同一模型的效用–公平性点。
- 例如 Adult/IID/Benign、seed123、10%真实 root：旧导出 ACC 来自第67轮，AEOD 来自第65轮，ASPD 来自第61轮。原始终轮 ACC 为0.825619；旧导出 ACC 为0.836443。原始记录没有丢失；问题在历史汇总口径。
- 方法记录数为 none120，Gaussian Copula、ForestDiffusion、SMOTE、CTGAN、TVAE 各144。这里只有三个独立种子，两个分布和两个攻击不能被算成十二个独立种子。
- 另生成210条种子内四场景均值和70条设置汇总：先在每个seed内平均双分布×双攻击，再计算三个seed的均值和sample SD。对应 `joint_per_seed_210.csv` / `joint_setting_summary_70.csv`，不把场景当作独立重复。

## 仍不能宣称完成的部分

数值链不等于生成器实现链。扫描归档中的117个相关文本源码文件，仅在两个驱动的方法清单中找到 ForestDiffusion；归档的三版 `reproduce_paper_tables.py` 不含其实现，CLI 也不接受该方法。成功运行日志和144份输出说明当时有相应记录，但不能补出当时的生成器代码、依赖版本、拟合样本/模型/缓存身份。原始记录也没有逐次不可变源码哈希。不得用现在的其他生成器替代，或声称已复现 ForestDiffusion。

`synthetic_joint_raw.csv` 是可追溯的历史同轮汇总，但仍基于已评价的最后十轮选择；它不等于冻结终轮或未触碰测试集的结果。尚未找到提交版 Fig.3 每一个绘图点对应的生成脚本/输入清单，因此不能认定该图一定使用旧的逐列最优值，也不能认定一定使用新 joint 值。PCA 控制还涉及另一个历史 suite，未被本840行审计冒称覆盖。

审计入口为 `audit.py`；完整结果是 `acceptance.json`，逐条来源、配置和选择轮次为 `per_record_840.csv`。`initial_failed_assumption.json` 保留发现旧导出并非终轮数据的首次检查证据。验收只覆盖数值追溯；生成器和最终图来源维持 PARTIAL。
