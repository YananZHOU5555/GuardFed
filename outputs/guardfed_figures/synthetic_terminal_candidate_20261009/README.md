# Fig.3终轮四面板审阅候选

**AUTHOR-REVIEW CANDIDATE；尚未采用，未修改原图或投稿稿。** 图内显式标记 HISTORICAL terminal round70、n=1 seed123，并保留来源限制。

- `fig3_terminal_candidate.png`：审阅预览；`.svg`和`.pdf`：同一四面板矢量图。Adult/COMPAS两行，ACC vs AEOD/ASPD两列。每数据集完整13设置，六生成器颜色+形状、1+9实心/5+5空心，10%真实基线黑色星形。数值重合处不抖动、不改值。
- `terminal_records_260.csv`：260条同轮三指标、原JSONL行SHA、原归档行号、runID、配置SHA及方法/比例历史标签。
- `terminal_points_26.csv`：26设置的三指标等权十场景均值，ACC另附百分数；`point_lineage_260.csv`逐点连接十条来源。
- `CAPTION.txt`：英文审阅caption；`build_candidate.py`：小型离线复算/绘图脚本。
- `verification.json`及`VISUAL_REVIEW.json`：数值和视觉检查；`INPUTS.json`固定原260JSONL、CSV、acceptance/verification、原封条与原Fig3预览SHA。

260条原行SHA/runID均与原CSV一致；每条`metrics`精确等于`last10_metrics`中第70轮完整三指标，且与CSV三个terminal字段精确一致。未使用CSV旧择优字段、joint轮次或FairScore。26设置各含2分布×5攻击，78个均值标量经独立NumPy从原JSONL终轮重算，最大误差1.1102230246251565e-16。

这里的十场景不是十seed，不计算sampleSD/CI。全部记录只有历史seed123；历史test已可见，终轮数值修正不能使它成为未触碰评价。每条只确认同轮指标身份，未恢复checkpoint二进制SHA、逐次不可变源码/dependency/cache身份。ForestDiffusion的实际执行实现仍未恢复；PCA-Gaussian只是历史标签，归档候选是均值/协方差采样与投影，没有显式PCA分解，不能宣称忠实原法复现。这些限制同时写在图脚注和英文caption中。

负结果保留。例如COMPAS：10%真实基线ACC64.2819%、AEOD0.109154、ASPD0.031457；TVAE1+9为58.8445%、0.153481、0.097901，三指标均差。CTGAN1+9提高ACC至66.1501%，但AEOD0.307666/ASPD0.358739更差。只能讨论实际取舍，不能写所有合成设置普遍更优或upper-right普遍最佳；本图gap越低越好。

仅生成可审阅修正候选。未SSH、训练、推理、下载、重新拟合、检索旧631来源、改Git/canonical或选择最终科学主张。原图仅用于只读风格/标签核对，不冒称恢复了原绘图脚本。绘图前已成功运行PDF artifact marker一次。最初读取封条时将list误当mapping的本地schema错误保留在initial_schema_failure.json；发生在导出/绘图之前，改为原path字段映射后通过，不涉及原证据或指标变化。

复算（项目根目录，仅写本目录）：

```powershell
python -B tmp/guardfed_synthetic_terminal_figure_candidate_20261009/build_candidate.py
```

PNG及PDF渲染已人工视觉检查：四面板、图例、轴名与限制脚注可读，无截断；接近点按真实位置保留，不添加不实分离或趋势线。投稿采用及相应正文修订由作者另行决定。
