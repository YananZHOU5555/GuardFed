# CelebA 九方法三视图描述表（valid only）

真实累计900条已接受记录，完整覆盖9方法 × IID/non-IID × 5场景 × 10种子；每条记录的raw、native、shared-calibration均来自同一个round-70 checkpoint。主面板使用91001–91010；补充面板固定使用91002–91010和91005–91010。所有面板均为均值 ± 样本标准差（ddof=1），ACC以百分数展示，AEOD/ASPD保留原尺度。AEOD的实际实现是绝对TPR差。

| 视图 | 10种子双分布表 | 9种子双分布表 | 6种子双分布表 |
| --- | --- | --- | --- |
| raw | [Markdown](raw/celeba_iid_noniid_ten_seed.md) / [PNG](raw/celeba_iid_noniid_ten_seed.png) / [TeX](raw/celeba_iid_noniid_ten_seed.tex) | [Markdown](raw/celeba_iid_noniid_exclude_selection.md) / [PNG](raw/celeba_iid_noniid_exclude_selection.png) / [TeX](raw/celeba_iid_noniid_exclude_selection.tex) | [Markdown](raw/celeba_iid_noniid_matching_six.md) / [PNG](raw/celeba_iid_noniid_matching_six.png) / [TeX](raw/celeba_iid_noniid_matching_six.tex) |
| native | [Markdown](native/celeba_iid_noniid_ten_seed.md) / [PNG](native/celeba_iid_noniid_ten_seed.png) / [TeX](native/celeba_iid_noniid_ten_seed.tex) | [Markdown](native/celeba_iid_noniid_exclude_selection.md) / [PNG](native/celeba_iid_noniid_exclude_selection.png) / [TeX](native/celeba_iid_noniid_exclude_selection.tex) | [Markdown](native/celeba_iid_noniid_matching_six.md) / [PNG](native/celeba_iid_noniid_matching_six.png) / [TeX](native/celeba_iid_noniid_matching_six.tex) |
| shared calibration | [Markdown](shared_calibration/celeba_iid_noniid_ten_seed.md) / [PNG](shared_calibration/celeba_iid_noniid_ten_seed.png) / [TeX](shared_calibration/celeba_iid_noniid_ten_seed.tex) | [Markdown](shared_calibration/celeba_iid_noniid_exclude_selection.md) / [PNG](shared_calibration/celeba_iid_noniid_exclude_selection.png) / [TeX](shared_calibration/celeba_iid_noniid_exclude_selection.tex) | [Markdown](shared_calibration/celeba_iid_noniid_matching_six.md) / [PNG](shared_calibration/celeba_iid_noniid_matching_six.png) / [TeX](shared_calibration/celeba_iid_noniid_matching_six.tex) |

每个视图目录另含10种子IID、non-IID单独表及900条简化source snapshot，共15组三种格式表。TeX为沿用原表形式的论文插入片段，需`multirow`和`graphicx`；本交付已核对全部TeX数值与Markdown一致，并完成全部PNG视觉检查，未将TeX片段编译成独立文档。

raw对全部方法使用未校准argmax（margin > 0）；native对GuardFed-AD2+使用原clean-training-root组阈值，对8个baseline使用原生argmax；shared calibration对9方法均使用已封存的clean-training-root组阈值。本次没有重新拟合阈值，没有CNN推理，也没有读取test标签或执行test评价。

实际推理来源为CPU434、GPU466；原训练环境为torch 2.11.0+cu128的886条和cu130的14条。推理runtime、原配置、checkpoint/result/job/source/data合约哈希均逐ID保留。该表不构成统一设备比较或70轮训练环境等价证明；三视图平行呈现，native/shared正式主口径仍待亚楠决定。九方法覆盖完整不等于17方法、机制研究或正式冻结评价全部完成。

核心证据：

- [records_three_views_900.json](records_three_views_900.json)：900条完整身份、原inventory记录、配置/环境、三视图9指标、组计数、规则/阈值、strict/offserver/archive/receipt/array绑定。
- [records_three_views_2700.csv](records_three_views_2700.csv)：2700条ID×view长表，不含替代表性能的score。
- [summary_statistics.json](summary_statistics.json)：三视图 × 10/9/6固定种子 × 90格的全精度均值/样本SD及展示值。
- [collector_chain.json](collector_chain.json)、[archive_member_SHA256_audit.json](archive_member_SHA256_audit.json)：实际累计链和88个archive的8395成员实际字节复核。
- [coverage_alias_environment.json](coverage_alias_environment.json)：完整900格、原source-method别名映射、旧表一对一ID连接、设备/环境以及全部常量预测ID。
- [verification.json](verification.json)、[independent_display_audit.json](independent_display_audit.json)：900条离机核验、独立NumPy ddof=1及全部序列化展示值/页边距核验。
- [old_native_snapshot_sampleSD_roundoff.json](old_native_snapshot_sampleSD_roundoff.json)：原snapshot中94个SD的末位浮点差，最大2.7755575615628914e-17；原900条native三指标、810个均值及旧5张native表全部1080个展示单元格精确一致。没有调整数值或扩大原1e-12核验容差。
- [adapter_provenance.json](adapter_provenance.json)、[renderer_presentation_only.diff](renderer_presentation_only.diff)、[minimum_change_review.md](minimum_change_review.md)：原函数复用和仅展示层的差异。

所有负结果和常量预测均保留。raw/native各有29条常量预测；shared calibration有28条；逐ID清单见coverage文件，不能由低gap推断模型有效。失败CPU记录`FairGuard_IID_F-Flip_seed91009`仍无效；当前该ID接受的是原诊断GPU数组，经既有明确import/strict/offserver链接受。本目录保存了两次本地adapter检查失败日志：一次是原`FedAA`与表名`FedAA-DDPG`的schema差异，一次是旧snapshot SD末位差的过严bitwise断言；均无新科学执行、无证据损坏，现已按原别名映射和原renderer核验方式解决。

复现命令（从项目根目录执行，仅写本目录）：

```powershell
python -B tmp/celeba_nine_method_three_view_tables_20261009/build_three_view_tables.py
python -B tmp/celeba_nine_method_three_view_tables_20261009/audit_displayed_tables.py
```

原统计与表格函数直接从旧`outputs/guardfed_tables/celeba_nine_method_final_20261004/build_tables.py`以AST读取执行，不运行该旧脚本的顶层写操作和额外pairwise汇总。数学公式、预测/评价函数、容差、种子集合均未修改。新目录之外未写文件、未操作服务器/Git/canonical状态，交付后停止。
