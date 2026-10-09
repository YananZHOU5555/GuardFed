# C40 实际四场景三视图表，待根独立采用

实际 after36 已 root 采用的4条 S-DFA 终轮三视图记录，加旧表已有6条，形成四个 IID 场景各10对 Full/minus_C。80记录、648均值/样本SD标量、324展示单元、720指标和1920计数字段通过；旧60记录、486标量、243展示单元及顺序一致。未产生额外推理。

新增 S-DFA 10seed 的 minus_C−Full：native/shared ACC −0.077012百分点、AEOD +0.000499452、ASPD +0.000219272；raw 为 −0.034731百分点、+0.002001521、+0.001439339。三项均值方向支持 Full，但差异小且无显著性检验；其它场景负结果保留，不能推导 C 普遍必要。10/9/6 子集统一，不选最佳 seed。

80记录 native/shared 三指标及group confusion counts相同，仅描述当前结果。Full replay CPU3/GPU37，C为CPU40，训练均cu128；不宣称环境等价。AEOD是绝对TPR差。历史配置选择、validation/test暴露、root-only校准和主终点未定保持披露；非最终test、非C全部场景完成。

首build于统计及输出前因准备封条list/dict schema失败；原源码/10成员封条未变。独立V2只改封条遍历，逆替换逐字相同；list正向及dict/错SHA/错路径拒收通过。这些补充fixture实际在成功record-only构建之后记录。另交接编写初次误用group_counts字段而KeyError，于交接文件创建前退出，改用原真实group_confusion_counts；表/数值验证不受影响。上述工程过程保留，未改科学body/阈值/统计。
