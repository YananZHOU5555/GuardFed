# 最小来源差异

原根采用的92 source seal为`4ae84bc75ac08902ea6787bd525270950945f6e5166887010b41b5d40df61c7d`。INPUTS逐路径绑定原builder、原九场景snapshot/ROOT、原统计源、after92库存/bridge/seals、原Full900数据/ROOT与native100 ROOT/table，不绑定或猜测未来8条采用SHA。

原`accepted_increment`按原模块调用，无复制/修改；`receipt_identity`、`normalized`、`canonical`按原AST提取，无修改。`statistic`、`summarize`由原evidence_v4导入，无修改。新build只有离线scope/adoption门、原184记录追加8control+8Full引用、旧行精确守卫与原输出展示逻辑。

`panels.py`仅修改完整scene列表和错误文字。`verify_numeric.py`将原184/27行/1458标量与2 partial边界改为200/30行/1620标量且无partial；追加的计数一致性复算取自原92根审阅公式，仅作独立检查、不产出新指标定义。准确变更见patch。

ROOT_ADOPTION实际schema若不满足明确92→100来源链，直接拒收，不能临时补键或放宽科学守卫。只在根实际review到齐后运行。
