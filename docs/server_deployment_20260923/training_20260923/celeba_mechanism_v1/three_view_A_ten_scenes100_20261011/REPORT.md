当前：A100十场景表已由root采用；下列交接报告的pending描述是采用前历史。当前范围以ROOT_VERIFICATION.json为准。

# A100实际三视图表交付：待root采用

实际build与finish各一次exit0，root随后唯一执行原verify_saved一次exit0；我未重跑root verifier。200条记录/100对终轮checkpoint覆盖IID和non-IID各五场景。9个原raw/native/shared×10/9/6面板保持同seed、同checkpoint。所有记录只读自已root采用300及原Full900身份链；没有CNN、阈值拟合、训练或test。

实际原检查结果：逐场景1620 mean/SD标量、五IID162、五nonIID162、balanced十场景162，合2106；1053展示单元、1800计数派生指标与4800原始混淆计数检查。最大差1.4210854715202004e−14≤原1e−12。旧A90的180记录对象字节/顺序、1458逐场景标量、729展示单元及IID_SEED_FIRST原文件字节保持。跨场景汇总先在每seed内等权平均场景，再对10/9/6 seed计算原sample SD(ddof1)，不把场景当独立seed。所有数值来自保存的原checker输出，不以计划计数替代实测。

新non-IID Sp-DFA的配对差(-A−Full)不能解释为A必需或始终改善。native/shared十seed：ACC−0.15754769215281783pp、AEOD+0.004918243551123458、ASPD−0.0013123213927118183；删除A降低准确率、增大AEOD，但ASPD较低。六seed方向反转为ACC+0.11493095753426748pp、AEOD−0.0003668245810185729、ASPD+0.0018715476200814412。raw十seed三项均值对Full有利，raw六seed则准确率略降而两差异均值降低。全部九行原mean/SD和精确JSONpointer在ROOT_READY_SUMMARY.json，全部场景/汇总方向在DIRECTIONS.json。没有显著性、每seed改善、因果或不可或缺结论。

实际环境：Full100回放5CPU/95GPU、训练98cu128/2cu130；minus_A100回放100CPU、训练100cu128。逐记录环境/配置/模型/source绑定保留。native/shared指标和计数在本批相同，但不是独立确认。recipe选择seed91001、validation开发暴露和历史official-test暴露保留；本表没有final-test性能，也没有选择最终主终点。A100完成仅限此control，其他五image-control variants、全17基线和提交正文集成并未完成。

root verifier之后按root明确授权，只把TABLES.md汇总总标题“Five IID scenes: seed-first aggregate”改成“Across-scene summaries (seed-first)”。TABLES_HEADER_EDIT.json/diff记录修改前后SHA及一处逆恢复；所有数字与表格body原字节不变，JSON/tex/source均未改，未重跑verifier。此修正避免总标题统摄non-IID/balanced时歧义。36个LaTeX table fragments尚未编译。

原15成员SOURCE_PREPARATION封条保留，准备阶段字符串转义/封条种类错误与adopter字符串唯一性拒绝均保留；它们发生于科学执行前或未执行的adopter准备中，不使已产生科学数据失效。root_adopter_prepared/adopt_root.py只经过编译和原member/copy loop AST比较，没有执行。未来实际adopter须显式传delivery seal/成员数/handoff/root命令/保存结果SHA。canonical、STATE、entry和Git未改。
