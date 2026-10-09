原32条 CosineFairnessHybrid 验证搜索执行副本（已冻结配置，尚待 root exact scope/seal 审批）。

科学定义和 body/driver/writer/5科学源与已接受CUDA4保持逐字节一致。仅 runtime protocol 冻结、执行路径/服务身份、scope/job绑定、历史门检凭据和本说明更新；没有新方法或新参数。

范围：LR{.0005,.001}×lambda{5,20}×tau{.1,.2} 共8候选，IID alpha5000/non-IID alpha5×Benign/S-DFA4条件，seed91001，70round，train162770/root16277/valid19867。候选lambda/tau同时传入adapter与legacy config，原逐轮oracle使用相同参数。

选择：每候选先跨全部4条件平均冻结score，方法内最大值；完全并列candidate字典序。另保留accuracy冠军、三指标Pareto及全部原始候选/负结果；n=1不报sampleSD/显著性，不按指标或条件挑seed。

资源：GPU0 UUID da357477-30a7-fddc-344b-a20513b9a2d0，CPU104计算1线程/nice10/idleIO/max1。派发前90s内重新核GPU显存≥4GiB/RecoveryNone/无duplicate/无CPU104紧绑重叠/实际CPU预算；不重启或抢占健康队列。任何失败保留并停，不自动重试。

门检：已有CPU4与CUDA4严格闭环，CUDA两pair模型张量/全部每轮指标攻击诊断/RNG exact。3round的4CUDA终轮均预测class0（ACC .5166859616449389/AEOD0/ASPD0/positive_rate0），不是性能优势。prerequisites引用已离机原件，不重复推理/备份原模型。

批准门：runtime_protocol和screen_scope的FROZEN只表示固定参数。原driver仍要求绑定exact scope+source seal的外部APPROVED_screen.json及90s资源proof；当前没有该批准文件。新副本gate_scope保持PREPARED，gate_runs/screen_runs不存在。APPROVED_screen_DRAFT只供审阅，状态故意不匹配执行批准要求。

边界：不启动formal100/test/新seed；原metadata loader可能物化含test尾部的标签/划分元数据，不读取test图像/推理/拟合/选参，不称untouched-test。
