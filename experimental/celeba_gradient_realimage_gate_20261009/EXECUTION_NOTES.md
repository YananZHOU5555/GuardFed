# 执行补充说明（2026-10-09）

主代理已审阅并授权本独立4条真实图像三轮探索pilot；仅限Fed-NGA/Huber各IID Benign、non-IID S-DFA。资源为CPU104–111、8计算线程、CPU-only、nice10和idle IO，一个长期前台gate由独立supervisor管理，autostart=false、autorestart=false。原800服务、原64screen/5项正式决定、Hybrid PID10424与CPU8–15保持不变；不得自动重试失败或调整参数。

**修正元数据访问边界：** 原冻结`src/celeba_data.py`第44行载入完整official split的`Smiling`和`Male`元数据，包括test尾部。虽然训练图像仅来自train、评价图像仅来自valid，本次实际运行不能声称完全不读取test标签元数据，也不能声称untouched test。

本次沿用原加载器，允许完整split元数据materialization；不读取test图像用于训练/推理，不在test上拟合、评分或选参。既有`test_evaluated=false`只表示没有test评价，不能解释为test元数据从未加载。原REPORT“不读取test”的字样属于准备说明，实际边界由本执行说明及附带到dispatch/provenance的元数据披露限定。

原22个准备封存成员及SETUP_SHA256不改。本说明、独立启动器与supervisor配置属于新增执行附件；批准收据绑定本说明SHA，gate的每条provenance将完整保存该收据。短程探索门检并不批准正式5项决定或64项搜索；完成前只报告实测进程与梯度证据，不称PASS。

启动前曾因原gate的physical-repo路径约束拒绝当前共享data symlink；当时无dispatch、无训练输出、无服务启动。失败证据留在`prelaunch_identity_failure.json`。主代理明确允许6个现有数据输入登记：4个RGB64缓存及官方属性/划分文件；都只允许`/workspace/GuardFed-revision/`下对应精确目标、原expected SHA和实测size。新`shared_cache_wrapper.py`仅处理这些已绑定输入，其余仍执行原repo内路径守卫；before/after仍全SHA核验，原22文件与数据字节不变。
