本包仅准备，不部署、不派发、不执行GPU/CPU推理、不登记新结果。当前九方法900唯一严格离机接受分母仍为424；机制800不在此范围。原872服务已因真实native数值分歧failstop，不能重启或自动重试。

`manifest.json` 是冻结900库存减去原424 collector的精确476项补集，按原任务顺序逐ID绑定科学cell、历史job ID、inventory record SHA、config/data/source身份以及model/result/rawjob的原archive/member SHA和实际storage路径。合法不同cell即使权重SHA相同仍各自保留。476严格分为：

- 465项：原chunks037..079未执行。建议新鲜GPU valid-only执行，复用原config.device=cuda、batch64、原v2数据/root/推理/校准/评分科学代码及原v4输入映射。
- 10项：原chunk036 CPU strict通过，native三指标差0；其数组/receipt已保全在65成员失败归档中，尚未独立登记。建议先由root审阅显式复用，避免重算；本包不接受。
- 1项：FairGuard_IID_F-Flip_seed91009。原CPU结果仍失效。新GPU科学体native三指标差0，14成员离机及后续保存数组比较已完成；此诊断仍不是cohort接受。本包仅引用该独立证据，复用决定保留null。

对应的42条已接受来源证明及archive SHA直接引用原424 collector，不重扫或重包旧归档，不重包旧900模型。原seed、统计10/9/6口径和正式协议PREPARED_NOT_FROZEN状态保持；native容差固定1e-12。

`gpu_replay_body.py` 与已实际运行的诊断科学体字节完全一致（SHA2aafb0d08b2a0c84bbcfc224638dc216329f72d450facf37fd2e50d1cb0f0ad5）。`SCIENCE_BODY_DIFF.patch` 对原v2仅改变模型传到cuda:0及四个receipt字段。root重建、valid19867、原checkpoint载入、阈值拟合、tie规则、评分、native守卫、无optimizer/gradients及权重不变检查逐字保留。此body保留诊断标签和“不接受cohort”声明，不能拿它的成功直接增加424。

`BOOTSTRAP_ATTEMPT2_DIFF.patch` 是已运行attempt2相对失败attempt1的原始精确差异，供新worker入口复用：先绑定批准的单CPU/单GPU UUID，导入同一个原v2 replay模块；导入后先断言CUDA未初始化，恢复被原导入覆盖的CUDA_VISIBLE_DEVICES和OMP/MKL/OpenBLAS=1，只set_num_threads(1)，断言interop已经1，不再次set_num_interop_threads。v3/v4必须引用同一sys.modules['replay']实例。保留导入期原intra/env=8的实际披露，不伪称从未发生。

attempt2 GPU科学执行成功后，其第一版保存数组比较因JSON阈值键类型失败，因此不能把该整个wrapper称为PASS。后续独立offline compare_saved_v2.py已有离机证明；源和最终交付SHA列于INPUT_PINS.json。未来strict检查应采用原v3.accept中从保存root margins重新fit_views、canonical比较fit、再用新拟合的fits生成预测的科学段，不能把JSON反序列化后的threshold字符串键直接送进预测，也不能假设native一定有fit。

当前原v3/v4 strict接受器硬要求CPU device/count，且诊断receipt/proof schema与普通worker不同。执行之前还需root审阅独立、显式版本绑定的最小GPU worker及strict入口差异，单独批准上述11项的复用登记和465项GPU资源/执行范围。不能修改旧接受器或伪造CPU receipt，也不能自动用本提案生成approval。本包没有服务配置、安装器、GPU启动CLI或自动队列；这是当前明确未完成的执行依赖。

`INPUT_CONTRACT.json` 列出未来执行、接受与恢复边界。任何native数值、source/model/map/ID身份、root/train/valid或资源守卫失败立即停并保留partial；不放宽容差、不换seed、不做CNN重跑诊断或选结果。新attempt仅跳过独立严格接受并离机、按唯一科学cell登记的ID；旧partial/诊断必须先获得显式新版本接受证明才能计入跳过集合。

本地复核命令（仅标准库，绝不导入Torch）：

```powershell
python tmp/celeba_valid_recovery_prepared_20261009/prepare.py --check
```

检查实际冻结22项输入、900唯一科学cell、424/476精确补集、465/10/1分区、GPUbody声明差异及已知interop故障边界，拒收重复/漏ID、与424重叠、错路径/模型SHA和无审阅复用。`selfcheck.json` 是本地结构验证，不是新的科学实验、900 GPU验收或性能表。
