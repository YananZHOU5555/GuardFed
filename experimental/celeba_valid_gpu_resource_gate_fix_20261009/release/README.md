本包是本机实现交付，未部署、未推理、未启动服务，未接受任何新cohort。当前424仍不变。仅root另行SHA绑定审阅可以启用CLI；APPROVAL_TEMPLATE.json的PREPARED状态、空ID和false权限默认全部拒收。首个候选新ID是 `FairGuard_IID_FedSA_seed91003`，并非全465授权。

`recovery.py`提供六个薄入口：单chunk最多11个ID顺序执行（同时仅1个GPU worker）、单worker、原科学strict、原增量archiver、精确10 CPU partial导入、精确1 GPU诊断导入。没有服务配置、bulk循环或自动重试。每个worker独占CPU105、Torch/BLAS和interop各1线程、nice10/idleIO、实际hostGPU0 UUID；使用现有cu128 Python，不新增依赖。

`gpu_replay_body.py`与实际诊断科学体SHA2aafb0d08b2a0c84bbcfc224638dc216329f72d450facf37fd2e50d1cb0f0ad5字节一致。模型.to(cuda:0)、原诊断receipt标签、signal.alarm(1800)、数值不一致时先保存NPZ/receipt再raise均原样保留。root16277、train162770、valid19867、batch64、所有源/模型/映射/原job/70round/checkpoint/fit/tie/原native1e-12守卫不变。CNN前后即时核实际quota、RAM≥8GiB、GPU0 UUID/RecoveryAction及线程/训练健康。

`STRICT_DIFF.patch`是相对原v3.accept的全部body差异：pass标签改为真实diagnostic GPU标签，device/count改为cuda:0/1，增加显式GPU receipt/proof/source/runtime/metadata/bootstrap/resource守卫。其余原科学函数代码逐字保留。`saved_science.py`直接保留原accept内17行完整科学段，诊断显式导入也重建root、重新fit_views、canonical核fit、用新拟合fits复算三视图预测/全部混淆计数/原native，不将JSON字符串threshold键直接传入预测，不假设native一定有fit。

bootstrap复用已成功attempt2：先绑定CPU105和批准UUID，导入同一原v2模块；断言CUDA未初始化后恢复被原导入覆盖的CUDA_VISIBLE_DEVICES及OMP/MKL/OpenBLAS=1，仅set_num_threads(1)，interop只断言已有1，不重复setter；v3/v4共享sys.modules['replay']。原导入期intra/env8如实写bootstrap收据，硬CPU始终限于105。coordinator和worker不再次增加nice，避免权限失败。

任何worker或新chunk数值/身份/资源失败即停，未执行后续ID保持未执行；旧输出、partial、CPU原失败和两次诊断工程失败不覆盖。main外层contract/preflight/accept错误也保存command/error/traceback：仅在外部root review SHA可信且指定的新真实输出命名空间写失败收据；无approval仅stderr，不创建科学输出。正常退出只表示完成执行，必须另做strict、离机archive/member验证和显式版本登记。

10 CPU partial入口复用原v4严格检查全部原11，要求恰有已知1项CPU失效、恰有原10项通过且native差0；逐ID核原65成员归档所绑定的receipt/数组SHA，不重算CNN。GPU诊断入口核原保存receipt/数组、diagnostic wrapper、attempt2 run_once/root授权/source seal、bootstrap、资源、FINAL_DELIVERY和离机/修复比较链，并复算保存数组科学段；实际count1/UUID/nice、无optimizer/gradients、root/valid-only及权重守卫均检查。二者输出eligible IDs及pending offserver/registration，不能改原424账本或原CPU失效。所有导入权限仍false。

`backup`复用原bounded_remaining.archive_chunk，成功归档必须externally SHA-bound完整strict输出；失败仅failure_命名，不会伪装成功。只打包该chunk新增arrays/receipts/config/log/proof及sourcefreeze，不打包旧900模型。原archiver逐成员SHA/size和源文件稳定性检查保留；远端归档仍不等于离机完成。本包不修改原collector；root后续需显式添加GPU版本、每ID receipt/array/source/proof/archive SHA后才能增加唯一接受分母。

424既有CPU接受与新GPU补集须保留逐IDdevice和source-version provenance。混合device的raw/shared汇总不能直接称为统一设备最终公平比较或最终test；本阶段只补实现/数值验证。正式native/shared主终点及其余final协议决定保持pending，不替用户选择，也不重算既有424。历史cu128/cu130、driver差异及seed91001选择历史继续披露。

本地检查（无Torch导入、无CNN）：

```powershell
python tmp/celeba_valid_gpu_recovery_implementation_20261009/selfcheck.py
```

结果见selfcheck.json/LOCAL_CHECK.log。已核原科学AST/17行、实际保存GPU数组9指标24混淆3预测规则、29项边界拒收以及固定native检查拒绝一张样本差；包括即时RAM恰8GiB通过、少1字节拒收、quota不足/无上限、UUID漂移、RecoveryAction和线程越界拒收。不宣称目标Linux运行已验证。

部署依赖见DEPLOYMENT_DEPENDENCIES.json。root需要逐SHA安装本完整包到 `/workspace/guardfed_checks/celeba_valid_gpu_recovery_implementation_20261009`，以及已封476提案到 `/workspace/guardfed_checks/celeba_valid_recovery_prepared_20261009`；原v2/v3/v4、evaluator/core/CNN/loader、精确900 storage-map链/源文件和只读artifacts保持原路径。只有审阅导入11时才需要对应历史actual输出，import_inputs只有现有证明/源码快照，无模型/预测数组重包。首次root仅填写外部新ROOT_REVIEW.json的单个新ID授权，不能编辑包内模板。

以下是未来root审阅后首1命令；本次没有执行。新attempt1父目录应真实存在且为空，chunk_000不得预建。ROOT_REVIEW.json在包外，只批准该ID，execute_new465=true、两个import=false，绑定实际PACKAGE_SHA/UUID；review本身需保存到批准attempt父目录之外，避免“空parent”约定冲突。

```bash
PY=/workspace/guardfed_envs/celeba-cu128-20261009/bin/python
PKG=/workspace/guardfed_checks/celeba_valid_gpu_recovery_implementation_20261009
REVIEW=/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/ROOT_REVIEW_FIRST1.json
OUT=/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/attempt1/chunk_000
# PACKAGE_SHA and REVIEW_SHA come from root's independently verified exact bytes.
nice -n 10 ionice -c 3 "$PY" "$PKG/recovery.py" run-chunk --review "$REVIEW" --review-sha256 "$REVIEW_SHA" --package-sha256 "$PACKAGE_SHA" --ids FairGuard_IID_FedSA_seed91003 --output "$OUT"
nice -n 10 ionice -c 3 "$PY" "$PKG/recovery.py" accept --review "$REVIEW" --review-sha256 "$REVIEW_SHA" --package-sha256 "$PACKAGE_SHA" --batch "$OUT/batch" --output "$OUT/strict_acceptance.json"
# STRICT_SHA comes from the just-closed exact strict_acceptance.json.
nice -n 10 ionice -c 3 "$PY" "$PKG/recovery.py" backup --review "$REVIEW" --review-sha256 "$REVIEW_SHA" --package-sha256 "$PACKAGE_SHA" --stage "$OUT" --chunk-index 0 --strict-sha256 "$STRICT_SHA"
```

任一命令非零立即停止，不顺序执行后续命令或盲重试。失败stage可在同样root边界下调用backup --failure保全，不接受其中partial。首1真实strict和离机闭环后，余下执行范围仍需root新授权；当前没有任何全465授权。
