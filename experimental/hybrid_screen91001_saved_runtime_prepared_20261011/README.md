# screen91001 saved-runtime：仅源码准备

原单项运行包 `9be2ce96b4548144903e9e928567efa96565326917e2363f1ad11712ca44015d` 与原 check_saved.py `9408f64db4fb68d53eee4d882d69f3d0a3a0025baedbcb163823b3794a328934` 未改。唯一记录是 `CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91001_screen`：既有搜索结果复用，保留 selection_seed/phase/environment，不是正式新增；不新增 partition hash。

最小派生原 Hybrid8 check_linux/contract/transport，科学验算仍直接调用已部署原 check_saved.py。Linux 是1次原 cached-root refit、0 CNN；Windows 是原已通过 canonical-only wrapper 的namespace/package/checker SHA映射，0 fit/0 CNN，不导入Linux runtime_originals。旧 Hybrid8 Windows首次 resource 导入失败和随后实际通过均保留，不改其证据。没有新通用验算器。

## 执行前真实门

root需等 FL Pool32、Hybrid8、screen91001 三服务均rc3/EXITED，零对应 candidate，CPU110所有task无≤32窄mask交叠，无现存saved checker。根用新实测source/resource结果生成post-replay preflight，路径必须在 `/workspace/guardfed_checks/`，调用时≤300秒；不能把旧preflight改UTC复用。

必需：utc，guide_sha256=`42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa`，package_sha256为本单项9be2…，gate_result_sha256为实际单项完成gate；下列字段必须有真实证据为true：fl_evaluation_service_exited、hybrid_replay_service_exited（本screen）、previous_Hybrid8_service_exited、previous_replay_workers_absent、cpu110_all_thread_free、source_model_data_hashes_verified、gpu_health_verified、cgroup_and_memory_headroom_verified、storage_headroom_verified。cpu110字段指实测无窄亲和reservation，不宣称宿主独占。

远端helper重新检查guide/package/checker pin，三服务EXITED、候选/checker缺席、所有TID、CPU110/nice10/idleIO/CUDA隐藏、quota/RAM/disk。transport也要求三服务EXITED、零candidate与CPU110无窄占有。原300秒运输超时与失败停止不重发保留。

## 根审后调用（本轮未执行）

```powershell
python -B tmp/hybrid_screen91001_saved_runtime_prepared_20261011/check_linux.py --gate-result-sha256 <ACTUAL_GATE_SHA> --post-replay-preflight <ACTUAL_REMOTE_PREFLIGHT_PATH> --post-replay-preflight-sha256 <ACTUAL_PREFLIGHT_SHA> --allow-original-cached-root-refit --report-dir tmp/hybrid_screen91001_saved_root_execution_20261011/saved001
python -B tmp/hybrid_screen91001_saved_runtime_prepared_20261011/transport.py --gate-result-sha256 <ACTUAL_GATE_SHA> --linux-proof-sha256 <ACTUAL_LINUX_SHA> --destination F:/YananResearchStorage/GuardFed/hybrid_screen91001_saved_20261011/attempt001 --report-dir tmp/hybrid_screen91001_saved_root_execution_20261011/saved001
python -B tmp/hybrid_screen91001_saved_runtime_prepared_20261011/windows_runtime_wrapper.py --mode windows-saved-output --allow-saved-output-zero-fit --package-sha256 9be2ce96b4548144903e9e928567efa96565326917e2363f1ad11712ca44015d --gate-result-sha256 <ACTUAL_GATE_SHA> --transport-proof tmp/hybrid_screen91001_saved_root_execution_20261011/saved001/TRANSPORT_VERIFICATION.json --transport-proof-sha256 <ACTUAL_TRANSPORT_SHA> --linux-proof-sha256 <ACTUAL_LINUX_SHA> --metadata-npz F:/YananResearchStorage/GuardFed/added_cnn_exact3_valid_20261010/attempt001/verified_extract/metadata.npz --gate-dir F:/YananResearchStorage/GuardFed/hybrid_screen91001_saved_20261011/attempt001/verified_extract/bundle --output F:/YananResearchStorage/GuardFed/hybrid_screen91001_saved_20261011/attempt001/WINDOWS_SAVED_OUTPUT_CHECK.json
```

5 payload=gate、metadata receipt、Linux whole、单receipt、单NPZ；另transport manifest共6 ZIP成员。无模型/图像/旧批下载；F写前Yanan 2TB/Healthy/容量门保留。Windows wrapper原样提取original replay canonical、检查rt字段仅canonical/metadata、要求windows-saved-output和zero-fit标志，暂时替换c.runtime_originals为canonical-only后恢复；没有更换科学函数/容差。

SOURCE_CHECK仅6文件compile、源码rt/schema/5+1布局、原local ZIP与server成员循环字节对照、Windows小函数AST、raw逆恢复；没有运行科学检查、fit、forward、数组读取、SSH或transport。actual gate/Linux/transport pins均null。原单项consumer中两条旧错误提示仍写exact8，但实际身份、成员集合和refit数量已是exact1；本任务按要求不改运行包，不影响拒收条件。
