# 根审阅后使用；本交付未执行

原V1科学metadata检查已封存；本V2仅实际执行部署predicate回归，未重跑科学门检。下列完整检查入口已同步新字段，可按需使用：
```powershell
python -B tmp/celeba_gradient_screen64_v2_20261010/check_prepared.py
```

部署保留整个目录的 snapshot sibling布局；`--repo`是原冻结core的逻辑仓库根。不要把共享cache/原权重重新打包。

实际Linux入口（占位参数必须由root真实预检提供；当前没有可执行审批文件）：
```bash
CUDA_VISIBLE_DEVICES=1 PYTHONDONTWRITEBYTECODE=1 taskset -c 105 nice -n 10 ionice -c 3 python -B /workspace/celeba_gradient_screen64_v2_20261010/run_queue.py \
  --repo <actual-frozen-core-logical-repo> \
  --out /workspace/celeba_gradient_screen64_v2_results_20261010 \
  --resource-preflight <actual-root-resource-preflight.json> \
  --resource-sha256 <actual-sha256>
```

预检JSON由root实际测量后提供：status=`ROOT_GRADIENT64_V2_RESOURCE_PREFLIGHT_PASS`; at_unix(≤90s); cpu_ids=[105]; cpu_threads=max_workers=1; cuda_visible_device="1"; nice≥10; idle_io=true; no_restricted_cpu105_owner/no_duplicate_worker/no_restricted_CPU_overlap/source_data_hashes_verified=true; existing_nominal_compute_threads/actual_quota_cores; gpu_free_memory_mib≥4096及gpu_uuid; guide_sha256; 本包FILES_SHA256.json的package_seal_sha256; AUTHOR_DECISIONS.json的author_decisions_sha256; test_authorized=automatic_retry_authorized=false。不要从本机MOCK fixture复制成actual PASS。

`--single-job`是串行入口的内部子进程模式，仅由已过资源门的parent调用。直接科学worker入口同原法存在，但root实际派发应走上述queue资源门。已有输出/任何失败拒绝覆盖，失败即停止，不自动重试。

完成项由原 `snapshot/gradient_bridge_20261010/accept_result.py::checked_result(job_path, output)` 严格核验；后续离机原archive/member/hash验证与root采用尚未执行。64全部严格离机前不汇总选recipe。

V2 CPU规则：扫描所有进程的所有线程，只排斥 affinity 核数≤16 且与 CPU105 重叠的既有限制性 reservation。较大mask仅表示线程可在这些CPU调度，不能据此宣称CPU105独占或空闲。root实际预检必须另外记录 broad-mask 线程信息、CPU实际测量和全工作负载预算，再产生上述 fresh proof；本包没有实际资源PASS。
