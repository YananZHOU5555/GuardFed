# Hybrid missing8 saved-runtime helpers — PREPARED, NOT EXECUTED

这批仅 IID Benign seeds91003..91010，跳过已接受91002；91001另有screen身份缺口，未纳入。固定运行包 FILES_SHA256=f3b197ce786a3f391f820baf8767fa25fba2d2387002acf654bced7dc8e95239。原 check_saved.py=1410e344f07cb7d5987dcf773de4dab968bb3f03d21b378f59704e67074bd924、原whole科学=d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745 未改。

## Root 后续顺序调用（本任务未调用）

先等实际 exact8 GATE_RESULT、Hybrid及FL Pool32服务均EXITED、零candidate、CPU110所有线程无窄亲和冲突。root使用实际只读resource/source/stat观察生成 SHA-bound 的Linux post-replay-preflight，文件放在/workspace/guardfed_checks/内，且调用时UTC≤300秒。必需字段为：

- utc、guide_sha256=42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa、package_sha256（上述f3包）、gate_result_sha256（实际）。
- fl_evaluation_service_exited、cpu110_all_thread_free、hybrid_replay_service_exited、source_model_data_hashes_verified、gpu_health_verified、cgroup_and_memory_headroom_verified、storage_headroom_verified 均实际true。

cpu110_all_thread_free指枚举所有task后没有覆盖CPU110的≤32逻辑CPU reservation；宽mask宿主活动仍可能发生。这不是host独占或持续监控。root应保留原resource样本/测量区间及source/data哈希+stat来源，不能把旧启动preflight改UTC复用。helper额外现场复核guide、两服务rc3/EXITED、零candidate/checker、全部task窄mask、当前cgroup quota/RAM/disk、CPU110/nice10/idleIO/CUDA隐藏；原check_saved.py仍执行原package/data/artifact检查及其原fresh门，不改科学body。

```powershell
python -B tmp/hybrid_missing8_saved_runtime_prepared_20261011/check_linux.py --gate-result-sha256 <ACTUAL_GATE_SHA256> --post-replay-preflight <ACTUAL_LINUX_POST_PREFLIGHT_PATH> --post-replay-preflight-sha256 <ACTUAL_POST_PREFLIGHT_SHA256> --allow-original-cached-root-refit --report-dir tmp/hybrid_missing8_saved_root_execution_20261011/saved001
```

远端只调用已部署的 /workspace/guardfed_checks/celeba_hybrid_three_view_missing8_20261011/source/check_saved.py --mode linux-whole；输出为base/LINUX_SAVED_CHECK.json，原started/failure/output保持。8次原cached-root refit、CNN0；不是零fit或Windows whole。stdout中的原summary转存stderr，传回的whole报告原字节保存；root端command/stdout/stderr/exit均保留，超时只记录unknown、不重发。

成功后仅一次运输（仍须fresh CPU110无冲突，两服务退出）：

```powershell
python -B tmp/hybrid_missing8_saved_runtime_prepared_20261011/transport.py --gate-result-sha256 <ACTUAL_GATE_SHA256> --linux-proof-sha256 <ACTUAL_LINUX_WHOLE_SHA256> --destination F:/YananResearchStorage/GuardFed/hybrid_missing8_saved_20261011/attempt001 --report-dir tmp/hybrid_missing8_saved_root_execution_20261011/saved001
```

19 payload=GATE+metadata+Linuxwhole+8receipt+8NPZ，另1 TRANSPORT_MANIFEST，ZIP总20。无模型/图像/cache/旧批重复下载。F写入前验证Yanan 2TB/Healthy/余量；原ZIP成员唯一/安全路径/原member SHA/size/checkpoint receipt–array链接、重读server成员不变、ZIP完整性与离机每成员hash原逻辑保留。输出proof status=HYBRID_MISSING8_F_SAVED_MEMBERS_SHA_PASS，含actual package/gate/Linux SHA、archive path/SHA/bytes、members（19）和verified_extract。Windows原consumer无需改源码。

transport保留原explicit --verify-existing能力，仅原exit0/非timeout、完整archive/COMMAND/STDERR可作只读验证；不是自动重试。任意非零/timeout/科学失败停止，原现场不删、不覆盖。

后续Windows只用当前运行包原consumer、零fit角色：

```powershell
python -B tmp/hybrid_missing8_pool32_runtime_prepared_20261011/check_saved.py --mode windows-saved-output --allow-saved-output-zero-fit --package-sha256 f3b197ce786a3f391f820baf8767fa25fba2d2387002acf654bced7dc8e95239 --gate-result-sha256 <ACTUAL_GATE_SHA256> --transport-proof tmp/hybrid_missing8_saved_root_execution_20261011/saved001/TRANSPORT_VERIFICATION.json --transport-proof-sha256 <ACTUAL_TRANSPORT_PROOF_SHA256> --linux-proof-sha256 <ACTUAL_LINUX_WHOLE_SHA256> --metadata-npz F:/YananResearchStorage/GuardFed/added_cnn_exact3_valid_20261010/attempt001/verified_extract/metadata.npz --gate-dir F:/YananResearchStorage/GuardFed/hybrid_missing8_saved_20261011/attempt001/verified_extract/bundle --output F:/YananResearchStorage/GuardFed/hybrid_missing8_saved_20261011/attempt001/WINDOWS_SAVED_OUTPUT_CHECK.json
```

只保存原预测/阈值/9metrics/24counts/3rules的一致性，不宣称Windows重新fit一致；旧FL的Windows失败不删除、不扩展到Hybrid。最终科学采用由root另审。当前所有future Gate/Linux/transport/preflight SHA均null，所有实际执行/新科学采用0。

SOURCE_CHECK仅5入口compile、原FL派生raw逆恢复、39运行包成员未变、19/20布局、原localZIP和server member/receipt块原字节相同，以及4个纯schema拒收fixture。没有SSH、fit、数组读写、科学检查或未来结果；无新评审框架。
