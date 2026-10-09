以下均为root审阅封存后在远端执行的候选命令；本阶段未执行。完整新包部署到同名checks目录；复用现有cu128 Python、原476 proposal、V2 guard release与v2/v3/v4/storage/data，无新增依赖。依赖SHA见manifest、INPUT_BASIS；科学依赖沿用V2 release/DEPLOYMENT_DEPENDENCIES.json及原recovery contract。

root在包外创建ROOT_REVIEW_REMAINING440_RESOURCE_GUARD_V2.json，按README填充外部审批字段并独立冻结SHA；QUEUE_PACKAGE_SHA须取本包已审外部PACKAGE_SHA256.json SHA，不能信任未经审阅的新自签名值。review模板里的queue_manifest_sha256和其余冻结链不得改动。target output parent保持不存在；执行根目录已存在，用于supervisor日志/lock/review。

```bash
PY=/workspace/guardfed_envs/celeba-cu128-20261009/bin/python
PKG=/workspace/guardfed_checks/celeba_valid_gpu_remaining440_resource_gate_v2_prepared_20261009
REVIEW=/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/ROOT_REVIEW_REMAINING440_RESOURCE_GUARD_V2.json
# 由root填已审外部常量；模板本身没有执行权限。
QUEUE_PACKAGE_SHA=__ROOT_REVIEWED_QUEUE_PACKAGE_SHA256__
REVIEW_SHA=__ROOT_REVIEWED_EXTERNAL_REVIEW_SHA256__
"$PY" "$PKG/remaining.py" inspect --review "$REVIEW" --review-sha256 "$REVIEW_SHA" --package-sha256 "$QUEUE_PACKAGE_SHA"
```

inspect仅核源码/元数据/review/guide身份，不导入Torch或执行CNN。root确认新scope与实时资源后，在包外生成conf，用两SHA常量替换conf.template的占位符，只targeted reread/update/start guardfed_celeba_valid_gpu_remaining440_resource_gate_v2_20261009；不start旧464，不创建新巡检。program的run命令等效为：

```bash
/usr/bin/nice -n 10 /usr/bin/ionice -c 3 "$PY" "$PKG/remaining.py" run --review "$REVIEW" --review-sha256 "$REVIEW_SHA" --package-sha256 "$QUEUE_PACKAGE_SHA"
```

不能同时运行手动run和supervisor。队列仅fresh一次，失败不得原样restart；保留原失败与partial。后续root离机按新V2源绑定逐chunk验收，本wrapper不登记任何cohort。当前旧460无需重验/重推理来启动此准备包。
