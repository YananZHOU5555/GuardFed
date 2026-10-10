# exact3 严格离机验收准备与实际结束观察

实际观察：2026-10-10T12:11:46.998347Z，服务 EXITED；三条均有 `NATIVE_VALID_REPLAY_PASS` receipt，`GATE_RESULT=EXACT3_VALID_INTERFACE_PASS_NOT_ROOT_ADOPTED`，8 个成员齐全，无 FAILURE，stdout/stderr 均空。`FINAL_OBSERVATION.json` 保存完整小 receipt 的 JSON 值与全部原件 SHA/bytes；并非原件字节副本，不以本地重序列化 SHA 冒充原件 SHA。三份数组仍在服务器，未下载到 E。

源码已准备并仅通过语法/静态检查，**未执行以下离机 root 重建或 cached-root fit，尚未科学采用**。

## 最小复用

`saved_binding.py` 从原 900 证据的 `062_saved_science.py` 提取 `check_saved`，源片段 SHA 与 AST 保持原样。仅提供外部身份/路径适配的 `mapping_paths/full_hashes/mapped_functions`；记录仍由原已采用 `bridge.identity_record` 复验，未重写阈值或指标公式。其科学调用链仍为原 `rebuild_root → fit_views → predict_views → evaluate_frozen_predictions → check_native`。

这条原函数检查 root IDs 与分区/噪声审计、root-only fit 的全部参数与 diagnostics、三视图逐项预测数组、9 个指标及 24 个混淆计数、native `1e-12`、同 checkpoint 的权重前后 identity。source/checkpoint/result/job/config 另外与固定 exact3 identity 对齐。实际重测的本地 artifact 前/后哈希明确标作本地观察，不虚构远端 wrapper 的 artifact proof。

原 `evidence.py.check_run` 的科学数组段可复用，但其 900/GPU worker proof/queue wrapper 不适用于当前 exact3，不能整函数直接套用。该旧程序的 `root_refit_verified_in_original_remote_strict` 也是旧队列声明，不能转移到新三条。当前准备入口真正调用原 `check_saved` 才能补上独立 cached-root refit。

## Root 后续执行

先在 F=`Yanan 2TB`、Healthy、容量足够时，只下载 exact3 的 8 个成员与原 `metadata.npz`，独立核服务器/离机成员 SHA 和大小。无需下载 images.npy 或重复下载模型；原模型/result/job 的 F 副本已由原 bridge 登记。三条必须全部闭合，失败/常量/负结果一并保留；不得改容差、换 checkpoint、重选 NGA recipe 或自动重试。

随后 root 可显式执行（`<...>` 换成其实际 F 目录）：

```powershell
python -B tmp/celeba_added_cnn_exact3_scientific_acceptance_20261010/verify_offserver.py --bundle "F:/YananResearchStorage/GuardFed/<exact3-original-8-members>" --metadata-npz "F:/YananResearchStorage/GuardFed/<original-metadata.npz>" --output "tmp/celeba_added_cnn_exact3_scientific_acceptance_20261010/ROOT_EXECUTED_SAVED_ARRAY_CHECK.json" --allow-original-cached-root-refit
```

依赖原科学环境中的 numpy/pandas/Torch/scipy/scikit-learn；入口隐藏 CUDA、1 CPU thread，只从保存的 margins 重算 root 阈值，不实例化 CNN、不重新训练、不访问 test 标签。原 metadata SHA 固定为 `161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091`。标签读取复用原 `read_prefix`，仅物化 182,637 个 train+valid 前缀；metadata 的 image_id/split 可全读。

`SOURCE_PINS.json` 固定原函数及文件来源；候选的全部封存成员在执行前重核。`src/data_loader.py` import 依赖固定 SHA，实际 root 验证环境版本进入结果。当前 CLI 只生成 pending-root-adoption 报告，transport验收及最终科学采用仍由 root 独立绑定；输出已存在会拒绝覆盖。

准备范围：一次运行中观察到 FLGMM 首条完成，随后一次合理间隔的结束观察；未 busy polling、未写远端、未执行 fit/CNN/训练。此交付只证明准备来源与实际文件存在，严格数组/root-fit 结果待 root 执行。
