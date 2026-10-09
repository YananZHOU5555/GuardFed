# 九方法 900 项 valid-only 重放：v3 封存入口

本目录只为已有九方法库存的 900 个 checkpoint 准备严格存储绑定和有界重放。已完成的真实图像证据仍是上一层封存 v2 的两条 Full canary；两条 native 三指标与原接受结果差均为 0。v3 新图像推理为 **0**，900 未派发，正式评价协议仍为 `PREPARED_NOT_FROZEN`。

本目录不覆盖另 800 项新机制 controls。机制 900（800 new + 100 Full）须在真实 70 轮接受后另行适配；其 Full100 与本库存重叠，两队列合并最多 1700 个独特 checkpoint。这里的 raw/native/shared 是 valid 上的实现复核，不能记作 final test 或机制后处理完成。

## 封存字节和实际验证

- `replay_v3.py`: `abb4560dd1c6752b9017a739381d2c54e68f52e278075c8e225a1c67df35cc6e`
- `selfcheck_v3.py`: `452f8eb4e8228b69163bd41bc41ba43c1c5901b9bc0b1a4b12984fe82e6ecab3`
- v2 科学推理：`8476d651bd281d97d6fed42feac5569b45ab0e881b047a7962201e6c0b300803`
- 原库存：`3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd`
- storage map：`e160416101ccb82b42224c9ff1bb337de45ef72fa251197063d1c15e2910f949`
- restore bundle 收据：`0cfa391b0d317622551d4c6526f2d14ec33c3dda0f3e9e523066bd3718cbcf3a`
- 实机 restore acceptance：`5114c2cd96e5b8ffaf46e40a341619dd8b3547f89d417263c19c6e7f1f33bf77`
- 冻结 core：`cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed`

本地 selfcheck 已通过 4 组、12 个拒收案例：真实 900/2700 map 与 2400 bundle 完整性，重复/漏 seed、路径越界、来源 archive/member SHA 篡改、恢复链错绑；从原归档提取真实非 Full 原结果，经原 `checked_result` 完整接受，再换入另一个真实 checkpoint 拒收；root ID 和原 job output 篡改拒收；两条 v2 显式复用、重复 ID 拒收与 partial n。没有用模拟性能数字填表，没有做 v3 图像推理。

首个实机只读 `inspect` 拒收了缺失的 12 种原 adapter 源码 SHA。原始 `.log` 和 `.json.failure.json` 保留，后者 SHA 为 `81865be8225e6343b9cca092bd278cb8b94a2e13b3c5c10bf50c1324a8d8f881`。主代理恢复精确源文件后，用新输出名完成只读门检：**42 项共享输入前后相等，2700 artifacts 全部 SHA/大小匹配**。实机收据 `inspect_900_after_exact_source_restore_verified.json` SHA 为 `c6320a614eceb216b1bc94a6f5bf7c835be36fd34bf4c5b617ed17ac8efc8e4b`；离机再次逐项对照原库存/map 与全部 12 种 adapter SHA，通过证据见 `remote_inspection_verification.json`。

一次外层 `/usr/bin/time` 不存在使 Python 未启动，该原始 log 同样保留；移除非必要计时 wrapper 后，以新的 output/log 名运行上述验收，没有安装依赖、修改代码或重试图像推理。

## 存储与科学身份

Full100 继续读原路径（98 expanded、2 tuning）；其余 800 只读独立 `artifact_store`，每项 checkpoint/result/raw job 都与原库存和原归档成员 SHA/大小相等。原库存与原 job 字节及历史 output 不改；原 `checked_result` 仅通过单独 runtime 字典定位已绑定的新存储。v2 完整校验和推理函数的代码对象原样复用，私有 globals 只绑定读取路径。

900 原训练环境为 886 cu128 + 14 cu130；Full100 子集为 98 cu128 + 2 cu130。原路径的 892 expanded + 8 tuning 与这个运行时分类不同。当前重放环境是 cu128 CPU，已测的两条 v2 和计划 26 条都选原 cu128；不据此声称其余 cu130 项等价。seed91001 曾参与 recipe 选择，库存中 90 项带该标记，不能把全部十 seed 称为独立留出。`cohort_history.json` 原样列出这些边界，没有新的性能测量。

`inspect` 全核 2700 文件与库存中所有科学输入的源码、数据 SHA，并复核前后身份。adapter 按库存 12 种 SHA 在 `repo/deployment/baseline_adapters_20260928` 内搜索精确 `.py/.json`；缺 SHA 明确拒收。原源码可由主代理从已核归档恢复到该树独立子目录，不覆盖同名异字节文件，不静默忽略缺源。

批次开始/结束各完整 hash 共享源码和数据；每 worker 开始/结束复核共享输入 stat/解析路径，以及自身三文件完整 SHA。保持原 root/train/valid ID、分区与样本数；图像是原 RGB64，valid19867 + 原 clean-train root16277，CPU 原 CNN，无 optimizer/梯度，权重前后逐 tensor 不变。标签解析只到 train+valid 前缀；身份 hash 不解释 test 标签。

`accept` 再调用原结果验收、重建真实 root，核保存 margin/ID/数组 SHA，从 root margin 重新拟合三视图并重算预测/混淆矩阵和指标。原 native 容差固定 `1e-12`，raw margin > 0 的 tie0 与校准 >= threshold 保持原差异。退出码为零必须再严格接受；失败证据不删、不自动重试。

## 只读 inspect 和显式批次

远端环境与公共参数：

```bash
PY=/workspace/guardfed_envs/celeba-cu128-20261009/bin/python
CHECK=/workspace/guardfed_checks/celeba_final_valid_replay_20261009
RESTORE=/workspace/guardfed_checks/celeba_validation900_restore_20261009
COMMON=(--inventory "$CHECK/inputs/model_inventory.json"
  --storage-map "$RESTORE/storage_map.json"
  --storage-map-sha256 e160416101ccb82b42224c9ff1bb337de45ef72fa251197063d1c15e2910f949
  --restore-receipt "$RESTORE/restore_bundle.json"
  --restore-receipt-sha256 0cfa391b0d317622551d4c6526f2d14ec33c3dda0f3e9e523066bd3718cbcf3a
  --restore-acceptance "$RESTORE/restore_acceptance.json"
  --restore-acceptance-sha256 5114c2cd96e5b8ffaf46e40a341619dd8b3547f89d417263c19c6e7f1f33bf77
  --repo /workspace/GuardFed-celeba-expanded)
"$PY" "$CHECK/v3/replay_v3.py" inspect "${COMMON[@]}" \
  --output "$CHECK/v3/inspect_900_after_exact_source_restore.json"
```

以下仅为主代理审阅身份与资源、明确授权该 phase 后的执行命令；准备本入口未执行它：

```bash
"$PY" "$CHECK/v3/replay_v3.py" run "${COMMON[@]}" \
  --ids FedAvg_IID_Benign_seed91001 --workers 1 --max-wall-seconds 1800 \
  --output "$CHECK/v3/phase1_useful"
"$PY" "$CHECK/v3/replay_v3.py" accept "${COMMON[@]}" \
  --batch "$CHECK/v3/phase1_useful" --output "$CHECK/v3/phase1_acceptance.json"
```

`run` 是前台有界程序，外部服务管理由主代理决定；这里不创建监督服务。默认一 worker，显式输入 IDs，首个失败停止本批新增派发，仅终止自己创建的子进程。所有现存及后续线程限制在分配的八 CPU，Torch/OMP/MKL8、interop1、nice10、idle I/O；辅助 OS 线程多于 8 不算失败。`--all` 是另一项显式授权，当前不使用。

## 实测计划与去重收集

`throughput_plan.json` 固定了 1/2/4/8/11 共 26 个互斥新 IDs，覆盖九方法、双分布和三个场景；排除两条已接受 v2 canary。每条均为原 cu128 checkpoint。phase 逐次严格接受后才推进，不预填耗时/吞吐。当前实际 worker 上限 11，CPU16..103；CPU104..111 留给独立梯度 pilot，FLGMM0..7、Hybrid8..15，加正式 GPU worker8 threads，名义总预算120 < 已见 quota122.88。实现保留 max12；未测12不得声称12最优。主代理按实测 accepted jobs/total batch wall、有效 CPU、RSS、线程 affinity 和 formal 前后 round/失败数选择并发；这些观测不构成受控训练减速实验。

累计收集只接外部审阅后的 acceptance SHA。示例中须把占位 SHA 换成真实审阅值：

```bash
"$PY" "$CHECK/v3/replay_v3.py" collect \
  --inventory "$CHECK/inputs/model_inventory.json" --include-sealed-v2 \
  --acceptance "$CHECK/v3/phase1_acceptance.json" REVIEWED_ACCEPTANCE_SHA256 \
  --output "$CHECK/v3/cumulative_after_phase1.json"
```

`collect` 校验全部 52 项 v2 seal，再显式复用两个真实 canary；批次间重复/外来 IDs 拒收。只有 900 个不同库存 ID 全接受才标完成；partial 保存 n 与 missing IDs。两项 v2 目前有效，v3 新推理 0，任何 overall900 完成均为 false。

本目录的 `FILES_SHA256` 独立封存 v3，上一层 52 项清单和所有 sealed evaluator/mechanism v1/v2 文件保持原样。未验证项是 v3 首次真实 worker/runner 执行及后续全部 900 项图像重放；不得用 selfcheck、恢复成功或输入身份核验替代真实推理验收。
