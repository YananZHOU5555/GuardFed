# 精确三条 CNN 三视图图像接口候选

**PREPARED / SOURCE ONLY / NOT DISPATCH AUTHORIZED。** 本包没有执行 CNN、`rebuild_root`、shared fit、图像读取、SSH 或科学接受；`CHECK_RESULTS.json` 的 PASS 仅指本地源码与元数据门检。三个代表记录取自原桥已注册且 root 已采用的记录，不重选 recipe：

| Method | Frozen accepted ID | 原实验范围 |
|---|---|---|
| FLGMM | `FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91005_fullcoverage` | 原 fullcoverage 中一个结果 |
| CosineFairnessHybrid | `CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage` | 原 fullcoverage 中一个结果 |
| Fed-NGA-gradient | `FedNGA_eta0.01_non-IID_Benign_seed91001_screen` | 原 screen 中一个结果，不是新获胜 recipe |

Huber 无本 registry 中的 70 轮 strict/offserver/root proof，拒收；LoGoFair 不是普通 CNN native，拒收。本包不是 full100、主终点选择、排名或 final test。

## 最小复用及原证明

`originals/bridge.py` 保留 root 已采用桥的原字节，ROOT SHA 为 `be5e6c949254960f55b765ec976f1e681a26450416f156ca237965dedc87fe69`。每条仍走该桥的原 strict→offserver→root、raw index、job/source/config、70 轮 valid19867、checkpoint 与所有已接受 artifact SHA 身份校验。原 GPU provenance 保留在 `original_training_provenance`；本候选不在 CPU 调用 GPU checker，不替换 CUDA 查询或伪造 CUDA strict。

原 900 `replay_one` 的旧方法 `validate_original` 无法接收新增方法，因此只在私有 AST 中替换为已采用外部身份票据，checkpoint 调用改为固定服务器映射，并改两处报告标签。`RUNTIME_BINDING_DIFF.patch` 给出全部差异；逆恢复 AST exact。原 17 个科学函数（包括 `rebuild_root`、`check_native`、`extract_and_predict`、shared fitter 接口）source SHA exact，另 7 个 runtime/helper 函数 AST unchanged。数学公式不复制重写。

原 core/root采样/client分区/root噪声、CPU 推理、同 checkpoint 三视图、root-only shared fit、valid 标签只在冻结 predictions 后评分、模型前后 tensor identity 和 native `1e-12` fail-stop 均沿用原实现。native/raw 为 `margin > 0`，shared 为 `margin >= group_threshold`；shared 七字段原样。CPU 与原 CUDA native 不一致会保存差异并停止，不能降低容差或称运行时等价。

## 文件及跨主机路径

`MANIFEST.json` 是固定三 ID、原配置/环境/证明/模型 SHA、明确 Windows/F→Linux 的路径契约。`path_map` 的 `package` 仅指需复制的本包 source 和小证明；`server` 指服务器原件。Windows/F 原路径作为来源 ID 保留，不作为 Linux 路径使用。

- FL 原件：`/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage/runs/<ID>/`。
- Hybrid 原件：`/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/stage/{jobs,runs}/`。
- Fed-NGA 原 job：`/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010/jobs/<ID>.json`；原结果：`/workspace/celeba_gradient_screen64_v2_results_20261010/<ID>/`。
- 原 repo/data cache：`/workspace/GuardFed-celeba-expanded/data/celeba/derived/rgb64_v1/`，留服务器，只 hash 原件并按原接口 mmap；只 materialize train+valid labels prefix、root 图像和 valid 图像，test 图像/标签不进入推理或拟合。

本包没有复制 result/raw diagnostics、模型、数组、cache 或压缩档到 E。F 原件只做 byte SHA 和 JSON 身份只读检查；F 卷实测见 `STORAGE_CHECK.json`。未来输出只能留服务器；若后续下载 arrays/model/log/archive，须再次实核 F=`Yanan 2TB`/Healthy/容量，落 `F:/YananResearchStorage/GuardFed/`，不得 E fallback。

## 未来实际执行契约（本次未执行）

root 先独审本包，然后复制整个小源码包到 `/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010/source/`，不复制 F 权重或 data cache。Linux preflight 必须逐一实测 MANIFEST 的服务器原路径/SHA、selected producer 已离开、无重复 gate、原服务健康、GPU健康、cgroup/RAM/磁盘、实际可用 CPU mask 及全部线程冲突。

候选绑定 **CPU120–127 / 8 threads / interop1 / nice≥10 / idle I/O / CUDA hidden / max1 process / loader0**。原 CPU112–119 被 remaining620 占用，故仅工程资源重绑定；120–127 尚未实测可用。原 `live_snapshot/resource_gate` 继续要求机制服务 RUNNING、quota≥32、memory headroom≥8GiB、所有自有线程不逃逸。额外 fresh root preflight 要求包含本 gate 的 nominal reservations≤实测quota，不以 logical128 或历史122.88代替当前测量。

实际 root source review 至少含 `source_adoptable=true, package_sha256=<本包封条SHA>`。实际 Linux preflight 须 `status=ROOT_LINUX_EXACT3_PREFLIGHT_PASS`、UTC 距启动≤300s、`cpu_affinity=[120..127]`、`eligible_cpus`、`actual_quota_cores`、`nominal_reserved_cores_including_gate`，以及下面全部 true：`no_duplicate_gate, all_thread_cpus_free, source_model_data_hashes_verified, selected_producers_quiescent, services_healthy, gpu_health_verified, cgroup_and_memory_headroom_verified, storage_headroom_verified`。

实际 root authorization 须 `status=ROOT_AUTHORIZED_EXACT3_VALID_IMAGE_INTERFACE_GATE`，绑定真实 `package_sha256, manifest_sha256, source_review_sha256, linux_preflight_sha256`、原序 exact三ID、CPU120–127、`device=cpu,max_processes=1,test=false`。**本包不提供填了未来 SHA 的 authorization/preflight，也不把 source review 当执行授权。**

通过上述实际检查后，root 可用以下命令（所有 SHA 必须替换为真实文件值；本包没有运行）：

```bash
env CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
 taskset -c 120-127 ionice -c 3 nice -n 10 \
 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B \
 /workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010/source/candidate.py \
 --package-sha256 "$PACKAGE_SHA" \
 --source-review /workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010/ROOT_SOURCE_REVIEW.json --source-review-sha256 "$SOURCE_REVIEW_SHA" \
 --preflight /workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010/LINUX_PREFLIGHT.json --preflight-sha256 "$PREFLIGHT_SHA" \
 --authorization /workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010/AUTHORIZATION.json --authorization-sha256 "$AUTHORIZATION_SHA" \
 --output /workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010/outputs/attempt001
```

root 需在独立服务器 attempt 中保存 stdout/stderr；源码/授权/资源阶段拒收同样保留，不重入旧 output、不自动重试。原 scientific/native 差异失败保存后停止。成功文件的状态也只能是 `EXACT3_VALID_INTERFACE_PASS_NOT_ROOT_ADOPTED`，`root_adopted_new=0`；后续离机/member/array复核及 root 采用是另外的实际步骤。

## 本地已完成检查与未检查

`python -B check_preparation.py` 已一次完成：3 个原身份 exact、17 个聚焦拒收、17 个原科学函数 source SHA、7 个 runtime AST exact、replay_one 逆恢复 exact、Python 编译。仅源码/元数据，不 import Torch、不调用科学函数。已有原桥64门检以其 root review 为来源，不重复冒充新实验。

未进行 Linux 导入/真实路径和资源观察、root 重建、图像推理、shared fit、native 差异测量或离机科学接受。这些均是实际派发前后必须完成的事实，不是本 source-only PASS 的结论。
