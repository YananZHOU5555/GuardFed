# old800 fixed3: saved-output 离机验收源准备

仅 SOURCE_ONLY。没有 SSH、F 访问、数组/模型读取、推理、拟合或科学验收；未来 actual hashes 与授权均未填写。原 attempt001 因 `queue.py` 遮蔽 stdlib queue，在 Torch import 前退出，证据保留于 root 原目录。本包只接新的 attempt002；不会把旧失败升为 accepted。

固定三项，顺序不变：

1. `FedAvg_IID_Benign_seed91001`
2. `FedAA-DDPG_policy0.001_keep16_local0.001_IID_Benign_seed91001`
3. `LASA_s0.3_l2_lr0.001_IID_Benign_seed91001_screen`

服务器 BASE 固定 `/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002`，caller 是 `run_queue.py`；对应 service `guardfed_uniform_cuda800_canary_v2_valid`。完整800 cohort 的现有 SHA 为 `d39e29333c40e8a5b63f1344487225abac4ba83216dc32e75f64badc87bc0b71`，仍须由 root 在真实输入中绑定。Full100 单独复用，本包不包含 FL/LoGo 或 future800 结果。

## root 后续一次执行

先确认原 Linux whole 三项均已真实成功，原 producer 已退出，无 canary failure；保存 root source review，字段至少 `source_adoptable: true` 和本包 `source_seal_sha256`。在本包外写实际 `ROOT_INPUTS.json`，填写 template 的部署、cohort、COMPLETE、三份 ACCEPTED SHA 与 source review。运输使用 `root_authorized_transport=true`；审计再绑定真实 transport proof/archive SHA 并令 `root_authorized_audit=true`。占位 null/false 不能执行。

从仓库根目录调用（参数均由 root 实际绑定）：

```powershell
python -B tmp/old800_cuda3_saved_audit_prepare_20261011/transport.py --inputs <actual-transport-inputs.json> --inputs-sha256 <actual-sha> --source-seal-sha256 <FILES_SHA256.json-sha>
python -B tmp/old800_cuda3_saved_audit_prepare_20261011/audit_saved.py --inputs <actual-audit-inputs.json> --inputs-sha256 <actual-sha> --source-seal-sha256 <FILES_SHA256.json-sha>
```

各一次，root 外层必须保存命令/source SHA/退出码/stdout/stderr。任何错误或 timeout 停止，不自动重发。`actual/STARTED.json`、成功或 FAILURE 已存在时审计拒绝重试；transport 的 F 目录/远端 export 亦必须不存在，部分现场保留。输入检查在 audit STARTED 之前失败时由外层命令证据保留，不能称已进入科学 loop。

## 最小归档与验收

未来运输在 CPU110 单线程、CUDA hidden、nice10/idleIO 下执行原 guide SHA 与全线程窄 owner 检查。只对这三个已闭合 producer 查活跃，不要求其它500评价退出。每次 bulk 前用原 `guardfed_local_storage.check_bulk_storage` 验 F 为 Yanan 2TB/Healthy/容量；archive 和解包仅写 `F:/YananResearchStorage/GuardFed/uniform_cuda_old800_20261011/canary3_attempt002`，无内盘 fallback。

归档显式包含三条完整原 inventory 对象的 compact selection、原 author/approval/deployment/queue/start/resource/source seals、三份 ACCEPTED/receipt/binding/saved acceptance、三份预测 NPZ，以及原 scalar metadata.npz。NPZ 含 root/valid margins 与三视图 prediction；label 仅从原 metadata 读取 train/root 与 valid 前缀。不给 8MB 完整800 scope、9MB inventory、已备份模型、图像、训练日志建立副本；scope 以真实 SHA、完整800有序 ID hash 和所选三个全对象追溯。archive/member 数只读实际 index，不预设。

原 CUDA7 成功工具的 safe member/hash/extraction 尾部与 guide/CPU110 资源门 raw bytes 保持；改动只作用于 namespace、固定三项、原800身份/字段与 explicit file mapping。完整逆恢复在 `*_SOURCE_INVERSE.json`。保留一次本地静态替换重叠失败 `SOURCE_PREPARATION_FAILURE.json`；它发生在生成运输源码前，不是科学或传输失败。

Windows 只用原 evaluator/replay 函数及原 saved-science 的七条 AST 语句重算保存预测的三指标和八 base counts，不调用 fit/CNN。保留原 dtype、native/raw `margin > 0`、shared `margin >= threshold`、同 checkpoint、原 root ID/valid19867/train162770、原70轮训练身份和 native≤1e-12。FedAA/LASA 的 method/config 已按原800记录绑定，不采用 Full/FL 的科学 schema。原已拟合 payload 的 SHA/有限性与 JSON group key 恢复照原规则核验，不在 Windows 重拟合。

若未来成功，范围仅 3 records / 27 metric scalars / 72 integer base counts / 9 prediction rules，仍需 root 独立接纳。不能称 Windows whole refit、CPU/CUDA bitwise 训练等价、完整800、论文表或 final test。既有 CPU 或混合路径表与原失败继续独立保留。
