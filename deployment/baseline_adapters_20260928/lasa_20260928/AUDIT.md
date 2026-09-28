# LASA 纯聚合adapter与检查记录

2026-09-28；状态：CPU固定输入检查通过；尚未接入worker、没有真实图像训练/GPU确定性或正式结果。

## 来源与默认参数

作者[官方仓库](https://github.com/JiiahaoXU/LASA)，commit `8477367a4e8708cde264f7572805040c650af59f`；对应[WACV2025论文](https://openaccess.thecvf.com/content/WACV2025/papers/Xu_Achieving_Byzantine-Resilient_Federated_Learning_via_Layer-Adaptive_Sparsified_Model_Aggregation_WACV_2025_paper.pdf)。本目录 `upstream/` 保存原始 `algorithms/defense/lasa.py`、`utils/mask_help.py`、`main.py`，没有修改。哈希在 `verification.json`。

官方 main.py 默认：`sparsity=0.3`、`lambda_n=1.0`、`lambda_s=1.0`。sparsity指丢弃比例。不能把GuardFed act_keep_ratio混成LASA调参。

调用：`delta, diagnostics = lasa_aggregate(updates, sparsity=.3, lambda_n=1., lambda_s=1.)`。输入为同名同形同dtype/device的参数差分字典列表（FP32/FP64），输出可直接加到global模型的差分；不修改输入、global模型或其他运行状态。不接收root数据、样本数或攻击者数量，也不强制选n−f。

## 保留的官方计算行为

1. 拼接每个客户端所有参数，按整体L2范数的torch lower median裁剪。
2. 每客户端将2D/4D权重拼接，以 `int(total*(1-sparsity))` 的top-k末位作阈值，严格 `abs(weight)>threshold`；因此ties会多丢弃，不能改成>=。1D bias不掩码。sparsity=0跳过阈值稀疏化。
3. 层范数用float32，再以numpy median/std构造modified z-score，按 `<lambda_n` 选；符号比例统计乘`1-sparsity`，同样构造分数并按 `<lambda_s` 选。两集合交集为空才回退所有客户端。
4. 官方 `clipped_local_updates` 引用的字典在后续被稀疏函数原地修改，**实际聚合的是裁剪且稀疏后的更新**。adapter保留实际行为；tie案例验证最终矩阵也归零。没有擅自改成“稀疏用于筛选、dense用于聚合”。
5. 全相同norm/sign时std=0，官方NaN比较产生空过滤集，再走交集/回退。adapter保留该选择行为，诊断用JSON null表示非有限分数，避免在JSON写NaN。

诊断保留每客户端整体范数、clip、零范数索引、掩码层、阈值与请求保留数量、各层实际非零量；每层norm/sign统计、median/std、分数、两个过滤集、最终集合及是否fallback；参数与来源commit随诊断返回。

## 明确的适配与数值边界

- **设备/副作用适配**：mask随输入device，不硬编码cuda；整体范数的lower median只在CPU求值，避免严格CUDA确定性模式下median带索引算子不可用；只返回delta；输入克隆；输出顺序固定，排序客户端集合不会改成员。这些不是新的过滤规则，GPU整体路径仍待门检。
- **零整体范数修复**：原代码0/0导致NaN；adapter将零向量保持零。其计入median裁剪及后续官方选择，没有删除客户端。`zero_norm_clients`给出证据。median为0时所有更新被裁成0，这是保留原裁剪定义的结果。
- **非有限输入拒绝**：原版仅删除NaN输入却仍以原num_selected_users建索引，Inf还可能继续传播；adapter对NaN/Inf及范数/输出溢出报错。不会静默重试、删客户端或挑选剩余结果。
- **不合法配置拒绝**：sparsity须[0,1)，lambda正且有限；top-k保留数0时报错；正sparsity但无2D/4D参数时报错。无整数buffer支持，调用者须只传参数差分；需要BN模型时另定buffer协议，不能默认套用。
- **局部全零sign保留原法退化处理**：符号统计0/0导致该层sign过滤为空，然后按原有空交集fallback。它不是零整体范数修复的隐藏延伸。

以上每项需要随正式实现说明，不声称未经改动的官方逐字拷贝。普通有限、非退化输入按官方行为对齐。

## 已运行门检

运行 `python tmp/celeba_baselines/lasa_20260928/check_adapter.py`，CPU torch2.8.0+cpu。

5组官方逐层bitwise输出/选择集合对齐：默认随机CNN张量、不稀疏、高稀疏、top-k ties、相同向量零std；另外检查全零/混合零范数明确修复、NaN/Inf和非法配置拒绝、输入不变和严格JSON诊断。官方oracle仅在内存移除mask创建的`.cuda()`并插入只读集合记录；保留全部选择/聚合运算，磁盘官方文件未改。

## 接入与有界调参建议（未运行）

新增明确方法ID和独立三参数字段，进入run身份、manifest及checkpoint元数据；聚合时直接调用adapter，输入必须是已施加冻结攻击的模型差分。现有老LASA启发式结果不得覆盖。先冻结两GPU小CNN/两轮canary检查、原始raw报告及shared校准派生结果身份，再进入验证搜索。

可从作者默认开始，以独立验证数据检查周围小网格，例如sparsity `{0,0.3,0.5}`，lambda_n/lambda_s `{1,2}`；这是**建议的12配置探索**，不是论文推荐最优，也不是已授权/已启动队列。先做默认与1–2个邻近配置的短门检能避免全网格运行接口错误；性能选择仍按整体冻结协议，保留失败和負结果。

还未完成：GPU确定性、runtime开销、真实图像的攻击注入身份、70轮checkpoint验收、多seed科学结果。CPU算子通过不能代替这些证据。
