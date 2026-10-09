# LoGoFair 虚拟人口映射提案

状态：**PREPARED_NOT_APPROVED**。已生成真实样本 ID 的映射及支持检查；没有批准或冻结人口定义，没有 LoGoFair 拟合、CNN 推理或性能评价。

四个 seed91001 条件（IID/non-IID × Benign/S-DFA）的 accepted FedAvg margin cache 均在本机共享校准归档中找到，并核原 archive/member/cache SHA。四者的 root/valid ID 顺序身份及 root 标签/敏感属性一致，因此共用一个 `mapping.npz`，无需复制四套相同数组。模型只引用既有 accepted checkpoint SHA 与归档，不载入、不重打包；后三个模型没有在本任务重新读取权重字节。

映射预先固定为 SHA256(`GuardFed/LoGoFair/virtual-cohort/v1` + NUL + image_id 的正整数十进制 ASCII)，取全部32字节大端整数 mod20。不加入 seed、条件、label、Male 或 score，也不为均衡人数搜索 domain。相同 image ID 在四条件中始终属于同组。NPZ 恰有四个 int64 列：`root_image_id`、`root_client_id`、`valid_image_id`、`valid_client_id`；这里的 client_id 只是既有接口名称，其语义是**虚拟 cohort，绝非真实训练 client**。

人口支持检查：root16277、valid19867；20组 root 人数774–871，valid人数926–1050。每个 cohort 的两个 Male 组都包含两类 root label，80个 root(y,Male)格的最小人数116；没有缺敏感组或缺两类标签。完整80行（四条件×20组）在 `cohort_counts.csv`，逐条件结构在 `population_support.json`。

读取边界：对 margin NPZ 只解码 `root_y` 和 `root_sensitive`；从既有 accepted replay NPZ 只解码 `root_image_ids` 与 `valid_image_ids`。原文件确实包含 valid 标签、敏感属性和 score 字节，整体 SHA 会覆盖这些字节，但代码未取出、解码或评价这些字段。root label/Male 只用于支持检查，不参与映射生成。未导入 Torch、原 bridge 或 netcal。

仅凭人口支持，四个条件已具备未来真实 score-only 小轮门检的样本前提，**不能称门检已通过**。本阶段不读取 score 或拟合阈值，所以精确 threshold tie、分数退化、Beta MLE 收敛和数值稳定性均未评估；两类标签齐全不保证这些条件成立。后续应保留原失败规则，不能通过池化、换 domain、改 beta 或随机 tie 处理消除失败。

下一步需要作者裁定是否接受这种虚拟人口解释，再单独批准真实 score-only 小轮门检。若要求真实训练 client 人口，本提案不适用，需要原训练 client 的独立 holdout 与同源评价协议。原 bridge/protocol/reuse 清单和32草案逐 SHA 保持不变，草案两个 mapping SHA 仍是 null，依然不可执行；本提案 metadata 的 approved=false 同样禁止正式入口使用。

`prepare_population.py` 包含身份、原顺序、同ID重现、反序等变、错误域/类型/重复ID/错顺序/错误cohort拒收与原32草案不变检查。生成入口拒绝覆盖已有提案。`INPUT_BINDINGS.json` 给出四缓存、模型、原结果、源码、配置、root/valid身份与本机输入路径；未补造任何缺失SHA或新的科学结果。
