# Fed-NGA 待集成聚合组件

本目录只实现[原论文公式(9)](https://arxiv.org/html/2408.09539v1)的聚合算子，并非完整 Fed-NGA 复现。没有修改现有 core、服务器或训练协议，没有开始训练。

`normalized_gradient_step(gradients, counts, server_eta)` 接受 `[客户端, 参数]` 梯度矩阵，返回可加到当前模型参数上的负梯度步长：

`step = -server_eta * sum_i (n_i / sum(n)) * g_i / ||g_i||`。

输入不变；输出保留输入 dtype/device。内部双精度归一和加权避免 float32 极小梯度范数下溢；此数值选择应记录为实现细节。每客户端范数使用整个参数向量，不逐层归一。零梯度显式扩展为零方向，其样本数仍保留在总数中，不能删除后重新分配权重。没有合并方向再归一、中位范数恢复或 epsilon 截断非零梯度。

运行CPU检查：`python tmp/celeba_baselines/fednga/fednga.py`。覆盖手算不等权二维例子、符号、客户端梯度缩放不变性、相反方向抵消、零梯度/极小梯度、零权重、不修改输入和错误输入。此检查不证明完整训练方法或GPU复现已通过。

## 接入现有 run_experiment 的最小方案

审计入口为 `tmp/publish-5090/scripts/reproduce_paper_tables.py` 中 `run_experiment`（1617行附近），`train_local_model`（564行附近），`apply_update`（268行附近）。实际图像分支行号可能变化，应按函数定位。

1. 新建独立方法标识，例如 `Fed-NGA-gradient`，不覆盖旧 `Fed-NGA` 分支或结果。新增独立 `server_eta` 及预先确定的日程字段到配置/run身份/manifest；它不是客户端 Adam learning_rate 的别名。
2. 该方法绕过 `train_local_model`：在本轮同一个 global 参数点，计算每客户端损失梯度。可用逐mini-batch累积实现全客户端均值梯度，但期间**不能 optimizer.step**。按实际样本数加权各batch平均损失，不能不等长batch简单平均。原文默认经验风险，若使用现有重加权样本损失必须明示为目标改动。BN/Dropout等模型状态策略也需固定；当前CNN是否有这些模块需接入时核查。
3. 以稳定 `named_parameters` 顺序展平真实梯度，保留参数名/shape映射。`None`梯度采用显式零填充或报错策略；不要把非参数buffer一起归一。每个客户端起点/模型状态一致，不能累积上一个客户端梯度。
4. 训练数据攻击继续使用已冻结的数据变换。模型更新攻击必须另外实现梯度空间的发送消息语义并验收。现有 `apply_foe_if_needed` 接收模型差分，不能直接套用后仍宣称原法；更不能把Adam多步delta除以LR当本轮梯度。Benign canary可先验证严格路径；攻击版在梯度消息语义冻结前不进入正式队列。
5. 对真实上传梯度调用本组件，按同一参数顺序还原返回向量。现有 `apply_update` 做加法，所以直接加这个已经含负号的 step，不能再取反。若复用全state更新器，非参数buffer显式零更新；优先仅更新参数。
6. 保留统一CNN、数据分区、轮数、评价实现；日志记录每轮 server_eta、counts、零梯度客户端、梯度/步长范数及源码/配置身份。服务器步长用验证集独立有界搜索。全梯度与本地Adam多步的计算/优化预算不同，结果表需披露。

接入验收：固定小线性模型，手算每客户端同点梯度后检查一轮参数；验证按mini-batch累积与全batch梯度等价；确认轮内各客户端起点一致；再做数据/攻击身份、终轮、配置、seed、checkpoint及设备确定性检查。上述集成检查尚未执行。
