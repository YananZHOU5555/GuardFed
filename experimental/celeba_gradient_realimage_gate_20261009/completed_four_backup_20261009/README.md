# 四项梯度真实图像门检已严格接受并离机核验

Fed-NGA-gradient 和 Huber-BRFL-gradient 各完成 IID/Benign、non-IID/S-DFA 两项真实 CelebA 三轮探索门检：完整 train162770、clean root16277、valid19867，4 CANARY、12实际轮、240条同点客户端梯度。原封存 `gate.main summarize` / `checked` 全部通过；攻击符号 oracle、样本加权/同点身份、零 local optimizer step、有限终轮模型和同 checkpoint native 重预测一致。

86成员增量 archive 全体 SHA/size 已通过原封存 v4 验证器在不同主机核验；原 protected 源码/数据 after hash 一致，正式 GPU 队列继续增长。原失败和负结果保留。此归档不重复备份图像缓存。

这些是短程接入门检，**科学性能表记录为0**，五项正式协议选择仍未冻结。四项均为恒定负类预测（ACC0.5166859616、AEOD0、ASPD0），不能作为方法性能或公平性优势证据。没有开启正式64搜索、70轮训练、额外 seed 或 test 评价。

原 loader 会 materialize 全 split 属性/划分元数据，包括 test 尾部；未读取 test 图像、test 推理/拟合/选参，不能称 untouched test。具体身份、结果、耗时、原源数据 hash 和 GPU 同期进展见 `strict_delivery.json`；离机证明见 `offserver_verification.json`。
