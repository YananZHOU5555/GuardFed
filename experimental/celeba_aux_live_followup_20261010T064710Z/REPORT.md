# 单次辅助队列来源绑定观测

UTC **2026-10-10T06:48:08.473049+00:00**：FLGMM为31/96终轮，原root采用28，本次未接受差集3（IID FedSA seed91010；IID S-DFA seed91002/91003）。活动为IID S-DFA seed91004 round65、seed91005 round19，63 pending；原4复用单列。Hybrid为30/32终轮，原root采用27，差集3；当前lam20/tau0.2/lr0.001 non-IID Benign round9，另1 pending。完整ID见 FINDINGS.json。

两服务RUNNING、冻结源所有成员匹配、未见失败；CPU106全线程无窄亲和owner，无现存collector。先核guide实际SHA `42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa`。实际共享LATEST、FL root/previous离机proof、Hybrid chain/root均按原SHA绑定，具体入口在 INPUT_PINS.json。

FL新增3未达4，Hybrid未齐32；按本轮边界正常停止。**0 collect、0 download、0新接受、0 recipe选择、0 bulk写入**，LATEST/STATE/总览/Git均未修改，无CNN/训练。未发生工具故障或重试。以后若需收取须另核当时CPU和F卷label/Healthy/容量；本轮没有大文件写入。

只调用一次SSH，完整复用此前read-only远端body；本地仅修正已知pending整数的读取并写本轮独立目录。观测字段来自同一次调用内的顺序读取，不是全服务器原子快照，也不替代strict或离机科学接受。
