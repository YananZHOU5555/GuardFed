# exact3 Linux 只读预检

PASS：实测 UTC 2026-10-10T12:05:28.650377+00:00，供 root 在 300 秒新鲜窗口内绑定。没有启动实验或写服务器。

37 个原地文件（2,527,797,281 bytes）全部精确 SHA 匹配，含 images.npy；单 CPU511、nice19、idle I/O 读取。随后刷新所有37文件的 resolved path/size/mtime 身份，小文件再hash，未重复扫描大图像文件；部署的原38封存成员全部匹配。

exact3 原 producer/failure/重复 gate 均无。五服务实际 PID 与入口一致。主队列从11:38的252完成推进至当前256，活动8、失败0；旧8任务全部轮次增长。GPU Recovery None/None，温度[64, 62]℃，OOM计数0，内存余量440698990592 bytes，磁盘余量1060283940864 bytes。

实测 cpu.max=12287999/100000，即122.87999核；eligible mask 0–511是调度许可，不是CPU配额。名义预算76核：主8worker各6 OS线程计48，FL2/Hybrid1/gradient1，remaining620预留8，exact3预留8，管理与I/O额外8。319个休眠main coordinator库线程披露；名义预算不是全部库线程同时唤醒的硬峰值上界。

CPU120–127无窄mask保留冲突；本次有CPU tick增长的宽mask线程，其last_cpu均不在120–127。宽mask允许未来迁移，所以该检查不宣称物理排他占用。

日志、命令、guide SHA、observer源码SHA和原始stdout/stderr均保留同目录。读实测supervisord include=/etc/supervisor/conf.d/*.conf；拟新增guardfed_added_cnn_exact3_gate.conf此前不存在。Linux preflight只证明此时间窗的部署/资源门，native1e-12与科学接受仍待真正执行。
