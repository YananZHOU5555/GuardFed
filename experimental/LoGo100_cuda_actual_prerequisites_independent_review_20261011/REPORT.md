# LoGo100 原始证据迁移与独立 CPU 环境：V1 源审

结论：712 成员的紧凑身份闭包和流式接收守卫通过本地源/元数据审查；V1 直接执行前需修两处工程边界。未迁移、安装或执行科学核验，未访问 F 文件。

1. `transfer_originals_proposal.py:40–42` 在流式 BrokenPipe 或等待超时的异常路径只终止进程并重新抛错，尚未写入正常路径第35–38行的 SSH stdout/stderr 和 command/exit 记录。远端拒收原因可能丢失。建议异常时终止并排空子进程，独占保存原异常、stdout/stderr、退出码和命令；保留部分远端目录，禁止自动重试。
2. `create_cpu_environment_proposal.sh:7,11–14` 仅拒绝已存在的 venv，未拒绝悬空符号链接；managed Python 目录也未要求全新，且 `uv python find 3.10` 没有断言返回解释器属于该专属 managed 目录。建议两个目录都拒绝 `-e/-L`，安装后核所选解释器的真实路径包含在专属目录，再创建 `--copies` venv。无需改变依赖版本或科学规则。

通过的边界：21 成员封条与全成员 SHA/大小正确；100 条原 LoGo 记录按原 EXACT100 顺序保持，700 个七角色 artifact SHA 一致，另有12个去重源码成员，总712件/51,278,594B。inventory 原路径为702个F文件、10个E小文件；审核仅核保存的身份元数据和作者原哈希记录，未再次读取这些物理文件。路径拼写 F:/ 与 F:\ 在 Windows 语义相同，原科学文件没有重写。

接收端核指南、批准/索引 SHA、固定新 server namespace、磁盘余量；流式索引在先，精确712 regular-only 成员、禁绝对/穿越路径、独占创建、流式与落盘双重 SHA、集合完整性均保留。发送端新核 F 实体卷 label/Healthy/Operational/UniqueId 并逐件 SHA；不生成本机临时压缩包，内部盘只写精简记录。该物理 F 守卫在此仅为已读源码，未来执行必须重新实际通过。

环境提案固定 Python3.10、CPU torch2.8.0+cpu、numpy1.26.4、netcal1.3.6、scikit-learn1.7.1、scipy1.15.3、pandas2.3.1、pyro-ppl1.9.2、gpytorch1.15.2；后五项有原 Windows 环境实读 metadata 支持。Python patch、传递依赖和跨 Linux 的原 pickle 兼容仍未测。仅安装独立 managed Python/venv，不触碰现有 driver、CUDA 环境或训练服务。原 LoGo 固定 fit seed1719、20 image-ID 虚拟 cohort 与 CPU 后处理限制保持；不是实际 client 公平性，也无跨 runtime 等价结论。

执行前仍需 root 的 fresh CPU113/SMT369、所有 PID/task 受限 mask、nominal quota、protected 服务与指南门，以及 OS affinity/thread1/nice10/idleIO。08:25:50 的54 PID/642 task 观察只证明当时无冲突，不是资源预留；安装 shell 不自行实现这些门。FedAvg CUDA100 真实 root/cache、迁移独立采用、安装和 pickle/import 接口检查亦须分别真实闭合。

本次仅编译5个 Python 源且未导入科学模块，三个 proposal 文件均为 LF。审核辅助检查曾因直接比较 Windows 路径字符串而报 AssertionError；改用 PureWindowsPath 并保留 SHA 比较后通过。这是审查记录层错误，未改作者源、原数据或科学规则。

角色：独立于本包作者 calibration_attribution；此前参与 old800 API、FL caller 源准备及 LoGo/FedAvg cache 源审。V1 旧封条保持有效；后续只审两处 successor 差分。不存在迁移/安装/科学接受 PASS。
