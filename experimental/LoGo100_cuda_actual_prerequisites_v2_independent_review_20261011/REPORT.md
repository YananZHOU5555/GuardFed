# LoGo100 prerequisite V2：限定差分独审

结论：source_adoptable=true。V1 两项执行边界缺口已修复，没有新的源级阻塞；这不构成迁移、安装、资源授权或科学接受。

发送端第13–25、53–64行保留原异常，BrokenPipe/超时后有界 drain 原 stdout/stderr，最后独占保存 argv、时间、exit、异常及 SHA；若最终 drain 仍超时，明确记录残余错误和可得输出，不能称完整。Popen 启动失败也进入 finally。已完成 communicate 的非零返回不重复读取输出。既有三份日志/符号链接拒收，无自动重试，远端部分现场保留。

环境脚本第7–8、13–20行同时拒绝 venv/managed Python 的既有路径和悬空链接；`uv python find` 返回解释器经 `readlink -f` 后必须处于专属 managed 目录，否则停止。Python3.10、CPU torch2.8.0+cpu、numpy1.26.4、netcal1.3.6 与其余固定依赖、`venv --copies` 均未变。Linux Python patch/传递依赖/原 Windows pickle 兼容尚待真实安装和接口门检。

独立静态检查：10 个封存成员 SHA/大小全符；INVERSE 的三个旧 preimage 原始字节与 V1 全等，新 SHA 对应当前文件；receiver 全文件原字节相同；sender 的逐成员 size/SHA→tar 循环及 need/sha 函数 raw+AST 均同原；3个 Python 源 compile 通过，proposal 仍为 LF。只读了作者6个假进程/metadata fixture源码和保存结果，未再次运行它们、transfer main、安装或任何科学函数。旧712成员/51,278,594B身份结论沿用已封 V1 独审，没有再次访问 F 物理文件。

root 实际执行前需绑定 V2 sender/receiver/inventory/独审 SHA、fresh 新 server destination 与授权；fresh 核指南、F 实体卷和 CPU113/SMT369/all-task/nominal/protected 服务，OS affinity/thread1/nice10/idleIO。08:25旧观察不可当通行或预留。迁移/安装每步一次，失败原件与部分现场保留；安装的实际 stdout/stderr/argv/exit 和 package/executable 哈希由 root wrapper 保存。FedAvg CUDA100/root cache、迁移独立接纳、安装及 pickle/import 门须各自真实闭合，不能由本源审推导。

LoGo 原 source/job/protocol、20虚拟cohort、固定fitseed1719及CPU后处理限制保持；没有 model/array deserialize、CNN、fit、test、STATE/Git、SSH或F访问。独立于本包作者 calibration_attribution；角色沿 V1 披露。
