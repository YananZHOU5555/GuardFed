# FLGMM100 v2限定复审：两项P2已修复

结论：可采用当前v2源码进入root实际绑定与canary阶段。本次没有新增阻塞项；这不是部署批准，也不证明实际GPU同horizon数值/RNG等价。原32实际recipe采用凭据由root另行持有，本次没有重新选择recipe或重验该科学结果链。

v2封条为`fc9cd4133345d74949d22cddca12868393b8ac6ac6920d2f1deb293611584c30`。核验全部28成员及before/after SHA后，原23继承文件仅3项改变：`bind_stage.py`、`screen_common.py`、`run_fullcoverage.py`；其余20项字节完全一致。新增5项均为v2说明/检查文件。两个入口文件的唯一修改是最先显式拒绝`sys.flags.optimize`，共享入口在导入worker之前拒收。queue改动仅为初始化严格成功数、严格成功后递增、将该数写入进度。实际差异见`ACTUAL_SOURCE_DIFF.patch`。

1. 原P2进度问题已修复。执行实际`run`函数AST，注入惰性path/process依赖，没有创建训练子进程。7个已严格接受skip、1个失败退出、1个接受peer：报告8成功而非9；exit0但缺严格terminal输出、另1个接受peer：报告1成功。两种失败均停止派发、等待peer闭合、不调用summary。正向94个已接受skip加2个成功退出：报告96，summary仅调用一次。
2. 原P2优化模式问题已修复。五个实际CLI入口`bind_stage/run_canaries/run_one/run_fullcoverage/canary_reference`各运行普通`--help`正向通过；各以`-O`、`-OO`、`PYTHONOPTIMIZE=1`、`PYTHONOPTIMIZE=2`执行时，20次均在显式guard处非零退出，记录正确拒收消息。共25个本机help子进程，不调用绑定或科学运行函数。

原`source/worker.py`科学body、adapter/作者源、`rng_capture.py`、`run_canaries.py`比较器、`canary_reference.py`horizon/RNG仪器、`source/accept_result.py`严格接受器、recipe/job定义、原scope协议模板均字节不变。未重复原review已通过的系统审计。

证据：`CHECKS.json`含实际队列fixture结果及25入口stdout/stderr，`audit_v2.py`是本次有限检查源码，`SOURCE_PINS.json`绑定原/新封条及本次比较输入。初次读取错把差异文件名写成`V2_DIFF.patch`，只读路径探针失败保存在`READ_PROBE_FAILURE.json`；实际文件`V2_SOURCE_DIFF.patch`随后正常读取，无源修改或科学执行。

限制：未SSH、未导入Torch/NumPy科学模块、未执行CNN/训练/真实canary、未绑定stage、未改accepted cohort/STATE/canonical/Git。这里的通过只解决两项工程P2；Linux资源、原4 legacy实际选中记录及两个真实同horizon比较仍由root按原冻结入口核验，既有canary和严格接受规则保持不变。
