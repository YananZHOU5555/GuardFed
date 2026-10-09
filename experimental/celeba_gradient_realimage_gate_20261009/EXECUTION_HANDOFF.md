# 4条真实梯度探索门检已启动，尚未完成

实测时间1791530814.9532144：独立服务`guardfed_celeba_gradient_realimage_gate` RUNNING，PID12370，CPU104–111、nice10、idle IO、CUDA_VISIBLE_DEVICES为空。torch实际计算线程8；OS记录30线程包含库线程池，启动146.23秒累计942.83CPU秒，平均6.45核，RSS3127928KiB。这是实际过程测量，不能换算为科学结果或提速结论。

第一项Fed-NGA/IID Benign已产生真实梯度。client0完整7351样本，CE分母7351，optimizer0步，共同全局点未改变；raw/upload范数0.016974168732564677、攻击消息oracle逐位相同，梯度SHA`f783243c45add8f9caa2fd15678850485a275cdf66a30271400a70f0f0bc4382`。记录中的默认`actual_foe_mode=state`只是Benign客户端的未使用配置；`actual_attack_types=[]`，没有执行state攻击。此时共4条client梯度，之后离机快照捕获7条，均不是完整一轮/终轮验收。

原800队列实测完成5、活动8、失败0；与启动前同job轮次比较确有增长，活动轮次为67/65/66/14/9/11/4/3。原服务、冻结源码、训练参数、Hybrid进程与CPU8–15未改。原22封存项在部署前后和首梯度归档时再次SHA通过。

当前完整图像门检验收数0，科学表记录0；Fed-NGA第二场景及两个Huber运行尚未交付完成证据。不把首梯度、PID或显存称PASS。

## 执行差异与保留证据

原gate因数据symlink resolve到repo外而前置拒收，尚无训练输出；六个共享输入的目标、size、原SHA全部核实一致后，主代理复核并批准独立wrapper（SHA`3c61385450d99824a3983627e61b98c672c3016788b8016d7830435a0b0b8946`）。只允许这六个精确目标，其他路径仍用原守卫。失败记录与最初执行附件版本保留于远端`prelaunch_identity_failure.json`、`prelaunch_history/`及原部署归档，原22文件不改。

额外资源探针最初假定cgroup-v1而没有找到quota；主机实测是cgroup-v2，`cpu.max=12287999 100000`，配额122.87999核，内存约71.6GB/519.2GB；批准的最大120线程计划低于配额。该探针错误没有修改训练方法或重启任务。正式启动使用主代理已明确分配的资源，随后实际PID/affinity/nice/IO均核验。

完整split的Smiling/Male元数据可被加载，包括test尾部；本次没有test图像推理、拟合、评分或选参，不能称untouched test。此边界已完整写入dispatch和每项provenance，详见EXECUTION_NOTES。

## 离机备份与后续入口

`launch_snapshot_20261009/`保留批准收据、launch preflight/receipt、FIRST_GRADIENT_EVIDENCE、首项raw job/provenance/source身份/资源与7条梯度审计。17成员归档SHA及16内容成员SHA在本机通过；归档SHA：`0033c79f9f5431a177e9afa1f11b9a8bd6c41e9e217f24bb4a706a4ad835b649`。原准备SETUP SHA仍为`c2c9325f7c2be4fc9dbd5164990f6e880a132bd26661e79e2b9d524ffbaca0cc`。

服务已启动，不再手动再次run/start；autorestart=false、startretries=0。仅按主代理现有检查读取服务、各run梯度/round增长与failure；不创建新监督任务。完成后先核所有4条终轮与same-checkpoint repredict、源/数据前后身份及原GPU增长，再离机备份全部模型/raw evidence。

主代理可在服务自然结束后，从该远端目录使用下面的只验收入口，显式注册已批准路径后调用原summarize门；此命令当前未执行：

```bash
cd /workspace/guardfed_checks/celeba_gradient_realimage_gate_20261009
/workspace/guardfed_envs/celeba-cu128-20261009/bin/python -c 'import shared_cache_wrapper as w; w.bind_shared_paths(); import gate,sys; sys.argv=["gate.py","summarize","--dispatch-receipt","dispatch_receipt.APPROVED.json"]; gate.main()'
```

若数值/身份/Huber求解失败，保留failure、partial与诊断，不自动调整阈值、eta、seed或重试。原64候选与5项正式决定保持PREPARED/UNRESOLVED，本4pilot不能自动批准正式搜索。
