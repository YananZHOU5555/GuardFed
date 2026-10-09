# 精确11项机制三视图 valid 重放准备

**PREPARED_NOT_APPROVED**。没有派发、SSH/网络、CNN、张量载入、训练或 test；没有 runtime installer、服务或资源分配。旧 next8 全17封存成员保持原字节，后续只提议审阅本11项，不再派发历史 next8。

## 精确输入

- `minus_U_non-IID_F Flip_seed91001` 至 `minus_U_non-IID_F Flip_seed91010`（保留空格）；
- `minus_U_non-IID_FedSA_seed91001`。

原生71库存减去已闭合60，严格选择11。原60及先前原生68记录逐JSON不变；其他729 pending只有ID，不造模型SHA。原70轮、alpha5、full train162770、valid19867、root-only、原checkpoint/source/data/adapter/config/训练runtime身份均保留。

本次新增3来自真实归档 `b9de831bed4854445a0b5ad7b82426c5b7aa6dded4e40cb1a951c3033abc9aee`：用原 evidence_v4.verify_archive 重核全部54成员，再用原 terminal_checks/partition_identity 核3份原结果。原8沿已验证 next8 封条及其94成员校验证据复用，本次不重包/复制/载入模型。原始结果/job成员SHA与重建inventory身份索引明确分开；runtime只继承原result记录。

60项重放闭合证据实际SHA `a7a563390de455914b33cf62064d674347e0c26754a778b9a344ac99bdc4fac3` 已核，与新原生71严格/离机证据不是同一种结果。Full100仅引用；这里未合并Full三视图验收，标MISSING直到连接原始SHA绑定的合格批次，不新推理、不复制Full权重。

## 可独立复核

1. 按 `FILES_SHA256.json.members` 逐文件SHA256及size核对。封条自身SHA由交付消息给出。
2. 在本机新建空检查输出目录，运行 `python -B tmp/celeba_mechanism_valid_incremental_next11_20261009/selfcheck.py --output-dir <该空目录>`。它只读取身份、核源函数和执行标明为fixture的标量门限/拒收检查；不导入torch/numpy，不调用bridge runtime。
3. 读 `MINIMAL_SOURCE_DIFF.patch`（相对已批准next37）和 `source_reuse_proof.json`。原16科学函数与桥接10函数逐源/AST精确复用；两个身份函数仅scope/count/排除边界替换，反向还原后逐源相等。

`selfcheck.json`记录42拒收PASS：包括历史8代替当前11、FedSA误配F Flip模型、新nonIID错误alpha/攻击/root、旧60重入、pending/Full/test/GPU、放宽native1e-12等。原check_native函数的1e-13通过/1e-11拒收使用明确标记的标量fixture，不是实验结果。

`prepare.py` 为本机库存重建入口，采用独占创建输出，已存在文件不会覆盖；无需重新运行才能复核交付。原科学预测、root拟合、raw/native/shared规则无改动。

## 后续界限

下一阶段仅建议复用已有fresh-child顺序协调器：单进程8线程、CPU112–119、nice10、idleIO、CUDA隐藏。这里未测实时配额或占用，未批准使用这些CPU。主代理需审封条和准确11项、核资源/空输出并给外部SHA批准后才能运行；模板保持拒收状态。不得自动重试或放宽门禁，也不加入本71快照之后的新模型。
