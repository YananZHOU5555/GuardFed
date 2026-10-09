# CelebA valid-only 原图重放前置验证

本目录只做终轮模型预测实现检查。正式最终评价仍为`PREPARED_NOT_FROZEN`，四项决策未冻结；未执行test、未启动900批次、未训练、未创建服务或监督机制。现有sealed evaluator、mechanism evidence v1/v2及冻结core保持原字节。

两条真实CPU canary均已通过。本机核原始归档SHA与27成员SHA，并用此前封存接受的valid标签缓存独立复数三视图18项指标与48个混淆计数，全部精确一致；没有重新推理700或900模型。32项运行输入的SHA、大小与resolved路径在重放前后保持一致，两个模型的所有权重张量SHA不变、无optimizer/梯度。详见`offserver_verification.json`与`remote_receipts/`原始收据。

| 检查对象 | valid/root图片数 | native最大误差 | 实际耗时 | 平均有效CPU核 | 峰值RSS |
| --- | ---: | ---: | ---: | ---: | ---: |
| Full IID Benign seed91001 | 19867 / 16277 | 0 | 108.022秒 | 6.581 | 2489160 KiB |
| Full non-IID Benign seed91002 | 19867 / 16277 | 0 | 121.936秒 | 6.577 | 2489160 KiB |

运行前8个、推理后16个辅助OS线程均绑定同一CPU集合16–23。formal八条活动训练在影响快照间分别推进2–3轮，0 failed，服务持续RUNNING；没有对照实验，不能定量声称零慢速影响。两条通过不证明其他898条或CUDA/CPU数值普遍等价。

`replay.py`支持原九方法900库存的逐条CPU valid重放，必须显式给`--ids`或`--all`。本轮只授权两条原cu128 Full Benign：IID seed91001与non-IID seed91002。每条读取完整valid19867张RGB64缓存图像及原clean-train root16277张，使用原CNN、batch64、FP32和冻结阈值搜索。原config的CUDA训练身份不改，模型推理明确放在CPU。native与原接受结果逐指标比较，固定误差上限`1e-12`，任一失败保存证据并停止，不自动重试。

raw视图保留`margin>0`且tie为class0；native AD2+与shared视图使用`margin>=group_threshold`。校准仅接收原clean-train root标签；valid标签在fits和predictions固定后才进入评分。三视图是验证检查，不作新性能表。没有调用或伪造sealed evaluator的final dispatch gate。

读取标签时只打开metadata.npz的Smiling/Male NPY头与前182637条标量，结束位置是valid末行；不载入整个标签数组或test尾部。不执行原全量loader。官方属性文件与库存其他输入仅作不解码的字节SHA核验。train/root/valid ID、client counts、分组support、全部原source/adapter/job/result/checkpoint身份均核；读前读后复核文件SHA、大小和resolved路径。仓库原CelebA路径是既有symlink，允许已登记输入指向原缓存，仍拒绝绝对/父级路径注入。

Linux入口使用已有`/workspace/guardfed_envs/celeba-cu128-20261009/bin/python`，不安装依赖。检查进程nice10、idle IO；把自身所有现存线程绑定启动前允许CPU的`sorted(...)[16:24]`，后续线程继承同一8核边界。Torch/OMP/MKL为8，OpenBLAS/NumExpr与interop为1，loader workers为0，CUDA隐藏。记录实际OS线程数、CPU时间、有效核数、RSS、GPU及formal队列快照；辅助线程数不代表更多可用CPU核。训练服务、并发和GPU worker不改。

本地`selfcheck.py`四组通过、14项拒收：独立900网格与身份、标签前缀/test哨兵/unsafe object payload、输入字节和路径篡改、真实原train元数据重建两条root与client分区、root ID篡改、固定容差及非有限指标。它不含真实图像推理，远端实测以独立收据为准。原始symlink拒收栈、当时代码/plan/selfcheck保存在`history/symlink_precheck_failure/`；该失败发生在推理之前。

本机独立核对首次把scorer的`group_confusion_counts`键误写为`groups`，出现KeyError；原始失败与当时代码保留在`history/offserver_verifier_key_failure/`。只修验证helper的读取键及已提取相同字节的复核方式后通过，没有重跑canary或改变任何原始收据/容差。复核入口：`python selfcheck.py --receipts`；四组结构检查入口：`python selfcheck.py`。

后续900需要显式受信storage_map接入，详见[STORAGE_MAP_CONTRACT.md](STORAGE_MAP_CONTRACT.md)。当前实测v2仅从原库存路径读取；其他800的独立artifact store尚未接入此版本。不能把历史output改写到库存，不能把准备/恢复完成算作原图重放完成。

本轮远端命令（输出必须为未用的新目录）：

```bash
/workspace/guardfed_envs/celeba-cu128-20261009/bin/python \
  /workspace/guardfed_checks/celeba_final_valid_replay_20261009/inspect_inputs.py

/workspace/guardfed_envs/celeba-cu128-20261009/bin/python \
  /workspace/guardfed_checks/celeba_final_valid_replay_20261009/replay.py run \
  --plan /workspace/guardfed_checks/celeba_final_valid_replay_20261009/inputs/valid_replay_plan_v2.json \
  --repo /workspace/GuardFed-celeba-expanded \
  --output /workspace/guardfed_checks/celeba_final_valid_replay_20261009/canary2_cpu_v2 \
  --ids GuardFed-AD2+_IID_Benign_seed91001 GuardFed-AD2+_lr0.0005_drop0.005_Benign_seed91002 \
  --max-wall-seconds 1800
```

原inventory字节与prepared protocol/evaluator副本位于`inputs/`，plan v2绑定本轮replay source SHA。`inputs/valid_replay_plan.json`是失败前plan，保留追溯，不能用于当前代码。输入计划只是valid前置检查身份记录，不是正式最终协议冻结。
