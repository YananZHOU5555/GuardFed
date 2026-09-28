# 独立真实图像pilot入口（未执行）

`run_pilot.py` 接受 `--repo/--job/--out`。只临时包装冻结core的 `aggregate_round`：方法 `LASA-official` 调本目录adapter，其余调用不变。训练、攻击、root、CNN、valid评价、progress callback和终轮checkpoint都由实际repo的 `core.run_experiment` 完成；finally恢复原函数。不修改core文件、主worker或服务器。

**必须标明是LASA官方聚合适配于local Adam模型差分**。此名称区分旧LASA-core启发式，不能据此宣称原论文完整训练协议一模一样。原生输出无额外阈值校准；共享校准以后另作同checkpoint派生评价。

两份job已冻结：`job_Benign.json`、`job_S-DFA.json`。均为non-IID alpha5，seed91001，3轮，20客户端/4恶意槽，batch64，local Adam LR .001，1 local epoch，full train162770、official valid19867；root10%，原冻结攻击/重加权参数继承base_job。adapter三个参数独立保存在job.adapter，默认 .3/1/1。数据限额均0。脚本主动拒绝改变成70轮、test或数据子集，避免把pilot悄悄当正式运行器。

job.source_hashes取已核验完整覆盖base_job中的核心源码与CelebA图像缓存/元数据/原始标注身份；job.adapter_source_hashes冻结adapter.py和run_pilot.py。启动时逐文件SHA校验（images.npy完整读取一次，可能花一些时间）。repo中缓存symlink允许，按真实文件内容SHA判定。哈希不符报错，不能就地自动刷新hash。修改入口后须在审阅新版本后明确重建job，不能绕过源身份检查。

主代理review后可复制**整个本目录**到服务器指定deployment目录；以下为具体调用示例，未执行。`--out`必须是未存在的新目录，重复执行不会覆盖旧pilot或自动重跑。

```bash
CUDA_VISIBLE_DEVICES=0 GUARDFED_CPU_THREADS=4 /workspace/GuardFed/.venv/bin/python /workspace/GuardFed-celeba-expanded/deployment/lasa_20260928/run_pilot.py --repo /workspace/GuardFed-celeba-expanded --job /workspace/GuardFed-celeba-expanded/deployment/lasa_20260928/job_Benign.json --out /workspace/GuardFed-celeba-expanded/results/revision_20260928/lasa_pilot_v1/Benign
```

```bash
CUDA_VISIBLE_DEVICES=1 GUARDFED_CPU_THREADS=4 /workspace/GuardFed/.venv/bin/python /workspace/GuardFed-celeba-expanded/deployment/lasa_20260928/run_pilot.py --repo /workspace/GuardFed-celeba-expanded --job /workspace/GuardFed-celeba-expanded/deployment/lasa_20260928/job_S-DFA.json --out /workspace/GuardFed-celeba-expanded/results/revision_20260928/lasa_pilot_v1/S-DFA
```

资源示例不是并发调度指令，应由主代理与现有共享校准任务协调。CUBLAS配置在import torch前设置，core图像路径负责strict deterministic/noTF32/cudnn设置；入口末尾核验strict模式实际开启。LASA整体norm median使用CPU值以避开已知CUDA median限制，但其余GPU路径仍须本pilot实测。

成功产物：`job.json`、`provenance.json`、`progress.json`、`model.pt`、`diagnostics.json`、`result.json`、`acceptance.json`。逐轮诊断含客户端范数、mask阈值与非零量、层norm/sign集合、fallback及原法参数；每次返回更新仍走原core apply_update。失败写`failure.json`并抛错退出，不循环重试。结果包含adapter源码身份及delta适配说明，终轮checkpoint与result/diagnostics均存SHA。

验收检查：必须3个完整轮次和3条聚合诊断、full train/valid合同、有限ACC/AEOD/ASPD、strict模式以及model文件身份；原core也执行攻击audit。这里的PASS只表示单seed3轮接口和有限值通过，不代表收敛、有效防御、优于基线或正式结果；父任务应再读取诊断确认选中集合/稀疏度/攻击是否合理。

本地已做：入口help、语法与offline wrapper检查（两份job/源码哈希/边界、LASA拦截/其它方法原样转发）。没有加载真实CelebA、没有CPU或GPU训练、没有连接服务器。运行 `python tmp/celeba_baselines/lasa_20260928/check_pilot.py` 可重复离线检查。
