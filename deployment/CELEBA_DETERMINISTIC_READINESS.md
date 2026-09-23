# CelebA确定性执行：正式前验收完成

当前训练源码 db80872。旧38ed607探索结果与strict v1的CUDA median失败全部保留。新执行使用strict deterministic、关闭cuDNN benchmark/TF32；仅strict CUDA坐标median走CPU同语义操作，AD2+算法与训练超参数未更改。

已完成四项验收：
- GF/FA各自在GPU0/1的fulltrain/valid首轮checkpoint逐tensor bitwise相同；全部轨迹与候选/权重诊断exact。
- 3轮matched双路/四路及四路/八路全部checkpoint、指标、聚合诊断exact。
- 六条新strict攻击2轮小valid canary全部完成，敏感属性翻转与FedSA实际生效，无label改动。两轮常数预测只说明管线通过，不作鲁棒性结论。
- 两方法固定10轮fulltrain/valid学习核对完成，不取最佳轮、不调参。GF raw/cal ACC=0.74334/0.72638，AUC=0.81410；FA=0.73771/0.71209，AUC=0.81221。多数类ACC=0.51669，两者非常数预测。

## 吞吐

四任务3轮：双路123.29s，四路81.49s，吞吐1.513倍。
八任务3轮：四路162.72s，八路111.98s，吞吐1.453倍；八路GPU峰值约4018MiB/卡，平均利用率约81%。
建议总并发8作为实测起点；每GPU4worker。测试仅GF/FA Benign alpha5，单次顺序测量含启动/哈希/加载，不能当作全部正式方法/攻击的精确70轮速度预测。容器CPU统计包含其他任务。

## 正式草案

results/revision_20260923/celeba_formal_v1/manifest.draft.json 共240唯一条件：
4方法 × 3攻击 × IID/non-IID × 10seed。
固定70轮、Adam lr=.001、batch64、root=.1、官方完整train/test，无增强；候选/筛选/重加权/攻击参数继承原协议。
上述ready快照生成时尚未freeze或启动正式训练，也未读取test做评估。父代理随后已审阅通过，负责freeze并以总并发8启动，先3seed的72条件，再完成240条件；实时状态以正式manifest/status为准。所有source与cache哈希已冻结；旧70轮明确为旧执行探索证据，不冒充新执行结果。

关键文件：
- results/revision_20260923/celeba_formal_v1/readiness.json
- results/revision_20260923/celeba_deterministic_acceptance_v2/comparison.json 与 comparison.md
- results/revision_20260923/celeba_formal_v1/canary/validation.json
- deployment/celeba_checks/deterministic_learning10_analysis/evaluation.json 与 evaluation.md


## 协议操作与字段解释

父代理负责版本提交、推送、freeze和正式启动。本子任务所有诊断已经结束，不再启动任何训练，也不再修改prepare或已记录哈希的源码。

正式执行入口为 scripts/run_revision_ablation.py，使用 results/revision_20260923/celeba_formal_v1/manifest.json。
第一批使用 run 的 --first-seeds 3 --concurrency 8，覆盖SEEDS前3个种子123/456/789，共72条件；成功结果经现有checkpoint/source/config检查后自动复用。随后去掉 --first-seeds 限制继续相同manifest，补齐240条件，不能另建相同结果混入独立样本数。失败原地保留，先查明原因，不隐藏失败或更换seed。

compas_preprocessing_version=train_only 是从通用ExperimentConfig继承的COMPAS专用字段。CelebA分支在载入COMPAS loader之前已返回，不读取该字段；CelebA预处理应以RGB64 cache manifest、缓存SHA、官方split/属性元数据及每次image_data_contract为准。无需为这个无效字段重跑任何canary。图像始终以Smiling为目标、Male为敏感元数据，不将敏感列拼进像素输入。

终轮70的主报告口径保留既有方法实现：GuardFed校准指标，其他方法原始指标；同checkpoint raw/cal补充诊断应分列注明，不能混列排序。所有性能重复、10轮学习pilot、2轮canary及旧执行探索结果均不计入正式10seed统计。
