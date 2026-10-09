# CelebA mechanism evidence tools

只读验收、统计与差集备份；不训练、不冻结dispatch、不接test、不调度，也不把进程退出当完成。使用现有 `worker.checked(original, job, job_path)` 和冻结 `scripts/run_revision_ablation.py:checked_result`，仅补足完整cohort、逐轮diagnostics、支持分母及独立Full身份核对。

可优先运行Full100严格只读验收。依赖为Python≥3.9、现有PyTorch（CPU加载state_dict即可）、当前冻结 `scripts/run_revision_ablation.py`、原机制manifest/PROTOCOL及manifest首个正式job，以及已发布的独立 `final_evaluation_prepared_20261009/model_inventory.json`。该库存文件SHA必须为 `3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd`，可从本地原文件复制到服务器任意只读路径；不需要重新恢复100旧job文件。

服务器例子（由主代理执行；本工具不访问服务器）：

```bash
python -B evidence.py inspect-full \
  --repo /workspace/GuardFed-celeba-expanded \
  --stage /workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1 \
  --full-inventory /path/to/verified/model_inventory.json \
  --output /path/to/new/full-inspection-snapshot
```

每次output必须是新目录，保留旧快照。只有 `FULL_REUSE_VERIFIED_100` 和退出码0表示100条全部通过；缺失、无效或保留failure均非通过。核对全部科学输入的实际SHA（core、loader、CNN、cache builder、manifest/metadata/images/available、官方attribute/split），原归档result/checkpoint/job身份、70轮轨迹与diagnostics、none mask、seed/alpha/scene、162770train/19867valid及分区/根样本ID、终轮三指标同checkpoint、有限权重。历史worker实现hash保留在输出，允许与当前worker不同；科学数据输入不允许漂移。

`selfcheck.py` 的首个本地检查使用一份真实已接受Full的原始归档model/result字节，调用原 `checked_result`，通过正门及16种科学输入/root/模型/failure拒收检查。全部四组本地验收通过，合计34个拒收边界：真实Full归档、800条真实设计结构/手算统计、新结果原checked路径及root-ID绑定、差集备份篡改/重复ID/混checkpoint。测试使用的人工数值仅是软件断言，不是机制实验结果，没有渲染缺失800正式数据的示意表。没有执行图像推理，本地验收不等于服务器Full100验收。

Full历史为98条cu128、2条cu130；新controls预期cu128，driver595与历史环境不同。排除2条cu130 Full的敏感性分析要同时移除相应配对比较：完整cohort时剩98条Full、784个新control配对；non-IID Benign与S-DFA场景均为n=9，跨全部10场景和non-IID五攻击的完整seed面板均为n=9，IID五攻击仍n=10。Seed91001参与recipe选择；其他9个也已观察过validation，不能称前瞻独立确认。保留负效应、恒定预测与固定balanced控制的RNG边界，不声称每项机制都必要。

`evidence.py` 是已封存v1，SHA `1c0961ae991d75d32d3269e967ac6bfdcdfb783afafcfcc965445a99a179617b`，主代理已用它完成服务器Full100只读门检；本工具没有访问服务器。其原字节保持不变，dispatch receipt继续引用该v1证明。后续结果验收/统计/备份使用独立 `evidence_v2.py`，SHA `fd3064dd81ab1a312e854cebbf89d592cb5d08054a1ecd6bdd7744f3a039c90d`。v2补上新control到同seed、分布、场景Full库存的root/train/eval IDs直接绑定、client分区数量、root clean/synthetic支持，遇到已存sensitive支持时必须一致；没有修改原manifest/job/worker/adapter/PROTOCOL。两版本职责与来源明确分开。

后续900记录的验收与统计：

```bash
python -B evidence_v2.py inspect \
  --repo /workspace/GuardFed-celeba-expanded \
  --stage /workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1 \
  --adapter-dir /workspace/GuardFed-celeba-expanded/deployment/celeba_mechanism_20261009 \
  --full-inventory /path/to/verified/model_inventory.json \
  --output /path/to/new/inspection-snapshot
```

输出 `inspection.json` 保存accepted_new_ids/reused_ids、invalid、pending、保留failure/无效原始结果及审计的身份；`statistics.json` 与 `per_scene_summary.csv` 给出每分布×5场景×9variants原报告native ACC百分比/AEOD绝对TPRgap/ASPD的均值、sampleSD(ddof1)、n及complete。`per_seed.csv` 保留逐seed值，`paired_per_seed.csv` 记录variant−Full差（ACC为百分点）。跨场景先在每seed内平均，只让完整场景面板进入跨seed统计，缺失面板保留但不填均值。统计中的reported_native指原结果报告值，未执行新的raw/native/shared后处理。只有100Full、0新正式结果时，其他variant为n=0/均值空，整体为PARTIAL；不会成为900完成。任意运行退出码0只说明快照可读，必须检查status/count/invalid/pending。

首次增量备份使用显式新账本；之后去掉 `--initialize-ledger`，沿用同一已核账本与新archive名：

```bash
python -B evidence_v2.py backup \
  --inspection /path/to/inspection-snapshot/inspection.json \
  --ledger /path/to/backups/verified_ledger.json \
  --archive /path/to/backups/new-increment.tar.gz \
  --initialize-ledger
```

先验证账本中每个原receipt SHA、archive SHA和每成员SHA及前后链，再只打包accepted_new_ids与已备份集合的差。包含新model/result/job/config/log/variant ledger/acceptance、sourcefreeze和本快照；不重复旧Full权重，只保存其原恢复链引用。dispatch receipt以完整原始字节复制，`strict_prerequisites` 及任何新增字段原样保留。failure-only新证据可以单独生成零模型增量；重复调用无新证据时不创建空归档。若原始文件在验收后变动、日志未纳入快照或源不匹配则拒绝，应新建inspect快照，不覆盖旧快照/归档/partial/failure。不要把新空账本用于已有备份的阶段，否则无法判断既有差集。

复制archive及receipt到另一台机器后，显式做恢复核验：

```bash
python -B evidence_v2.py verify \
  --archive /local/copied/new-increment.tar.gz \
  --receipt /local/copied/new-increment.tar.gz.receipt.json \
  --output /local/new/offserver-verification.json
```

核验全部成员后才可以把该copy纳入已验证恢复链。备份命令仅标记 `VERIFIED_LOCAL_BACKUP`，不宣称已离机；verify记录source_host/verification_host及different_host_observed，主代理仍需结合实际目标位置核对。写归档阶段失败保留已生成的partial及失败receipt；前置校验失败不生成归档。不覆盖，也不自动重试。

`FILES_SHA256`覆盖本目录持久文件，排除自身和bytecode。局限：本地只验收了软件行为与一份已接受历史Full，未测新800正式结果、未做GPU等价或当前批次离机恢复；900模型raw/native/shared后处理与整个返修完成均仍是另阶段。
