# FLGMM screen 启动与增量备份交接

当前已启动的是32项70轮valid-only搜索。服务 `guardfed_celeba_flgmm_screen`，PID18899；2026-10-09T09:28:52Z首次实际记录：GPU0/PID18911 round2，GPU1/PID18912 round1，CPU1/nice10，0完整接受、2活动、30等待。主800仍PID9179/8活动。首轮不是科学样本完成，后续以真实progress与严格接受为准。

- 服务器独立release：`/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2`
- release seal：`aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4`
- 正常supervisor配置：`/etc/supervisor/conf.d/guardfed_celeba_flgmm_screen.conf`
- 前台wrapper：`/opt/supervisor-scripts/guardfed_celeba_flgmm_screen.sh`
- 原v1失败release：同父目录的`release`，保持不动；失败解释已在v2封存evidence内。
- 本地release源码：`E:/OneDrive/文档/GuardFed/tmp/celeba_flgmm_screen_20261009_v2_frozen_release`
- 本地source/startup备份链：本目录`BACKUP_CHAIN.json`。

已离机验证启动source archive的67成员（66封存输入＋seal），及预检、授权、启动和首进度4项实际receipt。当前没有已接受模型，`accepted_job_ids=[]`；不要把启动备份称为32结果备份完成。

增量接受入口沿用已封存的`screen_common.accepted`，内部调用原`source/accept_result.checked_result`，再核source/data before/after与模型身份。例如在release_v2中，用已存在cu128 Python执行只读收集：

```python
from pathlib import Path
from screen_common import HERE, accepted, local_identity, repo_identity
protocol, manifest = local_identity()
repo_identity(Path('/workspace/GuardFed-celeba-expanded'), protocol)
accepted_ids = []
for item in manifest['jobs']:
    if accepted(item, HERE / 'runs' / item['id']) is not None:
        accepted_ids.append(item['id'])
```

任何checker异常必须保留并报告，不在此处跳过或改统计。新备份取`accepted_ids - BACKUP_CHAIN.accepted_job_ids`，沿用主代理现有增量归档/离机逐member SHA验收流程。纳入新增任务完整目录及已关闭log：model、result、job、config/provenance、acceptance、screen_identity、progress、state、diagnostics；只补尚未备份输入或receipt。保存成员路径/大小/SHA、archive SHA、上份备份引用和新accepted IDs，目标验证通过后再更新链。失败与部分任务另作诊断证据保留，不列accepted IDs，不打包增长中的活动模型或日志为完整产物。

32完成后生成的summary仍标`COMPLETE_ACCEPTED_BACKUP_PENDING`；全量严格验收和完整离机链通过才报告阶段完成。不自动发起100项确认或final test，不改候选/评分/seed，不自动retry/partial resume。
