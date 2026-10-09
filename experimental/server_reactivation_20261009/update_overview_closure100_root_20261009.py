"""Refresh current human entrypoints while retaining historical suffix bytes."""
from pathlib import Path
import hashlib, json
ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'docs/server_deployment_20260923/training_20260923'
CHECKS=TRAIN/'server_reactivation_20261009'
sha=lambda b:hashlib.sha256(b).hexdigest()
state=json.loads((TRAIN/'TRAINING_STATE.json').read_bytes())
main=state['celeba_mechanism_v1']; live=json.loads((CHECKS/'latest_formal_live.json').read_bytes())
assert main['scientific_results_offserver_verified']==104 and main['three_view_new_models_offserver_verified']==100
assert main['latest_paired_three_view_table']['complete_scenes']==10
overview=ROOT/'docs/返修实验总览.md'
marker='以下为 2026-10-04'.encode()
before,history=overview.read_bytes().split(marker,1)
assert sha(marker+history)=='08b94e001b4ff78474706db0a63cc2c1b626fe5664c01edfafbaefb53fe9b029'
text=before.decode('utf8')
text=text.replace('18:08 UTC实测；18:12 UTC验收','18:29 UTC实测；18:36 UTC验收')
text=text.replace('69/65℃','68/66℃').replace('CPU11.02/122.88核，RAM75.14GB','CPU17.61/122.88核，RAM76.11GB')
text=text.replace('104/800新增已严格离机，其中minus_U100完整、minus_C4部分；现有已审论文表仍为9场景，十场景native表正在验收',
    '104/800新增已严格离机，其中minus_U100完整、minus_C4部分；十场景native表540统计标量独立验收')
text=text.replace('[九场景native表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim92_20261009/TABLES.md)',
    '[十场景native表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009/rendered/TABLES.md)')
text=text.replace('92份minus_U已验收离机；9完整场景、90对模型；2对不完整Sp-DFA保留不入均值。1458统计及1656计数指标独立核验',
    '100份minus_U已验收离机；十完整场景、100对模型、200记录；1620统计、1800计数指标及810展示单元独立核验')
text=text.replace('[九场景三视图论文表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim92_20261009/snapshot92/TABLES.md)',
    '[十场景三视图论文表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md)')
text=text.replace('18:08实测及后续收尾是发布后本机更新','18:29实测、104份native及100份机制三视图收尾是发布后本机更新')
text=text.replace('三视图独立完成九场景，不代表其余七variant或全部800训练完成。',
    '删除U的三视图独立完成十场景，旧184记录/243统计行精确保持；不代表其余七variant或全部800训练完成。')
overview.write_bytes(text.encode('utf8')+marker+history)
assert sha(overview.read_bytes().split(marker,1)[1])==sha(history)
handoff=CHECKS/'MONITOR_HANDOFF.md'
marker2=b'# Historical handoff snapshots'
_,history2=handoff.read_bytes().split(marker2,1)
assert sha(marker2+history2)=='f9dfedaf289de16f1226d2356ccfa451034e2e5e05b19348f0c08095e5a0a1ee'
published=state['latest_publication_verification']
top=f'''# GuardFed 当前巡检交接

主训练实测{live['checked_utc']}；辅助来源快照17:42/17:43 UTC，验收记录在后续时间闭合。下一次须重新核实时状态。

- 服务器ssh -p60350 root@89.22.197.55，实例52183675；repo/workspace/GuardFed-celeba-expanded。先读/etc/vast-agents-guide.md，SHA42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa。用户授权sglang停止，文件/模型保留；213.224.31.105不自动切回。
- 主机制guardfed_celeba_mechanism_formal：{live['queue_completed']}观测终轮，104/800严格验收离机，100Full显式复用；{len(live['active'])}活动/{live['pending']}等待/{len(live['failed'])}失败，轮次{'/'.join(str(r.get('progress',{}).get('round')) for r in live['active'])}。实测JSON latest_formal_live.json SHA{sha((CHECKS/'latest_formal_live.json').read_bytes())}。
- 双5090利用率100%/100%、68/66℃、RecoveryNone；CPU{live['cpu_used_cores_2sec']:.2f}/{live['cpu_quota_cores']:.2f}核，RAM{live['memory_used_bytes']/1e9:.2f}GB，余量{live['disk_free_bytes']/1e12:.3f}TB，OOM0/近期错误0。冻结8并发，不因低CPU或交接瞬时低GPU重启。
- Native最新差集12严格离机，总104=U100完整+C4部分；archive1f8fed22d210028b18aa4682bacd80ec24c4c3ef1f9d461b26e108bb06a20f37，ledger8fc6b8b1ea9fdb4591c3da5e83ebe51284803720479c3d436f3a8c737430999d。只打包已验收ID差集，不重复旧Full/权重。
- 删除U三视图已100/100严格离机，after92最后8正常EXITED/0残留失败，ROOT9050eb059a797c70f0ca977294989b5ae5757286dbc85b36d529012cb5ab72ee，archive235837afdd336df8db7e3f224c6598f6f9b0a216a2deb10dcf11f3cf54987577。原92/Full不重推，C4排除；全部旧闭合服务禁止重启。
- 完整十场景三视图表celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md已独立验收，ROOT20d1031448701938063a398fc5416e404b1e8f0549807c85dab0c0204618a2a0；1620stats/1800计数指标/810显示cells，原184records/243rows精确。native100另核540stats。Full5CPU95GPU/98cu1282cu130，controls100CPU/cu128；负结果、validation选择/历史test暴露保留，不声称每项不可或缺/整个机制完成。
- FLGMM26/32、组合10/32已严格验收离机；来源快照分别26终轮/2活动/4等待与10终轮/1活动/21等待，不当当前进度。LATEST_BACKUP各链已绑定；未完整选recipe或启动formal100/test。
- 原after82审批数量错误CNN前0完成/空输出和FLGMM旧chain字段collector失败原证据保留；独立V2分别准确10/7一次通过，不重启旧失败目录。
- 九方法900三视图/9页PDF、2052校准统计、旧TableII480原值追溯均验收；Fig3候选未采纳且原执行身份不足，提交版正文源项目待路径。
- 仍待800机制剩余696严格验收及其余七variant三视图、8方法完整覆盖/忠实规格和作者待决正式协议、冻结最终评价及正文/rebuttal。U100完成不等于全部返修完成，不从方案派发test或新方法。
- 原三小时聊天任务guardfed-training-health当前PAUSED；本会话无automation_update接口，未编辑调度器或新建cron/Windows监督。supervisor训练持续运行不等于聊天定时巡检已恢复。
- 最新已验证Git{published['commit']}，{published['committed_blobs_sha256_verified']}blob；十场景收尾在该发布之后。以最新publication_verified凭据核实际远端，不把本机修改算已推送。

下次先读RUNNING/STATE/最新冻结协议，合并检查SSH、正确worker和真实轮次增长、错误/OOM/双GPU/实际cgroup资源。只有source/data/jobhash一致、无重复worker且恢复机制严格跳过已验收结果时，才有限恢复外部中断；代码/数值错误保存证据，不循环重试、不改科学配置/seed/指标/driver/实例、不购买资源。仅对重要变化、故障、完成或需用户处理通知。

'''
handoff.write_bytes(top.encode('utf8')+marker2+history2)
assert sha(handoff.read_bytes().split(marker2,1)[1])==sha(history2)
print(json.dumps(dict(status='CURRENT_OVERVIEW_AND_HANDOFF_UPDATED_HISTORY_BYTES_EXACT',overview_sha256=sha(overview.read_bytes()),handoff_sha256=sha(handoff.read_bytes()))))
