"""Refresh current human entrypoints while retaining historical suffix bytes."""
from pathlib import Path
import hashlib, json, re
ROOT=Path(__file__).resolve().parents[1]
TRAIN=ROOT/'docs/server_deployment_20260923/training_20260923'
CHECKS=TRAIN/'server_reactivation_20261009'
sha=lambda b:hashlib.sha256(b).hexdigest()
state=json.loads((TRAIN/'TRAINING_STATE.json').read_bytes())
main=state['celeba_mechanism_v1']; live=json.loads((CHECKS/'latest_formal_live.json').read_bytes())
assert main['scientific_results_offserver_verified']>=104 and main['three_view_new_models_offserver_verified']>=100
assert main['latest_paired_three_view_table']['complete_scenes']==10
overview=ROOT/'docs/返修实验总览.md'
marker='以下为 2026-10-04'.encode()
before,history=overview.read_bytes().split(marker,1)
assert sha(marker+history)=='08b94e001b4ff78474706db0a63cc2c1b626fe5664c01edfafbaefb53fe9b029'
text=before.decode('utf8')
published=state['latest_publication_verification']
text=re.sub(r'当前更新：[^。]+。', f'当前更新：{live["checked_utc"]}实测；辅助来源时间见下方，验收数字各按实际凭据。', text, count=1)
text=re.sub(r'主机制训练健康推进：\d+项观测到终轮，\d+/800项新增结果已严格验收并离机备份，\d+项活动、\d+项等待、\d+项当前失败',
    f'主机制训练健康推进：{live["queue_completed"]}项观测到终轮，{main["scientific_results_offserver_verified"]}/800项新增结果已严格验收并离机备份，{len(live["active"])}项活动、{live["pending"]}项等待、{len(live["failed"])}项当前失败',text,count=1)
gpu_temps='/'.join(row.rsplit(',',1)[-1].strip() for row in live['gpu_csv'].strip().splitlines())
gpu_usage='/'.join(row.split(',')[2].strip().replace(' ','') for row in live['gpu_csv'].strip().splitlines())
resource_row=(f'| 服务器与资源 | `ssh -p60350 root@89.22.197.55`，实例52183675；sglang已停止，文件模型保留。'
    f'双5090利用率{gpu_usage}，{gpu_temps}℃，Recovery None；CPU{live["cpu_used_cores_2sec"]:.2f}/{live["cpu_quota_cores"]:.2f}核，'
    f'RAM{live["memory_used_bytes"]/1e9:.2f}GB，余量{live["disk_free_bytes"]/1e12:.3f}TB，OOM0 | '
    '[实测凭据](server_deployment_20260923/training_20260923/server_reactivation_20261009/latest_formal_live.json) |')
fo=state['flgmm_screen32_20261009']['latest_readonly_terminal_observation']
ho=state['hybrid_screen32_20261009']['latest_readonly_terminal_observation']
aux_row=(f'| FLGMM / 组合控制 | 已严格验收离机26/32与10/32；{fo["checked_utc"]}只读实测分别'
    f'{fo["observed_complete"]}/{ho["observed_complete"]}终轮、{fo["active"]}/{ho["active"]}活动、{fo["pending"]}/{ho["pending"]}等待、失败0。未验收终轮不计接受，尚未选recipe | TRAINING_STATE对应搜索记录及备份链 |')
text='\n'.join(resource_row if line.startswith('| 服务器与资源 |') else aux_row if line.startswith('| FLGMM / 组合控制 |') else line for line in text.split('\n'))
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
git_row=(f'| Git / 三小时巡检 | 最近证据发布已推送{published["commit"][:7]}，'
    f'{published["committed_blobs_sha256_verified"]}份blob与实际远端核验通过；聊天任务当前PAUSED，supervisor训练持续运行 | '
    f'[发布核验](server_deployment_20260923/training_20260923/{published["proof_path"]})、'
    '[巡检交接](server_deployment_20260923/training_20260923/server_reactivation_20261009/MONITOR_HANDOFF.md) |')
text='\n'.join(git_row if line.startswith('| Git / 三小时巡检 |') else line for line in text.split('\n'))
native_row=(f'| 机制native消融 | {main["scientific_results_offserver_verified"]}/800新增已严格离机；U100完整，C部分{main["scientific_results_offserver_verified"]-100}。已审U十场景native表保持封存，新C均值须独立验收 | [十场景native表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009/rendered/TABLES.md) |')
if main.get('C_native_single_scene_table'):
    native_row=(f'| 机制native消融 | {main["scientific_results_offserver_verified"]}/800新增已严格离机；U100完整。C IID Benign十seed表另核54统计，F Flip2仍不入均值；取舍及9/6面板方向变化保留 | [U十场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009/rendered/TABLES.md)、[C单场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_C_Benign10_20261009/TABLES.md) |')
views_row=(f'| 机制三视图 | U100完整，另C单项门检{main.get("C1_valid_gate",{}).get("offserver_new_accepted",0)}已严格离机；U十场景1620统计、1800计数指标和810展示单元已核，C1不算完整C场景 | [十场景三视图论文表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md) |')
if main.get('C_after1_valid_replay'):
    C_now=main['C_after1_valid_replay']
    views_row=(f'| 机制三视图 | U100完整、C累计{main["three_view_counts_by_variant"]["minus_C"]}离机；准确C补集11状态{C_now["status"]}、新增接受{C_now["offserver_new_accepted"]}。U十场景表保持；C三视图统计表尚未验收 | [U十场景三视图表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md) |')
if main.get('C_three_view_single_scene_table'):
    views_row='| 机制三视图 | U100十场景完整；C12离机，IID Benign十seed单场景三视图表另核162统计/81单元/216计数指标。F Flip2对仅coverage，其余C未齐 | [C单场景三视图表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_Benign10_20261009/snapshot/TABLES.md) |'
reply_row=('| 英文回复 | 24条原意见逐字、37数值pointer及37链接核验；完整稿纳入U100十场景与900校准解释，保留全部pending，正文未应用 | [完整回复草稿](server_deployment_20260923/revision_20260923/rebuttal_integrated100_20261009/rebuttal_integrated_20261009.md)、[完整正文插入候选](server_deployment_20260923/revision_20260923/rebuttal_integrated100_20261009/manuscript_insertions_integrated_20261009.md) |')
text='\n'.join(native_row if line.startswith('| 机制native消融 |') else views_row if line.startswith('| 机制三视图 |') else reply_row if line.startswith('| 英文回复 |') else line for line in text.split('\n'))
overview.write_bytes(text.encode('utf8')+marker+history)
assert sha(overview.read_bytes().split(marker,1)[1])==sha(history)
handoff=CHECKS/'MONITOR_HANDOFF.md'
marker2=b'# Historical handoff snapshots'
_,history2=handoff.read_bytes().split(marker2,1)
assert sha(marker2+history2)=='f9dfedaf289de16f1226d2356ccfa451034e2e5e05b19348f0c08095e5a0a1ee'
top=f'''# GuardFed 当前巡检交接

主训练实测{live['checked_utc']}；辅助来源快照{fo['checked_utc']}。下一次须重新核实时状态。

- 服务器ssh -p60350 root@89.22.197.55，实例52183675；repo/workspace/GuardFed-celeba-expanded。先读/etc/vast-agents-guide.md，SHA42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa。用户授权sglang停止，文件/模型保留；213.224.31.105不自动切回。
- 主机制guardfed_celeba_mechanism_formal：{live['queue_completed']}观测终轮，{main['scientific_results_offserver_verified']}/800严格验收离机，100Full显式复用；{len(live['active'])}活动/{live['pending']}等待/{len(live['failed'])}失败，轮次{'/'.join(str(r.get('progress',{}).get('round')) for r in live['active'])}。实测JSON latest_formal_live.json SHA{sha((CHECKS/'latest_formal_live.json').read_bytes())}。
- 双5090利用率{gpu_usage}、{gpu_temps}℃、RecoveryNone；CPU{live['cpu_used_cores_2sec']:.2f}/{live['cpu_quota_cores']:.2f}核，RAM{live['memory_used_bytes']/1e9:.2f}GB，余量{live['disk_free_bytes']/1e12:.3f}TB，OOM0/近期错误0。冻结8并发，不因低CPU或交接瞬时低GPU重启。
- Native累计{main['scientific_results_offserver_verified']}=U100完整+C{main['scientific_results_offserver_verified']-100}部分；当前差集与恢复链见STATE.incremental_science_backups。只打包已验收ID差集，不重复旧Full/权重。
- C1单项评价门检严格离机{main.get('C1_valid_gate',{}).get('offserver_new_accepted',0)}，ROOT d045665b066dafc25f9970adfdffef9c9a8a388575ec87b9b54d5dcabfa65cab；9指标/24计数/3规则、40归档成员核验，native偏差0。原U100与Full不重推，C1不计作C十seed完整场景。
- C补集准确11已实际启动，状态{main.get('C_after1_valid_replay',{}).get('status','未启动')}，新增离机接受{main.get('C_after1_valid_replay',{}).get('offserver_new_accepted',0)}；独立namespace tmp/celeba_mechanism_valid_C_after1_20261009。观测/备份/采用分别用tmp/observe_mechanism_C_after1_root_20261009.py、backup_mechanism_C_after1_root_20261009.py、adopt_mechanism_C_after1_root_20261009.py；只在正常终态11/0worker后备份验收，不重推旧101/Full。C单场景native表已核54统计，ROOT77d046d695d9a36988b7ecbae0a37256389eb92413a70d74d0a838805a6d8872；9/6方向变化保留，三视图表尚须另验。
- 删除U三视图已100/100严格离机，after92最后8正常EXITED/0残留失败，ROOT9050eb059a797c70f0ca977294989b5ae5757286dbc85b36d529012cb5ab72ee，archive235837afdd336df8db7e3f224c6598f6f9b0a216a2deb10dcf11f3cf54987577。原92/Full不重推，C4排除；全部旧闭合服务禁止重启。
- 完整十场景三视图表celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md已独立验收，ROOT20d1031448701938063a398fc5416e404b1e8f0549807c85dab0c0204618a2a0；1620stats/1800计数指标/810显示cells，原184records/243rows精确。native100另核540stats。Full5CPU95GPU/98cu1282cu130，controls100CPU/cu128；负结果、validation选择/历史test暴露保留，不声称每项不可或缺/整个机制完成。
- FLGMM26/32、组合10/32已严格验收离机；{fo['checked_utc']}来源绑定只读实测分别{fo['observed_complete']}终轮/{fo['active']}活动/{fo['pending']}等待与{ho['observed_complete']}终轮/{ho['active']}活动/{ho['pending']}等待，失败0；新终轮未验收不得计入接受。LATEST_BACKUP各链已绑定；未完整选recipe或启动formal100/test。
- FLGMM final6闭合准备已独立源审，tmp/celeba_flgmm_final6_closure_20261009封条1d7cc11f95727a57478dd8575f65170024b36df98187030ac2de05e1e08cf6b9、独立源审f41bc996196c06772de23b578dd6f522c877b5ee8d877a601aa12ecdcc5e62e8。仅在原schema实际32终轮/无producer快照后调用tmp/execute_flgmm_final6_root_20261009.py，不以31终轮启动；CPU106单线程/nice10/idle、原loop保持、一次严格备份/本机验收后另root登记并汇总，不启动formal100/test。
- 原after82审批数量错误CNN前0完成/空输出和FLGMM旧chain字段collector失败原证据保留；独立V2分别准确10/7一次通过，不重启旧失败目录。
- 九方法900三视图/9页PDF、2052校准统计、旧TableII480原值追溯均验收；Fig3候选未采纳且原执行身份不足，提交版正文源项目待路径。
- 完整24英文回复和正文插入候选已升级U100/900校准证据，ROOT6dfb210c5d53c2badbe4fb53220008e784430fa33f9e68f0ebb81968ce18c391；24comments/37数值与链接核验，仍为作者审阅稿，正文未应用。
- 仍待800机制剩余{800-main['scientific_results_offserver_verified']}严格验收及其余七variant三视图、8方法完整覆盖/忠实规格和作者待决正式协议、冻结最终评价及正文/rebuttal。U100完成不等于全部返修完成，不从方案派发test或新方法。
- 原三小时聊天任务guardfed-training-health当前PAUSED；本会话无automation_update接口，未编辑调度器或新建cron/Windows监督。supervisor训练持续运行不等于聊天定时巡检已恢复。
- 最新已验证Git{published['commit']}，{published['committed_blobs_sha256_verified']}blob。提交后状态入口更新在本机保存，不把后续本机修改算已推送。

下次先读RUNNING/STATE/最新冻结协议，合并检查SSH、正确worker和真实轮次增长、错误/OOM/双GPU/实际cgroup资源。只有source/data/jobhash一致、无重复worker且恢复机制严格跳过已验收结果时，才有限恢复外部中断；代码/数值错误保存证据，不循环重试、不改科学配置/seed/指标/driver/实例、不购买资源。仅对重要变化、故障、完成或需用户处理通知。

'''
handoff.write_bytes(top.encode('utf8')+marker2+history2)
assert sha(handoff.read_bytes().split(marker2,1)[1])==sha(history2)
print(json.dumps(dict(status='CURRENT_OVERVIEW_AND_HANDOFF_UPDATED_HISTORY_BYTES_EXACT',overview_sha256=sha(overview.read_bytes()),handoff_sha256=sha(handoff.read_bytes()))))
