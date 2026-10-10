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
fl_accepted=state['flgmm_screen32_20261009']['offserver_accepted70round_jobs']
hy_accepted=state['hybrid_screen32_20261009']['offserver_accepted70round_jobs']
aux_row=(f'| FLGMM / 组合控制 | 已严格验收离机{fl_accepted}/32与{hy_accepted}/32；{fo["checked_utc"]}只读实测分别'
    f'{fo["observed_complete"]}/{ho["observed_complete"]}终轮、{fo["active"]}/{ho["active"]}活动、{fo["pending"]}/{ho["pending"]}等待、失败0。FLGMM已按冻结规则选Tg20/L2/lr0.001，n=1；组合未选recipe，100项确认未启动 | TRAINING_STATE对应搜索记录及备份链 |')
if state.get('flgmm_fullcoverage_v2_20261009',{}).get('formal100_started'):
    aux_row=aux_row.replace('组合未选recipe，100项确认未启动',f'FLGMM七项短程严格离机后已启动96新+4复用，新增已严格离机{state["flgmm_fullcoverage_v2_20261009"].get("new_accepted",0)}/96、双GPU各1线程；组合未选recipe/100项未启动')
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
if main.get('C_after12_valid_replay',{}).get('offserver_new_accepted')==8:
    views_row=views_row.replace('C12离机','C20离机').replace('F Flip2对仅coverage','F Flip十对完整评价已核，配对表待另验收')
    native_row=native_row.replace('F Flip2仍不入均值','F Flip十seed数据齐备、独立配对表待核')
if main.get('C_three_view_two_scene_table'):
    views_row='| 机制三视图 | U100十场景完整；C20的IID Benign/F Flip各十seed三视图表已独立核验324统计/162单元/360计数指标，旧Benign精确保持，其余八C场景未齐 | [C两场景三视图表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_two_scenes_20261009/snapshot/TABLES.md) |'
    native_row=native_row.replace('F Flip十seed数据齐备、独立配对表待核',
        'F Flip十seed的native表已随两场景三视图表独立验收')
    views_row=views_row.replace('U100十场景完整；C20的',f'U100十场景完整；C累计{main["three_view_counts_by_variant"]["minus_C"]}离机，其中C20的')
if main.get('C_three_view_three_scene_table'):
    views_row='| 机制三视图 | U100十场景完整；C36离机，IID Benign/F Flip/FedSA三场景表已独立核验486统计/243单元/540计数指标，旧两场景精确保持；S-DFA6单列排除，其他七C场景未齐 | [C三场景三视图表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_three_scenes_20261009/snapshot/TABLES.md) |'
    native_row=(f'| 机制native消融 | {main["scientific_results_offserver_verified"]}/800新增已严格离机；U100完整，C36部分。C三场景native已随三视图表独立验收，全部10/9/6面板及正负取舍保留 | [U十场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009/rendered/TABLES.md)、[C三场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_three_scenes_20261009/snapshot/TABLES.md) |')
if main.get('C_three_view_four_scene_table'):
    views_row='| 机制三视图 | U100十场景完整；C40的IID Benign/F Flip/FedSA/S-DFA各十seed四场景表已独立核验648统计/324单元/720计数指标，旧三场景精确保持；其余六C场景及其他六个变体待完成 | [C四场景三视图表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_four_scenes_20261010/snapshot/TABLES.md) |'
    native_row=(f'| 机制native消融 | {main["scientific_results_offserver_verified"]}/800新增已严格离机；U100完整、C40部分。C四场景native随三视图表独立验收；10/9/6面板及正负取舍均保留 | [U十场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009/rendered/TABLES.md)、[C四场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_four_scenes_20261010/snapshot/TABLES.md) |')
reply_row=('| 英文回复 | 24条原意见逐字、37数值pointer及37链接核验；完整稿纳入U100十场景与900校准解释，保留全部pending，正文未应用 | [完整回复草稿](server_deployment_20260923/revision_20260923/rebuttal_integrated100_20261009/rebuttal_integrated_20261009.md)、[完整正文插入候选](server_deployment_20260923/revision_20260923/rebuttal_integrated100_20261009/manuscript_insertions_integrated_20261009.md) |')
if state['latest_rebuttal_draft'].get('complete_C_scenes')==2:
    reply_row=reply_row.replace('37数值pointer及37链接核验；完整稿纳入U100十场景与900校准解释',
        '新增C20的22数值pointer/12范围环境事实与41链接核验；完整稿纳入U100十场景、C两场景与900校准解释').replace(
        'rebuttal_integrated100_20261009/','rebuttal_integrated_C20_20261009/')
if state.get('latest_rebuttal_addendum') and state['latest_rebuttal_addendum']['complete_C_scenes']>state['latest_rebuttal_draft'].get('complete_C_scenes',0):
    reply_row='| 英文回复 | 24原意见完整C20稿封存保留；C30独立英文补稿已核98数值pointer/49展示值/25范围事实/9链接，保留反例与未完成边界；均为作者审阅稿，正文未应用 | [完整回复草稿](server_deployment_20260923/revision_20260923/rebuttal_integrated_C20_20261009/rebuttal_integrated_20261009.md)、[C30新增回复](server_deployment_20260923/revision_20260923/rebuttal_C30_addendum_20261009/C30_REVIEWER_ADDENDUM.md) |'
    if state['latest_rebuttal_addendum']['complete_C_scenes']==4:
        addendum=state['latest_rebuttal_addendum']
        reply_row=(f"| 英文回复 | 24原意见完整稿保持；C40独立英文补稿已核{addendum['scalar_pointer_checks']}数值pointer/{addendum['display_cells_checked']}展示值/{addendum['scope_fact_checks']}范围事实/{addendum['links_checked']}链接，全部取舍与未完成边界保留；作者审阅稿，正文未应用 | [完整回复草稿](server_deployment_20260923/revision_20260923/rebuttal_integrated_C20_20261009/rebuttal_integrated_20261009.md)、[C40新增回复](server_deployment_20260923/revision_20260923/rebuttal_C40_addendum_20261010/C40_REVIEWER_ADDENDUM.md) |")
    if state['latest_rebuttal_addendum']['complete_C_scenes']==5:
        addendum=state['latest_rebuttal_addendum']
        reply_row=(f"| 英文回复 | 24原意见完整稿保持；C50英文补稿已核{addendum['scalar_pointer_checks']}数值pointer/{addendum['display_cells_checked']}展示值/{addendum['scope_fact_checks']}范围事实/{addendum['links_checked']}链接，保留Sp-DFA取舍和9/6seed面板反转；作者审阅稿，正文未应用 | [完整回复草稿](server_deployment_20260923/revision_20260923/rebuttal_integrated_C20_20261009/rebuttal_integrated_20261009.md)、[C50新增回复](server_deployment_20260923/revision_20260923/rebuttal_C50_update_20261010/C50_REVIEWER_ADDENDUM.md) |")
if main.get('C_three_view_five_scene_table'):
    views_row='| 机制三视图 | U100十场景完整；C50的IID五场景各十seed论文表已独立核验810统计/405单元/900计数指标及162个先seed内平均场景的汇总标量，旧四场景精确保留；五个non-IID C场景及其他六变体待完成 | [C五场景三视图表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010/snapshot/TABLES.md) |'
    native_row=(f'| 机制native消融 | {main["scientific_results_offserver_verified"]}/800新增已严格离机；U100完整、C50的IID五场景完成。10/9/6面板及正负取舍保留，尚未最终test | [U十场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009/rendered/TABLES.md)、[C五场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010/snapshot/TABLES.md) |')
if state['latest_rebuttal_draft'].get('complete_C_scenes')==5:
    reply_row='| 英文回复 | 24条审稿人原话完整稿已合并C五个IID场景；11处可逆修改、38个C数值pointer、19个范围/环境事实、24项方向及44链接核验通过，旧两份文档可逐字恢复；保留全部反例和pending，作者审阅稿，正文未应用 | [完整回复草稿](server_deployment_20260923/revision_20260923/rebuttal_integrated_C50_20261010/rebuttal_integrated_20261009.md)、[完整正文插入候选](server_deployment_20260923/revision_20260923/rebuttal_integrated_C50_20261010/manuscript_insertions_integrated_20261009.md) |'
if main.get('C_after50_valid_replay',{}).get('offserver_new_accepted')==6:
    views_row=views_row.replace('U100十场景完整；C50的','累计156模型离机：U100、C56；C50的').replace('五个non-IID C场景及其他六变体待完成','non-IID Benign仅6/10单列排除，五个non-IID C场景及其他六变体待完成')
if main.get('C_after56_valid_replay',{}).get('offserver_new_accepted')==4:
    views_row='| 机制三视图 | 累计160模型离机：U100完整、C60已评价；五IID及non-IID Benign各十seed齐备，六场景表须独立采用；其余四个non-IID C场景及六变体未完成 | [既有C五场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010/snapshot/TABLES.md) |'
if main.get('C_three_view_six_scene_table'):
    views_row='| 机制三视图 | U100完整；C60六场景60对/120记录表独立采用：五IID及non-IID Benign，972统计/486单元/1080计数指标/2880计数通过；旧100记录/810统计/405单元/162个IID seed-first标量保持。其余四个non-IID C场景及六变体未完成 | [C六场景三视图表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010/snapshot/TABLES.md) |'
    native_row=(f'| 机制native消融 | {main["scientific_results_offserver_verified"]}/800新增已严格离机；U100完整、C60六场景native随三视图表独立验收，10/9/6面板及全部负结果保留；未运行最终test | [U十场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009/rendered/TABLES.md)、[C六场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_six_scenes_20261010/snapshot/TABLES.md) |')
if state['latest_rebuttal_draft'].get('complete_C_scenes')==6:
    reply_row='| 英文回复 | 完整C60作者审阅稿已纳入五IID及non-IID Benign；24原意见逐字、10处可逆修改、36数值pointer/18展示值/27方向/50链接通过，旧C50两全文可逐字恢复；保留负结果及P1–P6，正文未应用、最终test未运行 | [完整回复草稿](server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010/rebuttal_integrated_20261009.md)、[完整正文插入候选](server_deployment_20260923/revision_20260923/rebuttal_integrated_C60_20261010/manuscript_insertions_integrated_20261009.md) |'
if main.get('C_three_view_seven_scene_table'):
    views_row='| 机制三视图 | U100十场景完整；C70七场景70对/140记录表独立采用：五IID及non-IID Benign/F Flip，1134统计/567单元/1260指标/3360计数通过，旧120/972/486/162保持；其余三non-IID C场景及六变体未完成 | [C七场景三视图表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_seven_scenes_20261010/snapshot/TABLES.md) |'
    native_row=(f'| 机制native消融 | {main["scientific_results_offserver_verified"]}/800新增已严格离机；U100完整、C70七场景native随三视图表验收，10/9/6面板和全部取舍保留；未运行最终test | [U十场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009/rendered/TABLES.md)、[C七场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_seven_scenes_20261010/snapshot/TABLES.md) |')
    reply_row=reply_row.replace('正文未应用、最终test未运行','C70另表已验收、尚未合入该封存全文；正文未应用、最终test未运行')
text='\n'.join(native_row if line.startswith('| 机制native消融 |') else views_row if line.startswith('| 机制三视图 |') else reply_row if line.startswith('| 英文回复 |') else line for line in text.split('\n'))
new_stage_note=''
if state.get('flgmm_fullcoverage_v2_20261009',{}).get('canary_runner_started'):
    new_stage_note+='FLGMM七项短程检查队列已实际启动，完成验收仍待原科学检查和离机备份；96新+4复用的70轮覆盖未启动。'
    if state['flgmm_fullcoverage_v2_20261009'].get('formal100_started'):
        new_stage_note=new_stage_note.replace('FLGMM七项短程检查队列已实际启动，完成验收仍待原科学检查和离机备份；96新+4复用的70轮覆盖未启动。',
            f'FLGMM七项短程已通过原严格、全状态/RNG比较及315成员离机核验；96新+4复用70轮valid完整覆盖已启动，双GPU worker首轮已实测，新增70轮离机接受{state["flgmm_fullcoverage_v2_20261009"].get("new_accepted",0)}/96。')
if main.get('C_after12_valid_replay'):
    new_stage_note+='新增8项C/IID F Flip终轮valid三视图CPU评价已实际启动，旧112和Full不重推；新增验收与完整F Flip表仍待完成。'
    if main['C_after12_valid_replay']['offserver_new_accepted']==8:
        new_stage_note=new_stage_note.replace('新增8项C/IID F Flip终轮valid三视图CPU评价已实际启动，旧112和Full不重推；新增验收与完整F Flip表仍待完成。',
            '新增8项C/IID F Flip终轮三视图已正常完成并严格离机采用：89归档成员、72指标/192计数/24规则通过，native偏差0；累计U100+C20，旧112/Full不重推，F Flip配对表待独立验收。')
text=re.sub(r'\n当前新增执行：[^\n]*\n','\n',text)
if main.get('C_three_view_two_scene_table'):
    new_stage_note=new_stage_note.replace('F Flip配对表待独立验收。','两场景三视图论文表已独立采用，324统计/162单元/360计数指标，旧Benign精确保持。')
if main.get('C_after20_valid_replay',{}).get('offserver_new_accepted')==5:
    new_stage_note+='另5项C/IID FedSA终轮三视图已严格离机采用，68归档成员、45指标/120计数/15规则通过，native偏差0；累计U100+C25，FedSA仍仅5/10、不计完整场景均值。'
if main.get('C_after25_valid_replay'):
    c3=main['C_after25_valid_replay']
    new_stage_note+=f'后续准确3项FedSA（91002/04/07）状态{c3["status"]}，新增离机接受{c3["offserver_new_accepted"]}；原125与Full不重推。'
    if c3['offserver_new_accepted']==3:
        new_stage_note+='本批54归档成员、27指标/72计数/9规则通过，native偏差0；当前累计U100+C28，FedSA仍仅8/10、不计完整场景均值。'
if main.get('C_after28_valid_replay'):
    c28=main['C_after28_valid_replay']
    new_stage_note+=f'后续准确8项（FedSA两项及S-DFA六项）状态{c28["status"]}，新增离机接受{c28["offserver_new_accepted"]}；原128与Full不重推。'
    if c28['offserver_new_accepted']==8:
        new_stage_note+='本批89归档成员、72指标/192计数/24规则通过，native偏差0；当前累计U100+C36，FedSA十seed齐备，S-DFA仅6/10不入均值，三场景配对表仍须独立验收。'
if state.get('flgmm_fullcoverage_v2_20261009',{}).get('new_accepted'):
    new_stage_note=new_stage_note.replace('新70轮离机接受仍0。',f'新增70轮已严格离机并经root采用，累计{state["flgmm_fullcoverage_v2_20261009"]["new_accepted"]}/96；4复用另计，不构成十seed场景均值。')
if main.get('C_three_view_three_scene_table'):
    fl_now=state.get('flgmm_fullcoverage_v2_20261009',{})
    new_stage_note=(f'FLGMM完整验证覆盖新增严格离机{fl_now.get("new_accepted",0)}/96，4复用另计；Hybrid已验收{hy_accepted}/32、未选完整recipe。准确8项C checkpoint三视图评价已正常退出并严格离机，89归档成员/72指标/192计数/24规则通过、native偏差0；累计U100+C36。IID Benign/F Flip/FedSA三场景表独立采用486统计/243单元/540指标，旧两场景精确保持；S-DFA仅6/10单列排除，其他七C场景及其余机制仍待完成。新表未重推Full、未重训、未运行test。')
if main.get('C_after36_valid_replay',{}).get('offserver_new_accepted')==4:
    new_stage_note=(f'FLGMM完整验证覆盖新增严格离机{state.get("flgmm_fullcoverage_v2_20261009",{}).get("new_accepted",0)}/96，4复用另计；Hybrid已验收{hy_accepted}/32、尚未选完整recipe。新增准确4项C checkpoint三视图已正常退出、0残留并严格离机：61归档成员/36指标/96计数/12规则通过，native偏差0，累计U100+C40。IID Benign/F Flip/FedSA/S-DFA各10seed齐备；旧136与Full不重推、未重训、未运行test。')
if main.get('C_three_view_four_scene_table'):
    new_stage_note+='四场景配对表已独立采用：648均值/样本SD标量、324展示单元及720计数指标通过，旧三场景60记录/486统计/243单元及原S-DFA6记录精确保留。其他六C场景及六个变体未完成，保留全部取舍和10/9/6面板。'
if main.get('C_three_view_five_scene_table'):
    new_stage_note=(f'FLGMM新增严格离机{state.get("flgmm_fullcoverage_v2_20261009",{}).get("new_accepted",0)}/96，4复用另计；Hybrid已验收{hy_accepted}/32、尚未选recipe。最新三项C评价已正常退出、0残留并严格离机：54归档成员/27指标/72计数/9规则通过，native偏差0，累计U100+C50。IID五场景各10seed配对表已独立采用：810均值/样本SD标量、405展示单元、900计数指标及162个seed-first跨场景汇总标量通过，旧80记录/648统计/324展示精确保留。五个non-IID C场景及其他六变体仍待完成；保留全部取舍、环境/选择史和10/9/6面板，未重推Full、重训或运行test。')
if main.get('C_after50_valid_replay'):
    C56=main['C_after50_valid_replay']
    new_stage_note+=('后续6项non-IID Benign checkpoint三视图也已严格离机，75归档成员/54指标/144计数/18规则通过，native偏差0，累计U100+C56；该场景仅6/10，不纳入完整场景均值。' if C56['offserver_new_accepted']==6 else
        '后续6项non-IID Benign checkpoint三视图已实际启动，严格离机接受仍0；不纳入完整场景均值。')
if main.get('C_after56_valid_replay',{}).get('offserver_new_accepted')==4:
    new_stage_note=(f'FLGMM新增严格离机{state.get("flgmm_fullcoverage_v2_20261009",{}).get("new_accepted",0)}/96，4复用另计；Hybrid已验收{hy_accepted}/32、尚未选recipe。最新准确4项non-IID Benign三视图已正常退出并严格离机，61归档/60内容成员、36指标/96计数/12规则通过，native偏差0，累计U100+C60=160；原156和Full不重推。五IID及non-IID Benign各10seed齐备，六场景表须另核。四个其他non-IID C场景及六变体未完成；未重训/运行test。')
if main.get('C_three_view_six_scene_table'):
    new_stage_note=new_stage_note.replace('六场景表须另核。','六场景表已独立采用：120记录/972均值样本SD标量/486单元/1080指标/2880计数；旧100记录/810统计/405展示及162个IID seed-first标量保持。全部10/9/6面板、负结果、混合CPU/GPU与环境/选择史保留，不计算不平衡六场景总均值。C60表是独立入口，完整英文作者审阅稿仍为C50，未纳入C60、未应用正文。')
if main.get('C_three_view_six_scene_table',{}).get('incorporated_into_full_rebuttal'):
    new_stage_note=new_stage_note.replace('C60表是独立入口，完整英文作者审阅稿仍为C50，未纳入C60、未应用正文。','C60证据已纳入完整英文作者审阅稿，24原意见及C50旧值保持；10处可逆修改/36数值pointer/27方向/50链接通过，提交版正文未应用。')
if main.get('C_three_view_seven_scene_table'):
    new_stage_note=(f'FLGMM新增严格离机{state.get("flgmm_fullcoverage_v2_20261009",{}).get("new_accepted",0)}/96，4复用另计；Hybrid已验收{hy_accepted}/32、尚未选recipe。十项non-IID F Flip终轮三视图正常退出并严格离机，累计U100+C70=170；103归档成员/90指标/240计数/30规则通过，native偏差0，旧160/Full不重推。C70七场景表已独立采用：140记录/1134统计/567单元/1260指标/3360计数/630配对指标，旧120/972/486和162个IID汇总标量保持。不计算不平衡七场景总均值；Full5CPU/65GPU对C70CPU，环境/选择史及负结果保留。其他三non-IID C场景及六变体仍缺。完整英文作者审阅稿仍为C60，第七场景尚未合入，正文未应用、test未运行。')
if main.get('C_three_view_eight_scene_table'):
    views_row='| 机制三视图 | U100十场景完整；C80八场景80对/160记录已独立采用：五IID及non-IID Benign/F Flip/FedSA，1296统计/648单元/1440指标/3840计数/720配对指标通过，旧140/1134/567及162 IID汇总保持；其余两non-IID C场景及六变体未完成 | [C八场景三视图表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_eight_scenes_20261010/snapshot/TABLES.md) |'
    native_row=(f'| 机制native消融 | {main["scientific_results_offserver_verified"]}/800新增严格离机；U100完整、C80八场景native随三视图表验收，全部10/9/6面板和性能取舍保留；未运行最终test | [U十场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_interim100_20261009/rendered/TABLES.md)、[C八场景表](server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_eight_scenes_20261010/snapshot/TABLES.md) |')
    text='\n'.join(views_row if line.startswith('| 机制三视图 |') else native_row if line.startswith('| 机制native消融 |') else line for line in text.split('\n'))
    new_stage_note=(f'FLGMM新增严格离机{state.get("flgmm_fullcoverage_v2_20261009",{}).get("new_accepted",0)}/96，4复用另计；Hybrid已验收{hy_accepted}/32。新增十项non-IID FedSA终轮三视图正常退出、103成员严格离机，累计U100+C80=180，旧170/Full不重推。C80八场景表已独立采用，1296统计/648单元/1440指标/3840计数/720配对指标通过，旧140记录/1134统计/567单元和162 IID汇总保持，不计算八场景总均值。删除C的native/shared FedSA差ACC+0.410pp、AEOD+0.00190、ASPD−0.0000093，9/6面板ASPD变号及全部负结果保留。Full5CPU/75GPU对C80CPU，环境与选择史披露；剩余两non-IID C场景及六变体未完成。完整英文稿仍为C60，C70/C80尚未合入，正文未应用、test未运行。')
elif main.get('C_after70_valid_replay',{}).get('offserver_new_accepted')==10:
    new_stage_note+=('后续non-IID FedSA准确十项终轮三视图也已正常退出、0残留worker，103成员/90指标/240计数/30规则与根验收通过，native偏差0，累计U100+C80=180；旧170与Full不重推。第八场景表尚待独立统计采用，C checkpoint覆盖只余non-IID S-DFA/Sp-DFA。')
if new_stage_note:text+='\n当前新增执行：'+new_stage_note+'\n'
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
- C补集准确11已实际启动，状态{main.get('C_after1_valid_replay',{}).get('status','未启动')}，新增离机接受{main.get('C_after1_valid_replay',{}).get('offserver_new_accepted',0)}；独立namespace tmp/celeba_mechanism_valid_C_after1_20261009。观测/备份/采用分别用tmp/observe_mechanism_C_after1_root_20261009.py、backup_mechanism_C_after1_root_20261009.py、adopt_mechanism_C_after1_root_20261009.py；只在正常终态11/0worker后备份验收，不重推旧101/Full。C单场景native表已核54统计，ROOT77d046d695d9a36988b7ecbae0a37256389eb92413a70d74d0a838805a6d8872；9/6方向变化保留；C IID Benign三视图表状态{main.get("C_three_view_single_scene_table",{}).get("status","待独立采用")}，其余C场景未齐。
- 删除U三视图已100/100严格离机，after92最后8正常EXITED/0残留失败，ROOT9050eb059a797c70f0ca977294989b5ae5757286dbc85b36d529012cb5ab72ee，archive235837afdd336df8db7e3f224c6598f6f9b0a216a2deb10dcf11f3cf54987577。原92/Full不重推，C4排除；全部旧闭合服务禁止重启。
- 完整十场景三视图表celeba_mechanism_v1/three_view_interim100_20261009/snapshot100/TABLES.md已独立验收，ROOT20d1031448701938063a398fc5416e404b1e8f0549807c85dab0c0204618a2a0；1620stats/1800计数指标/810显示cells，原184records/243rows精确。native100另核540stats。Full5CPU95GPU/98cu1282cu130，controls100CPU/cu128；负结果、validation选择/历史test暴露保留，不声称每项不可或缺/整个机制完成。
- FLGMM{fl_accepted}/32、组合{hy_accepted}/32已严格验收离机；{fo['checked_utc']}来源绑定只读实测分别{fo['observed_complete']}终轮/{fo['active']}活动/{fo['pending']}等待与{ho['observed_complete']}终轮/{ho['active']}活动/{ho['pending']}等待，失败0；新终轮未验收不得计入接受。LATEST_BACKUP各链已绑定；选recipe状态以STATE的独立汇总采用凭据为准，未启动formal100/test。
- FLGMM final6已一次严格验收、离机成员SHA核验并root登记，原26不重训/打包；actual_20261009T194424Z/ROOT_ADOPTION_REVIEW.json SHA66097564f346b0dd5a0194ea7bd8c1413d51936819db086283f1d9a99a85738e。32→8冻结规则汇总已独立及root采用，ROOT_SUMMARY_ADOPTION.json SHAe602761016e199da157862da3f24c9f9d0f191cfde10fad49074540c672b4a7f，选Tg20/L2/lr0.001，score前两差0.0000497792；ACC冠军不同、六Pareto及全部候选保留。n=1不作SD/显著性，formal100/test未启动，原搜索与final6不得重跑。
- 原after82审批数量错误CNN前0完成/空输出和FLGMM旧chain字段collector失败原证据保留；独立V2分别准确10/7一次通过，不重启旧失败目录。
- {new_stage_note} 实际入口分别tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_CANARY_STARTUP.json及tmp/celeba_mechanism_valid_C_after12_20261009/execution_candidate/ROOT_STARTUP_OBSERVATION.json；后续只读观测，不盲重启。
- 九方法900三视图/9页PDF、2052校准统计、旧TableII480原值追溯均验收；Fig3候选未采纳且原执行身份不足，提交版正文源项目待路径。
- 完整24英文回复和正文插入候选已升级U100/900校准证据，ROOT6dfb210c5d53c2badbe4fb53220008e784430fa33f9e68f0ebb81968ce18c391；24comments/37数值与链接核验，仍为作者审阅稿，正文未应用。
- 仍待800机制剩余{800-main['scientific_results_offserver_verified']}严格验收及其余七variant三视图、8方法完整覆盖/忠实规格和作者待决正式协议、冻结最终评价及正文/rebuttal。U100完成不等于全部返修完成，不从方案派发test或新方法。
- 原三小时聊天任务guardfed-training-health当前PAUSED；本会话无automation_update接口，未编辑调度器或新建cron/Windows监督。supervisor训练持续运行不等于聊天定时巡检已恢复。
- 最新已验证Git{published['commit']}，{published['committed_blobs_sha256_verified']}blob。提交后状态入口更新在本机保存，不把后续本机修改算已推送。

下次先读RUNNING/STATE/最新冻结协议，合并检查SSH、正确worker和真实轮次增长、错误/OOM/双GPU/实际cgroup资源。只有source/data/jobhash一致、无重复worker且恢复机制严格跳过已验收结果时，才有限恢复外部中断；代码/数值错误保存证据，不循环重试、不改科学配置/seed/指标/driver/实例、不购买资源。仅对重要变化、故障、完成或需用户处理通知。

'''
if state.get('flgmm_fullcoverage_v2_20261009',{}).get('formal100_started'):
    top=top.replace('未启动formal100/test','FLGMM96新+4复用已启动、组合100未启动；test未启动').replace('formal100/test未启动','FLGMM96新+4复用已启动、组合100未启动；test未启动')
handoff.write_bytes(top.encode('utf8')+marker2+history2)
assert sha(handoff.read_bytes().split(marker2,1)[1])==sha(history2)
print(json.dumps(dict(status='CURRENT_OVERVIEW_AND_HANDOFF_UPDATED_HISTORY_BYTES_EXACT',overview_sha256=sha(overview.read_bytes()),handoff_sha256=sha(handoff.read_bytes()))))
