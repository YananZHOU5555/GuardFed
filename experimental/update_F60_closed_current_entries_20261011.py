"""Root metadata reconciliation only; consumes adopted evidence and preserves history."""
from pathlib import Path
from datetime import datetime, timezone
import argparse, hashlib, json
R=Path(__file__).resolve().parents[1]
T=R/'docs/server_deployment_20260923/training_20260923'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def ref(p):return dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
def pin(p,h):
    p=R/p;assert sha(p)==h,(p,h);return p
def create(p,v):
    with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
ap=argparse.ArgumentParser();ap.add_argument('--editorial-adoption',required=True);ap.add_argument('--editorial-sha256',required=True);a=ap.parse_args()
E=pin(a.editorial_adoption,a.editorial_sha256);ed=read(E)
sp=T/'TRAINING_STATE.json';assert sha(sp)=='488fcf5140b7eb08fe8b5434ffe4a55b334b2d92755ee090e1cfd17f35251412'
before=sp.read_bytes();s=json.loads(before)
N=pin('docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T002530Z/ROOT_DELTA_VERIFICATION.json','396e40654b6dfc75029d4a7a42dbfafe4caaf111a872ae2ece5045f6d1203418');n=read(N)
V=pin('tmp/celeba_mechanism_remaining_F_nonIID_Benign10_root_adoption_20261011/ROOT_ADOPTION.json','ca558d6e558d26afe6b172b65eded83b51ad02eedddc0f9e2d2cbadf7cdf5a11');v=read(V)
I=pin(v['records_index_path'],'16299ca4ade394a1b89edb3f9230845cda7ce826c783645b9aada4ea359a5c37');ix=read(I)
P=pin('docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_F_six_scenes60_20261011/ROOT_VERIFICATION.json','cbba6a6cdfb9b861202422f867c847800474961cd02784dd207f87f51c386164');p=read(P)
O=pin('docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_five_queue_20261011T004147Z.raw.json','58909812013a8f0aa2c9a1e2c0ecbca3782f0e7ab86d54fd5125b31c0a351464');o=read(O)
draft=pin(ed['draft_path'],ed['draft_sha256'])
assert ed['accepted_native']==ed['accepted_three_view']==360 and ed['F_complete_scenes']==6
assert n['total_new_strict_and_offserver']==v['cumulative_accepted']==360
assert p['actual_views360_root_sha256']==sha(V) and p['paired_models']==60 and p['records']==120
assert n['new_ids']==v['accepted_new_ids'] and len(n['new_ids'])==10
now=datetime.now(timezone.utc).isoformat()
m=s['celeba_mechanism_v1'];remaining=s['mechanism_remaining620_valid_20261010']
assert m['new_completed']==m['scientific_results_offserver_verified']==m['three_view_new_models_offserver_verified']==350
assert len(m['incremental_science_backups'])==49 and len(m['three_view_accepted_ids'])==350
assert ix['all_ids']==m['three_view_accepted_ids']+v['accepted_new_ids']
assert remaining['new_offserver_accepted']==len(remaining['accepted_ids'])==170
assert not set(remaining['accepted_ids'])&set(v['accepted_new_ids'])
q=o['main_mechanism']['queue'];nr=len(q['completed']);ar=len(q['active'])
assert nr==360 and ar==8 and q['pending']==432 and not q['failed']
m.update(new_completed=360,scientific_results_completed=360,scientific_results_strictly_accepted=360,scientific_results_offserver_verified=360,native_unaccepted=440,new_started=nr+ar,new_started_observation_utc=o['utc'],queue_completed_observed=nr,queue_active_observed=ar,queue_pending_observed=q['pending'],queue_failures_observed=0,queue_observation_utc=o['utc'],queue_observation_path=ref(O)['path'],latest_native_root=ref(N),latest_native_root_proof_path=ref(N)['path'],latest_native_root_proof_sha256=sha(N),science_acceptance_inspection=(N.parent/'inspection/inspection.json').relative_to(T).as_posix(),latest_science_backup_sha256=n['archive_sha256'],latest_science_backup_members_verified=110)
m['incremental_science_backups'].append(dict(archive_sha256=n['archive_sha256'],new_ids=n['new_ids'],members_verified=110))
assert len(m['incremental_science_backups'])==50
m.update(three_view_new_models_accepted=360,three_view_new_models_offserver_verified=360,three_view_accepted_ids=ix['all_ids'],three_view_counts_by_variant=dict(minus_U=100,minus_C=100,minus_A=100,minus_F=60),three_view_root_proof_sha256=sha(V),latest_three_view_root=ref(V),latest_three_view_index=ref(I)['path'],latest_three_view_index_sha256=sha(I),three_view_scope_limit='U/C/A each100, F60: five IID scenes plus non-IID Benign. Remaining four nonIID F scenes40 plus four controls400. All accepted views are validation; no universal necessity/significance/final-test claim.')
m['F_three_view_six_scene_table']=dict(root_proof_path=ref(P)['path'],root_proof_sha256=sha(P),table_path=p['table_path'],paired_models=60,complete_scenes=6,complete_IID_scenes=5,complete_nonIID_scenes=['Benign'],displayed_records=120,seed_panels=[10,9,6],mean_SD_scalars=972,display_cells=486,replay_devices=dict(Full=p['Full_replay_devices'],minus_F=p['minus_F_replay_devices']),validation_only=True,final_test=False,incorporated_into_full_rebuttal=True,whole_rebuttal_complete=False)
remaining.update(new_offserver_accepted=180,root_adoption_path=ref(V)['path'],root_adoption_sha256=sha(V),latest_accepted_new_ids=v['accepted_new_ids'],F_complete_scenes=v['complete_F_scenes'],F_partial_scenes=[],F_partial_seed_ids=[],F_six_scene_table_adopted=True)
remaining['accepted_ids']+=v['accepted_new_ids'];assert len(remaining['accepted_ids'])==180
remaining['latest_measured_observation']=dict(**ref(O),utc=o['utc'],remote_closed=o['remaining620']['remote_closed_n'],offserver_accepted=180)
obs=dict(**ref(O),utc=o['utc'],main_completed=nr,main_active=ar,main_pending=q['pending'],failed=0,FLGMM_new_terminal=len(o['FLGMM']['terminal_ids']),gradient_terminal=len(o['gradient64']['terminal_ids']),Hybrid_new_terminal=len(o['Hybrid96']['terminal_records']),remaining620_remote_closed=o['remaining620']['remote_closed_n'],counts_are_observation_only=True,new_acceptance=0,source_hash_scope='Small source/data pins unchanged; large images prior full SHA plus current stat only',scope='Terminal observation distinct from strict/offserver acceptance. FL/Hybrid screen4 references are separate.')
s['latest_five_queue_readonly_observation']=obs
s['latest_main_queue_readonly_observation']=dict(**ref(O),utc=o['utc'],completed=nr,active=ar,pending=q['pending'],failed=0,observation_only=True,science_acceptance_created=0)
fl=s['flgmm_fullcoverage_v2_20261009'];hy=s['hybrid100_fullcoverage_20261010'];fo=o['FLGMM'];ho=o['Hybrid96']
fl['prior_coverage_observational_aliases_20261011']={k:fl[k] for k in ['latest_readonly_observation_path','latest_readonly_observation_sha256','latest_observed_utc','coverage_observed_utc','active','pending','active_rounds','new_completed_observed','worker_pids','coverage_service_observed']}
fl.update(latest_readonly_observation_path=ref(O)['path'],latest_readonly_observation_sha256=sha(O),latest_observed_utc=o['utc'],coverage_observed_utc=o['utc'],active=len(fo['active']),pending=fo['pending'],active_rounds=[x['round'] for x in fo['active']],new_completed_observed=len(fo['terminal_ids']),worker_pids=[x['pid'] for x in o['resources']['processes'] if any('/celeba_flgmm_fullcoverage_v2_20261009/stage/run_one.py' in arg for arg in x['argv'])],coverage_service_observed=fo['service'],latest_readonly_terminal_observation=dict(**ref(O),utc=o['utc'],new_terminal=len(fo['terminal_ids']),new_accepted=77,reuse_separate=4),latest_measured_observation=dict(checked_utc=o['utc'],terminal_observed=len(fo['terminal_ids']),snapshot_path=ref(O)['path'],snapshot_sha256=sha(O),acceptance_unchanged_by_observation=True))
hy.update(latest_readonly_terminal_observation=dict(**ref(O),utc=o['utc'],new_terminal=len(ho['terminal_records']),new_accepted=12,reuse_separate=4),latest_measured_observation=dict(checked_utc=o['utc'],terminal_observed=len(ho['terminal_records']),snapshot_path=ref(O)['path'],snapshot_sha256=sha(O),acceptance_unchanged_by_observation=True))
# Existing item proofs only: this is a cutoff, never whole14 acceptance.
G=R/'tmp/gradient_source_freeze_deployment_index_20261011';items=[]
for i in range(14):
    matches=sorted(G.glob(f'gate14_job{i:02d}_actual*/ITEM_PROOF.json'))
    assert len(matches)<=1
    if not matches:break
    z=read(matches[0]);assert z['index']==i and z['acceptance']['status']=='PASS'
    items.append(dict(index=i,job_id=z['job_id'],**ref(matches[0])))
gpath=R/'tmp/gradient_source_freeze_deployment_index_20261011/GATES_F60_CUTOFF.json'
g=dict(status='ACTUAL_INDIVIDUAL_CPU3_GATES_ONLY_NOT_WHOLE14_ACCEPTANCE',utc=now,completed=len(items),total=14,items=items,scientific_table_records=0,whole_comparison=False,offserver_whole14=False,coverage192_started=False,automatic_retry=False,test=False)
assert 7<=len(items)<14
gs=s['gradient200_fullcoverage_source_preparation_20261010'];gs.update(exact14_completed_at_cutoff=len(items),exact14_partial_progress=dict(path=gpath.relative_to(R).as_posix()),limitations=f'Frozen192new+8reuse; {len(items)}/14 original individual CPU3 gates PASS at metadata cutoff. Whole saved comparison/offserver/root closure and fresh resource review pending. No192 performance records or final test. Two initial resource preflight refusals preserved; no scientific failure retried.')
s['prior_working_F50_draft_fixed_cutoff']=s['latest_working_rebuttal_draft']
s['latest_working_rebuttal_draft']=dict(status='ROOT_UNIFIED_COMPLETE24_F60_FL81_GRADIENT64_WORKING_REBUTTAL_ADOPTED_FOR_AUTHOR_REVIEW',entry=ref(draft)['path'],sha256=sha(draft),root_proof=ref(E),comments_verbatim=24,content_cutoff=dict(native=360,three_view=360,F_pairs=60,F_scenes=6,FL_native_new=77,FL_three_view=81,FL_complete_scene_records=80,Hybrid_three_view=10,gradient_search=64),F60_incorporated=True,FL81_incorporated=True,gate14_incorporated_as_scientific_results=False,author_review_only=True,submitted_manuscript_edited=False,final_test=False,whole_rebuttal_complete=False)
s['current_editorial_adoption_source']=ref(E)
s['last_health_check_historical_before_F60']=s['last_health_check']
s['last_health_check']=dict(**ref(O),checked_utc=o['utc'],server='89.22.197.55:60350',instance=52183675,phase='multiple_authorized_queues',queue_completed=nr,active=ar,pending=q['pending'],failed=0,resources=o['resources']['cgroup'],GPU=o['resources']['GPU'],disk_free_bytes=o['resources']['server_workspace_disk_bytes']['free'],scope='Readonly snapshot; no scientific acceptance comes from PID/RUNNING/utilization.')
s['last_health_check_scope']='Current compact combined snapshot at checked_utc. All scientific counters link independent strict/offserver/root proofs; historical previous snapshot retained.'
s['active_services']=[o['main_mechanism']['service']['stdout'].split()[0]]+[x['service']['stdout'].split()[0] for x in [fo,o['remaining620'],ho] if 'RUNNING' in x['service']['stdout']]
s['updated_unix']=datetime.now(timezone.utc).timestamp();s['local_work_verified_utc']=now;s['current_entry_writer']=dict(**ref(Path(__file__)),utc=now)
prefix=f'''# CURRENT: GuardFed返修实验

入口更新：{now}；最近合并实测：{o['utc']}。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded。先遵守/etc/vast-agents-guide.md。sglang已按授权停止，其他项目不动。

| 阶段 | strict＋离机SHA＋root已采用 | 最近实测（不能代替验收） |
|---|---:|---|
| 机制训练 | 新360/800，Full100另复用 | 360终轮、8活动、432等待、失败0；活动20–54轮 |
| 机制三视图 | U100/C100/A100/F60，共360 | remaining620远端闭合180，离机采用180 |
| FLGMM覆盖 | 新77/96，screen4另复用 | 新83终轮、2活动；三视图81、八完整场景表80，部分1另存 |
| Fed-NGA/Huber搜索 | 全64已验收并按冻结规则选参 | 服务自然退出；扩展192尚未启动 |
| 组合基线覆盖 | 新12/96，screen4另复用 | 新22终轮、1活动；IID Benign三视图10已采用 |

机制native360根证明：{ref(N)['path']}；新10为non-IID Benign去F，110原归档成员已核验，旧450原记录和49账本前缀不变，现50条链。三视图360根证明：{ref(V)['path']}；旧350对象/顺序不变，新10的原保存数组核90指标/240计数/30规则、native差0，92归档成员和离机SHA通过；零新增训练/拟合/CNN。

F60六场景论文表：{p['table_path']}。五IID场景(alpha5000)加non-IID Benign(alpha5)，每场景10配对；raw/native/shared和固定10/9/6面板、sampleSD(ddof1)。120记录、972均值/SD标量、486显示单元、1080指标和2880计数结构检查通过；旧F50的100记录、810标量、405单元精确保持。新non-IID Benign的native/shared去F−Full在10/9seed是准确率略升且两公平差距略增，在6seed准确率方向反转；raw也保留此反转。IID Sp-DFA/S-DFA和其他场景原取舍保留，不作普遍必要性、显著性或独立因果主张。尚缺四non-IID F场景40和四类控制400，共440条三视图，不能称全部返修完成。

FLGMM表：outputs/guardfed_tables/celeba_flgmm_eight_scenes80_20261011/TABLES.md。non-IID FedSA10 native ACC89.15±0.88%、AEOD0.0428±0.0049、ASPD0.1088±0.0055；shared88.77±0.81%、0.0087±0.0079、0.0689±0.0061，准确率代价保留。原Windows47重拟合失败、资源/传输拒绝和单个groupKL差−2.168404344971009e−19保留；后续Windows审计保存输出零拟合，不称跨平台逐位等价。

梯度搜索64：冻结四条件score均值/精确并列字典序选NGA eta0.001与Huber eta0.3/T0=0.01/M0。Huber27/32、NGA12/32恒负预测，全部负结果、准确率冠军与Pareto保留，n=1无SD/显著性。192新＋8复用覆盖源码已冻结；原真实图像CPU3门检本入口截止{len(items)}/14单项PASS，整体验收/离机保存/资源复核仍待完成。短门检不是70轮性能或GPU等价证明。FFlip梯度桥接为元数据空效应，初次index0/index4资源拒绝保留，不重试科学失败。

当前24条原话统一英文工作稿：{ref(draft)['path']}。固定native/views360、F60、FL77新/81三视图/80整场景、gradient64、Hybrid表10；24原话块38行与顺序精确保持，仅供作者审阅，未应用提交版正文。最新Git验证仍为3edbd954d91bbc5e4993546950202645d82ccbb0（529 blob/远端通过），其截止native/views350/F50；本轮F60增量与新稿尚未发布，不能冒称已上传。

Huber恒等投影依用户决定标CNN项目适配，不沿用原理论保证。LoGoFair固定image_id的20虚拟cohort＋官方人口统计平等后处理，标人口适配，不称真实训练client公平性。十方法native1000和九方法三视图900/2052归因已采用；17方法目标仍未齐。FedWA/SmartFL/FedDNA忠实规格、主终点/最终test边界、匹配LaTeX、Fig3来源仍待解决，已有问题不重复问。partition2仅核19962 image_id元数据；未读最终标签/像素、未最终推理。validation选择历史、旧test暴露、cu128/cu130与driver/重放设备差异继续披露。

最近实测双GPU100/100%，67/61°C，6036/5039MiB，RecoveryAction None；cgroup内存76.32/519.17GB，OOM0，服务器磁盘余1.060TB。CPU有界诊断见tmp/cpu_throughput_readonly_20261011/ROOT_DIAGNOSIS.json；未证明最优吞吐，未改并发/数值/实例。

大型证据优先留服务器；本机只按差集取必要产物到F:/YananResearchStorage/GuardFed，写前核Yanan 2TB/Healthy/容量，无内置盘回退。E仅代码/配置/索引/精简报告，模型/数组/数据/归档不进Git。三小时聊天任务guardfed-training-health仍PAUSED，现无automation_update接口；未另建cron/Windows监督，supervisor训练独立继续。

当前写入器：{Path(__file__).relative_to(R).as_posix()}。旧current前缀另存，本段下方历史原字节保持。

'''.encode('utf8')
targets=[(T/'RUNNING.md',b'# HISTORICAL:','203700dc6c349a10fc18f09e271df6db05f0aba0fcdf42617cce1f0bf6328a44'),(T/'REBUTTAL_COMPLETION_20261009.md',b'## Historical accepted increment','3aa956a830219050d59e9577c87d5b29b66520d26bee018d54335b9cd1630080'),(R/'docs/返修实验总览.md','以下为 2026-10-04'.encode(),'08b94e001b4ff78474706db0a63cc2c1b626fe5664c01edfafbaefb53fe9b029'),(T/'server_reactivation_20261009/MONITOR_HANDOFF.md',b'# Historical handoff snapshots','f9dfedaf289de16f1226d2356ccfa451034e2e5e05b19348f0c08095e5a0a1ee'),(T/'celeba_mechanism_v1/EXECUTION.md',b'# HISTORICAL PREPARATION SNAPSHOT','f1056563ae8ac5d19dd0d8a076541298ebff298ccde73ac951954b9e83dde7cd')]
oldparts=[]
for path,marker,tailpin in targets:
    raw=path.read_bytes();at=raw.index(marker);tail=raw[at:];assert hashlib.sha256(tail).hexdigest()==tailpin
    oldparts.append((path,raw[:at],tail,tailpin))
out=R/'tmp/root_F60_current_entries_20261011';out.mkdir(exist_ok=False)
create(out/'PRIOR_STATE_SHA_AND_CURRENT_PREFIXES.json',dict(state_sha256=hashlib.sha256(before).hexdigest(),entries=[dict(path=ref(path)['path'],prefix=old.decode('utf8'),prefix_sha256=hashlib.sha256(old).hexdigest(),historical_tail_sha256=h) for path,old,tail,h in oldparts],prior_last_health_check=s['last_health_check_historical_before_F60']))
create(gpath,g);gs['exact14_partial_progress']['sha256']=sha(gpath)
with sp.open('w',encoding='utf8',newline='\n') as f:json.dump(s,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')
for path,old,tail,h in oldparts:
    path.write_bytes(prefix+tail);assert path.read_bytes().startswith(prefix) and hashlib.sha256(path.read_bytes()[len(prefix):]).hexdigest()==h
check=read(sp);cm=check['celeba_mechanism_v1']
assert cm['new_completed']==cm['scientific_results_offserver_verified']==cm['three_view_new_models_offserver_verified']==len(cm['three_view_accepted_ids'])==360
assert cm['latest_native_root']['sha256']==cm['latest_native_root_proof_sha256']==sha(N)
assert check['latest_working_rebuttal_draft']['sha256']==sha(draft)
create(out/'ROOT_METADATA_ADOPTION.json',dict(status='ROOT_F60_CLOSED_EVIDENCE_STATE_AND_FIVE_CURRENT_ENTRIES_RECONCILED',utc=now,source=ref(Path(__file__)),state_before_sha256=hashlib.sha256(before).hexdigest(),state_after=ref(sp),native=ref(N),views=ref(V),table=ref(P),editorial=ref(E),observation=ref(O),entries=[ref(path) for path,_,_,_ in oldparts],historical_tails_unchanged=True,previous_current_prefixes_preserved=True,new_scientific_calculation=0,new_training=0,new_fit=0,test=False,Git_mutations=0,whole_rebuttal_complete=False))
print(json.dumps(dict(status='PASS',native=360,views=360,F_pairs=60,gradient_individual_gates=len(items),state_sha256=sha(sp),proof_sha256=sha(out/'ROOT_METADATA_ADOPTION.json'))))
