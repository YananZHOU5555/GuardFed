"""Root metadata update from already adopted evidence; no science or remote mutation."""
from pathlib import Path
import ast,copy,datetime,hashlib,json
R=Path(__file__).resolve().parents[1];T=R/'docs/server_deployment_20260923/training_20260923'
H=R/'tmp/root_native380_FL90_gradient14_NGA_current_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
def save(p,v):
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
assert not H.exists();H.mkdir()
sp=T/'TRAINING_STATE.json';before=sp.read_bytes();assert sha(sp)=='1e9119a17226a302bf7fe33293a63dd746744d22c0480a7677b2f3533b18faa8'
native=T/'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T024637Z/ROOT_DELTA_VERIFICATION.json'
fl=R/'tmp/fl90_root_closure_source_prepare_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
ft=R/'outputs/guardfed_tables/celeba_flgmm_nine_scenes90_20261011/ROOT_VERIFICATION.json'
gate=R/'tmp/gradient14_root_closure_actual_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
nga=R/'tmp/gradient96_parent_bootstrap_prepare_20261011/actual_parent_correction001/ROOT_RUNTIME_START_VERIFIED.json'
editorial=R/'tmp/rebuttal_FL90_editorial_prepare_20261011/ROOT_EDITORIAL_ADOPTION.json'
obs=T/'server_reactivation_20261009/root_five_queue_20261011T030032Z.raw.json'
pub=T/'publication_closed_increment66_verified_20261011.json'
for p,pin in [(native,'0d479b88b1728e32bf204cc98117efb0361431c215f0804137819fc70b0aa6f5'),(fl,'0b07a8775bf43809d00ea251a832261f59a161fd61872d09dd8ba5f2530eb932'),(ft,'efdb6913217500de34a62e1ad904feb84982741a8cb864a19d18e88d913ec1d5'),(gate,'b4ba8c8921bc6b63cfec5996149fb59a9abae41dfaba1e3b345c3d3d8b4c2bf5'),(nga,'0567c629cdd440b05f1c5c5cfb7bffad4c3c21d1df8abf90d0577c4a6ae86e84'),(editorial,'5d99df629b4991b0802beb30c0967906f0190a92e6be4acefc94c902abb34d1c'),(obs,'13d245e87ffa6526b67606fb599eac7db559c78f50422db66c7833eea7e6ca93'),(pub,'890d2c79e759de68ba7290060473d7aa8e9694b4296f86849997c9860bcba7de')]:assert sha(p)==pin,p
np,fp,tp,gp,rp,ep,o,pubp=[read(p) for p in [native,fl,ft,gate,nga,editorial,obs,pub]]
assert np['root_adopted'] and np['total_new_strict_and_offserver']==380
assert fp['root_adoption'] and fp['FLGMM_total_three_view_records']==90
assert gp['all14_offserver_verified'] and gp['scientific_table_records']==0
assert rp['runtime_dispatch_verified'] and rp['completed70_scientific_records']==0 and not rp['Huber_started']
draft=R/ep['candidate_path'];assert sha(draft)==ep['candidate_sha256']
s=json.loads(before);old=copy.deepcopy(s);m=s['celeba_mechanism_v1'];q=o['main_mechanism']['queue'];now=datetime.datetime.now(datetime.timezone.utc).isoformat()
m.update(new_completed=380,scientific_results_completed=380,scientific_results_strictly_accepted=380,scientific_results_offserver_verified=380,latest_native_root_proof_path=ref(native)['path'],latest_native_root_proof_sha256=sha(native),latest_native_root=ref(native),queue_completed_observed=len(q['completed']),queue_observation_utc=o['utc'],queue_observation_path=ref(obs)['path'],queue_observation_sha256=sha(obs),new_started=len(q['completed'])+len(q['active']),new_started_observation_utc=o['utc'],native_unaccepted=800-380)
m['latest_native380_accepted_delta']=dict(root=ref(native),new_ids=np['new_ids'],new_accepted=10,old470_exact=np['old470_records_exact'],new_table_created=False)
m['three_view_scope_limit']='Adopted U100/C100/A100/F70=370 saved models, Full100 reused. Native380 is newer; exact FedSA10 saved export/F80 remain pending. Controls400 plus F30 not yet adopted. Validation only.'
ob=dict(**ref(obs),utc=o['utc'],new_acceptance=0,counts_are_observation_only=True,main_completed=len(q['completed']),main_active=len(q['active']),main_pending=q['pending'],main_failed=len(q['failed']),FL_terminal_new=len(o['FLGMM']['terminal_ids']),Hybrid_terminal_new=len(o['Hybrid96']['terminal_records']),remaining_remote_closed=o['remaining620']['remote_closed_n'],NGA_runtime_start_observation=rp['actual_observation'])
s['latest_five_queue_readonly_observation']=ob
s['FLGMM_SDFA9_three_view_pending_20261011'].update(status='ROOT_EXACT9_SAVED_ADOPTED_FL90_NINE_SCENE_TABLE_ACCEPTED',started=True,new_three_view_accepted=9,prior_three_view_accepted=81,total_three_view_accepted=90,root=ref(fl),table_root=ref(ft),table_path=tp['table_path'] if 'table_path' in tp else 'outputs/guardfed_tables/celeba_flgmm_nine_scenes90_20261011/TABLES.md',reason='Original Linux whole checker and zero-fit Windows saved audit pass; one group_KL diagnostic difference retained. Historical Windows47 exact-refit failure remains failed.',test=False)
s['flgmm_fullcoverage_v2_20261009'].update(latest_three_view_root=ref(fl),three_view_records_accepted=90,latest_three_view_table_root=ref(ft),complete_three_view_scenes=9,observed_terminal_new=len(o['FLGMM']['terminal_ids']),scientific_native_new_accepted=86,explicit_screen_references=4,native_observed_not_accepted=max(0,len(o['FLGMM']['terminal_ids'])-86))
s['gradient14_current_dispatch_20261011'].update(status='ROOT_WHOLE14_CPU3_OFFSERVER_ADOPTED_NGA96_ACTUALLY_STARTED',updated_utc=now,individual_passes=14,whole14_accepted=True,whole14_root=ref(gate),coverage192_started=False,NGA96_started=True,Huber96_started=False,NGA_runtime_start_root=ref(nga),scientific_table_records=0,next_action='Continue unchanged NGA96 single-worker queue. Huber96 remains sequential and unstarted; require actual NGA completion and original strict/resource gates. Short14 deployment checks are not performance results.')
s['NGA96_runtime_dispatch_20261011']=dict(status='ACTUAL_ORIGINAL_RUNNER_FIRST_CHILD_OBSERVED_NOT_SCIENTIFIC_ACCEPTANCE',root=ref(nga),new_jobs=96,screen_reuse=4,workers=1,CPU=105,physical_GPU=1,latest_start_observation=rp['actual_observation'],scientific_accepted=0,Huber_started=False,automatic_retry=False,final_test=False)
entry=dict(status='ROOT_COMPLETE24_F70_FL90_HYBRID20_WORKING_REBUTTAL_ADOPTED_FOR_AUTHOR_REVIEW',entry=ref(draft)['path'],sha256=sha(draft),root_proof=ref(editorial),comments_verbatim=24,quote_lines=38,content_cutoff=ep['fixed_scientific_content_cutoff'],F70_incorporated=True,FL90_incorporated=True,Hybrid20_incorporated=True,gate14_incorporated_as_scientific_results=False,author_review_only=True,submitted_manuscript_edited=False,final_test=False,whole_rebuttal_complete=False)
s['prior_F70_FL81_working_rebuttal_preserved']=old['latest_working_rebuttal_draft'];s['latest_working_rebuttal_draft']=entry;s['latest_rebuttal_draft']=copy.deepcopy(entry)
s['current_accepted_increment_20261011'].update(native380=ref(native),FL_views90=ref(fl),FL_nine_scene_table90=ref(ft),gradient14_deployment_only=ref(gate),NGA96_runtime_only=ref(nga),working_complete24=ref(editorial),not_in_fixed_Git66=True,no_new_science_by_this_writer=True)
s['latest_publication_verification']=dict(commit=pubp['commit'],branch=pubp['branch'],verified_utc=pubp['verified_utc'],committed_blobs_sha256_verified=pubp['committed_blobs_sha256_verified'],acceptance_cutoff=pubp['acceptance_cutoff'],proof_path=pub.name,proof_sha256=sha(pub),earlier_publication_snapshot_is_historical=True)
s['health_monitor']['prior_record_before_current_refresh']=copy.deepcopy(old['health_monitor'])
s['health_monitor'].update(status='PAUSED_LAST_VERIFIED_TOOL_UNAVAILABLE',scheduler_verified=False,scheduler_verified_in_current_turn=False,current_training='main mechanism, FLGMM, Hybrid and NGA96 observed; Supervisor queues are independent of desktop monitor',note='Last actual scheduler inspection found PAUSED; automation_update remains unavailable. User authorization to keep monitoring persists, but enabled status is not claimed. No replacement scheduler created.',updated_utc=now)
s['active_services']=['guardfed_celeba_mechanism_formal','guardfed_celeba_flgmm_fullcoverage','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_fednga_fullcoverage_v1']
s['current_entry_writer']=dict(**ref(__file__),utc=now);s['local_work_verified_utc']=now;s['updated_unix']=datetime.datetime.now(datetime.timezone.utc).timestamp()
tree=ast.parse((R/'tmp/refresh_native370_FL86_Hybrid20_root_20261011.py').read_text());assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='entries' for x in n.targets));ns={'T':T,'R':R};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<existing historical guards>','exec'),ns)
pending=[];priorprefix=None
for p,marker,pin in ns['entries']:
 raw=p.read_bytes();i=raw.index(marker.encode());tail=raw[i:];assert hashlib.sha256(tail).hexdigest()==pin
 if priorprefix is None:priorprefix=raw[:i]
 assert priorprefix==raw[:i];pending.append((p,raw,tail,pin))
prefix=f'''# CURRENT — GuardFed revision evidence ({now})

用户已授权补全返修实验。Huber 采用恒等投影，明确 CNN 项目适配且不沿用原约束域理论保证；LoGoFair 使用固定图像 ID 的20虚拟 cohort，仅报告人口适配。重大主评价端点和 final-test 边界仍待作者决定；不能保证方法全面获胜。

当前严格采用：机制 native380，三视图370（U100/C100/A100/F70，Full100显式复用）；F70七场景表；FLGMM native86新增＋4旧screen引用、三视图90/九场景论文表；Hybrid native19新增、三视图20/两场景表；梯度调参64。native380新 FedSA10已严格核验和离机备份，但其三视图/F80在此截止尚未采用。保留所有负结果、恒定预测、失败和准确率代价。10/9/6固定种子面板，sampleSD(ddof1)，不做最佳单次对基线均值。

最新实测 {o['utc']}：主队列380/800终轮、8活动、412待运行、当前失败0；FL90/96新增终轮，Hybrid26/96新增终轮，远端remaining620已闭合200但不自动等于离机采用。双GPU100%、65/63°C、约6GB显存各卡，cgroup OOM0，磁盘余量约1.059TB。以上运行计数仅观察，科学采用以对应root proof为准。观察：{ref(obs)['path']}。

14个 CPU三轮部署门已严格通过及离机核验，457成员/14完整state/112张量；只证明该短程实现与身份检查，科学性能表记录0。Fed-NGA96已按冻结runner启动，02:50首任务第3轮、parent108479/child108499；70轮科学采用0。Huber96保持随后串行，尚未启动。NGA首次启动的输出父目录接线错误及一次有限修复证据保留，训练数值逻辑未改。旧服务器213实例不启队列，当前SSH60350 root@89.22.197.55。

完整24点回复工作稿：{ref(draft)['path']}（{sha(draft)}），24原话块/38引文行顺序exact。该稿固定native/views370、F70、FL90、Hybrid20及gradient64，故与较新的native380有明确截止差异；尚未应用提交LaTeX。FL新增S-DFA10 native ACC89.17±1.15%、AEOD0.0423±0.0048、ASPD0.1083±0.0105；共享校准ACC88.78±1.17%、AEOD0.0045±0.0039、ASPD0.0659±0.0080。一次group_KL诊断差−2.1684e−19保留，不称跨平台逐位校准相等。原Windows47 exact-refit失败仍保留为失败。

当前已验证Git66：{pubp['commit']}，455个committed blobs逐个SHA及remote分支通过。其固定截止native/views370、F70、FL81views、Hybrid20；新native380/FL90/whole14/NGA运行接线等待下一次独立发布，不改写Git66事实。

仍需：完成剩余机制和新基线多seed覆盖、后续三视图验收与表；FedWA/SmartFL/FedDNA忠实定义待补齐；最终主评价端点与冻结final-test决策；匹配提交版LaTeX及rebuttal整合。TableII旧480格追溯中的单次结果不能伪造SD。三小时监控最后实测为暂停，当前编辑工具不可用，不能声称已启用；服务器Supervisor独立继续。大型原始数据/权重/归档只留服务器或经检查的F:/YananResearchStorage/GuardFed，不回退内置盘。

本段写入器：{ref(__file__)['path']}。历史尾部逐字保留；本次只更新已有证据入口，无训练/统计/科研验收。

'''.encode('utf8')
(H/'PRIOR_STATE.json').write_bytes(before);(H/'PRIOR_PREFIX.md').write_bytes(priorprefix);(H/'CURRENT_PREFIX.md').write_bytes(prefix)
assert sp.read_bytes()==before;sp.write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8',newline='\n')
for p,raw,tail,pin in pending:
 assert p.read_bytes()==raw;p.write_bytes(prefix+tail);assert hashlib.sha256(p.read_bytes()[len(prefix):]).hexdigest()==pin
save(H/'ROOT_METADATA_ADOPTION.json',dict(status='ROOT_NATIVE380_FL90_WHOLE14_NGA_RUNTIME_CURRENT_ENTRIES_ADOPTED',utc=now,STATE_before_sha256=hashlib.sha256(before).hexdigest(),STATE_after=ref(sp),writer=ref(__file__),roots=[ref(p) for p in [native,fl,ft,gate,nga,editorial]],entries=[ref(p) for p,_,_,_ in pending],five_prefixes_identical=True,historical_tails_byte_exact=True,prior_STATE_preserved=True,new_science=0,Git66_actual_receipt=ref(pub),final_test=False,whole_rebuttal_complete=False))
print(json.dumps(dict(state_sha256=sha(sp),root_sha256=sha(H/'ROOT_METADATA_ADOPTION.json'))))
