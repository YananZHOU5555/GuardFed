"""Adopt actual F80/current-draft navigation only; preserve all scientific history."""
from pathlib import Path
import argparse,ast,copy,datetime,hashlib,json
R=Path(__file__).resolve().parents[1];T=R/'docs/server_deployment_20260923/training_20260923';H=R/'tmp/root_saved380_F80_editorial_current_20261011'
read=lambda p:json.loads(Path(p).read_bytes());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def ref(p):return dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
def save(p,v):
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
pa=argparse.ArgumentParser();pa.add_argument('--editorial-root',type=Path,required=True);pa.add_argument('--editorial-root-sha256',required=True);a=pa.parse_args()
edp=a.editorial_root.resolve();assert edp.is_relative_to(R) and sha(edp)==a.editorial_root_sha256;ed=read(edp)
assert ed['root_accepted'] and ed['quote_blocks']==24 and ed['quote_lines']==38 and ed['quote_order_exact']
assert not ed['final_test'] and not ed['whole_rebuttal_complete'];draft=R/ed['candidate_path'];assert sha(draft)==ed['candidate_sha256']
assert not H.exists();H.mkdir();sp=T/'TRAINING_STATE.json';before=sp.read_bytes();assert sha(sp)=='b1eef139b58147125c4984ab376c7aa9fa6707b424d17f6798f495c55f767665'
saved=R/'tmp/celeba_mechanism_remaining_F_nonIID_FedSA10_root_adoption_20261011/ROOT_ADOPTION.json';table=T/'celeba_mechanism_v1/three_view_F_eight_scenes80_20261011/ROOT_VERIFICATION.json';pub=T/'publication_closed_increment67_verified_20261011.json';obs=T/'server_reactivation_20261009/root_five_queue_20261011T031011Z.raw.json';nga=R/'tmp/gradient96_parent_bootstrap_prepare_20261011/actual_parent_correction001/OBSERVATION_20261011T030857Z.json'
for p,pin in [(saved,'64b37040dc477aa92b08656ae9f029b419607b699646b2b4e74ce135f4851c2b'),(table,'0b1a8b65d063253e1f09f60fda09e46af8758e9dcf5bcbadd103a22603f006a0'),(pub,'c58a5bf7804bcbceb30d2a3b515e23af4a40ad44b275ef284960b426037da1ae'),(obs,'adc4fe96a4e19ec6e50fa515154cf003e0c241777efe43fc1bad95a96aa93311'),(nga,'fe796ac56bd2f0a201440e84ced8cd687048e3ce799d03bd34133219b8bf65cf')]:assert sha(p)==pin,p
sv,tp,pb,o,no=[read(p) for p in [saved,table,pub,obs,nga]];assert sv['cumulative_accepted']==380 and tp['complete_scenes']==8 and tp['paired_models']==80
s=json.loads(before);old=copy.deepcopy(s);m=s['celeba_mechanism_v1'];q=o['main_mechanism']['queue'];now=datetime.datetime.now(datetime.timezone.utc).isoformat()
m.update(three_view_new_models_accepted=380,three_view_new_models_offserver_verified=380,three_view_root_proof_sha256=sha(saved),three_view_root_proof_path=ref(saved)['path'],latest_three_view_root=ref(saved),three_view_counts_by_variant={'minus_U':100,'minus_C':100,'minus_A':100,'minus_F':80},queue_completed_observed=len(q['completed']),queue_active_observed=len(q['active']),queue_pending_observed=q['pending'],queue_failed_observed=len(q['failed']),queue_observation_utc=o['utc'],queue_observation_path=ref(obs)['path'],queue_observation_sha256=sha(obs),new_started=len(q['completed'])+len(q['active']),new_started_observation_utc=o['utc'],native_observed_not_yet_accepted=max(0,len(q['completed'])-380))
idx=R/sv['records_index_path'];assert sha(idx)==sv['records_index_sha256'];iv=read(idx)
assert len(iv['all_ids'])==len(set(iv['all_ids']))==380 and iv['all_ids'][:370]==m['three_view_accepted_ids']
m.update(three_view_accepted_ids=iv['all_ids'],latest_three_view_index=sv['records_index_path'],latest_three_view_index_sha256=sha(idx))
nr=read(R/m['latest_native_root_proof_path']);old_backups=m['incremental_science_backups'];assert len(old_backups)==51 and not any(x['archive_sha256']==nr['archive_sha256'] for x in old_backups)
old_backups.append(dict(archive_sha256=nr['archive_sha256'],new_ids=nr['new_ids'],members_verified=nr['root_checked_archive_members']))
m['latest_native_ledger_sha256']=nr['ledger_sha256']
m['prior_F70_table_preserved']=m['F_three_view_seven_scene_table']
m['F_three_view_eight_scene_table']=dict(root_proof_path=ref(table)['path'],root_proof_sha256=sha(table),table_path=tp['table_path'],paired_models=80,complete_scenes=8,complete_IID_scenes=5,complete_nonIID_scenes=['Benign','F Flip','FedSA'],displayed_records=160,seed_panels=[10,9,6],mean_SD_scalars=1296,display_cells=648,validation_only=True,final_test=False,incorporated_into_full_rebuttal=True,whole_rebuttal_complete=False)
m['three_view_scope_limit']='U100/C100/A100/F80 adopted:380 models plus Full100 references; F80 eight complete scenes. Controls400 and final F20 remain. Validation only, no endpoint decision or final-test evaluation.'
s['latest_five_queue_readonly_observation']=dict(**ref(obs),utc=o['utc'],new_acceptance=0,counts_are_observation_only=True,main_completed=len(q['completed']),main_active=len(q['active']),main_pending=q['pending'],main_failed=len(q['failed']),FL_terminal_new=len(o['FLGMM']['terminal_ids']),Hybrid_terminal_new=len(o['Hybrid96']['terminal_records']),remaining_remote_closed=o['remaining620']['remote_closed_n'],NGA_readonly_observation=ref(nga))
s['NGA96_runtime_dispatch_20261011'].update(latest_readonly_observation=ref(nga),terminal_result_directories_observed=len(no['terminal_result_directories_observed']),latest_observed_rounds=no['progress'],scientific_accepted=0)
s['prior_F70_FL90_working_rebuttal_preserved']=old['latest_working_rebuttal_draft'];entry=dict(status='ROOT_COMPLETE24_F80_FL90_HYBRID20_WORKING_REBUTTAL_ADOPTED_FOR_AUTHOR_REVIEW',entry=ref(draft)['path'],sha256=sha(draft),root_proof=ref(edp),comments_verbatim=24,quote_lines=38,content_cutoff=ed['fixed_scientific_content_cutoff'],F80_incorporated=True,FL90_incorporated=True,Hybrid20_incorporated=True,gate14_incorporated_as_scientific_results=False,author_review_only=True,submitted_manuscript_edited=False,final_test=False,whole_rebuttal_complete=False)
s['latest_working_rebuttal_draft']=entry;s['latest_rebuttal_draft']=copy.deepcopy(entry)
s['current_accepted_increment_20261011'].update(views380=ref(saved),F80_eight_scene_table=ref(table),working_complete24=ref(edp),not_in_fixed_Git67=True,no_new_science_by_this_writer=True)
s['latest_publication_verification']=dict(commit=pb['commit'],branch=pb['branch'],verified_utc=pb['verified_utc'],committed_blobs_sha256_verified=pb['committed_blobs_sha256_verified'],acceptance_cutoff=pb['acceptance_cutoff'],proof_path=pub.name,proof_sha256=sha(pub),earlier_publication_snapshot_is_historical=True)
s['active_services']=['guardfed_celeba_mechanism_formal','guardfed_celeba_flgmm_fullcoverage','guardfed_celeba_mechanism_remaining620_valid_v2a','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_fednga_fullcoverage_v1']
s['current_entry_writer']=dict(**ref(__file__),utc=now);s['local_work_verified_utc']=now;s['updated_unix']=datetime.datetime.now(datetime.timezone.utc).timestamp()
tree=ast.parse((R/'tmp/refresh_native370_FL86_Hybrid20_root_20261011.py').read_text());assignment=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='entries' for x in n.targets));ns={'T':T,'R':R};exec(compile(ast.Module(body=[assignment],type_ignores=[]),'<existing historical guards>','exec'),ns)
pending=[];priorprefix=None
for p,marker,pin in ns['entries']:
 raw=p.read_bytes();i=raw.index(marker.encode());tail=raw[i:];assert hashlib.sha256(tail).hexdigest()==pin
 if priorprefix is None:priorprefix=raw[:i]
 assert priorprefix==raw[:i];pending.append((p,raw,tail,pin))
prefix=f'''# CURRENT — GuardFed revision evidence ({now})

用户已授权补全返修实验。Huber采用恒等投影，标为CNN项目适配且不沿用原约束域理论保证；LoGoFair采用固定图像ID的20虚拟cohort，仅称人口适配。主评价端点和final-test边界仍待作者决定，不能保证方法全面获胜。

严格采用：机制native/views380（U100/C100/A100/F80，Full100显式复用）；F80八场景/160记录表，10/9/6种子及sampleSD(ddof1)、paired差异完整，1296统计标量/648显示格；旧F70的140对象、1134统计标量、567单元原样保留。FLGMM native86新增＋4screen引用、三视图90及九场景表；Hybrid native19新增、三视图20/两场景表；gradient搜索64。控制400和F20尚未闭合，不称整个机制完成。

最新五队列实测{o['utc']}：主队列383/800终轮、8活动、409待运行、失败0；FL90/96新增终轮，Hybrid26/96新增终轮，三视图远端203闭合但本机仅380总采用，观察不等于验收。记录：{ref(obs)['path']}。两张5090正常，资源以该实时快照为准，低瞬时利用率不重启健康任务。

14个CPU三轮部署门已严格离机核验（457成员、14完整state、112张量），性能表0记录。NGA96冻结runner运行，{no['utc']}首任务70轮已落result，下一任务第9轮，当前failure0；科学接受仍0，不能把终轮文件存在当验收。Huber96未启动，按原定随后串行。首次输出父目录接线失败及一次有限修复保留，训练源码/数学/参数/seed未改。当前SSH60350 root@89.22.197.55；213旧实例不启队列。

完整24点工作稿：{ref(draft)['path']}（{sha(draft)}）。24原话块/38引文行顺序exact，稿固定native/views380/F80、FL90、Hybrid20、gradient64；从保存JSON复制新FedSA六行统计，不重算。non-IID FedSA十seed Full native ACC88.41±0.92%、AEOD0.0078±0.0056、ASPD0.0616±0.0141；去F−Full ΔACC−0.239±1.297pp、ΔAEOD+0.00359±0.01201、ΔASPD+0.00047±0.01507。方向随面板变化和旧F70负结果全部保留，不称普遍必要或显著。FL新S-DFA校准的准确率代价及−2.1684e−19 group_KL诊断差、旧Windows47 exact-refit失败保留；不称跨平台逐位一致。

Git67已验证远端：{pb['commit']}，196个committed blobs逐个SHA和remote通过。其固定截止native380/views370/F70、FL90、Hybrid20、14部署及NGA启动；F80/new views10为下一次发布，不改写旧receipt。当前本机证据更近，Git状态只依真实publication receipt。

仍待：剩余机制与新基线完整多seed、三视图验收和IID/non-IID表；FedWA/SmartFL/FedDNA忠实定义；最终主评价端点与冻结final-test决策；匹配提交版LaTeX和正文/rebuttal整合。TableII旧480格追溯中的单次值不能伪造SD。原1390及已完成队列不重跑。当前稿仅作者审阅，未修改提交LaTeX或启用最终测试。

三小时桌面监控最后实际为暂停，当前automation_update不可用，不能声称启用；服务器Supervisor独立继续，无新cron/Windows任务或其他监督机制。大数据、模型、归档留服务器或已检查的F:/YananResearchStorage/GuardFed（Yanan 2TB），不回退内置盘；本机仅代码/配置/必要索引与精简报告。

写入器：{ref(__file__)['path']}。本次只更新已验收证据入口，无训练/fit/新科学检查。五入口历史尾部逐字保留。

'''.encode('utf8')
(H/'PRIOR_STATE.json').write_bytes(before);(H/'PRIOR_PREFIX.md').write_bytes(priorprefix);(H/'CURRENT_PREFIX.md').write_bytes(prefix)
assert sp.read_bytes()==before;sp.write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8',newline='\n')
for p,raw,tail,pin in pending:
 assert p.read_bytes()==raw;p.write_bytes(prefix+tail);assert hashlib.sha256(p.read_bytes()[len(prefix):]).hexdigest()==pin
save(H/'ROOT_METADATA_ADOPTION.json',dict(status='ROOT_SAVED380_F80_COMPLETE24_CURRENT_ENTRIES_ADOPTED',utc=now,STATE_before_sha256=hashlib.sha256(before).hexdigest(),STATE_after=ref(sp),writer=ref(__file__),roots=[ref(p) for p in [saved,table,edp]],entries=[ref(p) for p,_,_,_ in pending],five_prefixes_identical=True,historical_tails_byte_exact=True,prior_STATE_preserved=True,new_science=0,Git67_actual_receipt=ref(pub),final_test=False,whole_rebuttal_complete=False))
print(json.dumps(dict(state_sha256=sha(sp),root_sha256=sha(H/'ROOT_METADATA_ADOPTION.json'))))
