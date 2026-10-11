"""Record measured/adopted increments and one unresolved replay failure."""
from pathlib import Path
import ast,copy,datetime,hashlib,json,re
R=Path(__file__).resolve().parents[1];T=R/'docs/server_deployment_20260923/training_20260923';V=R/'docs/server_deployment_20260923/revision_20260923'
H=R/'tmp/root_native400_native11_replay_failure_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
ref=lambda p:dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
assert not H.exists()
sp=T/'TRAINING_STATE.json';before=sp.read_bytes()
assert sha(sp)=='683efac6352578b176607df998e3c5ac320b6d4c7cafd1f3a769a50fabffa699'
s=json.loads(before);old=copy.deepcopy(s)
np=T/'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261011T050221Z/ROOT_DELTA_VERIFICATION.json'
fp=R/'tmp/fl_native_final10_prepared_20261011/ROOT_ADOPTION_REVIEW.json'
tp=R/'outputs/guardfed_tables/celeba_eleven_method_native_20261011/ROOT_REVIEW.json'
pp=R/'outputs/guardfed_tables/celeba_eleven_method_native_pdf_20261011/ROOT_REVIEW.json'
dp=R/'tmp/rebuttal_native11_root_adoption_20261011/ROOT_EDITORIAL_ADOPTION.json'
ob=T/'server_reactivation_20261009/root_five_queue_20261011T050158Z.raw.json'
ng=R/'tmp/gradient96_parent_bootstrap_prepare_20261011/actual_parent_correction001/OBSERVATION_20261011T045803Z.json'
failure=R/'tmp/fl_final10_completed_aux_runtime_prepare_20261011/ACTUAL_NATIVE_ARTIFACTS.json'
for p,pin in [(np,'7ac1c9232a9383891a927a1906b213c57791a4912146f4f37d63e54b8decacb6'),(fp,'ed7d317530fb6c7181509b1eb798312d65fa7955b70831316e1b0abb34277712'),(tp,'69e9c479c061a1aa428157b6f1d121aa5dd72a48e0e6d1d16dd21ac296391d02'),(pp,'9bd262a38fce788799b37d503fbd951b56faaacc73fbeb211db875c6af4af4d3'),(dp,'fe030787cad1ff87da2d094e943c38cf3a82c1d0aac9f2facd7c1e1e7a8ebd25'),(ob,'01a1fc53f064f15c22f710e51e658e6c4e81d023cd0d5a25e89536d16c9cdd34'),(ng,'70eea961ecd95da3634abb524d9181565b68d1a3e91bb9f5579b55913fdce7f0'),(failure,'80b0c83eded8f65cce9fde3c3f5ef84955154705287cabd227a05840a76ffbab')]:assert sha(p)==pin,p
n,f,tab,pdf,draft,obs,nga,diag=map(read,[np,fp,tp,pp,dp,ob,ng,failure])
assert n['root_adopted'] and n['total_new_strict_and_offserver']==400 and n['root_checked_archive_members']==190
assert f['accepted_total']==96 and f['reused_separately']==4 and f['native_coverage_complete']
assert tab['records']==1100 and tab['methods']==11 and pdf['root_adopted'] and draft['comment_blocks']==24
q=obs['main_mechanism']['queue'];assert len(q['completed'])==400 and not q['failed']
m=s['celeba_mechanism_v1'];assert m['scientific_results_strictly_accepted']==380 and m['three_view_new_models_accepted']==380
for k in ['new_completed','scientific_results_completed','scientific_results_strictly_accepted','scientific_results_offserver_verified']:m[k]=400
m.update(new_started=len(q['completed'])+len(q['active']),latest_native_root=ref(np),latest_native_root_proof_path=ref(np)['path'],latest_native_root_proof_sha256=sha(np),science_acceptance_inspection=(np.parent/'inspection/inspection.json').relative_to(T).as_posix(),latest_science_backup_sha256=n['archive_sha256'],latest_science_backup_members_verified=190,latest_native_ledger_sha256=n['ledger_sha256'],queue_completed_observed=400,queue_active_observed=len(q['active']),queue_pending_observed=q['pending'],queue_failures_observed=0,queue_observation_utc=obs['utc'],queue_observation_path=ref(ob)['path'])
m['incremental_science_backups'].append(dict(archive_sha256=n['archive_sha256'],new_ids=n['new_ids'],members_verified=190))
m['latest_native400_accepted_delta']=dict(root=ref(np),new=20,accepted_native=400,saved_views_unchanged=380,old480_records_exact=True,parent52_ledger_prefix_exact=True,Full100_reused_not_retrained=True)
m['three_view_scope_limit']='Native U/C/A/F100 each accepted (400); saved views U/C/A100+F80 (380). Final F20 saved closure stopped on seed91004 numerical mismatch. Controls400 still train; no F100 three-view table accepted.'
fl=s['flgmm_fullcoverage_v2_20261009'];assert fl['new_accepted']==86 and fl['three_view_records_accepted']==90
assert f['accepted_job_ids'][:86]==fl['accepted_new_ids']
fl.update(status='COMPLETE_NATIVE100_STRICT_OFFSERVER_ROOT_SAVED90_LAST10_PENDING',new_accepted=96,scientific_native_new_accepted=96,accepted_new_ids=f['accepted_job_ids'],active=0,pending=0,active_rounds=[],worker_pids=[],new_completed_observed=96,observed_terminal_new=96,native_observed_not_accepted=[],coverage_verified_cells_including_reuse=100,latest_full70_root_adoption_path=ref(fp)['path'],latest_full70_root_adoption_sha256=sha(fp),latest_readonly_observation_path=ref(ob)['path'],latest_readonly_observation_sha256=sha(ob),latest_observed_utc=obs['utc'],coverage_service_observed=obs['FLGMM']['service'])
fl['latest_backup']=dict(accepted_total=96,root_adoption_path=ref(fp)['path'],root_adoption_sha256=sha(fp),next_collector_previous_path='F:/YananResearchStorage/GuardFed/fl_native_final10_prepared_20261011/batch/OFFSERVER_ACCEPTANCE.json',next_collector_previous_sha256=f['offserver_acceptance_sha256'])
fl['latest_native_final10']=dict(root=ref(fp),archive_sha256=f['archive_sha256'],archive_members_verified=107,old86_ordered_prefix_exact=True,native_complete100=True,saved_views_still90=True)
s['celeba_native_eleven_method_table_20261011']=dict(status=tab['status'],root=ref(tp),table=tab['canonical_table'],methods=11,evaluation_records=1100,view='native',seed_panels=[10,9,6],remaining_native_methods=['Fed-NGA','Huber','CosineFairness','FedWA','SmartFL','FedDNA'],original1000_exact=True,primary_endpoint_selected=False,full17_complete=False,final_test=False)
s['celeba_native_eleven_method_PDF_20261011']=dict(root=ref(pp),pdf=pdf['pdf'],pages=3,display_cells=990,main_page='ten-seed IID/non-IID',sensitivity_pages=[9,6],IEEE_manuscript_fit_verified=False)
rem=s['mechanism_remaining620_valid_20261010'];assert rem['new_offserver_accepted']==190
saved_root=R/'tmp/celeba_mechanism_remaining_F_nonIID_FedSA10_root_adoption_20261011/ROOT_ADOPTION.json'
assert sha(saved_root)=='64b37040dc477aa92b08656ae9f029b419607b699646b2b4e74ce135f4851c2b'
saved=read(saved_root);index_path=R/saved['records_index_path'];assert sha(index_path)==saved['records_index_sha256']
index=read(index_path);assert len(index['new_archive']['all_transported_ids'])==saved['remaining620_new_accepted']==200
assert index['new_archive']['all_transported_ids'][:190]==rem['accepted_ids']
rem.update(new_offserver_accepted=200,accepted_ids=index['new_archive']['all_transported_ids'],root_adoption_path=ref(saved_root)['path'],root_adoption_sha256=sha(saved_root),latest_accepted_new_ids=saved['accepted_new_ids'],F_complete_scenes=saved['complete_F_scenes'],F_partial_scenes=[],F_partial_seed_ids=[],previous190_observational_alias_reconciled_to_existing_root200=True)
comparison=diag['receipt']['native_comparison'];assert not comparison['accepted'] and comparison['tolerance']==1e-12
rem.update(status='HALTED_ON_NATIVE_METRIC_MISMATCH_NO_AUTOMATIC_RETRY',remote_closed_observed=213,unaccepted_closed_observed=13,latest_failure_observation=ref(failure),current_worker_count_observed=0,failed_id=diag['id'],original_CPU_failure_preserved=True)
s['mechanism_replay_failure_20261011']=dict(status='NUMERICAL_MISMATCH_DIAGNOSIS_PENDING_NOT_EXTERNAL_INTERRUPTION',identity=diag['id'],evidence=ref(failure),comparison=comparison,runtime=diag['receipt']['runtime'],source_checkpoint_job_result_identity_verified=True,known_cause=False,tolerance_changed=False,retried=False,existing_saved380_unchanged=True,native400_still_valid=True,FL_final10_launch_refused=True,FL_final10_started=False)
hy=s['hybrid100_fullcoverage_20261010'];hy['latest_measured_observation']=dict(checked_utc=obs['utc'],terminal_observed=len(obs['Hybrid96']['terminal_records']),snapshot=ref(ob),acceptance_unchanged_by_observation=True)
gd=s['NGA96_runtime_dispatch_20261011'];gd.update(latest_readonly_observation=ref(ng),terminal_result_directories_observed=len(nga['terminal_result_directories_observed']),latest_observed_rounds=nga['progress'])
s['latest_NGA96_readonly_observation']=dict(**ref(ng),utc=nga['utc'],terminal_observed=len(nga['terminal_result_directories_observed']),accepted_new_unchanged=1,Huber_started=False)
s['latest_five_queue_readonly_observation']=dict(utc=obs['utc'],snapshot=ref(ob),main_completed=400,main_active=len(q['active']),main_pending=q['pending'],main_failed=0,FL_new_terminal=96,Hybrid_new_terminal=len(obs['Hybrid96']['terminal_records']),remaining_remote_closed=213,counts_are_observation_only=True,new_scientific_acceptance=0)
prior=copy.deepcopy(s['latest_working_rebuttal_draft']);cur=copy.deepcopy(prior)
cur.update(status='ROOT_COMPLETE24_NATIVE11_F80_FL90_HYBRID20_CURRENT_PDF',entry=draft['adopted_draft']['path'],sha256=draft['adopted_draft']['sha256'],root_proof=ref(dp),scientific_content_unchanged_from_parent=False,reporting_scope_correction_only=False,native11_incorporated=True,published_in_Git68=False)
cur['content_cutoff'].update(FL_native_new=96,native_methods=11,native_evaluation_records=1100)
s['prior_working_statistics_clarified_F80_draft']=prior
s['latest_working_rebuttal_draft']=copy.deepcopy(cur);s['latest_rebuttal_draft']=copy.deepcopy(cur)
s['current_editorial_adoption_source']=ref(dp)
reader=V/'AUTHOR_REVIEW_START_HERE.md';reader_before=reader.read_bytes();r=reader_before.decode('utf8')
assert r.count(Path(prior['entry']).name)==1;r=r.replace(Path(prior['entry']).name,Path(cur['entry']).name)
a='当前实验截止为机制380条、F80、FLGMM90和组合控制20；后续结果尚未写入该稿。'
b='该回复稿的机制截止仍为380条、F80、FLGMM三视图90和组合控制20；新增11方法native表（1100条评估记录、含完整FLGMM100）已写入。其后的机制native400验收和校准重放故障见实时入口，尚未写入该固定截止稿。'
assert r.count(a)==1;r=r.replace(a,b)
a='[十方法 native 表](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_ten_method_native_20261010/TABLES.md)及[三页论文表PDF](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_ten_method_native_pdf_20261010/celeba_ten_method_native.pdf)：IID/non-IID各5场景，每条件10种子，共1000条记录。'
b='[十一方法 native 表](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_eleven_method_native_20261011/TABLES.md)及[三页论文表PDF](E:/OneDrive/文档/GuardFed/outputs/guardfed_tables/celeba_eleven_method_native_pdf_20261011/celeba_eleven_method_native.pdf)：IID/non-IID各5场景，每条件10种子，共1100条方法–场景–种子评估记录，包含完整FLGMM100；不是1100个独立训练模型。'
assert r.count(a)==1;r=r.replace(a,b)
a='合计380个删除控制，Full100显式复用；不能说800个控制全部完成，也不能说每个评分项都不可或缺。'
b='原始删除控制已验收400条（U/C/A/F各100），保存三视图仍380条（F80）；最后F20的校准重放因一条数值核验失败停下。Full100显式复用；不能说800个控制全部完成，也不能说每个评分项都不可或缺。'
assert r.count(a)==1;r=r.replace(a,b)
a='单独展示，不混进已完成的十方法全覆盖表。'
b='三视图单独展示；FLGMM的完整native100已纳入十一方法表。'
assert r.count(a)==1;r=r.replace(a,b)
for link in re.findall(r'\[[^\]]+\]\((E:/[^)]+)\)',r):assert Path(link.split('#',1)[0]).is_file()
now=datetime.datetime.now(datetime.timezone.utc).isoformat()
tree=ast.parse((R/'tmp/refresh_native370_FL86_Hybrid20_root_20261011.py').read_text(encoding='utf8'))
node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='entries' for x in n.targets));ns={'T':T,'R':R};exec(compile(ast.Module(body=[node],type_ignores=[]),'<history guards>','exec'),ns)
entries=[];prefix=None
for p,marker,pin in ns['entries']:
    raw=p.read_bytes();i=raw.index(marker.encode());tail=raw[i:];assert hashlib.sha256(tail).hexdigest()==pin
    if prefix is None:prefix=raw[:i]
    assert raw[:i]==prefix;entries.append((p,raw,tail,pin))
H.mkdir();(H/'PRIOR_STATE.json').write_bytes(before);(H/'PRIOR_PREFIX.md').write_bytes(prefix);(H/'PRIOR_READER.md').write_bytes(reader_before)
reader.write_text(r,encoding='utf8',newline='\n')
s['author_review_reading_entry_20261011'].update(utc=now,reader=ref(reader),latest_reply=ref(dp),native11_table=ref(tp),native11_PDF=ref(pp),prior_appendix_reading_entry_hash_is_historical=True)
s['current_entry_writer']=dict(**ref(__file__),utc=now);s['local_work_verified_utc']=now;s['updated_unix']=datetime.datetime.now(datetime.timezone.utc).timestamp()
newprefix=f'''# CURRENT: GuardFed返修实验

核验入口更新：{now}。最近合并实测{obs['utc']}；NGA窄观测{nga['utc']}。服务器 ssh -p60350 root@89.22.197.55，实例52183675，repo /workspace/GuardFed-celeba-expanded；先遵守/etc/vast-agents-guide.md。sglang已按授权停止。

| 阶段 | 严格验收、离机SHA及root已采用 | 最新实测与边界 |
|---|---:|---|
| 机制训练 | 新400/800，Full100显式复用 | 实测400终轮、8活动、392等待、训练失败0；U/C/A/F各100 native完成，另四类400继续 |
| 机制三视图 | U100/C100/A100/F80，共380 | remaining620已远端闭合213，本机已采用200；seed91004去F/non-IID/Sp-DFA CPU重放数值失败，队列已停止；不得宣称F100闭合 |
| FLGMM | 96新+4screen=100 native完整 | 三视图仍90；最后10独立评价因配套服务失败状态被启动检查拒绝，未启动，正在准备隔离资源检查 |
| 主比较表 | 11方法native1100及三页PDF已采用 | IID/non-IID各5场景、每格10seed；9/6seed为敏感性，不是未触碰test；仍非目标17方法 |
| 组合控制 | native19、三视图20已采用 | 实测29新终轮；尚非完整100 |
| Fed-NGA/Huber | 搜索64、部署门检14、NGA新1已采用 | NGA实测7新终轮、第8项seed91009在60轮；96新+4复用，1worker CPU105/GPU1；Huber96未开始 |

本次机制native最后20仅non-IID S-DFA/Sp-DFA，终轮70/valid19867/source/data/config/checkpoint核验、190归档成员与离机SHA通过，旧480对象及52账本前缀保留，canonical账本53项已在本地/服务器同步采用。根：{ref(np)['path']}。保存三视图380、F80表与既有负结果不变；训练完成不能替代校准评价验收。

已保存故障证据：tmp/fl_final10_completed_aux_runtime_prepare_20261011/ACTUAL_NATIVE_ARTIFACTS.json。失败ID minus_F_non-IID_Sp-DFA_seed91004；训练native ACC/AEOD/ASPD=.8928877032/.0011269696/.0705992338，CPU重放=.8930890421/.0032383875/.0679697308，最大差.0026295030，原容差1e-12拒绝。源码、数据、模型/result/job SHA吻合；尚不能判定原因。不改容差、不抹除失败、不盲重试剩余620；主训练、NGA和Hybrid独立继续。已验收380与原生400仍有效，失败CPU数组不计入论文结果。

FLGMM完整native根 tmp/fl_native_final10_prepared_20261011/ROOT_ADOPTION_REVIEW.json；最后10归档107成员，旧86有序链不变。新增11方法表 outputs/guardfed_tables/celeba_eleven_method_native_20261011/TABLES.md，三页PDF outputs/guardfed_tables/celeba_eleven_method_native_pdf_20261011/celeba_eleven_method_native.pdf。1100为方法–场景–种子评估记录，LoGoFair复用FedAvg checkpoint，不能称1100次独立训练。1980场景统计数值/990显示格独立核验，误差至1.11e-16；旧1000对象和统计精确保留。balanced十seed：GF88.42%/.0097/.0610，FLGMM89.35%/.0439/.1119；准确率代价、LoGo更低ASPD及FLTrust更高准确率都保留，不称全面领先或显著。

当前完整24条英文回复：{cur['entry']}。38条审稿引文和既有统计范围修正保留；该稿机制固定截止380/F80、FL三视图90、Hybrid20，新增native11已写入；后来的native400/本次故障尚未写入固定截止稿。正文插入候选、14方法信息访问附录及作者入口在revision_20260923；尚未移入匹配投稿LaTeX、未做最终IEEE排版验收。旧TableII全部480格均已追溯（包括原44个N/E），实际n1=294/n10=180/n3=6，单次SD不可计算；历史选轮与round70修正候选分开，不能宣称旧值全是同checkpoint。

Huber采用用户同意的恒等投影，标CNN适配、不沿用约束域理论；LoGoFair固定image_id的20虚拟cohort仅人口适配，DP不是差分隐私。剩余六个native方法覆盖、其他图像控制400、共享校准闭合、忠实FedWA/SmartFL/FedDNA资料、主端点/最终评价边界、匹配LaTeX与最终发布仍待完成。官方test曾暴露，当前只valid；不挑seed、不用Full最佳对比消融均值、不混checkpoint指标、不删负结果。

最近双5090为100/100%、64/61°C、约5GB显存，RecoveryAction None；cgroup OOM0，内存约69.4/519.2GB、CPU配额122.87999核、磁盘余1.059TB。累计CPU计数不等于瞬时利用率，不据占用重启健康任务。大型证据留服务器或F:/YananResearchStorage/GuardFed，写前核Yanan2TB/Healthy/容量，无内置盘回退。Git68为f252db2c固定历史截止，本次native400/11方法/新全文尚未发布；不改旧发布回执。三小时聊天automation仍PAUSED，当前无可调用更新工具，未创建替代监督。

本入口只采用上述实际根证明和带时间观测；全部历史尾字节保持，原STATE/当前前缀另存在tmp/root_native400_native11_replay_failure_20261011。整个返修未完成。

'''.encode('utf8')
assert sp.read_bytes()==before;sp.write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8',newline='\n')
for p,raw,tail,pin in entries:
    assert p.read_bytes()==raw;p.write_bytes(newprefix+tail);assert hashlib.sha256(p.read_bytes()[len(newprefix):]).hexdigest()==pin
proof=dict(status='ROOT_NATIVE400_NATIVE11_AND_UNRESOLVED_REPLAY_FAILURE_RECORDED',utc=now,source=ref(__file__),state_before_sha256=hashlib.sha256(before).hexdigest(),state_after=ref(sp),reader=ref(reader),adopted_roots=[ref(p) for p in [np,fp,tp,pp,dp]],five_entries=[ref(e[0]) for e in entries],old_historical_tails_exact=True,old_current_prefix_and_state_preserved=True,native=400,saved=380,FL_native=100,FL_saved=90,main_methods=11,new_training_or_inference=0,whole_rebuttal_complete=False)
with (H/'ROOT_METADATA_ADOPTION.json').open('x',encoding='utf8',newline='\n') as f:json.dump(proof,f,ensure_ascii=False,indent=2);f.write('\n')
print(json.dumps(dict(status=proof['status'],state_sha256=sha(sp),root=ref(H/'ROOT_METADATA_ADOPTION.json'))))
