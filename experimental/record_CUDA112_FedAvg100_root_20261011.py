"""Record adopted FedAvg100, without promoting remote observations to evidence."""
from pathlib import Path
from datetime import datetime, timezone
import ast, hashlib, json, re
R=Path.cwd();T=R/'docs/server_deployment_20260923/training_20260923';V=T.parent/'revision_20260923'
D=R/'tmp/root_CUDA112_FedAvg100_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
ref=lambda p:dict(path=Path(p).relative_to(R).as_posix(),sha256=sha(p))
sp=T/'TRAINING_STATE.json';before=sp.read_bytes()
assert sha(sp)=='5df16f0cb785b87cc1f586033c6a7a953f5adbea3a6d676e620d2caadb46f10d' and not D.exists()
paths={
 'FedAvg_root':R/'tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
 'old3_root':R/'tmp/old800_cuda3_saved_audit_prepare_20261011/mixed_producer_v2/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json',
 'health':T/'server_reactivation_20261009/ROOT_COMPACT_20261011T083853Z.json',
 'cuda500':R/'tmp/uniform_cuda500_root_dispatch_20261011/RESUME2_OBSERVATION_20261011T083851Z.json',
 'cuda800':R/'tmp/baseline_old800_runtime_root_20261011_attempt003/OBSERVATION_20261011T083851Z.json',
 'FL':R/'tmp/FL100_cuda_actual_root_20261011/OBSERVATION_20261011T083851Z.json'}
assert sha(paths['FedAvg_root'])=='83a047fdd4d9c997c3ca31ff713de134575f1bcc6c75f6da579031f604efb083'
f=read(paths['FedAvg_root']);prior=read(paths['old3_root']);o=read(paths['health']);c=read(paths['cuda500']);b=read(paths['cuda800']);fl=read(paths['FL'])
assert f['accepted']==100 and f['archive_all_members_rehashed'] and not f['test_evaluation']
assert (f['metric_values_checked'],f['integer_base_counts_checked'],f['prediction_rules_checked'])==(900,2400,300)
assert len(set(f['ids'])&set(prior['ids']))==1 and len(set(f['ids'])|set(prior['ids']))==102
cp=c['progress.json'];assert len(cp['remote_saved_accepted_ids'])==77 and cp['pending']==423 and not cp['failed']
assert len(b['accepted'])+len(b['reused'])==126 and not b['failures']
assert len(fl['accepted'])==40 and not fl['failures']
assert len(o['main_queue']['completed'])==432 and len(o['main_queue']['active'])==8 and not o['main_queue']['failed']
now=datetime.now(timezone.utc).isoformat();s=json.loads(before);u=s['uniform_CUDA_evaluation_v1']
u.update(status='ROOT112_ADOPTED_THREE_FROZEN_CUDA_QUEUES_RUNNING',uniform_CUDA_scientific_accepted=112,
 root_adopted_old800_unique=102,FedAvg100_scientific_accepted=100,FedAvg100_root=ref(paths['FedAvg_root']),
 root_old800_deduplication='FedAvg single in prior3 overlaps FedAvg100; unique100+2=102',
 current_remote_saved_accepted=77,current_remote_new_after_resume=20,current_remote_pending=423,current_remote_failed=0,current_observation=ref(paths['cuda500']),
 old800_remote_saved_accepted=126,old800_remaining_queue_new_accepted=len(b['remaining_progress']['new_accepted_ids']),old800_current_pending=b['remaining_progress']['remaining'],old800_observation=ref(paths['cuda800']),
 FL100_server_saved_accepted=40,FL100_observation=ref(paths['FL']),LoGo_requires_FedAvg100_root_cache=True,
 LoGo_FedAvg_root_complete=True,LoGo_Linux_index_pending=True,LoGo_original_restore_and_environment_pending=True)
s['celeba_mechanism_v1'].update(queue_observation_utc=o['utc'],queue_observation_path=str(paths['health']),queue_observation_sha256=sha(paths['health']))
s['NGA96_runtime_dispatch_20261011'].update(latest_readonly_observation=ref(paths['health']),terminal_result_directories_observed=23,latest_observed_rounds=o['NGA']['active_progress'])
s['latest_NGA96_readonly_observation']=dict(**ref(paths['health']),utc=o['utc'],terminal_observed=23,accepted_new=19,active=o['NGA']['active_progress'],Huber_started=False)
s['hybrid100_fullcoverage_20261010'].update(latest_measured_observation=dict(utc=o['utc'],terminal_observed=35,source=ref(paths['health']),acceptance_unchanged_by_observation=True))
s['latest_current_resource_observation']=dict(utc=o['utc'],source_ref=ref(paths['health']),main_terminal=432,main_active=8,main_pending=360,main_failed=0,NGA_terminal_observed=23,NGA_new_accepted=19,Hybrid_terminal_observed=35,Hybrid_native_new_accepted=29,GPU_utilization=[100,100],GPU_C=[65,62],GPU_RecoveryAction=['None','None'],memory_current_B=int(o['cgroup']['memory.current']),memory_max_B=int(o['cgroup']['memory.max']),disk_free_B=o['disk_free_B'],cgroup_CPU_quota_cores=122.87999,cgroup_CPU_measured_cores=o['measured_cgroup_CPU_cores'],cgroup_OOM=0,not_new_scientific_acceptance=True)
tree=ast.parse((R/'tmp/refresh_native370_FL86_Hybrid20_root_20261011.py').read_text());node=next(n for n in tree.body if isinstance(n,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='entries' for x in n.targets));ns={'R':R,'T':T};exec(compile(ast.Module(body=[node],type_ignores=[]),'<tails>','exec'),ns)
entries=[];prefix=None
for p,marker,h in ns['entries']:
 raw=p.read_bytes();at=raw.index(marker.encode());assert hashlib.sha256(raw[at:]).hexdigest()==h
 if prefix is None:prefix=raw[:at]
 assert prefix==raw[:at];entries.append((p,raw,raw[at:],h))
text=prefix.decode('utf8');text=re.sub(r'入口更新：[^。]+。',f'入口更新：{now}。',text,count=1)
changes={
 '2026-10-11T08:18:34.423493+00:00':o['utc'],'2026-10-11T08:21:57.557521+00:00':c['utc'],'2026-10-11T08:20:37.041749+00:00':b['utc'],
 '本地采用7/500；服务器保存验收70/500':'本地采用7/500；服务器保存验收77/500',
 '现已新验收13':'现已新验收20',
 '本地采用3/800；服务器保存验收83/800':'本地去重采用102/800；服务器保存验收126/800',
 '新通过80、717等待':'新通过123、674等待',
 '本地采用3/100；服务器保存验收6/100':'本地采用3/100；服务器保存验收40/100',
 '08:20 UTC剩余97实际启动，08:22实测，失败0':'08:20 UTC剩余97实际启动，08:38实测，失败0',
 '最新21终轮':'最新23终轮',
 '本地CUDA采用13=7+3+3':'本地CUDA去重采用112=7+102+3',
 '其余尚只计服务器保存验收。':'FedAvg100现已逐项原Linux完整验收、离机900指标/2400计数/300规则和F盘1058成员SHA核验；root tmp/FedAvg_cuda100_saved_cache_closure_prepare_20261011/root_adoption/ROOT_SCIENTIFIC_ADOPTION.json。与先前三项的FedAvg单例重叠一项，八方法唯一采用102，不能相加成103。其余尚只计服务器保存验收。',
 '08:22实测6/100服务器保存验收':'08:38实测40/100服务器保存验收',
 'LoGo100调用端源独审通过，实际缓存/恢复/环境仍未就绪':'LoGo100调用端源独审通过，FedAvg100已根采用，Linux索引、旧原始后处理产物精确恢复和独立CPU环境仍待实际核验',
 'FL3/97、NGA19及本次08:18后观察尚未发布':'FL3/97、NGA19、FedAvg100采用及本次08:38观察尚未发布',
 '双5090 100/100%、66/62°C':'双5090 100/100%、65/62°C',
 'CPU实测11.99核/122.87999配额':'CPU实测12.97核/122.87999配额',
 '内存71.13/519.17GB':'内存71.25/519.17GB',
 '本次原STATE/入口/reader/protocol备份在tmp/root_CUDA10_two_queues_Git70_20261011':'本次原STATE/入口/reader备份在tmp/root_CUDA112_FedAvg100_20261011，协议此前快照保留在root_CUDA13_NGA19_FL97_Git71_20261011'}
for a,z in changes.items():assert a in text,a;text=text.replace(a,z)
reader=V/'AUTHOR_REVIEW_START_HERE.md';oldreader=reader.read_bytes();r=oldreader.decode('utf8')
r=r.replace('FL3/97和NGA19等后续增量尚未发布。','FL3/97、NGA19和FedAvg100等后续增量尚未发布。')
r=r.replace('新CUDA已有13项','新CUDA已有112项（去重）').replace('首阶段7/500、原八方法3/800、FLGMM3/100','首阶段7/500、原八方法102/800、FLGMM3/100')
r=r.replace('各带时刻实测服务器70/500（08:21）、83/800（08:20）、FLGMM6/100（08:22）','各带时刻实测服务器77/500、126/800、FLGMM40/100（均08:38）')
r=r.replace('服务器进度不等于全部离机采用，完整CUDA表尚未完成。','FedAvg100全部通过离机900指标、2400计数、300规则核对，1058个归档成员SHA已核验并根采用；与旧三项重叠一个FedAvg单例，原八方法采用102。LoGo尚待Linux缓存索引、原post_state精确恢复和匹配CPU环境实际核验。服务器进度不等于全部离机采用，完整CUDA表尚未完成。')
for link in re.findall(r'\[[^\]]+\]\((E:/[^)]+)\)',r):assert Path(link.split('#',1)[0]).is_file(),link
D.mkdir();(D/'PRIOR_STATE.json').write_bytes(before);(D/'PRIOR_READER.md').write_bytes(oldreader);(D/'PRIOR_PREFIX.md').write_bytes(prefix)
reader.write_bytes(r.encode());s['author_review_reading_entry_20261011'].update(utc=now,reader=ref(reader),uniform_CUDA_root_adopted=112)
s['current_entry_writer']=dict(**ref(__file__),utc=now);s['local_work_verified_utc']=now;s['updated_unix']=datetime.now(timezone.utc).timestamp()
assert sp.read_bytes()==before;sp.write_text(json.dumps(s,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf8',newline='\n')
for p,raw,tail,h in entries:
 assert p.read_bytes()==raw;p.write_bytes(text.encode()+tail);assert hashlib.sha256(p.read_bytes()[len(text.encode()):]).hexdigest()==h
proof=dict(status='ROOT_CUDA112_FEDAVG100_CURRENT_ENTRIES_RECORDED',utc=now,source=ref(__file__),state_before_sha256=hashlib.sha256(before).hexdigest(),state_after=ref(sp),reader=ref(reader),five_entries=[ref(x[0]) for x in entries],history_tails_exact=True,evidence={k:ref(p) for k,p in paths.items()},root_adopted_CUDA=112,old800_unique_accepted=102,remote500_saved_accepted=77,remote800_saved_accepted=126,remoteFL_saved_accepted=40,new_scientific_adoptions_in_metadata_writer=0,whole_rebuttal_complete=False)
p=D/'ROOT_METADATA_ADOPTION.json';p.write_text(json.dumps(proof,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n');print(json.dumps({'state':ref(sp),'metadata':ref(p)}))
