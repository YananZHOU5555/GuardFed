"""One source-only operational pool amendment; no live operations or science execution."""
from pathlib import Path
import ast,json,hashlib,marshal,difflib
H=Path(__file__).resolve().parent;O=H.parent/'fl_FFlip10_capacity_cpu11_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
pin=lambda p:dict(sha256=sha(p),bytes=Path(p).stat().st_size)
def save(p,v):
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def funcs(b):
 s=b.decode();return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
oldns=O.name;newns=H.name
oldprog='guardfed_flgmm_FFlip10_capacity_cpu11_valid';newprog='guardfed_flgmm_FFlip10_capacity_pool32_valid'
policy0='sampled_cpu_busy_le20pct_cgroup_plus16_lt_quota_v2';policy1='sampled_32core_pool_atleast16_pairs_le20pct_cgroup_plus16_lt_quota_v3'
seals=['FILES_SHA256.json','runtime/SOURCE_ONLY_FILES_SHA256.json','saved_v2/STATIC_SOURCE_SHA256.json'];oldseals=[read(O/s) for s in seals]
assert sha(O/seals[0])=='74023838ca591c56eab96a80f3607cb91f2f48c499c80edc80891eb4356bba8f'
for j,prefix in zip(oldseals,['','runtime/','saved_v2/']):
 for rel,v in j['files'].items():assert pin(O/(prefix+rel))==v
inverse={};diff=[];edits={};compiled=[]
def edit(rel,replacements):
 original=(O/rel).read_bytes();s=original.decode();nl='\r\n' if '\r\n' in s else '\n';s=s.replace('\r\n','\n');oldtext=s;applied=[]
 for a,b in replacements:
  n=s.count(a)
  assert n>0,(rel,a)
  s=s.replace(a,b);applied.append((a,b,n))
 restored=s
 for a,b,n in reversed(applied):
  assert restored.count(b)==n,(rel,b,n,restored.count(b))
  restored=restored.replace(b,a)
 assert restored==oldtext and restored.replace('\n',nl).encode()==original
 out=s.replace('\n',nl).encode();p=H/rel;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(out)
 inverse[rel]=True;edits[rel]=[dict(before=a,after=b,count=n) for a,b,n in applied]
 if out!=original:diff.extend(difflib.unified_diff(oldtext.splitlines(True),s.splitlines(True),fromfile='cpu11/'+rel,tofile='pool32/'+rel))
 if rel.endswith('.py'):compile(out,rel,'exec');compiled.append(rel)
 return s
base=[(oldns,newns),(oldprog,newprog)]
aff=[('range(11, 19)','range(32, 64)'),('range(11,19)','range(32,64)'),('narrow_cpu11_18_conflicts','narrow_pool32_and_siblings_conflicts'),('taskset -c 11-18','taskset -c 32-63')]
def applicable(rel,pairs):
 s=(O/rel).read_text();return [(a,b) for a,b in pairs if a in s]
# Candidate operational assertions only. Original per-record science remains untouched.
candidate=[("require(set(CPUS) <= set(p['eligible_cpus']), 'Requested CPUs outside actual eligible mask')","require(set(CPUS + list(range(288, 320))) <= set(p['eligible_cpus']) and p['pool_topology_verified'] is True, 'Pool/siblings outside actual eligible mask or topology changed')"),
("p['sample_monitored_cpu_count'] == 16","p['sample_monitored_cpu_count'] == 64"),
("p['sample_each_cpu_busy_fraction_max'] <= .20","p['quiet_pairs'] >= 16"),
("affinity.intersection(CPUS)","affinity.intersection(CPUS + list(range(288, 320)))"),(policy0,policy1)]
# Pair table preserves the original first-eight tick accounting and cgroup measurement.
metrics=[("assert targets==list(range(11,19)) and len(siblings)==len(set(siblings))==8 and not set(targets)&set(siblings)","assert targets==list(range(32,64)) and siblings==list(range(288,320))"),
("    return dict(interval_seconds=seconds,targets=targets,siblings=siblings,per_cpu=rows,\n", "    by_cpu={row['cpu']:row for row in rows}\n    pairs=[dict(target=cpu,sibling=cpu+256,target_busy=by_cpu[cpu]['busy_fraction'],sibling_busy=by_cpu[cpu+256]['busy_fraction'],quiet=by_cpu[cpu]['busy_fraction']<=.20 and by_cpu[cpu+256]['busy_fraction']<=.20) for cpu in targets]\n    return dict(interval_seconds=seconds,targets=targets,siblings=siblings,per_cpu=rows,physical_pairs=pairs,quiet_pairs=sum(pair['quiet'] for pair in pairs),\n")]
topold="""targets=list(range(11,19)); sibling_map={}
for cpu in targets:
    pair=cpu_set((P('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology/thread_siblings_list').read_text())
    assert cpu in pair and len(pair)==2
    sibling_map[str(cpu)]=sorted(pair)
siblings=sorted(set().union(*(set(v) for v in sibling_map.values()))-set(targets))
assert len(siblings)==8 and all(len(set(v)&set(targets))==1 for v in sibling_map.values())"""
topnew="""targets=list(range(32,64)); siblings=list(range(288,320)); sibling_map={}; core_topology={}
assert set(targets+siblings)<=set(eligible), 'Fixed pool or SMT sibling outside actual eligible mask'
for cpu in targets+siblings:
    topology=P('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
    pair=cpu_set((topology/'thread_siblings_list').read_text())
    target=cpu if cpu in targets else cpu-256
    assert pair=={target,target+256}, 'Fixed pool SMT topology drift'
    core_topology[str(cpu)]={'socket':int((topology/'physical_package_id').read_text()),'core':int((topology/'core_id').read_text())}
    if cpu in targets:sibling_map[str(cpu)]=sorted(pair)
assert len({core_topology[str(cpu)]['socket'] for cpu in targets+siblings})==1, 'Pool not one socket'
assert len({core_topology[str(cpu)]['core'] for cpu in targets})==32, 'Pool reuses physical cores'
assert all(core_topology[str(cpu)]==core_topology[str(cpu+256)] for cpu in targets), 'SMT physical identity drift'"""
dup="('fl_three_view_FFlip10_20261011','fl_three_view_FFlip10_cpu136_20261011','fl_FFlip10_capacity_guard_v2_20261011','fl_FFlip10_capacity_cpu11_20261011')"
observer=metrics+[(topold,topnew),('set(affinity)&set(range(11,19))','set(affinity)&(set(range(32,64))|set(range(288,320)))'),('narrow_cpu11_18_conflicts','narrow_pool32_and_siblings_conflicts'),(dup,dup[:-1]+",'"+newns+"')"),("'sibling_topology':sibling_map,","'sibling_topology':sibling_map,'core_topology':core_topology,'pool_topology_verified':True,")]
launcher=[("capacity['targets']==sorted(cpus) and len(capacity['siblings'])==8","capacity['targets']==sorted(cpus) and capacity['siblings']==list(range(288,320)) and len(capacity['per_cpu'])==64 and fresh['pool_topology_verified'] is True"),
("assert capacity['max_busy_fraction']<=.20, capacity","assert capacity['quiet_pairs']>=16 and len(capacity['physical_pairs'])==32, capacity"),
("each target and SMT sibling sampled busy<=20%","at least16 of32 distinct physical pairs have target and SMT sibling sampled busy<=20%"),
("'sample_each_cpu_busy_fraction_max':capacity['max_busy_fraction'],","'sample_each_cpu_busy_fraction_max':capacity['max_busy_fraction'],'quiet_pairs':capacity['quiet_pairs'],'physical_pairs':capacity['physical_pairs'],'pool_topology_verified':fresh['pool_topology_verified'],'core_topology':fresh['core_topology'],"),
(policy0,policy1),('CAPACITY_V2_LAUNCH_REFUSED_NO_RETRY','CAPACITY_POOL32_V3_LAUNCH_REFUSED_NO_RETRY')]
pending0='fresh Linux CPU11-18 and SMT siblings sampled capacity (not exclusive cores), narrow reservation, quota/memory/GPU/storage preflight'
pending1='fresh Linux fixed pool32-63 and SMT288-319 eligible/same-socket distinct-core topology; at least16 quiet physical pairs; no narrow reservation; quota/memory/GPU/storage preflight (sampled capacity, not exclusivity)'
def json_edits(rel):
 j=read(O/rel);obj=j['manifest'] if rel.startswith('runtime/') else j;assert obj['cpu_affinity']==list(range(11,19))
 indent=6 if rel.startswith('runtime/') else 4
 arr=lambda xs:',\n'.join(' '*indent+str(i) for i in xs)
 s=edit(rel,[(arr(range(11,19)),arr(range(32,64))),(pending0,pending1)])
 new=json.loads(s);sub=new['manifest'] if rel.startswith('runtime/') else new
 sub['cpu_affinity']=list(range(11,19));sub['authorizations_and_runtime_facts_pending']=[x.replace(pending1,pending0) for x in sub['authorizations_and_runtime_facts_pending']];assert new==j
for rel in oldseals[0]['files']:
 if rel=='candidate.py':edit(rel,applicable(rel,base+aff)+candidate)
 elif rel=='MANIFEST.json':json_edits(rel)
 else:edit(rel,[])
obj=dict(oldseals[0]);obj['status']='FROZEN_EXACT10_CAPACITY_POOL32_V3_SOURCE_ONLY_NOT_AUTHORIZED';obj['files']={r:pin(H/r) for r in obj['files']};save(H/seals[0],obj);pkg=sha(H/seals[0]);oldpkg=sha(O/seals[0])
for r in oldseals[1]['files']:
 rel='runtime/'+r
 if r=='observer.py':edit(rel,observer)
 elif r=='observer_payload.json':json_edits(rel)
 elif r=='OBSERVATION_REF.json':edit(rel,[])
 elif r=='launch_once.py':edit(rel,applicable(rel,base+aff)+launcher)
 else:edit(rel,applicable(rel,base))
for r in oldseals[2]['files']:
 rel='saved_v2/'+r;edit(rel,applicable(rel,base+[(oldpkg,pkg)]))
for seal,j,prefix in zip(seals[1:],oldseals[1:],['runtime/','saved_v2/']):
 obj=dict(j);obj['status']='POOL32_V3_SOURCE_ONLY_NOT_DISPATCHED_ACTUAL_OUTPUT_PINS_PENDING';obj['files']={r:pin(H/(prefix+r)) for r in obj['files']};save(H/seal,obj)
science=[]
for row in read(O/'SOURCE_CHECK.json')['science17']:
 rel=row['member'];name=row['name'];a=funcs((O/rel).read_bytes())[name];b=funcs((H/rel).read_bytes())[name]
 assert a==b and marshal.dumps(compile(a,'same_science','exec'))==marshal.dumps(compile(b,'same_science','exec'));science.append(row)
assert len(science)==17
of=funcs((O/'candidate.py').read_bytes());nf=funcs((H/'candidate.py').read_bytes());changed=[n for n in of if of[n]!=nf[n]];assert changed==['authorize','exclusive_cpus','run']
for n in of:
 if n not in changed:assert marshal.dumps(compile(of[n],'same_candidate','exec'))==marshal.dumps(compile(nf[n],'same_candidate','exec'))
# Original whole per-record sequence byte-exact after namespace reversal; first seed replay_one must return accepted before append/next.
a=of['run'].replace(O.name,H.name).replace('guardfed_flgmm_FFlip10_capacity_cpu11_valid','guardfed_flgmm_FFlip10_capacity_pool32_valid');assert a==nf['run']
assert read(H/'MANIFEST.json')['exact_ids'][0].endswith('seed91001_fullcoverage')
assert "require(comparison['accepted'], 'Native metrics exceed fixed tolerance; preserve evidence and stop without retry')" in (H/'originals/replay.py').read_text()
assert sha(H/'saved_v2/saved_science.py')=='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745'
assert (H/'runtime/OBSERVATION_REF.json').read_bytes()==(O/'runtime/OBSERVATION_REF.json').read_bytes()
assert read(H/'runtime/observer_payload.json')['manifest']==read(H/'MANIFEST.json')
assert 'timeout=300' in (H/'saved_v2/transport.py').read_text()
obs=(H/'runtime/observer.py').read_text();launch=(H/'runtime/launch_once.py').read_text()
splice=lambda s:next(n.value for n in ast.walk(ast.parse(s)) if isinstance(n,ast.Constant) and isinstance(n.value,str) and n.value.startswith('hashes=[];seen={}') and 'for previous in' in n.value)
assert splice(launch)==splice((O/'runtime/launch_once.py').read_text())
assert not (H/'runtime/observe_once.py').exists()
# Source-only new pair-count arithmetic boundary; no old fixtures/imports/CPU reads.
ns={};exec(funcs((H/'runtime/observer.py').read_bytes())['capacity_metrics'],ns)
targets=list(range(32,64));siblings=list(range(288,320));start={'monotonic':0.,'usage_usec':0,'proc_stat':{str(i):[0]*8 for i in targets+siblings}}
fixtures=[]
for quiet in [15,16,32]:
 end={'monotonic':2.,'usage_usec':24_000_000,'proc_stat':{}}
 for index,cpu in enumerate(targets):
  end['proc_stat'][str(cpu)]=[0,0,0,200,0,0,0,0] if index<quiet else [200,0,0,0,0,0,0,0]
  end['proc_stat'][str(cpu+256)]=[0,0,0,200,0,0,0,0]
 result=ns['capacity_metrics'](start,end,targets,siblings);assert result['quiet_pairs']==quiet and len(result['physical_pairs'])==32 and len(result['per_cpu'])==64
 assert (result['quiet_pairs']>=16)==(quiet!=15);fixtures.append(dict(quiet_pairs=quiet,eligible_pair_threshold=(quiet>=16)))
refusals=read(H.parent/'fl_FFlip10_capacity_guard_v2_20261011/PARENT_RUNTIME_PINS.json')['prior_refusals']
for n in ['fl_FFlip10_capacity_guard_v2_20261011','fl_FFlip10_capacity_cpu11_20261011']:
 p=H.parent/n/'runtime/LAUNCH_REFUSAL.json';refusals.append(dict(path=str(p),**pin(p)))
for row in refusals:assert pin(row['path'])=={k:row[k] for k in ['sha256','bytes']}
assert len(refusals)==5
save(H/'INVERSE_EDITS.json',dict(parent_namespace=oldns,new_namespace=newns,parent_package_sha256=oldpkg,new_package_sha256=pkg,edits=edits))
(H/'SOURCE_DIFF.patch').write_text(''.join(diff),encoding='utf8')
save(H/'PARENT_AND_REFUSAL_PINS.json',dict(parent_package=dict(path=str(O/seals[0]),**pin(O/seals[0])),preserved_refusals=refusals,prior_runtime_seal=dict(path=str(O/seals[1]),**pin(O/seals[1]))))
save(H/'SOURCE_CHECK.json',dict(status='PASS_SOURCE_ONLY_POOL32_OPERATIONAL_GUARD',package_sha256=pkg,inverse_exact=inverse,compiled=compiled,science17=science,candidate_changed_only=changed,per_record_run_loop_inverse_namespace_exact=True,first_seed91001_native_1e12_acceptance_before_next=True,metadata_changes_only_affinity_and_resource_pending=True,historical81ref_bytes_exact=True,fresh_stat_splice_exact=True,saved_consumer_inverse_exact=True,new_pair_count_fixtures=fixtures,old_fixtures_rerun=False,SSH=0,CPU_samples=0,CNN=0,fit=0,training=0,STATE=0,Git=0))
save(H/'HANDOFF.json',dict(status='SOURCE_ONLY_POOL32_V3_ROOT_REVIEW_REQUIRED',package_sha256=pkg,runtime_seal_sha256=sha(H/seals[1]),saved_seal_sha256=sha(H/seals[2]),source_check_sha256=sha(H/'SOURCE_CHECK.json'),CPU_affinity=targets,required_SMТ_siblings=siblings,threads=8,processes=1,capacity_policy=policy1,monitored_logical_cpus=64,required_quiet_physical_pairs=16,actual_topology_verified=False,actual_resources_eligible=False,actual_new_outputs=0,exact_ids=read(H/'MANIFEST.json')['exact_ids'],deploy_command=['python','-B',str(H/'runtime/deploy_once.py'),'--source-review-sha256','ACTUAL_ROOT_SOURCE_REVIEW_SHA256'],single_fresh_gate_command=['python','-B',str(H/'runtime/launch_once.py')],expected_root_source_review_schema=dict(source_adoptable=True,package_sha256=pkg),no_retry=True,previous_refusals=5,root_review_required=True))
(H/'OPERATIONAL_AMENDMENT.md').write_text('固定 affinity 池32–63，SMT siblings必须现场确认为288–319，64 logical CPUs全部eligible，32个不同physical cores且同socket。保持8 Torch计算threads、1interop thread、1process和nice10/idleIO；OS在池内调度，不按32线程计预算。启动门要求至少16对target及sibling各自采样busy<=20%，实际cgroup使用+8eval+8IO<真实quota，原nominal预算不变；无≤8线程窄mask与池或siblings重叠，duplicate保留全部5namespace，其余source/data/producer/GPU/内存磁盘门不变。每对及quiet数量落记录；这是采样余量，非独占/硬峰值保证。固定拓扑或eligible失败直接拒绝，不自动换核/重试。17science/FP32/exact10/recipe/native1e-12/whole/consumer及300s运输不变；第一seed91001原replay_one在返回前要求native accepted，失败阻止后9项且保留输出。仅资源策略变化，不预称数值等价。保留5旧拒绝。本目录尚无实际sample/部署/授权/输出；根审后唯一freshgate，不运行全系统survey。只做source compile/inverse及新增pair计数15/16/32边界，不重跑旧fixtures或科学验证。\n',encoding='utf8')
files={p.relative_to(H).as_posix():pin(p) for p in sorted(H.rglob('*')) if p.is_file()};save(H/'DELIVERY_FILES_SHA256.json',dict(status='SOURCE_ONLY_POOL32_V3_DELIVERY',files=files))
print(json.dumps(dict(package=pkg,runtime=sha(H/seals[1]),saved=sha(H/seals[2]),source_check=sha(H/'SOURCE_CHECK.json'),handoff=sha(H/'HANDOFF.json'),delivery=sha(H/'DELIVERY_FILES_SHA256.json'),members=len(files),compiled=len(compiled))))
