"""Derive only operational entrypoints; never deploy/sample/launch or touch science."""
from pathlib import Path
import ast,difflib,hashlib,json
D=Path(__file__).resolve().parent;C=D.parent;R=C.parents[1]
OLD=R/'tmp/hybrid_missing8_pool32_runtime_prepared_20261011/runtime'
H=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();J=lambda p:json.loads(p.read_bytes())
changes={};diffs=[];checks={}
def once(s,a,b):
    assert s.count(a)==1,a[:100]
    return s.replace(a,b)
def emit(name,s):
    old=(OLD/name).read_text();compile(s,name,'exec')
    inv=list(difflib.ndiff(old.splitlines(True),s.splitlines(True)))
    assert ''.join(difflib.restore(inv,1))==old and ''.join(difflib.restore(inv,2))==s
    changes[name]=inv;diffs.extend(difflib.unified_diff(old.splitlines(True),s.splitlines(True),fromfile='original/'+name,tofile='runtime/'+name))
    (D/name).write_text(s,encoding='utf8',newline='')
    checks[name]=dict(parent_sha256=H(OLD/name),candidate_sha256=H(D/name),compile=True,full_inverse_exact=True)
ns='hybrid_FFlip10_three_view_prepare_20261011';oldns='celeba_hybrid_three_view_missing8_20261011'
s=(OLD/'observer.py').read_text().replace('range(64,96)','range(128,160)')
s=once(s,"    assert targets==list(range(128,160)) and siblings==list(range(320,352))","    assert targets==list(range(128,160)) and len(siblings)==32 and not set(targets)&set(siblings)")
s=once(s,"    pairs=[dict(target=cpu,sibling=cpu+256,target_busy=by_cpu[cpu]['busy_fraction'],sibling_busy=by_cpu[cpu+256]['busy_fraction'],quiet=by_cpu[cpu]['busy_fraction']<=.20 and by_cpu[cpu+256]['busy_fraction']<=.20) for cpu in targets]","    pairs=[]\n    for cpu in targets:\n        sibling=next(v for v in sibling_map[str(cpu)] if v!=cpu)\n        pairs.append(dict(target=cpu,sibling=sibling,target_busy=by_cpu[cpu]['busy_fraction'],sibling_busy=by_cpu[sibling]['busy_fraction'],quiet=by_cpu[cpu]['busy_fraction']<=.20 and by_cpu[sibling]['busy_fraction']<=.20))")
s=s.replace("(set(range(128,160))|set(range(320,352)))","set(monitored_pool_cpus)")
s=s.replace("len(af)<=8 and 110 in af","len(af)<=32 and 110 in af")
start=s.index('targets=list(range(128,160)); siblings=');end=s.index('before,narrow=processes()',start)
s=s[:start]+'''targets=list(range(128,160)); siblings=[]; sibling_map={}; core_topology={}
def topology_pair(cpu):
    topology=P('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
    pair=cpu_set((topology/'thread_siblings_list').read_text())
    assert cpu in pair and len(pair)==2, 'Actual two-thread SMT topology required'
    return pair, {'socket':int((topology/'physical_package_id').read_text()),'core':int((topology/'core_id').read_text())}
for cpu in targets:
    pair,identity=topology_pair(cpu);other=next(v for v in pair if v!=cpu)
    sibling_map[str(cpu)]=sorted(pair);siblings.append(other);core_topology[str(cpu)]=identity
siblings=sorted(siblings)
assert len(set(siblings))==32 and not set(siblings)&set(targets)
monitored_pool_cpus=sorted(targets+siblings)
assert set(monitored_pool_cpus)<=set(eligible), 'Fixed pool or actual SMT sibling outside eligible mask'
for cpu in siblings:
    pair,identity=topology_pair(cpu);core_topology[str(cpu)]=identity
    target=next(v for v in pair if v in targets)
    assert pair==set(sibling_map[str(target)]) and identity==core_topology[str(target)], 'SMT identity drift'
assert len({x['socket'] for x in core_topology.values()})==1, 'Pool not one socket'
assert len({core_topology[str(cpu)]['core'] for cpu in targets})==32, 'Pool reuses physical cores'
excluded=set(range(11,19))|set(range(32,96))|set(range(102,120))
gate_smt=set()
for cpu in sorted(excluded):
    pair,_=topology_pair(cpu);excluded.update(pair)
    if cpu in range(88,96):gate_smt.update(pair-{cpu})
excluded_gate_smt=sorted(gate_smt)
assert len(excluded_gate_smt)==8 and not set(monitored_pool_cpus)&excluded, 'Reserved old/gate pool or SMT overlap'
''' +s[end:]
s=once(s,"duplicate=[r for r in after if any('hybrid_three_view_missing8' in a and '/source/candidate.py' in a for a in r['argv'])]","duplicate=[r for r in after if any('/source/candidate.py' in a and any(n in a for n in ('hybrid_three_view_missing8','hybrid_screen91001_single','hybrid_FFlip10_three_view_prepare_20261011')) for a in r['argv'])]")
s=s.replace("'pool_topology_verified':True,","'pool_topology_verified':True,'excluded_gate_smt_cpus':excluded_gate_smt,'excluded_reserved_and_smt_cpus':sorted(excluded),")
emit('observer.py',s)
s=(OLD/'observation_storage.py').read_text().replace('hybrid_missing8_pool32_runtime_prepared_20261011','hybrid_FFlip10_three_view_prepare_20261011').replace("hybrid_FFlip10_three_view_prepare_20261011/runtime'", "hybrid_FFlip10_three_view_prepare_20261011/runtime_v2'")
emit('observation_storage.py',s)
s=(OLD/'deploy_once.py').read_text().replace(oldns,ns)
# Deployment keeps the original fresh namespace + sealed source member body.
s=once(s,"source=old.read_text().replace('celeba_flgmm_three_view_closed_batch_20261011','"+ns+"')","source=\"assert __import__('hashlib').sha256(__import__('pathlib').Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'\\n\"+old.read_text().replace('celeba_flgmm_three_view_closed_batch_20261011','"+ns+"')")
s=once(s,"source=\"assert __import__", "assert sha(old)=='7a1eaff809262b4c2264529374d18433154a2b8ee856a6e6a18ca09549b25156'\nsource=\"assert __import__")
emit('deploy_once.py',s)
s=(OLD/'launch_once.py').read_text().replace(oldns,ns).replace('guardfed_hybrid_missing8_valid','guardfed_hybrid_FFlip10_valid').replace('ROOT_LINUX_HYBRID_MISSING8_PREFLIGHT_PASS','ROOT_LINUX_HYBRID_FFLIP10_PREFLIGHT_PASS').replace('ROOT_AUTHORIZED_HYBRID_MISSING8_CANARY_FIRST_VALID','ROOT_AUTHORIZED_HYBRID_FFLIP10_CANARY_FIRST_VALID').replace('ROOT_HYBRID_FINITE8','ROOT_HYBRID_FINITE10').replace('range(64,96)','range(128,160)').replace('taskset -c 64-95','taskset -c 128-159').replace("'Hybrid_missing8':8","'Hybrid_FFlip10':8")
old="m=read(C/'MANIFEST.json');ref=read(H/'OBSERVATION_REF.json');assert sha(ref['path'])==ref['sha256'];first=read(ref['path'])\ns=(H/'observer.py').read_text()\nfresh=remote(s,dict(read(H/'observer_payload.json'),previous_hashes=first['hashes']),'FRESH_OBSERVATION')"
new="""import argparse
parser=argparse.ArgumentParser();parser.add_argument('--root-inputs',type=Path,required=True);parser.add_argument('--root-inputs-sha256',required=True);args=parser.parse_args()
assert sha(args.root_inputs)==args.root_inputs_sha256
bound=read(args.root_inputs);assert bound['status']=='ROOT_AUTHORIZED_HYBRID_FFLIP10_FRESH_GATE_AND_START_ONCE' and bound['root_authorized'] is True
assert bound['package_sha256']==sha(C/'FILES_SHA256.json') and bound['manifest_sha256']==sha(C/'MANIFEST.json')
assert bound['science_review_sha256']==sha(C/'ROOT_SOURCE_REVIEW.json')
for name in ('observer.py','launch_once.py','deploy_once.py','observation_storage.py'):
 assert sha(H/name)==bound['runtime_sources'][name]
m=read(C/'MANIFEST.json');assert m['exact_ids']==['CosineFairness_lam20.0_tau0.1_lr0.001_IID_F Flip_seed'+str(n)+'_fullcoverage' for n in range(91001,91011)]
ref=bound['previous_hash_observation'];assert sha(ref['path'])==ref['sha256'];historical_hashes=read(ref['path'])
prior=bound['previous_main_observation'];assert sha(prior['path'])==prior['sha256'];prior_raw=read(prior['path'])
first={'main_workers':[{'id':Path(x['job']).stem,'progress':{'round':x['round']}} for x in prior_raw['main_mechanism']['workers']], 'main_queue':prior_raw['main_mechanism']['queue']}
s=(H/'observer.py').read_text()
fresh=remote(s,dict(guide_sha256='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa',manifest=m,previous_hashes=historical_hashes['hashes']),'FRESH_OBSERVATION')"""
s=once(s,old,new)
s=once(s,"capacity['siblings']==list(range(320,352))","capacity['siblings']==sorted({v for row in fresh['sibling_topology'].values() for v in row if v not in cpus})")
s=once(s," service=fresh['services'];assert service['returncode']==0 and len(service['stdout'].splitlines())==5 and all('RUNNING' in l for l in service['stdout'].splitlines())",""" service=fresh['services'];service_states={line.split()[0]:line.split()[1] for line in service['stdout'].splitlines()}
 expected_services={'guardfed_celeba_mechanism_formal','guardfed_celeba_flgmm_fullcoverage','guardfed_celeba_gradient_screen64_v2a','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_mechanism_remaining620_valid_v2a'}
 assert set(service_states)==expected_services and service['returncode'] in (0,3)
 assert service_states['guardfed_celeba_gradient_screen64_v2a']=='EXITED', 'Original64 must remain naturally exited'
 assert all(service_states[k]=='RUNNING' for k in expected_services-{'guardfed_celeba_gradient_screen64_v2a'})
 assert not [r for r in rows if any('celeba_gradient_screen64_v2_20261010' in x for x in r['argv'])], 'Old64 worker must be absent'
 assert not fresh['main_queue']['failed']""")
s=once(s,"elif x.get('progress')!=old[x['id']].get('progress'):main_growth.append(x['id'])","elif x.get('progress') and x['progress'].get('round',0)>old[x['id']]['progress']['round']:main_growth.append(x['id'])")
s=once(s," assert 0<len(main)<=8 and all(r['env'].get('OMP_NUM_THREADS')=='1' and r['env'].get('MKL_NUM_THREADS')=='1' for r in main)"," assert 0<len(main)<=8 and all(r['env'].get('OMP_NUM_THREADS')=='1' and r['env'].get('MKL_NUM_THREADS')=='1' for r in main)\n assert len(fresh['main_workers'])==len(main)==len(fresh['main_queue']['active'])\n assert {x['id'] for x in fresh['main_workers']}=={x['id'] for x in fresh['main_queue']['active']}\n assert all(x.get('progress') and 1<=x['progress'].get('round',0)<=70 for x in fresh['main_workers']), 'Missing/new main progress is a refusal, not health'")
s=s.replace("'gradient':1,","'gradient14_or_future_single_worker_reserved':8,")
s=once(s,"('celeba_gradient_screen64_v2_20261010',set([105])),",'')
s=once(s," assert int(fresh['cgroup']['memory.max'])", " gate_workers=[r for r in rows if any('/adapter.py' in a and 'gradient' in a for a in r['argv'])]\n assert len(gate_workers)<=1 and all(all(set(t['affinity'])<=set(range(88,96)) for t in r['threads']) for r in gate_workers), 'Unexpected gradient gate worker'\n assert int(fresh['cgroup']['memory.max'])")
s=once(s,"'same_socket_cpu_pool':True,","'same_socket_cpu_pool':True,'smt_sibling_cpus':capacity['siblings'],'excluded_gate_smt_cpus':fresh['excluded_gate_smt_cpus'],")
s=once(s,"len(first['hashes'])","len(historical_hashes['hashes'])")
s=once(s,"'cpu_affinity':sorted(cpus),'device':'cpu'","'cpu_affinity':sorted(cpus),'smt_sibling_cpus':capacity['siblings'],'excluded_gate_smt_cpus':fresh['excluded_gate_smt_cpus'],'device':'cpu'")
s=once(s,"Parent root explicit exact8 execution authorization for adopted Hybrid exact8 after actual source review; prior91002 excluded and first91003 must pass.","Actual SHA-bound root authorization for existing adopted Hybrid IID F Flip exact10 only; first91001 native<=1e-12 then9; no other jobs/test.")
s=once(s,"save('LINUX_PREFLIGHT.json',pre)","assert 0<=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(pre['utc'])).total_seconds()<=300\nsave('LINUX_PREFLIGHT.json',pre)")
line="s=(R/'tmp/celeba_flgmm_closed47_root_execution_20261011/START_RECEIPT_remote.py').read_text()"
s=once(s,line,"s=\"assert __import__('hashlib').sha256(__import__('pathlib').Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'\\n\"+(R/'tmp/celeba_flgmm_closed47_root_execution_20261011/START_RECEIPT_remote.py').read_text()")
s=once(s,"s=\"assert __import__", "assert sha(R/'tmp/celeba_flgmm_closed47_root_execution_20261011/START_RECEIPT_remote.py')=='762820e76a2cc8de4ee7e03b6c12e5b90e4ea2bf08d496945012b2d087a9fe36'\ns=\"assert __import__")
emit('launch_once.py',s)
(D/'SOURCE_DIFF.patch').write_text(''.join(diffs),encoding='utf8')
(D/'SOURCE_INVERSE.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
template=dict(status='TEMPLATE_NOT_AUTHORIZATION',root_authorized=False,package_sha256=H(C/'FILES_SHA256.json'),manifest_sha256=H(C/'MANIFEST.json'),science_review_sha256=None,runtime_sources={n:H(D/n) for n in checks},previous_hash_observation=dict(path=None,sha256=None),previous_main_observation=dict(path=None,sha256=None),actual_preflight=None,actual_authorization=None,cpu_targets=list(range(128,160)),actual_smt_siblings=None,compute_threads=8)
(D/'ROOT_RUNTIME_INPUTS.template.json').write_text(json.dumps(template,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
checks=dict(status='SOURCE_ONLY_OPERATIONAL_DERIVATION_COMPILE_AND_INVERSE_PASS',files=checks,science_package_sha256=H(C/'FILES_SHA256.json'),candidate_science_unchanged=True,actual_resources_bound=False,SSH=False,F_access=False,launched=False,scope='Only runtime adapters; original statistics/17 science functions and saved consumer are untouched.')
(D/'SOURCE_CHECK.json').write_text(json.dumps(checks,ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(dict(status=checks['status'],files=len(checks['files']))))
