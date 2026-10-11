"""Single fresh-resource authorization/start; original finite supervisor launcher."""
from pathlib import Path
import base64,datetime,hashlib,json,subprocess,sys
H=Path(__file__).resolve().parent;C=H.parent;R=C.parents[1]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,v):
 with (H/name).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def remote(source,payload,name,timeout=90):
 compile(source,name,'exec');(H/(name+'_remote.py')).write_text(source,encoding='utf8')
 code="import base64;exec(compile(base64.b64decode('"+base64.b64encode(source.encode()).decode()+"'),'<"+name+">','exec'))"
 cmd=['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=20','root@89.22.197.55','nice -n 10 ionice -c 3 python3 -c "'+code+'"']
 save(name+'_COMMAND.json',{'argv':cmd,'source_sha256':hashlib.sha256(source.encode()).hexdigest()})
 from observation_storage import run_capture
 result=run_capture(cmd,json.dumps(payload).encode(),name,timeout,bulk=(name=='FRESH_OBSERVATION'))
 return result['value']
FL_PACKAGE_SHA='699e24a9421e684f446346d0eb46252020806ee3250a3d677a3d90c7c821de39';FL_EXACT_IDS=['FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91001_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91002_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91003_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91004_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91005_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91006_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91007_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91008_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91009_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91010_fullcoverage']
assert not (H/'AUTHORIZATION.json').exists()
import argparse
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
fresh=remote(s,dict(guide_sha256='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa',manifest=m,previous_hashes=historical_hashes['hashes']),'FRESH_OBSERVATION')
try:
 assert fresh['source_model_data_hashes_verified'] and not fresh['narrow_hybrid_pool32_and_siblings_conflicts'] and not fresh['duplicate_gate']
 assert all(not x['failures'] and not x['live_producers'] for x in fresh['selected'])
 rows=fresh['guardfed_processes'];cpus=set(range(128,160))
 active_conflict=[{'pid':r['pid'],'tid':t['tid']} for r in rows for t in r['threads'] if (t.get('delta_cpu_ticks_2s') or 0)>0 and t['last_cpu'] in cpus]
 capacity=fresh['capacity_metrics']
 assert capacity['targets']==sorted(cpus) and capacity['siblings']==sorted({v for row in fresh['sibling_topology'].values() for v in row if v not in cpus}) and len(capacity['per_cpu'])==64 and fresh['pool_topology_verified'] is True
 assert capacity['quiet_pairs']>=16 and len(capacity['physical_pairs'])==32, capacity
 measured_with_allowance=capacity['cgroup_used_cores']+8+8
 fl=fresh['FL_replay_service'];workers=fresh['FL_replay_workers'];fl_name='guardfed_flgmm_FFlip10_capacity_pool32_valid'
 if fl['returncode']==0 and fl['stdout'].split()[:2]==[fl_name,'RUNNING']:
  assert len(workers)==1 and workers[0]['env'].get('OMP_NUM_THREADS')=='8' and workers[0]['env'].get('MKL_NUM_THREADS')=='8'
  assert all(set(t['affinity'])==set(range(32,64)) for t in workers[0]['threads'])
  assert sum(t.get('delta_cpu_ticks_2s') or 0 for t in workers[0]['threads'])>0, 'Concurrent FL worker has no measured tick growth'
 elif fl['returncode']==3 and fl['stdout'].split()[:2]==[fl_name,'EXITED']:
  assert not workers and fresh['FL_replay_gate']['status']=='FLGMM_EXACT10_THREE_VIEW_PASS_NOT_ROOT_ADOPTED'
  assert fresh['FL_replay_gate']['package_sha256']==FL_PACKAGE_SHA and fresh['FL_replay_gate']['exact_ids']==FL_EXACT_IDS
 else:raise AssertionError('Concurrent FL replay neither growing nor normally complete')
 service=fresh['services'];service_states={line.split()[0]:line.split()[1] for line in service['stdout'].splitlines()}
 expected_services={'guardfed_celeba_mechanism_formal','guardfed_celeba_flgmm_fullcoverage','guardfed_celeba_gradient_screen64_v2a','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_mechanism_remaining620_valid_v2a'}
 assert set(service_states)==expected_services and service['returncode'] in (0,3)
 assert service_states['guardfed_celeba_gradient_screen64_v2a']=='EXITED', 'Original64 must remain naturally exited'
 assert all(service_states[k]=='RUNNING' for k in expected_services-{'guardfed_celeba_gradient_screen64_v2a'})
 assert not [r for r in rows if any('celeba_gradient_screen64_v2_20261010' in x for x in r['argv'])], 'Old64 worker must be absent'
 assert not fresh['main_queue']['failed']
 old={x['id']:x for x in first['main_workers']};main_growth=[]
 for x in fresh['main_workers']:
  if x['id'] not in old:main_growth.append(x['id'])
  elif x.get('progress') and x['progress'].get('round',0)>old[x['id']]['progress']['round']:main_growth.append(x['id'])
 assert main_growth or len(fresh['main_queue']['completed'])>len(first['main_queue']['completed']),'No actual main progress'
 main=[r for r in rows if any('/deployment/celeba_mechanism_20261009/worker.py' in a for a in r['argv'])]
 assert 0<len(main)<=8 and all(r['env'].get('OMP_NUM_THREADS')=='1' and r['env'].get('MKL_NUM_THREADS')=='1' for r in main)
 assert len(fresh['main_workers'])==len(main)==len(fresh['main_queue']['active'])
 assert {x['id'] for x in fresh['main_workers']}=={x['id'] for x in fresh['main_queue']['active']}
 assert all(x.get('progress') and 1<=x['progress'].get('round',0)<=70 for x in fresh['main_workers']), 'Missing/new main progress is a refusal, not health'
 components={'main_all_current_OS_threads':sum(len(r['threads']) for r in main),'FL_shared_affinity':2,'Hybrid':1,'gradient14_or_future_single_worker_reserved':8,'remaining620_reserved':8,'Hybrid_FFlip10':8,'concurrent_FL_replay':8,'coordinator_IO_allowance':8}
 quota,period=fresh['cgroup']['cpu.max'].split();assert quota!='max';quota=int(quota)/int(period)
 assert sum(components.values())<=quota
 assert measured_with_allowance<quota, (measured_with_allowance,quota)
 for marker,mask in [('celeba_flgmm_fullcoverage_v2_20261009',set([102,103])),('celeba_hybrid_fullcoverage_v3_20261010',set([104])),('celeba_mechanism_remaining_evaluation_v2_20261010',set(range(112,120)))]:
  rr=[r for r in rows if any(marker in x for x in r['argv']) and any('python' in x for x in r['argv']) and not any('log-tee' in x for x in r['argv'])]
  assert rr,marker
  assert all(set(t['affinity'])<=mask for r in rr for t in r['threads']),(marker,rr)
 gate_workers=[r for r in rows if any('/adapter.py' in a and 'gradient' in a for a in r['argv'])]
 assert len(gate_workers)<=1 and all(all(set(t['affinity'])<=set(range(88,96)) for t in r['threads']) for r in gate_workers), 'Unexpected gradient gate worker'
 assert int(fresh['cgroup']['memory.max'])-int(fresh['cgroup']['memory.current'])>32*1024**3
 assert fresh['disk']['free']>20*1024**3
 assert fresh['gpu_recovery']['returncode']==0 and all(l.endswith('None') for l in fresh['gpu_recovery']['stdout'].strip().splitlines())
 assert fresh['gpu']['returncode']==0 and all(int(l.split(',')[-1])<85 for l in fresh['gpu']['stdout'].strip().splitlines())
 pre={'status':'ROOT_LINUX_HYBRID_FFLIP10_PREFLIGHT_PASS','utc':fresh['utc'],'cpu_affinity':sorted(cpus),'eligible_cpus':fresh['eligible_cpus'],'actual_quota_cores':quota,'nominal_reserved_cores_including_gate':sum(components.values()),'nominal_budget_components':components,'nominal_budget_not_hard_peak_bound':True,'no_duplicate_gate':True,'resources_eligible':True,'resources_eligible_definition':'No <=32-CPU narrow reservation on pool or SMT siblings, duplicate or selected producer; nominal budget within quota; at least16 of32 distinct physical pairs have target and SMT sibling sampled busy<=20%; cgroup measured usage+8 evaluator+8 IO allowance<quota; original source/data, memory, disk, GPU and service guards pass. This is sampled capacity, not exclusive cores.', 'capacity_policy':'sampled_32core_pool_atleast16_pairs_le20pct_cgroup_plus16_lt_quota_v3','sample_monitored_cpu_count':len(capacity['per_cpu']),'sample_each_cpu_busy_fraction_max':capacity['max_busy_fraction'],'quiet_pairs':capacity['quiet_pairs'],'physical_pairs':capacity['physical_pairs'],'pool_topology_verified':fresh['pool_topology_verified'],'same_socket_cpu_pool':True,'smt_sibling_cpus':capacity['siblings'],'excluded_gate_smt_cpus':fresh['excluded_gate_smt_cpus'],'FL_replay_runtime_verified':True,'FL_replay_service':fl,'FL_replay_workers':workers,'core_topology':fresh['core_topology'],'measured_cgroup_cores_with_gate_and_allowance':measured_with_allowance,'capacity_sample':capacity,'wide_active_thread_observations':active_conflict,'source_model_data_hashes_verified':True,'selected_producers_quiescent':True,'services_healthy':True,'gpu_health_verified':True,'cgroup_and_memory_headroom_verified':True,'storage_headroom_verified':True,'original_hash_observation_sha256':ref['sha256'],'stat_resource_refresh_sha256':read(H/'FRESH_OBSERVATION_REF.json')['sha256'],'historical_hash_reference_files':len(historical_hashes['hashes']),'current_verified_input_files':len(fresh['hashes']),'main_progressed_ids':main_growth,'package_sha256':sha(C/'FILES_SHA256.json'),'new_CNN':0,'new_fit':0,'test':False}
except BaseException as error:
 save('LAUNCH_REFUSAL.json',dict(status='HYBRID_POOL32_V3_LAUNCH_REFUSED_NO_RETRY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(error),observation_ref=read(H/'FRESH_OBSERVATION_REF.json'),resources_eligible=False,authorization_created=False,new_CNN=0,new_fit=0,new_training=0,new_three_views_accepted=0,retry_authorized=False))
 raise
assert 0<=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(pre['utc'])).total_seconds()<=300
save('LINUX_PREFLIGHT.json',pre)
auth={'status':'ROOT_AUTHORIZED_HYBRID_FFLIP10_CANARY_FIRST_VALID','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'package_sha256':sha(C/'FILES_SHA256.json'),'manifest_sha256':sha(C/'MANIFEST.json'),'source_review_sha256':sha(C/'ROOT_SOURCE_REVIEW.json'),'linux_preflight_sha256':sha(H/'LINUX_PREFLIGHT.json'),'exact_ids':m['exact_ids'],'canary_id':m['canary_id'],'cpu_affinity':sorted(cpus),'smt_sibling_cpus':capacity['siblings'],'excluded_gate_smt_cpus':fresh['excluded_gate_smt_cpus'],'device':'cpu','max_processes':1,'test':False,'delegated_root_authorization':'Actual SHA-bound root authorization for existing adopted Hybrid IID F Flip exact10 only; first91001 native<=1e-12 then9; no other jobs/test.'}
save('AUTHORIZATION.json',auth)
base='/workspace/guardfed_checks/hybrid_FFlip10_three_view_prepare_20261011';program='guardfed_hybrid_FFlip10_valid'
args=['--cpus',','.join(map(str,range(128,160))),'--package-sha256',auth['package_sha256'],'--authorization',base+'/AUTHORIZATION.json','--authorization-sha256',sha(H/'AUTHORIZATION.json'),'--preflight',base+'/LINUX_PREFLIGHT.json','--preflight-sha256',auth['linux_preflight_sha256'],'--source-review',base+'/ROOT_SOURCE_REVIEW.json','--source-review-sha256',auth['source_review_sha256'],'--output',base+'/outputs/attempt001']
config='[program:'+program+']\ncommand=/usr/bin/env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /usr/bin/taskset -c 128-159 /usr/bin/ionice -c 3 /usr/bin/nice -n 10 '+m['server_python']+' -B '+base+'/source/candidate.py '+' '.join(args)+'\ndirectory='+base+'\nautostart=false\nautorestart=false\nstartretries=0\nstartsecs=1\nstopasgroup=true\nkillasgroup=true\nstdout_logfile='+base+'/execution/stdout.log\nstderr_logfile='+base+'/execution/stderr.log\nstdout_logfile_maxbytes=0\nstderr_logfile_maxbytes=0\n'
payload={'base':base,'program':program,'package':auth['package_sha256'],'source_review_sha256':auth['source_review_sha256'],'config':config,'config_sha256':hashlib.sha256(config.encode()).hexdigest(),'files':{}}
for name in ['LINUX_PREFLIGHT.json','AUTHORIZATION.json']:
 b=(H/name).read_bytes();payload['files'][name]={'sha256':sha(H/name),'bytes':len(b),'base64':base64.b64encode(b).decode()}
assert sha(R/'tmp/celeba_flgmm_closed47_root_execution_20261011/START_RECEIPT_remote.py')=='762820e76a2cc8de4ee7e03b6c12e5b90e4ea2bf08d496945012b2d087a9fe36'
s="assert __import__('hashlib').sha256(__import__('pathlib').Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'\n"+(R/'tmp/celeba_flgmm_closed47_root_execution_20261011/START_RECEIPT_remote.py').read_text().replace('celeba_flgmm_three_view_closed_batch_20261011','hybrid_FFlip10_three_view_prepare_20261011').replace('guardfed_flgmm_closed47_valid',program).replace('ROOT_FLGMM_FINITE47','ROOT_HYBRID_FINITE10')
result=remote(s,payload,'START_RECEIPT');print(json.dumps({'status':result['status'],'utc':result['utc'],'cpu_budget':sum(components.values()),'quota':quota}))
