"""Single fresh-resource authorization/start; original finite supervisor launcher."""
from pathlib import Path
import base64,datetime,hashlib,json,subprocess,sys
H=Path(__file__).resolve().parent;R=H.parents[2];C=R/'tmp/fl_SpDFA10_final100_isolated_v3_20261011'
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

def remaining_guard(fresh, fixed):
 a=fresh['auxiliary']['remaining'];f=fixed['remaining_halted_failure']
 state={line.split()[0]:line.split()[1] for line in fresh['services']['stdout'].splitlines()}['guardfed_celeba_mechanism_remaining620_valid_v2a']
 rows=[r for r in fresh['guardfed_processes'] if any(fixed['remaining_base']+'/' in x for x in r['argv'])]
 # This is a pinned unresolved numerical failure, never normal completion or recovery.
 assert state=='EXITED' and not rows, 'Only the reviewed quiescent failed queue may be isolated'
 assert a['failure_sha256']==f['failure_files'] and a['queue_complete'] is None, 'New/changed failure or completion state'
 progress=a['progress'];assert progress['status']=='READY_FOR_ORIGINAL_STRICT' and progress['id']==f['id']
 assert progress['remote_closed_ids']==f['closed_ids'] and progress['remote_closed_n']==len(f['closed_ids'])
 assert a['closed_marker_ids']==sorted(f['closed_ids']), 'Closed prefix changed'
 assert a['binding_sha256']==f['binding_sha256'] and a['receipt_sha256']==f['receipt_sha256']
 assert a['checkpoint_sha256']==f['checkpoint_sha256'] and a['receipt_id']==f['id']
 assert a['native_comparison']['accepted'] is False, 'Failed native comparison replaced'
 assert not a['failed_task_closed'] and not a['failed_task_strict'], 'Failed task must remain unaccepted'
 return {'remaining_failure_evidence':a,'remaining_live_processes':len(rows),'remaining_state':state,'remaining_known_failure_isolated':True,'remaining_healthy':False,'remaining_complete':False,'remaining_failure_id':f['id'],'remaining_queue_failure_sha256':f['queue_failure_sha256'],'remaining_closed_marker_count':len(f['closed_ids']),'remaining_retry_authorized':False}

def auxiliary_guard(fresh, fixed):
 rows=fresh['guardfed_processes'];a=fresh['auxiliary']
 assert a['source_hashes']==fixed['source_hashes'], 'Producer source identities differ'
 states={line.split()[0]:line.split()[1] for line in fresh['services']['stdout'].splitlines()}
 def at_base(base):
  return [r for r in rows if any(base+'/' in x for x in r['argv']) and 'python' in Path(r['argv'][0]).name]
 def running_group(base,mask,scripts):
  rr=at_base(base);assert rr,base
  assert all(sum(base+'/'+s in r['argv'] for s in scripts)==1 for r in rr), 'Foreign stage producer'
  assert all(r['threads'] and all(set(t['affinity'])<=mask for t in r['threads']) for r in rr), 'Producer affinity changed'
  return rr
 flbase=fixed['FL_stage'];flrows=at_base(flbase);flstate=states['guardfed_celeba_flgmm_fullcoverage']
 assert not a['FL_failures'], 'Current FL failure preserved'
 if flstate=='RUNNING':
  running_group(flbase,{102,103},('run_fullcoverage.py','run_one.py'))
 elif flstate=='EXITED':
  q=a['FL_queue'];t=a['FL_terminal']
  assert not flrows and q['active']==[] and q['pending']==0 and q['completed_new']==96 and q['reused']==4 and q['failed'] is False, 'FL exit is not exact normal completion'
  assert t and t['status']=='100_STRICT_ACCEPTED_BACKUP_PENDING' and t['accepted_new']==96 and t['accepted_reused']==4
  assert t['manifest_sha256']==fixed['source_hashes'][flbase+'/manifest.json'] and t['package_sha256']==fixed['source_hashes'][flbase+'/PACKAGE_SHA256.json']
  new=[r['id'] for r in t['records'] if r['reused'] is False];old=[r['id'] for r in t['records'] if r['reused'] is True]
  assert new==fixed['FL_ids'] and old==fixed['FL_reused_ids'] and len(t['records'])==100 and all(r['rounds']==70 for r in t['records']), 'FL terminal ID set differs'
 else:raise AssertionError('FL neither RUNNING nor exact normally completed EXITED')
 assert states['guardfed_celeba_hybrid_fullcoverage']=='RUNNING'
 running_group(fixed['Hybrid_stage'],{104},('run_fullcoverage.py','run_one.py'))
 remaining=remaining_guard(fresh,fixed)
 ns=a['NGA_service'];assert ns['returncode']==0 and ns['stdout'].split()[:2]==[fixed['NGA_service'],'RUNNING'], 'NGA completion/absence requires separate reviewed evidence'
 # The frozen runner blocks on one preflight helper, then on one worker.
 # A helper is neither a second manager nor a science worker.
 ng=[r for r in rows if any(fixed['NGA_base']+'/' in x for x in r['argv'])]
 parents=[];children=[];helpers=[];options={}
 common={'--repo':'/workspace/GuardFed-celeba-expanded','--source-approval':fixed['NGA_source_approval'],'--source-approval-sha256':fixed['NGA_source_approval_sha256'],'--gate14-adoption':fixed['NGA_gate14'],'--gate14-adoption-sha256':fixed['NGA_gate14_sha256']}
 for r in ng:
  argv=r['argv'];assert argv[:2]==[fixed['NGA_python'],'-B'] and len(argv)>=3, 'NGA executable/prefix changed'
  assert argv[2] in (fixed['NGA_runner'],fixed['NGA_preflight']), 'Unknown NGA producer'
  tail=argv[3:];assert len(tail)%2==0 and all(x.startswith('--') for x in tail[::2]), 'Malformed NGA flags'
  flags=dict(zip(tail[::2],tail[1::2]));assert len(flags)*2==len(tail), 'Duplicate NGA flags'
  assert all(flags.get(k)==v for k,v in common.items()), 'NGA source/repo argv differs'
  options[r['pid']]=flags
  if argv[2]==fixed['NGA_preflight']:
   assert set(flags)==set(common)|{'--base','--out','--job','--controller-pid'} and flags['--base']==fixed['NGA_base'], 'NGA helper argv differs'
   helpers.append(r)
  elif '--single-job' in flags:
   assert set(flags)==set(common)|{'--out','--single-job','--resource-preflight','--resource-sha256','--lock-fd'}, 'NGA worker argv differs'
   children.append(r)
  else:
   assert set(flags)==set(common)|{'--out','--resource-preflight','--resource-sha256'}, 'NGA manager argv differs'
   assert flags['--resource-preflight']==fixed['NGA_initial_resource'] and flags['--resource-sha256']==fixed['NGA_initial_resource_sha256'], 'NGA actual002 bootstrap changed'
   parents.append(r)
 assert len(parents)==1 and len(children)+len(helpers)<=1 and len(options)==len(ng) and ns['stdout'].split()[3].rstrip(',')==str(parents[0]['pid']), 'NGA parent/worker/helper or supervisor PID differs'
 assert not a['NGA_failures'], 'Current NGA queue/child failure'
 for r in ng:
  flags=options[r['pid']];assert r['threads'] and all(t['affinity']==[105] for t in r['threads'])
  assert all(r['env'].get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','CUDA_VISIBLE_DEVICES')), 'NGA CPU/GPU/thread policy differs'
  if r in children or r in helpers:
   identity=flags['--single-job'] if r in children else flags['--job']
   assert identity in a['NGA_ids'] and r['ppid']==parents[0]['pid'] and flags['--out']==fixed['NGA_output']+'/'+identity, 'NGA subprocess parent/job/output changed'
   if r in helpers:assert flags['--controller-pid']==str(parents[0]['pid']), 'NGA helper controller changed'
   else:
    proof=Path(flags['--resource-preflight']);digest=flags['--resource-sha256']
    assert str(proof.parent).replace('\\','/')==fixed['NGA_output'] and proof.name.startswith('ROOT_RESOURCE_') and proof.name.endswith('.json'), 'NGA worker resource path changed'
    assert len(digest)==64 and all(c in '0123456789abcdef' for c in digest) and flags['--lock-fd'].isdigit(), 'NGA worker resource/lock metadata malformed'
  else:assert flags['--out']==fixed['NGA_output']
 # Legacy gradient gate must be absent; NGA's actual producer is the frozen run_queue.
 assert not [r for r in rows if any('/adapter.py' in x and 'gradient' in x for x in r['argv'])], 'Unexpected old gradient adapter'
 return {**remaining,'source_hashes':a['source_hashes'],'FL_state':flstate,'FL_normal_completion_verified':flstate=='EXITED','NGA_parent_pid':parents[0]['pid'],'NGA_children':len(children),'NGA_preflight_helpers':len(helpers),'NGA_observed_OS_threads':sum(len(r['threads']) for r in ng),'NGA_reserved_compute_threads':len(ng),'scientific_adoptions':0}

FL_PACKAGE_SHA='699e24a9421e684f446346d0eb46252020806ee3250a3d677a3d90c7c821de39';FL_EXACT_IDS=['FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91001_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91002_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91003_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91004_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91005_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91006_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91007_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91008_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91009_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91010_fullcoverage']
assert not (H/'AUTHORIZATION.json').exists()
import argparse
parser=argparse.ArgumentParser();parser.add_argument('--root-inputs',type=Path,required=True);parser.add_argument('--root-inputs-sha256',required=True);args=parser.parse_args()
assert sha(args.root_inputs)==args.root_inputs_sha256
bound=read(args.root_inputs);assert bound['status']=='ROOT_AUTHORIZED_FLGMM_SpDFA10_FRESH_GATE_AND_START_ONCE' and bound['root_authorized'] is True
assert bound['package_sha256']==sha(C/'FILES_SHA256.json') and bound['manifest_sha256']==sha(C/'MANIFEST.json')
assert bound['science_review_sha256']==sha(C/'ROOT_SOURCE_REVIEW.json')
for name in ('observer.py','launch_once.py','deploy_once.py','observation_storage.py','AUXILIARY_FIXED.json'):
 assert sha(H/name)==bound['runtime_sources'][name]
fixed=read(H/'AUXILIARY_FIXED.json')
m=read(C/'MANIFEST.json');assert m['cpu_affinity']==list(range(128,160)) and m['threads']==8;assert m['exact_ids']==['FLGMM_Tg20_L2.0_lr0.001_non-IID_Sp-DFA_seed'+str(n)+'_fullcoverage' for n in range(91001,91011)]
ref=bound['previous_hash_observation'];assert sha(ref['path'])==ref['sha256'];historical_hashes=read(ref['path'])
prior=bound['previous_main_observation'];assert sha(prior['path'])==prior['sha256'];prior_raw=read(prior['path'])
first={'main_workers':[{'id':Path(x['job']).stem,'progress':{'round':x['round']}} for x in prior_raw['main_mechanism']['workers']], 'main_queue':prior_raw['main_mechanism']['queue']}
s=(H/'observer.py').read_text()
fresh=remote(s,dict(guide_sha256='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa',manifest=m,previous_hashes=historical_hashes['hashes'],auxiliary_fixed=fixed),'FRESH_OBSERVATION')
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
  assert fresh['FL_replay_gate']['package_sha256']==FL_PACKAGE_SHA and [r['id'] for r in fresh['FL_replay_gate']['receipts']]==FL_EXACT_IDS
 else:raise AssertionError('Concurrent FL replay neither growing nor normally complete')
 service=fresh['services'];service_states={line.split()[0]:line.split()[1] for line in service['stdout'].splitlines()}
 expected_services={'guardfed_celeba_mechanism_formal','guardfed_celeba_flgmm_fullcoverage','guardfed_celeba_gradient_screen64_v2a','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_mechanism_remaining620_valid_v2a'}
 assert set(service_states)==expected_services and service['returncode'] in (0,3)
 assert service_states['guardfed_celeba_gradient_screen64_v2a']=='EXITED', 'Original64 must remain naturally exited'
 assert service_states['guardfed_celeba_mechanism_formal']=='RUNNING'
 auxiliary=auxiliary_guard(fresh,fixed)
 assert not [r for r in rows if any('celeba_gradient_screen64_v2_20261010' in x for x in r['argv'])], 'Old64 worker must be absent'
 assert not fresh['main_queue']['failed']
 old={x['id']:x for x in first['main_workers']};main_growth=[]
 for x in fresh['main_workers']:
  if x['id'] not in old:main_growth.append(x['id'])
  elif (type((x.get('progress') or {}).get('round')) is int and type((old[x['id']].get('progress') or {}).get('round')) is int
        and x['progress']['round']>old[x['id']]['progress']['round']):main_growth.append(x['id'])
 assert main_growth or len(fresh['main_queue']['completed'])>len(first['main_queue']['completed']),'No actual main progress'
 main=[r for r in rows if any('/deployment/celeba_mechanism_20261009/worker.py' in a for a in r['argv'])]
 assert 0<len(main)<=8 and all(r['env'].get('OMP_NUM_THREADS')=='1' and r['env'].get('MKL_NUM_THREADS')=='1' for r in main)
 assert len(fresh['main_workers'])==len(main)==len(fresh['main_queue']['active'])
 assert {x['id'] for x in fresh['main_workers']}=={x['id'] for x in fresh['main_queue']['active']}
 assert all(x.get('progress') and 1<=x['progress'].get('round',0)<=70 for x in fresh['main_workers']), 'Missing/new main progress is a refusal, not health'
 components={'main_all_current_OS_threads':sum(len(r['threads']) for r in main),'FL_shared_affinity':2,'Hybrid':1,'gradient_reserved_guard':8,'NGA96_compute':auxiliary['NGA_reserved_compute_threads'],'remaining620_reserved':8,'FLGMM_SpDFA10':8,'concurrent_FL_replay':8,'coordinator_IO_allowance':8}
 quota,period=fresh['cgroup']['cpu.max'].split();assert quota!='max';quota=int(quota)/int(period)
 assert sum(components.values())<=quota
 assert measured_with_allowance<quota, (measured_with_allowance,quota)
 assert int(fresh['cgroup']['memory.max'])-int(fresh['cgroup']['memory.current'])>32*1024**3
 assert fresh['disk']['free']>20*1024**3
 assert fresh['gpu_recovery']['returncode']==0 and all(l.endswith('None') for l in fresh['gpu_recovery']['stdout'].strip().splitlines())
 assert fresh['gpu']['returncode']==0
 gpu_rows=[[x.strip() for x in l.split(',')] for l in fresh['gpu']['stdout'].strip().splitlines()]
 assert len(gpu_rows)==2 and [int(x[0]) for x in gpu_rows]==[0,1] and all(int(x[-1])<85 and int(x[4])-int(x[3])>=2048 for x in gpu_rows), 'GPU temperature/free-memory gate'
 pre={'status':'ROOT_LINUX_FLGMM_EXACT10_PREFLIGHT_PASS','utc':fresh['utc'],'cpu_affinity':sorted(cpus),'eligible_cpus':fresh['eligible_cpus'],'actual_quota_cores':quota,'nominal_reserved_cores_including_gate':sum(components.values()),'nominal_budget_components':components,'nominal_budget_not_hard_peak_bound':True,'no_duplicate_gate':True,'resources_eligible':True,'resources_eligible_definition':'No <=32-CPU narrow reservation on pool or SMT siblings, duplicate or selected producer; nominal budget within quota; at least16 of32 distinct physical pairs have target and SMT sibling sampled busy<=20%; cgroup measured usage+8 evaluator+8 IO allowance<quota; original source/data, memory, disk, GPU and service guards pass. This is sampled capacity, not exclusive cores.', 'capacity_policy':'sampled_32core_pool_atleast16_pairs_le20pct_cgroup_plus16_lt_quota_v3','sample_monitored_cpu_count':len(capacity['per_cpu']),'sample_each_cpu_busy_fraction_max':capacity['max_busy_fraction'],'quiet_pairs':capacity['quiet_pairs'],'physical_pairs':capacity['physical_pairs'],'pool_topology_verified':fresh['pool_topology_verified'],'same_socket_cpu_pool':True,'smt_sibling_cpus':capacity['siblings'],'excluded_gate_smt_cpus':fresh['excluded_gate_smt_cpus'],'FL_replay_runtime_verified':True,'FL_replay_service':fl,'FL_replay_workers':workers,'core_topology':fresh['core_topology'],'measured_cgroup_cores_with_gate_and_allowance':measured_with_allowance,'capacity_sample':capacity,'wide_active_thread_observations':active_conflict,'source_model_data_hashes_verified':True,'selected_producers_quiescent':True,'services_healthy':False,'known_auxiliary_failure_isolated':True,'auxiliary_runtime_guard':auxiliary,'gpu_health_verified':True,'cgroup_and_memory_headroom_verified':True,'storage_headroom_verified':True,'original_hash_observation_sha256':ref['sha256'],'stat_resource_refresh_sha256':read(H/'FRESH_OBSERVATION_REF.json')['sha256'],'historical_hash_reference_files':len(historical_hashes['hashes']),'current_verified_input_files':len(fresh['hashes']),'main_progressed_ids':main_growth,'package_sha256':sha(C/'FILES_SHA256.json'),'new_CNN':0,'new_fit':0,'test':False}
except BaseException as error:
 save('LAUNCH_REFUSAL.json',dict(status='FLGMM_SpDFA10_POOL32_LAUNCH_REFUSED_NO_RETRY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(error),observation_ref=read(H/'FRESH_OBSERVATION_REF.json'),resources_eligible=False,authorization_created=False,new_CNN=0,new_fit=0,new_training=0,new_three_views_accepted=0,retry_authorized=False))
 raise
assert 0<=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(pre['utc'])).total_seconds()<=300
save('LINUX_PREFLIGHT.json',pre)
auth={'status':'ROOT_AUTHORIZED_FLGMM_CLOSED_EXACT10_THREE_VIEW','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'package_sha256':sha(C/'FILES_SHA256.json'),'manifest_sha256':sha(C/'MANIFEST.json'),'source_review_sha256':sha(C/'ROOT_SOURCE_REVIEW.json'),'linux_preflight_sha256':sha(H/'LINUX_PREFLIGHT.json'),'exact_ids':m['exact_ids'],'canary_id':m['exact_ids'][0],'cpu_affinity':sorted(cpus),'smt_sibling_cpus':capacity['siblings'],'excluded_gate_smt_cpus':fresh['excluded_gate_smt_cpus'],'device':'cpu','max_processes':1,'test':False,'delegated_root_authorization':'Actual SHA-bound root authorization for existing adopted FLGMM non-IID Sp-DFA exact10 only; first91001 native<=1e-12 then9; no other jobs/test.'}
save('AUTHORIZATION.json',auth)
base='/workspace/guardfed_checks/fl_SpDFA10_final100_isolated_v3_20261011';program='guardfed_flgmm_SpDFA10_isolated_v3_valid'
args=['--package-sha256',auth['package_sha256'],'--authorization',base+'/AUTHORIZATION.json','--authorization-sha256',sha(H/'AUTHORIZATION.json'),'--preflight',base+'/LINUX_PREFLIGHT.json','--preflight-sha256',auth['linux_preflight_sha256'],'--source-review',base+'/ROOT_SOURCE_REVIEW.json','--source-review-sha256',auth['source_review_sha256'],'--output',base+'/outputs/attempt001']
config='[program:'+program+']\ncommand=/usr/bin/env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /usr/bin/taskset -c 128-159 /usr/bin/ionice -c 3 /usr/bin/nice -n 10 '+m['server_python']+' -B '+base+'/source/candidate.py '+' '.join(args)+'\ndirectory='+base+'\nautostart=false\nautorestart=false\nstartretries=0\nstartsecs=1\nstopasgroup=true\nkillasgroup=true\nstdout_logfile='+base+'/execution/stdout.log\nstderr_logfile='+base+'/execution/stderr.log\nstdout_logfile_maxbytes=0\nstderr_logfile_maxbytes=0\n'
payload={'base':base,'program':program,'package':auth['package_sha256'],'source_review_sha256':auth['source_review_sha256'],'config':config,'config_sha256':hashlib.sha256(config.encode()).hexdigest(),'files':{}}
for name in ['LINUX_PREFLIGHT.json','AUTHORIZATION.json']:
 b=(H/name).read_bytes();payload['files'][name]={'sha256':sha(H/name),'bytes':len(b),'base64':base64.b64encode(b).decode()}
assert sha(R/'tmp/celeba_flgmm_closed47_root_execution_20261011/START_RECEIPT_remote.py')=='762820e76a2cc8de4ee7e03b6c12e5b90e4ea2bf08d496945012b2d087a9fe36'
s="assert __import__('hashlib').sha256(__import__('pathlib').Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'\n"+(R/'tmp/celeba_flgmm_closed47_root_execution_20261011/START_RECEIPT_remote.py').read_text().replace('celeba_flgmm_three_view_closed_batch_20261011','fl_SpDFA10_final100_isolated_v3_20261011').replace('guardfed_flgmm_closed47_valid',program).replace('ROOT_FLGMM_FINITE47','ROOT_FLGMM_FINITE10')
result=remote(s,payload,'START_RECEIPT');print(json.dumps({'status':result['status'],'utc':result['utc'],'cpu_budget':sum(components.values()),'quota':quota}))
