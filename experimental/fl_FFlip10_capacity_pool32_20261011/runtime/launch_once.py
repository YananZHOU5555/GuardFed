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
assert not (H/'AUTHORIZATION.json').exists()
m=read(C/'MANIFEST.json');ref=read(H/'OBSERVATION_REF.json');assert sha(ref['path'])==ref['sha256'];first=read(ref['path'])
s=(H/'observer.py').read_text();a=s.index('hashes=[];seen={}');b=s.index('hash_end=utc()',a)
s=s[:a]+'''hashes=[];seen={}
assert len(PAYLOAD['previous_hashes'])==81 and {x['path'] for x in PAYLOAD['previous_hashes']}==set(pins)
for previous in PAYLOAD['previous_hashes']:
 assert previous['match'] and previous['sha256']==pins[previous['path']]['sha256']
 p=P(previous['path']); st=p.stat()
 assert str(p.resolve())==previous['resolved'] and st.st_size==previous['bytes'] and st.st_mtime_ns==previous['mtime_ns'], 'Previously hashed input stat changed'
 hashes.append(dict(previous, stat_refreshed=True))
'''+s[b:]
fresh=remote(s,dict(read(H/'observer_payload.json'),previous_hashes=first['hashes']),'FRESH_OBSERVATION')
try:
 assert fresh['source_model_data_hashes_verified'] and not fresh['narrow_pool32_and_siblings_conflicts'] and not fresh['duplicate_gate']
 assert all(not x['failures'] and not x['live_producers'] for x in fresh['selected'])
 rows=fresh['guardfed_processes'];cpus=set(range(32,64))
 active_conflict=[{'pid':r['pid'],'tid':t['tid']} for r in rows for t in r['threads'] if (t.get('delta_cpu_ticks_2s') or 0)>0 and t['last_cpu'] in cpus]
 capacity=fresh['capacity_metrics']
 assert capacity['targets']==sorted(cpus) and capacity['siblings']==list(range(288,320)) and len(capacity['per_cpu'])==64 and fresh['pool_topology_verified'] is True
 assert capacity['quiet_pairs']>=16 and len(capacity['physical_pairs'])==32, capacity
 measured_with_allowance=capacity['cgroup_used_cores']+8+8
 service=fresh['services'];assert service['returncode']==0 and len(service['stdout'].splitlines())==5 and all('RUNNING' in l for l in service['stdout'].splitlines())
 old={x['id']:x for x in first['main_workers']};main_growth=[]
 for x in fresh['main_workers']:
  if x['id'] not in old:main_growth.append(x['id'])
  elif x.get('progress')!=old[x['id']].get('progress'):main_growth.append(x['id'])
 assert main_growth or len(fresh['main_queue']['completed'])>len(first['main_queue']['completed']),'No actual main progress'
 main=[r for r in rows if any('/deployment/celeba_mechanism_20261009/worker.py' in a for a in r['argv'])]
 assert 0<len(main)<=8 and all(r['env'].get('OMP_NUM_THREADS')=='1' and r['env'].get('MKL_NUM_THREADS')=='1' for r in main)
 components={'main_all_current_OS_threads':sum(len(r['threads']) for r in main),'FL_shared_affinity':2,'Hybrid':1,'gradient':1,'remaining620_reserved':8,'finite10':8,'coordinator_IO_allowance':8}
 quota,period=fresh['cgroup']['cpu.max'].split();assert quota!='max';quota=int(quota)/int(period)
 assert sum(components.values())<=quota
 assert measured_with_allowance<quota, (measured_with_allowance,quota)
 for marker,mask in [('celeba_flgmm_fullcoverage_v2_20261009',set([102,103])),('celeba_hybrid_fullcoverage_v3_20261010',set([104])),('celeba_gradient_screen64_v2_20261010',set([105])),('celeba_mechanism_remaining_evaluation_v2_20261010',set(range(112,120)))]:
  rr=[r for r in rows if any(marker in x for x in r['argv']) and any('python' in x for x in r['argv']) and not any('log-tee' in x for x in r['argv'])]
  assert rr,marker
  assert all(set(t['affinity'])<=mask for r in rr for t in r['threads']),(marker,rr)
 assert int(fresh['cgroup']['memory.max'])-int(fresh['cgroup']['memory.current'])>32*1024**3
 assert fresh['disk']['free']>20*1024**3
 assert fresh['gpu_recovery']['returncode']==0 and all(l.endswith('None') for l in fresh['gpu_recovery']['stdout'].strip().splitlines())
 assert fresh['gpu']['returncode']==0 and all(int(l.split(',')[-1])<85 for l in fresh['gpu']['stdout'].strip().splitlines())
 pre={'status':'ROOT_LINUX_FLGMM_EXACT10_PREFLIGHT_PASS','utc':fresh['utc'],'cpu_affinity':sorted(cpus),'eligible_cpus':fresh['eligible_cpus'],'actual_quota_cores':quota,'nominal_reserved_cores_including_gate':sum(components.values()),'nominal_budget_components':components,'nominal_budget_not_hard_peak_bound':True,'no_duplicate_gate':True,'resources_eligible':True,'resources_eligible_definition':'No narrow CPU reservation, duplicate or selected producer; nominal budget within quota; at least16 of32 distinct physical pairs have target and SMT sibling sampled busy<=20%; cgroup measured usage+8 evaluator+8 IO allowance<quota; original source/data, memory, disk, GPU and service guards pass. This is sampled capacity, not exclusive cores.', 'capacity_policy':'sampled_32core_pool_atleast16_pairs_le20pct_cgroup_plus16_lt_quota_v3','sample_monitored_cpu_count':len(capacity['per_cpu']),'sample_each_cpu_busy_fraction_max':capacity['max_busy_fraction'],'quiet_pairs':capacity['quiet_pairs'],'physical_pairs':capacity['physical_pairs'],'pool_topology_verified':fresh['pool_topology_verified'],'core_topology':fresh['core_topology'],'measured_cgroup_cores_with_gate_and_allowance':measured_with_allowance,'capacity_sample':capacity,'wide_active_thread_observations':active_conflict,'source_model_data_hashes_verified':True,'selected_producers_quiescent':True,'services_healthy':True,'gpu_health_verified':True,'cgroup_and_memory_headroom_verified':True,'storage_headroom_verified':True,'original_hash_observation_sha256':ref['sha256'],'stat_resource_refresh_sha256':read(H/'FRESH_OBSERVATION_REF.json')['sha256'],'hashed_files':len(first['hashes']),'main_progressed_ids':main_growth,'package_sha256':sha(C/'FILES_SHA256.json'),'new_CNN':0,'new_fit':0,'test':False}
except BaseException as error:
 save('LAUNCH_REFUSAL.json',dict(status='CAPACITY_POOL32_V3_LAUNCH_REFUSED_NO_RETRY',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),error=repr(error),observation_ref=read(H/'FRESH_OBSERVATION_REF.json'),resources_eligible=False,authorization_created=False,new_CNN=0,new_fit=0,new_training=0,new_three_views_accepted=0,retry_authorized=False))
 raise
save('LINUX_PREFLIGHT.json',pre)
auth={'status':'ROOT_AUTHORIZED_FLGMM_CLOSED_EXACT10_THREE_VIEW','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'package_sha256':sha(C/'FILES_SHA256.json'),'manifest_sha256':sha(C/'MANIFEST.json'),'source_review_sha256':sha(C/'ROOT_SOURCE_REVIEW.json'),'linux_preflight_sha256':sha(H/'LINUX_PREFLIGHT.json'),'exact_ids':m['exact_ids'],'cpu_affinity':sorted(cpus),'device':'cpu','max_processes':1,'test':False,'delegated_root_authorization':'Parent root explicit exact10 execution authorization after native67 adoption and actual source review; prior61 unchanged.'}
save('AUTHORIZATION.json',auth)
base='/workspace/guardfed_checks/fl_FFlip10_capacity_pool32_20261011';program='guardfed_flgmm_FFlip10_capacity_pool32_valid'
args=['--package-sha256',auth['package_sha256'],'--authorization',base+'/AUTHORIZATION.json','--authorization-sha256',sha(H/'AUTHORIZATION.json'),'--preflight',base+'/LINUX_PREFLIGHT.json','--preflight-sha256',auth['linux_preflight_sha256'],'--source-review',base+'/ROOT_SOURCE_REVIEW.json','--source-review-sha256',auth['source_review_sha256'],'--output',base+'/outputs/attempt001']
config='[program:'+program+']\ncommand=/usr/bin/env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1 /usr/bin/taskset -c 32-63 /usr/bin/ionice -c 3 /usr/bin/nice -n 10 '+m['server_python']+' -B '+base+'/source/candidate.py '+' '.join(args)+'\ndirectory='+base+'\nautostart=false\nautorestart=false\nstartretries=0\nstartsecs=1\nstopasgroup=true\nkillasgroup=true\nstdout_logfile='+base+'/execution/stdout.log\nstderr_logfile='+base+'/execution/stderr.log\nstdout_logfile_maxbytes=0\nstderr_logfile_maxbytes=0\n'
payload={'base':base,'program':program,'package':auth['package_sha256'],'source_review_sha256':auth['source_review_sha256'],'config':config,'config_sha256':hashlib.sha256(config.encode()).hexdigest(),'files':{}}
for name in ['LINUX_PREFLIGHT.json','AUTHORIZATION.json']:
 b=(H/name).read_bytes();payload['files'][name]={'sha256':sha(H/name),'bytes':len(b),'base64':base64.b64encode(b).decode()}
s=(R/'tmp/celeba_flgmm_closed47_root_execution_20261011/START_RECEIPT_remote.py').read_text().replace('celeba_flgmm_three_view_closed_batch_20261011','fl_FFlip10_capacity_pool32_20261011').replace('guardfed_flgmm_closed47_valid',program).replace('FINITE47','FINITE10')
result=remote(s,payload,'START_RECEIPT');print(json.dumps({'status':result['status'],'utc':result['utc'],'cpu_budget':sum(components.values()),'quota':quota}))
