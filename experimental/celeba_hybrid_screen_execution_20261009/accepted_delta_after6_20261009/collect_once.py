"""One snapshot of original Hybrid terminals; unchanged checked, no dispatch or inference."""
from pathlib import Path
import os,sys,json,hashlib,tarfile,datetime,subprocess,traceback
B=Path(__file__).resolve().parent;H=B.parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(H))
from driver import digest,read,functions,approve

def save(name,value):
 with (B/name).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def main():
 assert not (B/'live_snapshot.json').exists()
 assert digest('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
 assert set(os.sched_getaffinity(0))=={106} and os.getpriority(os.PRIO_PROCESS,0)>=10
 assert 'idle' in subprocess.check_output(['ionice','-p',str(os.getpid())],text=True)
 processes=[];nominal=0
 for proc in Path('/proc').iterdir():
  if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
  try:
   argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
   if not argv or 'python' not in Path(argv[0]).name:continue
   affinity=sorted(os.sched_getaffinity(int(proc.name)));assert not(len(affinity)<=16 and 106 in affinity),'CPU106 occupied'
   if any(s.startswith(('/workspace/guardfed_checks/','/workspace/GuardFed-')) for s in argv[1:]):
    env=dict(x.split('=',1) for x in (proc/'environ').read_text().split('\0') if '=' in x);n=max([int(env[k]) for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','GUARDFED_CPU_THREADS'] if env.get(k,'').isdigit()]+[1]);nominal+=n
    processes.append(dict(pid=int(proc.name),argv=argv,declared_threads=n,affinity=affinity if len(affinity)<=16 else {'count':len(affinity)}))
  except (FileNotFoundError,ProcessLookupError,PermissionError):continue
 quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split();assert quota!='max' and nominal+1<=int(quota)/int(period)
 assert digest(H/'FILES_SHA256.json')=='2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f'
 assert digest(H/'screen_scope.json')=='d76d5fdff375c58b3b42354fc4256e19b26b29f3fdb4387530f0ae8617920f94'
 assert digest(H/'runtime_protocol.json')=='bcc66477d22096eaf647e31f065f59ed6727716dcf01db814d5edcbe4ad525f1'
 scope,approval=approve('screen',H/'APPROVED_screen.json',digest(H/'APPROVED_screen.json'),dispatch=False)
 rows=[]
 for e in scope['jobs']:
  out=H/e['output'];p=read(out/'progress.json') if (out/'progress.json').exists() else None
  rows.append(dict(id=e['id'],terminal=(out/'acceptance.json').exists(),result_exists=(out/'result.json').exists(),round=p['round'] if p else None,failures=[str(f) for f in out.glob('failure*.json')]))
 service=subprocess.check_output(['supervisorctl','status','guardfed_celeba_hybrid_screen32'],text=True).strip()
 snapshot=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),rows=rows,service=service,processes=processes,nominal_threads=nominal,cpu_quota=int(quota)/int(period),helper_cpu=106,helper_nice=os.getpriority(os.PRIO_PROCESS,0),helper_IO='idle')
 save('live_snapshot.json',snapshot)
 assert digest(B/'PREVIOUS_CHAIN.json')=='231dda94d276174aaf488bb03aa71a8072b2bd0b4a0f63c241fa412e72d3fcd7' and digest(B/'AUTHORIZED_SNAPSHOT.json')=='e27bc9d6dde558ad3470edbb6906f0b44878bac25fca6e610cb8e9237cf52845'
 wanted=['CosineFairness_lam5.0_tau0.2_lr0.0005_non-IID_Benign_seed91001_screen', 'CosineFairness_lam5.0_tau0.2_lr0.0005_non-IID_S-DFA_seed91001_screen', 'CosineFairness_lam20.0_tau0.1_lr0.0005_IID_Benign_seed91001_screen', 'CosineFairness_lam20.0_tau0.1_lr0.0005_IID_S-DFA_seed91001_screen']
 previous=read(B/'PREVIOUS_CHAIN.json')
 assert set(wanted)<={r['id'] for r in rows if r['terminal'] and r['result_exists'] and r['round']==70} and not set(wanted)&set(previous['accepted_job_ids'])
 assert read(B/'EXACT_DELTA.json')['selected_ids']==wanted and previous['accepted_total']==6
 assert not(H/'screen_failure.json').exists() and not any(r['failures'] for r in rows)
 if not wanted:save('NO_NEW_TERMINAL.json',dict(status='NO_NEW_TERMINAL',accepted=0,snapshot_sha256=digest(B/'live_snapshot.json')));return
 import torch
 torch.set_num_threads(1)
 import body
 body.verify_scope(scope)
 scope=dict(scope,runtime_cuda_visible_device=approval['cuda_visible_device'],runtime_gpu_uuid=approval['gpu_uuid'])
 _,checked,_=functions(body,scope)
 records=[];files={}
 for e in scope['jobs']:
  if e['id'] not in wanted:continue
  result=checked(e,scope);assert result is not None and result['rounds']==70
  out=H/e['output'];job=read(H/e['job']);assert job['config']['seed']==91001
  assert job['config']['client_alpha']=={'IID':5000.0,'non-IID':5.0}[result['distribution']]
  for f in sorted(out.iterdir()):
   assert f.is_file() and not f.name.endswith('.tmp');files[str(f.relative_to(H))]=dict(sha256=digest(f),size=f.stat().st_size)
  files[e['job']]=dict(sha256=digest(H/e['job']),size=(H/e['job']).stat().st_size)
  records.append(dict(id=e['id'],rounds=result['rounds'],distribution=result['distribution'],attack=result['attack'],seed=result['seed'],alpha=result['alpha'],metrics=result['metrics'],evaluation_stats=result['evaluation_stats'],model_sha256=digest(out/'model.pt'),acceptance_sha256=digest(out/'acceptance.json'),undefined_sidecar_sha256=digest(out/'undefined_diagnostics.json'),original_provenance=read(out/'provenance.json')))
 body.verify_scope(scope)
 # The original queue uses one persistent producer and one shared stdout log; capture a labelled prefix, not a closed per-job log.
 log=Path('/var/log/guardfed_celeba_hybrid_screen32.log');(B/'service_log_prefix.txt').write_bytes(log.read_bytes())
 proof=dict(status='PARTIAL_ORIGINAL_STRICT_ACCEPTED_OFFSERVER_PENDING',previous_chain_sha256=digest(B/'PREVIOUS_CHAIN.json'),old_accepted=6,accepted_total=6+len(wanted),accepted_new=len(wanted),accepted_new_ids=wanted,planned=32,records=records,snapshot_sha256=digest(B/'live_snapshot.json'),source_seal_sha256=digest(H/'FILES_SHA256.json'),scope_sha256=digest(H/'screen_scope.json'),runtime_protocol_sha256=digest(H/'runtime_protocol.json'),driver_sha256=digest(H/'driver.py'),body_sha256=digest(H/'body.py'),writer_policy_sha256=digest(H/'writer_policy.py'),source_data_verified_before_after=True,source_host=os.uname().nodename,runtime=dict(python=sys.version,torch=torch.__version__,cuda_build=torch.version.cuda,CPU=sorted(os.sched_getaffinity(0)),threads=torch.get_num_threads(),gpu_name_for_original_checker=torch.cuda.get_device_name(0),cuda_context_initialized_for_original_device_metadata=torch.cuda.is_initialized()),no_CNN_training_or_test=True,no_selection=True,log_semantics='Shared still-running producer log prefix only. Each accepted result bytes hashed before/after archive; no active result included.',collector_sha256=digest(__file__))
 save('PARTIAL_ACCEPTANCE.json',proof)
 for name in ['collect_once.py','live_snapshot.json','PARTIAL_ACCEPTANCE.json','service_log_prefix.txt','PREVIOUS_CHAIN.json','PREVIOUS_LATEST.json','AUTHORIZED_SNAPSHOT.json','EXACT_DELTA.json','COLLECTOR_DIFF.patch','SOURCE_RECEIPT.json']:
  files['accepted_delta_after6_20261009/'+name]=dict(sha256=digest(B/name),size=(B/name).stat().st_size)
 save('MEMBERS.json',{'members':files})
 archive=B/'hybrid_after6_delta.tar.gz'
 with tarfile.open(archive,'x:gz') as t:
  for name in files:t.add(H/name,arcname=name,recursive=False)
  t.add(B/'MEMBERS.json',arcname='MEMBERS.json',recursive=False)
 for name,row in files.items():assert digest(H/name)==row['sha256'] and (H/name).stat().st_size==row['size'],'Changed terminal bytes'
 with tarfile.open(archive) as t:
  assert len(t.getnames())==len(set(t.getnames())) and set(t.getnames())==set(files)|{'MEMBERS.json'}
  for name,row in files.items():assert hashlib.sha256(t.extractfile(name).read()).hexdigest()==row['sha256'] and t.getmember(name).size==row['size']
 save('BACKUP_SHA256.json',dict(archive_sha256=digest(archive),archive_size=archive.stat().st_size,inventory_sha256=digest(B/'MEMBERS.json'),acceptance_sha256=digest(B/'PARTIAL_ACCEPTANCE.json'),member_count=len(files)+1,accepted_new_ids=wanted,source_seal_sha256=digest(H/'FILES_SHA256.json'),source_not_repackaged=True,original_strict_server_PASS=True))
 print(json.dumps(read(B/'BACKUP_SHA256.json')))
if __name__=='__main__':
 try:main()
 except BaseException as e:
  save('FAILURE.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False));raise
