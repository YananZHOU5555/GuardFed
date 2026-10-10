"""One future root call of the running package's unchanged whole checker; no CNN."""
import argparse,contextlib,datetime,hashlib,io,json,os,pathlib,runpy,shutil,subprocess,sys
P=pathlib.Path
base=P('/workspace/guardfed_checks/celeba_hybrid_three_view_missing8_20261011')
PACKAGE='f3b197ce786a3f391f820baf8767fa25fba2d2387002acf654bced7dc8e95239'
GUIDE='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
CHECK='1410e344f07cb7d5987dcf773de4dab968bb3f03d21b378f59704e67074bd924'
sha=lambda p:hashlib.sha256(P(p).read_bytes()).hexdigest()
read=lambda p:json.loads(P(p).read_bytes())
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--gate-result-sha256',required=True)
ap.add_argument('--post-replay-preflight',required=True,type=P)
ap.add_argument('--post-replay-preflight-sha256',required=True)
ap.add_argument('--allow-original-cached-root-refit',required=True,action='store_true')
a=ap.parse_args();assert __debug__
for pin in (a.gate_result_sha256,a.post_replay_preflight_sha256):assert len(pin)==64 and all(x in '0123456789abcdef' for x in pin)
assert sha('/etc/vast-agents-guide.md')==GUIDE and sha(base/'source/FILES_SHA256.json')==PACKAGE and sha(base/'source/check_saved.py')==CHECK
assert a.post_replay_preflight.is_absolute() and a.post_replay_preflight.is_relative_to('/workspace/guardfed_checks') and sha(a.post_replay_preflight)==a.post_replay_preflight_sha256
pre=read(a.post_replay_preflight)
assert all(pre[k] is True for k in ('fl_evaluation_service_exited','cpu110_all_thread_free','hybrid_replay_service_exited','source_model_data_hashes_verified','gpu_health_verified','cgroup_and_memory_headroom_verified','storage_headroom_verified'))
assert pre['guide_sha256']==GUIDE and pre['package_sha256']==PACKAGE and pre['gate_result_sha256']==a.gate_result_sha256
age=(datetime.datetime.now(datetime.timezone.utc)-datetime.datetime.fromisoformat(pre['utc'])).total_seconds();assert 0<=age<=300
assert os.sched_getaffinity(0)=={110} and os.getpriority(os.PRIO_PROCESS,0)>=10 and os.environ['CUDA_VISIBLE_DEVICES']==''
assert 'idle' in subprocess.run(['ionice','-p',str(os.getpid())],capture_output=True,text=True,check=True).stdout
quota,period=P('/sys/fs/cgroup/cpu.max').read_text().split();assert quota=='max' or int(quota)/int(period)>=1
memory_max=P('/sys/fs/cgroup/memory.max').read_text().strip();memory_current=int(P('/sys/fs/cgroup/memory.current').read_text())
assert memory_max=='max' or memory_current+512*1024**2<int(memory_max)
assert shutil.disk_usage(base).free>1024**3
for name in ('guardfed_hybrid_missing8_valid','guardfed_flgmm_FFlip10_capacity_pool32_valid'):
 q=subprocess.run(['supervisorctl','status',name],capture_output=True,text=True)
 assert q.returncode==3 and q.stdout.split()[:2]==[name,'EXITED'],q.stdout
for proc in P('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 try:
  command=(proc/'cmdline').read_bytes().decode(errors='replace').split('\0')
  assert not any(P(token).name=='candidate.py' and any(ns in token for ns in ('celeba_hybrid_three_view_missing8_20261011','fl_FFlip10_capacity_pool32_20261011')) for token in command),'Live replay worker'
  assert not any(P(token).name=='check_saved.py' for token in command),'Existing saved checker'
 except (FileNotFoundError,ProcessLookupError):pass
 for task in (proc/'task').glob('*'):
  try:
   affinity=set(os.sched_getaffinity(int(task.name)));assert not (len(affinity)<=32 and 110 in affinity),'Restricted-thread CPU110 overlap'
  except ProcessLookupError:pass
output=base/'LINUX_SAVED_CHECK.json';gate=base/'outputs/attempt001'
assert not any(output.with_suffix(s).exists() for s in ('.json','.started.json','.failure.json'))
assert sha(gate/'GATE_RESULT.json')==a.gate_result_sha256 and not list(gate.rglob('FAILURE.json'))
sys.path.insert(0,str(base/'source'))
sys.argv=[str(base/'source/check_saved.py'),'--mode','linux-whole','--cpu','110','--allow-original-cached-root-refit','--package-sha256',PACKAGE,'--gate-result-sha256',a.gate_result_sha256,'--post-replay-preflight',str(a.post_replay_preflight),'--post-replay-preflight-sha256',a.post_replay_preflight_sha256,'--gate-dir',str(gate),'--output',str(output)]
captured=io.StringIO()
try:
 with contextlib.redirect_stdout(captured):runpy.run_path(str(base/'source/check_saved.py'),run_name='__main__')
finally:
 print(captured.getvalue(),file=sys.stderr,end='')
sys.stdout.buffer.write(output.read_bytes())
