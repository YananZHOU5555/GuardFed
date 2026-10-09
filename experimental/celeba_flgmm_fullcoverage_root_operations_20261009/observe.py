"""One original-resource-style read-only snapshot; no acceptance or monitor installation."""
from pathlib import Path
import datetime,json,subprocess,sys
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
REMOTE=r'''
from pathlib import Path
from collections import deque
import datetime,hashlib,json,os,re,shutil,subprocess
BASE=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009');STAGE=BASE/'stage'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(Path('/etc/vast-agents-guide.md'))=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
def cmd(argv):
 r=subprocess.run(argv,capture_output=True,text=True,timeout=20);return dict(returncode=r.returncode,stdout=r.stdout,stderr=r.stderr)
processes=[]
for path in Path('/proc').glob('[0-9]*/cmdline'):
 try:
  argv=[x for x in path.read_bytes().decode(errors='replace').split('\0') if x]
  if argv and any(str(BASE) in x for x in argv):processes.append(dict(pid=int(path.parent.name),argv=argv,affinity=sorted(os.sched_getaffinity(int(path.parent.name)))))
 except (FileNotFoundError,ProcessLookupError,PermissionError):pass
rows=[];hashes={};source_ok=None;failures=[];errors=[]
package=STAGE/'PACKAGE_SHA256.json'
if package.exists():
 hashes['PACKAGE_SHA256.json']=sha(package);seal=read(package)
 changed=[n for n,h in seal['files'].items() if not (STAGE/n).is_file() or sha(STAGE/n)!=h];source_ok=not changed
 manifest=read(STAGE/'manifest.json');hashes['manifest.json']=sha(STAGE/'manifest.json');hashes['source/protocol.json']=sha(STAGE/'source/protocol.json')
 for kind,items,folder in [('new',manifest['jobs'],'runs'),('canary',manifest['preflight_jobs'],'preflight/runs')]:
  for item in items:
   out=STAGE/folder/item['id'];job=STAGE/'jobs'/item['job']
   rows.append(dict(id=item['id'],kind=kind,job_sha256=sha(job),job_matches=sha(job)==item['job_sha256'],
       progress=read(out/'progress.json') if (out/'progress.json').exists() else None,
       result_present=(out/'result.json').is_file(),acceptance_present=(out/'acceptance.json').is_file()))
 for item in manifest['reused_jobs']:
  if item['distribution']=='non-IID':
   out=STAGE/'preflight/references'/item['id'];rows.append(dict(id=item['id'],kind='reference_canary',progress=read(out/'progress.json') if (out/'progress.json').exists() else None,result_present=(out/'result.json').is_file()))
 failures=[str(p.relative_to(STAGE)) for p in STAGE.rglob('*.json') if 'failure' in p.name.lower()]
 pattern=re.compile(r'Traceback|RuntimeError|CUDA error|out of memory|OutOfMemory|\bnan\b|\binf\b|fatal',re.I)
 for path in STAGE.rglob('*.log'):
  with path.open(errors='replace') as stream:
   errors.extend(dict(log=str(path.relative_to(STAGE)),line=line.strip()[:1000]) for line in deque(stream,maxlen=100) if pattern.search(line))
else:changed=[]
main=cmd(['supervisorctl','status','guardfed_celeba_mechanism_formal'])
snapshot=dict(status='READONLY_OBSERVATION_NOT_SCIENTIFIC_ACCEPTANCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 stage_state='NOT_BOUND' if not package.exists() else 'NOT_STARTED' if not processes and not any(r.get('progress') or r.get('result_present') for r in rows) else 'OBSERVE_ROWS_AND_PROCESSES',
 source_hashes=hashes,source_members_match=source_ok,changed_source_members=changed,processes=processes,rows=rows,
 queue=read(STAGE/'queue_progress.json') if (STAGE/'queue_progress.json').exists() else None,
 failure_paths=failures,recent_log_error_matches=errors,
 canary_service=cmd(['supervisorctl','status','guardfed_celeba_flgmm_fullcoverage_canary']),
 coverage_service=cmd(['supervisorctl','status','guardfed_celeba_flgmm_fullcoverage']),main_service=main,
 old_flgmm_service=cmd(['supervisorctl','status','guardfed_celeba_flgmm_screen']),
 cpu_max=Path('/sys/fs/cgroup/cpu.max').read_text(),cpu_stat=Path('/sys/fs/cgroup/cpu.stat').read_text(),
 memory_current=Path('/sys/fs/cgroup/memory.current').read_text(),memory_max=Path('/sys/fs/cgroup/memory.max').read_text(),
 memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),disk_free_bytes=shutil.disk_usage(BASE if BASE.exists() else BASE.parent).free,
 gpu=cmd(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader']),
 scientific_acceptance_performed=False,no_remote_writes=True)
print(json.dumps(snapshot))
'''

def main():
 here=Path(__file__).resolve().parent;out=here/('observation_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ'));out.mkdir()
 try:
  result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -'],input=REMOTE.encode(),capture_output=True,timeout=75)
 except subprocess.TimeoutExpired as e:
  (out/'TIMEOUT.json').write_text(json.dumps(dict(error=repr(e),automatic_retry=False,stdout=(e.stdout or b'').decode(errors='replace'),stderr=(e.stderr or b'').decode(errors='replace')),indent=2));raise
 (out/'STDOUT.txt').write_bytes(result.stdout);(out/'STDERR.txt').write_bytes(result.stderr)
 (out/'RETURN_CODE.json').write_text(json.dumps(dict(returncode=result.returncode)))
 result.check_returncode();value=json.loads(result.stdout)
 (out/'SNAPSHOT.json').write_text(json.dumps(value,indent=2)+'\n');print(json.dumps(dict(path=str(out/'SNAPSHOT.json'),stage_state=value['stage_state'])))

if __name__=='__main__':main()
