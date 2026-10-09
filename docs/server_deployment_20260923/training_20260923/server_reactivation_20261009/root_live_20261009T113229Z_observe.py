from pathlib import Path
import datetime,hashlib,json,os,shutil,subprocess,sys,time
PREFIX='root_live_20261009T113229Z'
BASE=Path('/workspace/guardfed_checks/server_reactivation_20261009')
MAIN=Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1')
FL=Path('/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2')
HY=Path('/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009')
read=lambda p:json.loads(p.read_text())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
os.sched_setaffinity(0,[sorted(os.sched_getaffinity(0))[16]]);os.nice(10)
subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True,capture_output=True)
def proc(pid):
 p=Path('/proc')/str(pid)
 if not p.exists():return {'exists':False}
 try:
  stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  return {'exists':True,'start_ticks':int(stat[19]),'cpu_ticks':int(stat[11])+int(stat[12]),'affinity':sorted(os.sched_getaffinity(pid)),'nice':os.getpriority(os.PRIO_PROCESS,pid),'command':(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')}
 except (FileNotFoundError,ProcessLookupError):return {'exists':False}
def row(item,parent):
 p=parent/item['id']/'progress.json';x=read(p) if p.exists() else {}
 return {'id':item['id'],'pid':item['pid'],'round':x.get('round'),'progress_updated_unix':x.get('updated_unix',x.get('time')),'progress_sha256':sha(p) if p.exists() else None,'process':proc(item['pid'])}
mq=read(MAIN/'formal_queue_progress.json');fq=read(FL/'queue_progress.json');hs=read(HY/'screen_scope.json')
hcompleted=[];hactive=[]
for entry in hs['jobs']:
 out=HY/entry['output'];progress=out/'progress.json'
 if (out/'acceptance.json').is_file():hcompleted.append(entry['id'])
 elif progress.is_file():
  x=read(progress);pid=x.get('pid')
  if pid is not None and proc(pid)['exists']:hactive.append(row({'id':entry['id'],'pid':pid},HY/hs['run_parent']))
cpu={k:int(v) for k,v in (line.split() for line in Path('/sys/fs/cgroup/cpu.stat').read_text().splitlines())}
memory={k:Path('/sys/fs/cgroup'/Path(k)).read_text().strip() for k in ['memory.current','memory.max','memory.events']}
disk=shutil.disk_usage('/workspace')
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'monotonic':time.monotonic(),'main800':{'completed_observed_ids':mq['completed'],'completed_observed_n':len(mq['completed']),'pending':mq['pending'],'failed':mq['failed'],'active':[row(x,MAIN/'runs') for x in mq['active']],'queue_sha256':sha(MAIN/'formal_queue_progress.json'),'failure_files':[str(p) for p in (MAIN/'runs').glob('*/failure*.json')]},'FLGMM32':{'completed_observed_n':fq['completed'],'pending':fq['pending'],'active':[row(x,FL/'runs') for x in fq['active']],'failure_files':[str(p) for p in FL.glob('QUEUE_FAILURE*.json')]+[str(p) for p in (FL/'runs').glob('*/failure*.json')],'queue_sha256':sha(FL/'queue_progress.json')},'Hybrid32':{'acceptance_file_n_not_revalidated':len(hcompleted),'acceptance_file_ids':hcompleted,'active':hactive,'failure_files':[str(p) for p in HY.glob('screen_failure.json')]+[str(p) for p in (HY/'screen_runs').glob('*/failure*.json')],'scope_sha256':sha(HY/'screen_scope.json')},'services':subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_formal','guardfed_celeba_flgmm_screen','guardfed_celeba_hybrid_screen32','sglang'],capture_output=True,text=True).stdout.strip(),'cpu_max':Path('/sys/fs/cgroup/cpu.max').read_text().strip(),'cpu_stat':cpu,'memory':memory,'disk':{'total':disk.total,'used':disk.used,'free':disk.free},'gpu':subprocess.run(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout.strip(),'guide_sha256':sha(Path('/etc/vast-agents-guide.md')),'source_sha256':sha(Path(__file__)),'baseline872_touched':False,'canonical_latest_or_state_written':False,'sample_is_scientific_acceptance':False}
which=sys.argv[1];target=BASE/(PREFIX+'_'+which+'.json');assert which in ('before','after') and not target.exists()
if which=='after':
 before=read(BASE/(PREFIX+'_before.json'));dt=r['monotonic']-before['monotonic'];r['sample_seconds']=dt;r['effective_cgroup_cpu_cores']=(cpu['usage_usec']-before['cpu_stat']['usage_usec'])/1e6/dt;r['throttled_usec_delta']=cpu['throttled_usec']-before['cpu_stat']['throttled_usec'];r['round_changes']={}
 for role in ('main800','FLGMM32','Hybrid32'):
  prior={x['id']:x for x in before[role]['active']}
  r['round_changes'][role]=[{'id':x['id'],'before_round':prior[x['id']]['round'],'after_round':x['round'],'pid_unchanged':x['pid']==prior[x['id']]['pid'],'cpu_ticks_delta':x['process'].get('cpu_ticks',0)-prior[x['id']]['process'].get('cpu_ticks',0)} for x in r[role]['active'] if x['id'] in prior]
target.write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps({'receipt':str(target),'sha256':sha(target),'services':r['services'],'main_completed':r['main800']['completed_observed_n'],'main_rounds':[x['round'] for x in r['main800']['active']],'FL_completed':r['FLGMM32']['completed_observed_n'],'FL_rounds':[x['round'] for x in r['FLGMM32']['active']],'Hybrid_acceptance_files':r['Hybrid32']['acceptance_file_n_not_revalidated'],'Hybrid_rounds':[x['round'] for x in r['Hybrid32']['active']],'effectiveCPU':r.get('effective_cgroup_cpu_cores'),'round_changes':r.get('round_changes'),'gpu':r['gpu']},indent=2))

