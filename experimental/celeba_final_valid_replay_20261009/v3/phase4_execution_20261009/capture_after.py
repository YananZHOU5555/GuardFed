import json,os,subprocess,time
from pathlib import Path
phase=4
label='after'
base=Path('/workspace/guardfed_checks/celeba_final_valid_replay_20261009/v3')
root=Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1')
out=base/('phase'+str(phase)+'_execution_20261009')/('impact_'+label+'.json')
q=json.loads((root/'formal_queue_progress.json').read_text())
r={'unix':time.time(),'training_queue':q,'active_rounds':[],'replay_processes':[]}
for a in q.get('active',[]):
    p=root/'runs'/a['id']/'progress.json'
    r['active_rounds'].append(dict(a,progress=json.loads(p.read_text()) if p.exists() else None))
for p in Path('/proc').glob('[0-9]*/cmdline'):
    try:
        command=p.read_bytes().replace(b'\x00',b' ').decode()
        if '/v3/replay_v3.py worker ' not in command and '/v3/replay_v3.py run ' not in command:continue
        if ('phase'+str(phase)+'_useful') not in command:continue
        pid=int(p.parent.name)
        status={a:b.strip() for line in (p.parent/'status').read_text().splitlines() if ':' in line for a,b in [line.split(':',1)]}
        r['replay_processes'].append({'pid':pid,'command':command,'nice':os.getpriority(os.PRIO_PROCESS,pid),'threads':status['Threads'],'rss':status['VmRSS'],'thread_affinities':{t.name:sorted(os.sched_getaffinity(int(t.name))) for t in (p.parent/'task').glob('*')}})
    except (ProcessLookupError,FileNotFoundError):pass
for key,cmd in [('service',['supervisorctl','status','guardfed_celeba_mechanism_formal','guardfed_celeba_valid_phase'+str(phase)+'_20261009']),('gpu',['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'])]:
    p=subprocess.run(cmd,capture_output=True,text=True,timeout=15)
    r[key]={'returncode':p.returncode,'stdout':p.stdout,'stderr':p.stderr}
assert not out.exists()
out.write_text(json.dumps(r,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'phase':phase,'label':label,'completed_n':len(q['completed']),'failed_n':len(q['failed']),'active_rounds':[{ 'id':x['id'],'round':None if x['progress'] is None else x['progress']['round']} for x in r['active_rounds']],'replay_processes':[{k:p[k] for k in ('pid','nice','threads','rss')} for p in r['replay_processes']],'gpu':r['gpu']},indent=2))
