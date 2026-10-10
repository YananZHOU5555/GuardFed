assert __import__('hashlib').sha256(__import__('pathlib').Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
"""Read-only stdlib exact3 observer; payload supplied on stdin by local launcher."""
import datetime, hashlib, json, os, pathlib, shutil, subprocess, time

PAYLOAD = json.load(__import__('sys').stdin)
P = pathlib.Path
def utc(): return datetime.datetime.now(datetime.timezone.utc).isoformat()
def read(p): return json.loads(P(p).read_text())
def cmd(a):
    q = subprocess.run(a, capture_output=True, text=True, timeout=30)
    return {'argv':a, 'returncode':q.returncode, 'stdout':q.stdout, 'stderr':q.stderr}
def sha(p):
    h=hashlib.sha256()
    with P(p).open('rb') as f:
        for b in iter(lambda:f.read(8*1024**2),b''): h.update(b)
    return h.hexdigest()
def processes():
    rows=[]; all_narrow=[]
    for p in P('/proc').glob('[0-9]*'):
        pid=int(p.name)
        if pid==os.getpid(): continue
        try:
            argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            threads=[]
            for t in (p/'task').glob('*'):
                try:
                    affinity=sorted(os.sched_getaffinity(int(t.name)))
                    st=(t/'stat').read_text().rsplit(')',1)[1].split()
                    item={'tid':int(t.name),'affinity':affinity,'state':st[0],'cpu_ticks':int(st[11])+int(st[12]),'last_cpu':int(st[36])}
                    threads.append(item)
                    if len(affinity)<=8 and set(affinity)&set(range(120,128)): all_narrow.append({'pid':pid,'argv':argv,'thread':item})
                except (OSError,ValueError): pass
            if argv and any('guardfed' in a.lower() or 'celeba' in a.lower() for a in argv):
                env={}
                for e in (p/'environ').read_bytes().split(b'\0'):
                    key,_,val=e.partition(b'=')
                    if key.decode(errors='replace') in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS','CUDA_VISIBLE_DEVICES']:
                        env[key.decode()]=val.decode(errors='replace')
                rows.append({'pid':pid,'ppid':int((p/'stat').read_text().rsplit(')',1)[1].split()[1]),'argv':argv,'threads':threads,'env':env})
        except OSError: pass
    return rows,all_narrow

assert PAYLOAD is not None
started=utc(); eligible=sorted(os.sched_getaffinity(0))
guide=sha('/etc/vast-agents-guide.md'); assert guide==PAYLOAD['guide_sha256']
os.sched_setaffinity(0,{110})
m=PAYLOAD['manifest']; pins={}
for rel,h in m['runtime_repo_hashes'].items(): pins[str(P(m['server_repo'])/rel)]={'sha256':h}
for row in m['records']:
    for pin in row['runtime_artifacts'].values():
        name=pin['server_path']; v={'sha256':pin['sha256'],'bytes':pin['bytes']}
        assert name not in pins or pins[name]['sha256']==v['sha256'];pins[name]=v
hashes=[];seen={}
for previous in PAYLOAD['previous_hashes']:
 p=P(previous['path']); st=p.stat()
 assert str(p.resolve())==previous['resolved'] and st.st_size==previous['bytes'] and st.st_mtime_ns==previous['mtime_ns'], 'Previously hashed input stat changed'
 hashes.append(dict(previous, stat_refreshed=True))
hash_end=utc()
sv=['guardfed_celeba_mechanism_formal','guardfed_celeba_flgmm_fullcoverage','guardfed_celeba_gradient_screen64_v2a','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_mechanism_remaining620_valid_v2a']
services=cmd(['supervisorctl','status',*sv])
configs={}
for f in P('/etc/supervisor/conf.d').glob('*'):
    try:
        t=f.read_text()
        if any('[program:'+s+']' in t for s in sv): configs[str(f)]=t
    except OSError:pass
gpu=cmd(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader'])
gpu_health=cmd(['nvidia-smi','-q','-d','PAGE_RETIREMENT,ROW_REMAPPER'])
gpu_recovery=cmd(['nvidia-smi','--query-gpu=index,gpu_recovery_action','--format=csv,noheader'])
cg={n:(P('/sys/fs/cgroup')/n).read_text().strip() for n in ['cpu.max','cpu.stat','memory.current','memory.max','memory.events','cpuset.cpus.effective']}
disk=shutil.disk_usage('/workspace')
before,narrow=processes(); time.sleep(2); after,narrow_after=processes(); old={(r['pid'],t['tid']):t for r in before for t in r['threads']}
for r in after:
    for t in r['threads']:
        prior=old.get((r['pid'],t['tid']));t['delta_cpu_ticks_2s']=t['cpu_ticks']-prior['cpu_ticks'] if prior else None
selected=[]
for row in m['records']:
    out=P(row['runtime_output']);selected.append({'id':row['id'],'output':str(out),'failures':[str(f) for f in out.glob('failure*.json')]+([str(out/'FAILED.json')] if (out/'FAILED.json').exists() else []),'live_producers':[r['pid'] for r in after if any(row['id'] in a for a in r['argv'])]})
queuepath=P(m['server_repo'])/'results/revision_20261009/celeba_mechanism_v1/formal_queue_progress.json'
queue=read(queuepath); workers=[]
for r in after:
    argv=r['argv']
    if '--job' in argv:
        jp=P(argv[argv.index('--job')+1])
        try:
            job=read(jp);out=P(job.get('output',''))
            if 'celeba_mechanism_v1' in str(jp):
                progress=read(out/'progress.json') if (out/'progress.json').exists() else None
                workers.append({'pid':r['pid'],'job_path':str(jp),'id':job['id'],'progress':progress})
        except (OSError,KeyError,ValueError): pass
duplicate=[r for r in after if any('fl_three_view_after48_20261011/source/candidate.py' in a for a in r['argv'])]
lock_lines=[x for x in P('/proc/locks').read_text().splitlines() if 'guardfed' in x]
result={'status':'READONLY_OBSERVATION_NOT_PREFLIGHT_DECISION','started_utc':started,'hash_end_utc':hash_end,'utc':utc(),'guide_sha256':guide,'eligible_cpus':eligible,'observer_affinity':sorted(os.sched_getaffinity(0)),'observer_nice':os.getpriority(os.PRIO_PROCESS,0),'hashes':hashes,'unique_inodes_hashed':len(seen),'source_model_data_hashes_verified':all(x['match'] for x in hashes),'services':services,'supervisor_configs':configs,'gpu':gpu,'gpu_health':gpu_health,'gpu_recovery':gpu_recovery,'cgroup':cg,'disk':dict(total=disk.total,used=disk.used,free=disk.free),'guardfed_processes':after,'narrow_cpu120_127_conflicts':narrow_after,'selected':selected,'duplicate_gate':duplicate,'main_queue':queue,'main_workers':workers,'no_remote_writes':True,'Torch_imported':False,'CNN':0,'fit':0,'training':0}
print(json.dumps(result,ensure_ascii=False))
