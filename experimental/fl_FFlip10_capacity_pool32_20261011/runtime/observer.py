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
def cpu_set(text):
    result=set()
    for part in text.strip().split(','):
        ends=part.split('-'); lo=int(ends[0]); hi=int(ends[-1])
        assert len(ends) in (1,2) and 0<=lo<=hi
        result.update(range(lo,hi+1))
    return result

def capacity_snapshot(monitored):
    rows={}
    raw=P('/proc/stat').read_text()
    for line in raw.splitlines():
        fields=line.split()
        if fields and fields[0].startswith('cpu') and fields[0][3:].isdigit():
            cpu=int(fields[0][3:])
            if cpu in monitored:
                assert len(fields)>=9
                rows[str(cpu)]=[int(x) for x in fields[1:9]]
    assert set(map(int,rows))==set(monitored)
    cpu_stat=P('/sys/fs/cgroup/cpu.stat').read_text().strip()
    values={k:int(v) for k,v in (line.split() for line in cpu_stat.splitlines())}
    return dict(monotonic=time.monotonic(),proc_stat=rows,cgroup_cpu_stat=cpu_stat,usage_usec=values['usage_usec'])

def capacity_metrics(start,end,targets,siblings):
    assert targets==list(range(32,64)) and siblings==list(range(288,320))
    monitored=sorted(targets+siblings)
    assert set(start['proc_stat'])==set(end['proc_stat'])==set(map(str,monitored))
    seconds=end['monotonic']-start['monotonic']; assert 2.0<=seconds<=10.0
    usage=end['usage_usec']-start['usage_usec']; assert usage>=0
    rows=[]
    for cpu in monitored:
        before=start['proc_stat'][str(cpu)];after=end['proc_stat'][str(cpu)]
        assert len(before)==len(after)==8
        delta=[b-a for a,b in zip(before,after)]
        assert min(delta)>=0 and sum(delta)>0
        total=sum(delta); busy=total-delta[3]
        rows.append(dict(cpu=cpu,total_ticks=total,busy_ticks=busy,busy_fraction=busy/total))
    by_cpu={row['cpu']:row for row in rows}
    pairs=[dict(target=cpu,sibling=cpu+256,target_busy=by_cpu[cpu]['busy_fraction'],sibling_busy=by_cpu[cpu+256]['busy_fraction'],quiet=by_cpu[cpu]['busy_fraction']<=.20 and by_cpu[cpu+256]['busy_fraction']<=.20) for cpu in targets]
    return dict(interval_seconds=seconds,targets=targets,siblings=siblings,per_cpu=rows,physical_pairs=pairs,quiet_pairs=sum(pair['quiet'] for pair in pairs),
                max_busy_fraction=max(x['busy_fraction'] for x in rows),
                cgroup_used_cores=usage/(seconds*1e6),
                definition='first8 proc/stat fields only; user/nice already include guest, guest columns excluded; only idle is free, iowait and steal conservatively busy')

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
                    if len(affinity)<=8 and set(affinity)&(set(range(32,64))|set(range(288,320))): all_narrow.append({'pid':pid,'argv':argv,'thread':item})
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
for proc in P('/proc').glob('[0-9]*'):
 if int(proc.name)==os.getpid():continue
 for task in (proc/'task').glob('*'):
  try:
   af=set(os.sched_getaffinity(int(task.name)))
   assert not (len(af)<=8 and 110 in af),'Observer CPU110 narrowly reserved'
  except ProcessLookupError:pass
os.sched_setaffinity(0,{110})
m=PAYLOAD['manifest']; pins={}
for rel,h in m['runtime_repo_hashes'].items(): pins[str(P(m['server_repo'])/rel)]={'sha256':h}
for row in m['records']:
    for pin in row['runtime_artifacts'].values():
        name=pin['server_path']; v={'sha256':pin['sha256'],'bytes':pin['bytes']}
        assert name not in pins or pins[name]['sha256']==v['sha256'];pins[name]=v
hashes=[];seen={}
for path,pin in pins.items():
    try:
        p=P(path);st=p.stat();identity=(st.st_dev,st.st_ino,st.st_size,st.st_mtime_ns)
        actual=seen.get(identity)
        if actual is None: actual=sha(p);seen[identity]=actual
        hashes.append({'path':path,'resolved':str(p.resolve()),'bytes':st.st_size,'mtime_ns':st.st_mtime_ns,'sha256':actual,'expected_sha256':pin['sha256'],'match':actual==pin['sha256'] and ('bytes' not in pin or pin['bytes']==st.st_size)})
    except OSError as e: hashes.append({'path':path,'match':False,'error':str(e)})
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
targets=list(range(32,64)); siblings=list(range(288,320)); sibling_map={}; core_topology={}
assert set(targets+siblings)<=set(eligible), 'Fixed pool or SMT sibling outside actual eligible mask'
for cpu in targets+siblings:
    topology=P('/sys/devices/system/cpu')/('cpu'+str(cpu))/'topology'
    pair=cpu_set((topology/'thread_siblings_list').read_text())
    target=cpu if cpu in targets else cpu-256
    assert pair=={target,target+256}, 'Fixed pool SMT topology drift'
    core_topology[str(cpu)]={'socket':int((topology/'physical_package_id').read_text()),'core':int((topology/'core_id').read_text())}
    if cpu in targets:sibling_map[str(cpu)]=sorted(pair)
assert len({core_topology[str(cpu)]['socket'] for cpu in targets+siblings})==1, 'Pool not one socket'
assert len({core_topology[str(cpu)]['core'] for cpu in targets})==32, 'Pool reuses physical cores'
assert all(core_topology[str(cpu)]==core_topology[str(cpu+256)] for cpu in targets), 'SMT physical identity drift'
before,narrow=processes()
capacity_start=capacity_snapshot(targets+siblings); time.sleep(2); capacity_end=capacity_snapshot(targets+siblings)
capacity=capacity_metrics(capacity_start,capacity_end,targets,siblings)
after,narrow_after=processes(); old={(r['pid'],t['tid']):t for r in before for t in r['threads']}
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
duplicate=[r for r in after if any(any(marker+'/source/candidate.py' in a for marker in ('fl_three_view_FFlip10_20261011','fl_three_view_FFlip10_cpu136_20261011','fl_FFlip10_capacity_guard_v2_20261011','fl_FFlip10_capacity_cpu11_20261011','fl_FFlip10_capacity_pool32_20261011')) for a in r['argv'])]
lock_lines=[x for x in P('/proc/locks').read_text().splitlines() if 'guardfed' in x]
result={'status':'READONLY_OBSERVATION_NOT_PREFLIGHT_DECISION','started_utc':started,'hash_end_utc':hash_end,'utc':utc(),'guide_sha256':guide,'eligible_cpus':eligible,'observer_affinity':sorted(os.sched_getaffinity(0)),'observer_nice':os.getpriority(os.PRIO_PROCESS,0),'hashes':hashes,'unique_inodes_hashed':len(seen),'source_model_data_hashes_verified':all(x['match'] for x in hashes),'services':services,'supervisor_configs':configs,'gpu':gpu,'gpu_health':gpu_health,'gpu_recovery':gpu_recovery,'cgroup':cg,'disk':dict(total=disk.total,used=disk.used,free=disk.free),'guardfed_processes':after,'capacity_start':capacity_start,'capacity_end':capacity_end,'capacity_metrics':capacity,'sibling_topology':sibling_map,'core_topology':core_topology,'pool_topology_verified':True,'narrow_pool32_and_siblings_conflicts':narrow+narrow_after,'selected':selected,'duplicate_gate':duplicate,'main_queue':queue,'main_workers':workers,'no_remote_writes':True,'Torch_imported':False,'CNN':0,'fit':0,'training':0}
print(json.dumps(result,ensure_ascii=False))
