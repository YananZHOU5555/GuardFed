"""Appended to the observer function definitions; resource/stat refresh only."""
assert PAYLOAD is not None
base=PAYLOAD['previous_observation']; started=utc()
assert sha('/etc/vast-agents-guide.md')==PAYLOAD['guide_sha256']
eligible=sorted(os.sched_getaffinity(0));os.sched_setaffinity(0,{eligible[-1]})
identities=[]
for old in base['hashes']:
    p=P(old['path'])
    try:
        st=p.stat();same=(st.st_size==old['bytes'] and st.st_mtime_ns==old['mtime_ns'] and str(p.resolve())==old['resolved'])
        actual=sha(p) if st.st_size<=16*1024**2 else None
        identities.append({'path':str(p),'stat_identity_unchanged':same,'small_file_sha256':actual,'pass':same and (actual is None or actual==old['sha256'])})
    except OSError as e:identities.append({'path':str(p),'pass':False,'error':str(e)})
deployed=P('/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010/source')
package={'path':str(deployed),'seal_sha256':sha(deployed/'FILES_SHA256.json'),'members':[]}
for rel,pin in read(deployed/'FILES_SHA256.json')['files'].items():
    q=deployed/rel;package['members'].append({'path':rel,'pass':q.stat().st_size==pin['bytes'] and sha(q)==pin['sha256']})
sv=['guardfed_celeba_mechanism_formal','guardfed_celeba_flgmm_fullcoverage','guardfed_celeba_gradient_screen64_v2a','guardfed_celeba_hybrid_fullcoverage','guardfed_celeba_mechanism_remaining620_valid_v2a']
services=cmd(['supervisorctl','status',*sv]);before,_=processes();time.sleep(2);after,narrow=processes();old={(r['pid'],t['tid']):t for r in before for t in r['threads']}
active_wide=[];process_summary=[]
for r in after:
    groups={};active=[]
    for t in r['threads']:
        mask=tuple(t['affinity']);groups[mask]=groups.get(mask,0)+1
        prev=old.get((r['pid'],t['tid']));dt=t['cpu_ticks']-prev['cpu_ticks'] if prev else None
        if dt and dt>0:
            item={'pid':r['pid'],'tid':t['tid'],'affinity_n':len(mask),'last_cpu':t['last_cpu'],'delta_cpu_ticks_2s':dt};active.append(item)
            if len(mask)>8:active_wide.append(item)
    process_summary.append({'pid':r['pid'],'ppid':r['ppid'],'argv':r['argv'],'env':r['env'],'os_threads':len(r['threads']),'affinity_groups':[{'n_cpus':len(mask),'min':min(mask),'max':max(mask),'threads':n} for mask,n in groups.items()],'active_threads':active})
selected=[]
for row in PAYLOAD['manifest']['records']:
    out=P(row['runtime_output']);selected.append({'id':row['id'],'failures':[str(p) for p in out.glob('failure*.json')]+([str(out/'FAILED.json')] if (out/'FAILED.json').exists() else []),'live_producers':[r['pid'] for r in after if any(row['id'] in a for a in r['argv'])]})
cg={n:(P('/sys/fs/cgroup')/n).read_text().strip() for n in ['cpu.max','cpu.stat','memory.current','memory.max','memory.events','cpuset.cpus.effective']}
gpu=cmd(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu','--format=csv,noheader']);recovery=cmd(['nvidia-smi','--query-gpu=index,gpu_recovery_action','--format=csv,noheader'])
disk=shutil.disk_usage('/workspace');queue=read('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1/formal_queue_progress.json')
duplicate=[r['pid'] for r in after if any('celeba_added_cnn_three_view_gate_20261010/source/candidate.py' in a for a in r['argv'])]
result={'status':'FRESH_READONLY_RESOURCE_STAT_OBSERVATION','started_utc':started,'utc':utc(),'guide_sha256':PAYLOAD['guide_sha256'],'eligible_cpus':eligible,'identities':identities,'deployed_package':package,'services':services,'processes':process_summary,'narrow_conflicts':narrow,'active_wide_threads':active_wide,'selected':selected,'duplicate_gate':duplicate,'cgroup':cg,'gpu':gpu,'gpu_recovery':recovery,'disk':{'total':disk.total,'free':disk.free},'main_queue_counts':{'completed':len(queue['completed']),'failed':len(queue['failed']),'active':len(queue['active'])},'no_Torch':True,'no_CNN_fit_training':True,'no_remote_writes':True}
print(json.dumps(result))
