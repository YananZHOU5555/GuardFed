"""Root-authorized future fixed2 export. Hash/metadata only; no pickle, NumPy or Torch."""
from pathlib import Path
import json,hashlib,os,subprocess,tarfile,datetime
import metadata_gate as g
def main(x,refs):
    g.input_gate(x);g.need(x['root_authorized_transport'] is True,'No actual transport authority')
    g.need(g.sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa','Changed guide')
    g.need(os.sched_getaffinity(0)=={110},'Original transport CPU110 required')
    for proc in Path('/proc').glob('[0-9]*'):
        if int(proc.name)==os.getpid():continue
        try:
            for task in (proc/'task').iterdir():
                try:aff=os.sched_getaffinity(int(task.name))
                except ProcessLookupError:continue
                g.need(not (len(aff)<=16 and 110 in aff),'CPU110 occupied')
            argv=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
            if x['server_base']+'/worker.py' in argv:g.need(False,'LoGo producer still live')
        except (FileNotFoundError,ProcessLookupError):continue
    files={};paths={};protected={}
    def register(pin,name):
        p=Path(pin['path']);g.need(p.is_absolute() and '..' not in p.parts and p.resolve()==p and not p.is_symlink() and p.is_file() and g.sha(p)==pin['sha256'],'Changed live pin: '+name)
        if str(p) in paths:return p
        g.need(name not in files or files[name]==p,'Archive member collision');files[name]=p;paths[str(p)]=name;return p
    register(x['root_queue_auth'],'base/ROOT_QUEUE_AUTH.json');register(x['deployment_seal'],'base/DEPLOYMENT_SEAL.json')
    auth=g.read(files['base/ROOT_QUEUE_AUTH.json']);B=Path(x['server_base'])
    dep=g.read(files['base/DEPLOYMENT_SEAL.json'])
    for rel,item in dep['files'].items():
        p=B/rel;g.need(p.resolve().is_relative_to(B.resolve()) and g.sha(p)==item['sha256'],'Changed deployment source');protected[str(p)]=item['sha256']
    g.need(not (B/'canary/QUEUE_FAILURE.json').exists(),'Failed canary queue')
    complete=g.read(B/'canary/COMPLETE.json');g.need(complete['ids']==g.IDS and complete['server_saved_accepted'] is True,'Incomplete fixed2 Linux canary')
    register(g.pin(B/'canary/COMPLETE.json'),'base/CANARY_COMPLETE.json')
    service=subprocess.run(['supervisorctl','status',x['service']],capture_output=True,text=True,timeout=30)
    g.need(service.returncode in (0,3) and len(service.stdout.split())>=2 and service.stdout.split()[0]==x['service'] and service.stdout.split()[1]=='EXITED','Canary service not normally exited')
    g.need([r['post_id'] for r in refs['records']]==g.IDS,'Changed fixed2 references')
    for row in refs['records']:
        i=row['post_id'];out=B/'runs'/i;task=B/'tasks'/i
        g.need(not out.with_name(out.name+'.formal_failure.json').exists() and not list(out.glob('failure*.json')),'Preserved failed task')
        register(x['task_accepted'][i],'tasks/'+i+'/ACCEPTED.json')
        def load(pin):
            p=Path(pin['path'])
            if str(p).startswith(str(task)+'/'):name='tasks/'+i+'/'+p.name
            elif str(p).startswith(str(out)+'/'):name='runs/'+i+'/'+p.name
            elif pin==x['root_queue_auth']:name='base/ROOT_QUEUE_AUTH.json'
            else:name='original_jobs/'+i+'.json'
            return g.read(register(pin,name))
        v=g.verify_record(x,row,load);register(v['scores'],'runs/'+i+'/scores_predictions.npz')
        for role,pin in v['approval']['original_artifacts'].items():
            g.need(g.sha(pin['path'])==row['original_artifacts'][role]['sha256'],'Original restored artifact changed');protected[pin['path']]=pin['sha256']
        bridge=Path(v['approval']['runtime_dependencies']['bridge']['path'])
        for name,h in v['job']['local_hashes'].items():register({'path':str(bridge.parent/name),'sha256':h},'source/'+i+'/'+name)
        for role,pin in v['approval']['runtime_dependencies'].items():
            # Original source-only dependencies; not models/caches.
            register(pin,'source/dependencies/'+pin['sha256']+'/'+Path(pin['path']).name)
        register(v['approval']['adapter'],'source/adapter.py')
        register(g.pin(task/'worker_command.json'),'tasks/'+i+'/worker_command.json');g.need(g.read(task/'worker_command.json')['exit_code']==0,'Worker did not exit normally')
        # Hash frozen output files without copying them; F original hashes will be rechecked offline.
        for role in ['post_state.pkl','mapping.npz','mapping_metadata.json']:
            g.need(g.sha(out/role)==row['original_artifacts'][role]['sha256'],'Frozen output differs');protected[str(out/role)]=row['original_artifacts'][role]['sha256']
    for p in [x['FedAvg_CUDA_root'],x['FedAvg_CUDA_index']]:
        g.need(g.sha(p['path'])==p['sha256'],'Adopted CUDA proof changed');protected[p['path']]=p['sha256']
    export=B/'exports/LoGo_fixed2_saved_attempt001';g.need(not export.exists(),'No overwrite or retry');export.mkdir(parents=True)
    allowed_npz={'runs/'+i+'/scores_predictions.npz' for i in g.IDS}
    g.need(all(not n.endswith(('.pkl','.pt','.pth','.npy')) and (not n.endswith('.npz') or n in allowed_npz) for n in files),'Forbidden bulk payload')
    # Original transport archive construction body, unchanged apart from fixed archive filename.
    sha=g.sha
    members={n:{'sha256':sha(p),'bytes':p.stat().st_size} for n,p in files.items()}
    index=export/'ROOT_MEMBERS.json';assert not index.exists();index.write_text(json.dumps({'files':members,'ids':g.IDS,'server_paths':paths},indent=2)+chr(10))
    archive=export/'LOGO_FIXED2.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz') as tar:
        for n,p in files.items():tar.add(p,arcname=n,recursive=False)
        tar.add(index,arcname='ROOT_MEMBERS.json',recursive=False)
    assert all(sha(p)==members[n]['sha256'] for n,p in files.items())
    g.need(all(sha(p)==s for p,s in protected.items()),'Protected source/frozen artifact changed during export')
    print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),archive_path=str(archive),archive_sha256=sha(archive),archive_bytes=archive.stat().st_size,archive_members=len(files)+1,payload_members=len(files),member_index_sha256=sha(index),ids=g.IDS,service=service.stdout.strip(),no_pickle_download=True)))
