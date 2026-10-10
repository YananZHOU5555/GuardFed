"""One original-strict closed gradient result, CPU111 archive only; no training."""
import configparser, hashlib, importlib.util, io, json, os, socket, subprocess, sys, tarfile, time, traceback
from pathlib import Path

ID='FedNGA_eta0.001_IID_Benign_seed91001_screen'
PKG=Path('/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010')
REPO=Path('/workspace/GuardFed-celeba-expanded')
RUNS=Path('/workspace/celeba_gradient_screen64_v2_results_20261010')
DEST=Path('/workspace/guardfed_checks/celeba_gradient64_first_closed_adoption_20261010')
SEAL='11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced'
GUIDE='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
VERIFIER=Path('/workspace/guardfed_checks/celeba_mechanism_evidence_20261009/evidence_v4.py')
VERIFIER_SHA='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_text())
def save(p,v):
    with Path(p).open('x') as f:json.dump(v,f,indent=2,allow_nan=False)
def require(v,m):
    if not v:raise ValueError(m)
def command(argv):
    p=subprocess.run(argv,capture_output=True,text=True);return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
def live():
    owners=[];workers=[];broad=0
    for p in Path('/proc').iterdir():
        if not p.name.isdigit() or int(p.name) in (os.getpid(),os.getppid()):continue
        try:
            if (p/'stat').read_text().rsplit(')',1)[1].split()[0] in ('Z','X'):continue
            argv=[x.decode(errors='replace') for x in (p/'cmdline').read_bytes().split(b'\0') if x]
            if str(PKG/'run_queue.py') in argv:workers.append(dict(pid=int(p.name),argv=argv))
            for t in (p/'task').iterdir():
                cpus=os.sched_getaffinity(int(t.name))
                if len(cpus)<=16 and 111 in cpus:owners.append(dict(pid=int(p.name),tid=int(t.name),cpus=sorted(cpus)))
                elif 111 in cpus:broad+=1
        except (FileNotFoundError,ProcessLookupError):continue
    return dict(at_unix=time.time(),restricted_CPU111_owners=owners,broad_mask_threads_containing111=broad,workers=workers,
        service=command(['supervisorctl','status','guardfed_celeba_gradient_screen64_v2a']),queue=read(RUNS/'QUEUE_PROGRESS.json'))

require(sha('/etc/vast-agents-guide.md')==GUIDE,'guide changed')
require(os.sched_getaffinity(0)=={111} and os.getpriority(os.PRIO_PROCESS,0)==10,'CPU111 nice10 required')
require('idle' in command(['ionice','-p',str(os.getpid())])['stdout'].lower(),'idle IO required')
require(os.environ.get('CUDA_VISIBLE_DEVICES')=='','CUDA must be hidden for tensor validation')
require(not DEST.exists(),'Preserve existing collection')
DEST.mkdir()
try:
    before=live();save(DEST/'LIVE_BEFORE.json',before)
    require(not before['restricted_CPU111_owners'],'CPU111 reserved by another restricted thread')
    require(before['service']['returncode']==0 and 'RUNNING' in before['service']['stdout'],'Target queue not healthy')
    require(ID in before['queue']['strict_server_completed_ids'],'Exact item not completed')
    require(not any(ID in r['argv'] or str(RUNS/ID) in r['argv'] for r in before['workers']),'Selected producer still alive')
    quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split();require(quota!='max','Explicit CPU quota required')
    cores=int(quota)/int(period);memory=int(Path('/sys/fs/cgroup/memory.max').read_text())-int(Path('/sys/fs/cgroup/memory.current').read_text())
    require(cores>=24 and memory>=2*1024**3,'Insufficient collector CPU/RAM headroom')
    save(DEST/'RESOURCE.json',dict(pid=os.getpid(),cpu_ids=[111],threads=1,nice=10,io='idle',quota_cores=cores,RAM_headroom_bytes=memory,guide_sha256=GUIDE))
    require(sha(PKG/'FILES_SHA256.json')==SEAL,'Frozen package seal drift')
    for n,s in read(PKG/'FILES_SHA256.json')['files'].items():require(sha(PKG/n)==s,'Frozen member drift: '+n)
    sys.path.insert(0,str(PKG));import run_queue
    stage=run_queue.STAGE;sys.path.insert(0,str(stage));import worker
    run_queue.bind_shared_inputs(worker,REPO.resolve())
    entry=next(r for r in read(PKG/'jobs/manifest.json')['jobs'] if r['id']==ID);jobpath=PKG/'jobs'/entry['job']
    require(sha(jobpath)==entry['job_sha256'],'job changed');job=read(jobpath)
    worker.verify_hashes(REPO,job['source_hashes'])
    import torch
    torch.set_num_threads(1);torch.set_num_interop_threads(1)
    require(torch.__version__=='2.11.0+cu128','Original isolated environment required')
    from accept_result import checked_result
    result=checked_result(jobpath,RUNS/ID);require(result is not None,'No original strict result')
    files={}
    def add(name,path):
        path=Path(path);require(path.is_file() and not path.is_symlink(),'Invalid archive input '+str(path));files[name]=dict(path=path,sha256=sha(path),bytes=path.stat().st_size)
    for f in (RUNS/ID).iterdir():add('runs/'+ID+'/'+f.name,f)
    for n in read(PKG/'FILES_SHA256.json')['files']:add('source/package/'+n,PKG/n)
    add('source/package/FILES_SHA256.json',PKG/'FILES_SHA256.json')
    # Preserve a measured prefix of the active shared log, never relabel it as a closed per-job log.
    log_receipt=[]
    conf=Path('/etc/supervisor/conf.d/guardfed_celeba_gradient_screen64_v2a.conf')
    if conf.exists():
        add('source/supervisor.conf',conf);cfg=configparser.ConfigParser(interpolation=None);cfg.read(conf)
        for key in ('stdout_logfile','stderr_logfile'):
            value=cfg.get('program:guardfed_celeba_gradient_screen64_v2a',key,fallback=None)
            if not value or value.startswith('/dev/'):continue
            path=Path(value)
            if not path.exists():continue
            count=path.stat().st_size;require(count<=8*1024*1024,'Unexpected shared log size; bounded backup stops')
            data=path.open('rb').read(count);target=DEST/(key+'_prefix.log');target.write_bytes(data);add('logs/'+target.name,target)
            log_receipt.append(dict(source=str(path),prefix_bytes=len(data),sha256=sha(target),complete_per_job_log=False))
    proof=dict(status='ORIGINAL_CHECKED_RESULT_PASS',id=ID,rounds=result['rounds'],metrics=result['metrics'],checkpoint_sha256=sha(RUNS/ID/'model.pt'),job_sha256=sha(jobpath),
        validator_sha256=sha(stage/'accept_result.py'),package_seal_sha256=SEAL,source_data_files_rehashed=len(job['source_hashes']),
        runtime=dict(python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,collector_cuda_hidden=True,threads=1),original_training_provenance=result['provenance'],
        log_prefixes=log_receipt,source_host=socket.gethostname(),new_CNN=0,new_training=0,root_adopted=False,method_champion_claim=False)
    save(DEST/'ORIGINAL_STRICT.json',proof)
    after=live();save(DEST/'LIVE_AFTER.json',after)
    for n in ('ORIGINAL_STRICT.json','LIVE_BEFORE.json','LIVE_AFTER.json','RESOURCE.json'):add('evidence/'+n,DEST/n)
    require(sha(VERIFIER)==VERIFIER_SHA,'Original archive verifier changed');add('source/evidence_v4.py',VERIFIER)
    inventory=dict(accepted_new_ids=[ID],members={n:{k:r[k] for k in ('sha256','bytes')} for n,r in files.items()},old_models_repacked=0,selected_only=True,scientific_source_seal=SEAL)
    payload=(json.dumps(inventory,indent=2)+'\n').encode();archive=DEST/'gradient64_first_closed.tar.gz'
    with tarfile.open(archive,'w:gz') as t:
        info=tarfile.TarInfo('backup_inventory.json');info.size=len(payload);t.addfile(info,io.BytesIO(payload))
        for n,r in sorted(files.items()):
            require(sha(r['path'])==r['sha256'],'Input changed before backup');t.add(r['path'],arcname=n,recursive=False)
    require(all(sha(r['path'])==r['sha256'] for r in files.values()),'Input changed during backup')
    receipt=dict(archive_sha256=sha(archive),inventory_sha256=hashlib.sha256(payload).hexdigest(),accepted_new_ids=[ID],members=len(files)+1,source_host=socket.gethostname(),archive=str(archive),original_strict_sha256=sha(DEST/'ORIGINAL_STRICT.json'),scientific_offserver_accepted=0)
    spec=importlib.util.spec_from_file_location('original_archive_verifier',VERIFIER);v=importlib.util.module_from_spec(spec);spec.loader.exec_module(v)
    receipt['remote_member_verification']=v.verify_archive(archive,receipt);save(DEST/'backup_receipt.json',receipt)
    print(json.dumps(dict(status='REMOTE_STRICT_AND_ARCHIVE_PASS',receipt=receipt)),flush=True)
except BaseException as exc:
    save(DEST/'FAILURE.json',dict(error=repr(exc),traceback=traceback.format_exc(),automatic_retry=False));raise
