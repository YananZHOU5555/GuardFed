"""PREPARED-only single-model GPU diagnostic callable; no CLI/dispatch/training."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json,os,subprocess,sys

HERE=Path(__file__).resolve().parent
ID='FairGuard_IID_F-Flip_seed91009'
OUTPUT='/workspace/guardfed_checks/celeba_native_mismatch_diagnostic_execution_20261009/runs/'+ID
def require(ok,message):
    if not ok:raise ValueError(message)
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))

def validate_contract(a,inputs_sha,package_sha):
    require(a.get('status')=='APPROVED_EXACT_ONE_GPU_DIAGNOSTIC_ONLY','PREPARED is not execution approval')
    require(a.get('inputs_sha256')==inputs_sha and a.get('package_sha256')==package_sha,'Wrong source/input seal')
    require(a.get('id')==ID and a.get('output')==OUTPUT,'Foreign model/output')
    require(a.get('allowed_cpus')==[105] and a.get('compute_threads')==1 and a.get('max_processes')==1
        and a.get('host_gpu_index')==0 and a.get('logical_device')=='cuda:0','Only proposed oneCPU105/GPU0 execution')
    require(a.get('native_tolerance')==1e-12 and a.get('target_split')=='valid'
        and a.get('views')==['native','raw','shared_calibration'],'Scientific split/view/tolerance changed')
    for key in ['training_authorized','test_authorized','retry_authorized','cohort_inclusion_authorized','restart_old872_authorized']:
        require(a.get(key) is False,'Prohibited operation: '+key)
    require(a.get('gpu_uuid','').startswith('GPU-') and len(a.get('resource_receipt_sha256',''))==64,'Actual GPU/resource approval required')

def authorized(path,expected_sha):
    require(path is not None and expected_sha is not None and Path(path).is_file(),'External root exact approval absent')
    require(sha(path)==expected_sha,'External approval bytes changed')
    seal=read(HERE/'PACKAGE_SHA256.json')
    for name,row in seal['members'].items():
        p=HERE/name;require(p.is_file() and sha(p)==row['sha256'] and p.stat().st_size==row['bytes'],'Prepared package changed: '+name)
    a=read(path);validate_contract(a,sha(HERE/'INPUTS.json'),sha(HERE/'PACKAGE_SHA256.json'))
    return a

def resource_gate(torch,a,snapshot):
    require(torch.__version__=='2.11.0+cu128' and torch.get_num_threads()==1 and torch.get_num_interop_threads()==1,'Exact proposed runtime/one CPU thread required')
    require(torch.cuda.device_count()==1 and torch.cuda.current_device()==0,'Exactly one approved visibleGPU required')
    require(os.environ.get('CUDA_VISIBLE_DEVICES')==a['gpu_uuid'],'GPU visibility must bind actual hostGPU0 UUID')
    actual=subprocess.run(['nvidia-smi','--id=0','--query-gpu=uuid','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    require(actual==a['gpu_uuid'],'Host GPU0 identity changed')
    gpu_detail=subprocess.run(['nvidia-smi','--id=0','-q'],capture_output=True,text=True,check=True).stdout
    recovery=[x.split(':',1)[1].strip() for x in gpu_detail.splitlines() if 'GPU Recovery Action' in x]
    require(recovery==['None'],'Actual approvedGPU Recovery Action is not None')
    require(set(os.sched_getaffinity(0))=={105} and os.getpriority(os.PRIO_PROCESS,0)>=10,'CPU105/nice10 required')
    require(all(cpus==[105] for cpus in snapshot['thread_cpu_affinities'].values()),'Auxiliary thread escapedCPU105')
    require(torch.are_deterministic_algorithms_enabled() and not torch.backends.cudnn.benchmark and torch.backends.cudnn.deterministic
        and not torch.backends.cudnn.allow_tf32 and not torch.backends.cuda.matmul.allow_tf32,'Original strict FP32 deterministic settings changed')
    require(os.environ.get('CUBLAS_WORKSPACE_CONFIG')==':4096:8','Original CUBLAS workspace changed')
    require('RUNNING' in snapshot['service'].get('stdout',''),'Protected main service health unavailable')
    quota,period=snapshot['cgroup']['cpu.max'].split();require(quota!='max','Actual finite CPU quota required')
    proofpath=Path(a['resource_receipt']);require(sha(proofpath)==a['resource_receipt_sha256'],'Resource receipt changed')
    proof=read(proofpath)
    require(proof['no_duplicate_diagnostic'] is True and proof['allowed_cpu105_free'] is True and proof['gpu_recovery_action']=='None','Resource preflight did not pass')
    require(proof['nominal_threads_before']+1<=int(quota)/int(period) and proof['gpu_memory_free_bytes']>=2*1024**3,'Actual budget or GPU headroom insufficient')
    cg=snapshot['cgroup'];require(cg['memory.max']!='max' and int(cg['memory.max'])-int(cg['memory.current'])>=8*1024**3,'Insufficient memory headroom')

def diagnose(v4,core,original,cnn,evaluator,approval_path,approval_sha):
    """A future root-owned fresh child may invoke once after exact external approval."""
    a=authorized(approval_path,approval_sha) # refuses PREPARED before scientific imports/work
    require(sys.platform=='linux' and not sys.flags.optimize,'Unchanged Linux environment without -O required')
    inputs=read(HERE/'INPUTS.json');record=read(HERE/'MODEL_RECORD.json');v2=v4.v2
    for name,module in [('v4',v4),('v3',v4.v3),('v2',v2),('core',core),('cnn',cnn),('evaluator',evaluator)]:
        require(sha(module.__file__)==inputs['scientific_sources'][name]['sha256'],'Foreign scientific module '+name)
    require(sha(original.__file__)==record['source_hashes']['scripts/run_revision_ablation.py'],'Original checker source changed')
    proof=read(a['resource_receipt']);age=(datetime.now(timezone.utc)-datetime.fromisoformat(proof['utc'])).total_seconds()
    require(0<=age<=120,'Recheck actual resources immediately before the one diagnostic')
    require(all(os.environ.get(k)=='1' for k in ['OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS']),'One-thread environment required before import')
    require('idle' in subprocess.run(['ionice','-p',str(os.getpid())],capture_output=True,text=True,check=True).stdout,'Idle IO required')
    out=Path(a['output']);require(out.parent.is_dir() and not out.parent.is_symlink() and out.parent.resolve()==out.parent
        and not out.exists() and not out.with_name(out.name+'.diagnostic.json').exists(),'Fresh real parent/output required; no implicit retry')
    repo=Path('/workspace/GuardFed-celeba-expanded');paths=v4.expected_paths(record)
    pins={Path(inputs['scientific_sources'][k]['remote']):v['sha256'] for k,v in inputs['scientific_sources'].items()}
    pins.update({repo/k:v for k,v in record['source_hashes'].items()})
    pins.update({paths[k]:record[k]['sha256'] for k in ['checkpoint','result','raw_job']})
    pins.update({HERE/'INPUTS.json':sha(HERE/'INPUTS.json'),HERE/'MODEL_RECORD.json':inputs['record_sha256'],Path(approval_path):approval_sha})
    before=v4.v3.full_hashes(pins)
    _,mapped=v4.mapped_functions(record,paths,original)
    namespace=dict(mapped.__globals__)
    namespace['resource_gate']=lambda snapshot:resource_gate(v2.torch,a,snapshot)
    code=(HERE/'gpu_replay_body.py').read_text();exec(compile(code,str(HERE/'gpu_replay_body.py'),'exec'),namespace)
    replay=namespace['replay_one'];ids,y,s,metadata=v2.metadata(repo)
    error=None
    try:
        replay(core,original,cnn,evaluator,record,repo,ids,y,s,out,1800)
    except BaseException as exc:
        error={'type':type(exc).__name__,'message':str(exc)}
    after=v4.v3.full_hashes(pins)
    require(before==after,'Diagnostic original source/data/model/input changed')
    receipt=read(out/'receipt.json') if (out/'receipt.json').is_file() else None
    report=dict(status='DIAGNOSTIC_ONLY_RESULT_SAVED' if receipt is not None else 'DIAGNOSTIC_EXECUTION_FAILED',id=ID,error=error,
        source_before=before,source_after=after,metadata_read=metadata,native_comparison=receipt.get('native_comparison') if receipt else None,
        old_CPU_failure_still_invalid=True,accepted_for_cohort=False,old872_restart_authorized=False,new_training=0,test_inference=0)
    with out.with_name(out.name+'.diagnostic.json').open('x',encoding='utf-8') as f:json.dump(report,f,indent=2,allow_nan=False);f.write('\n')
    if error is not None and not (receipt and receipt['status']=='DIAGNOSTIC_NATIVE_MISMATCH' and error['message']=='Native metrics exceed fixed tolerance; preserve evidence and stop without retry'):
        raise RuntimeError(error)
    return report

if __name__=='__main__':
    print('PREPARED_NOT_AUTHORIZED: no CLI dispatcher, GPU/CNN/training execution disabled')
