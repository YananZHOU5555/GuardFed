"""Read frozen failure artifacts only; no NumPy/Torch, prediction or refit."""
from pathlib import Path
import datetime,hashlib,json,os,subprocess
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert __debug__ and sorted(os.sched_getaffinity(0))==[110] and os.getpriority(os.PRIO_PROCESS,0)>=10
assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'))
assert 'idle' in subprocess.check_output(['ionice','-p',str(os.getpid())],text=True).lower()
base=Path('/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010');run=base/'attempt1';identity='minus_F_non-IID_Sp-DFA_seed91004'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
bp=run/'tasks'/identity/'binding.json';assert sha(bp)=='489e9bb0beed5e9977d5a8e81b62fbe02a9cabe73d08ce72df0f73b8f1e712cd'
b=read(bp);r=b['record'];assert r['id']==identity
out=run/'runs'/identity;receipt=read(out/'receipt.json')
original=Path(r['runtime_paths']['result']);result=read(original)
aggregate=result['round_summaries'][-1]['aggregate']
print(json.dumps({'id':identity,'result_sha256':sha(original),'round':result['round_summaries'][-1]['round'],'aggregate_keys':list(aggregate),'aggregate':aggregate,'original_data_contract':result.get('data_contract'),'original_root_noise_audit':result['round_summaries'][-1].get('root_noise_audit'),'read_only':True,'CNN':0,'fit':0,'retry':False}))
