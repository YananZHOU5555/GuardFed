"""Bounded read-only actual-process and receipt progress snapshot; no scientific imports."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess

HERE = Path(__file__).resolve().parent
STAGE = HERE.parent / 'remaining_seven_prepared_20261009'
def read(path): return json.loads(Path(path).read_text(encoding='utf-8-sig'))
processes=[]
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try:
        argv=[x.decode(errors='replace') for x in (proc/'cmdline').read_bytes().split(b'\0') if x]
        if str(STAGE/'batch.py') not in argv: continue
        index=argv.index(str(STAGE/'batch.py')); role=argv[index+1]
        env=dict(x.split('=',1) for x in (proc/'environ').read_text().split('\0') if '=' in x)
        stat=(proc/'stat').read_text().rsplit(')',1)[1].split()
        processes.append(dict(pid=int(proc.name),role=role,argv=argv,cwd=str((proc/'cwd').resolve()),
            cpus=sorted(os.sched_getaffinity(int(proc.name))),nice=os.getpriority(os.PRIO_PROCESS,int(proc.name)),
            cuda_visible_devices=env.get('CUDA_VISIBLE_DEVICES'),process_state=stat[0],
            user_seconds=int(stat[11])/os.sysconf('SC_CLK_TCK'),system_seconds=int(stat[12])/os.sysconf('SC_CLK_TCK'),
            thread_environment={k:env.get(k) for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS')}))
    except (FileNotFoundError,ProcessLookupError,PermissionError): pass
now=datetime.now(timezone.utc)
completed=[read(path) for path in sorted(STAGE.glob('completed_*.json'))]
outputs=[]
for identity,path in read(STAGE/'SCOPE.json')['outputs'].items():
    out=Path(path)
    outputs.append(dict(id=identity,exists=out.exists(),artifacts=[dict(name=x.name,bytes=x.stat().st_size) for x in out.iterdir() if x.is_file()] if out.exists() else [],
        strict_acceptance_exists=(out/'strict_acceptance.json').exists(),bridge_failure=read(out.with_name(out.name+'.bridge_failure.json')) if out.with_name(out.name+'.bridge_failure.json').exists() else None))
report=dict(utc=now.isoformat(),hostname=os.uname().nodename,service=subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_valid_remaining7'],capture_output=True,text=True).stdout.strip(),
    processes=processes,completed=completed,outputs=outputs,batch_failure=read(STAGE/'batch_failure.json') if (STAGE/'batch_failure.json').exists() else None,
    batch_complete=read(STAGE/'batch_complete.json') if (STAGE/'batch_complete.json').exists() else None,
    source_seal_sha256=hashlib.sha256((STAGE/'FILES_SHA256.json').read_bytes()).hexdigest(),new_training=0,new_Full_inference=0,new_test_inference=0,
    logs=[dict(name=x.name,bytes=x.stat().st_size,tail=x.read_text(errors='replace')[-1200:]) for x in sorted((STAGE/'logs').glob('*.log'))])
target=HERE/('live_'+now.strftime('%Y%m%dT%H%M%SZ')+'.json')
assert not target.exists();target.write_bytes((json.dumps(report,indent=2,allow_nan=False)+'\n').encode())
print(json.dumps(dict(snapshot=str(target),service=report['service'],processes=processes,strictly_accepted=len(completed),failure=report['batch_failure'])),flush=True)
