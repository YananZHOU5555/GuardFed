"""Preserve a bounded, read-only view of the first GPU queue failstop."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'tmp/celeba_valid_gpu_remaining464_execution_20261009'
code = r'''from pathlib import Path
import datetime,hashlib,json
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
base=Path('/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/remaining464_attempt1')
stage=base/'chunk_002'
files=[base/'queue_failure.json',base/'chunk_002.commands.log']
files += sorted(p for p in stage.rglob('*') if p.is_file() and p.suffix in ('.log','.json') and 'runs/' not in str(p))
files += sorted((stage/'batch/runs').glob('*.log'))[-3:]
rows=[]
for p in files:
    raw=p.read_bytes()
    rows.append({'path':str(p),'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest(),'tail':raw[-40000:].decode(errors='replace')})
print(json.dumps({'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'read_only':True,'files':rows,'run_members':[p.name for p in sorted((stage/'batch/runs').iterdir())] if (stage/'batch/runs').exists() else []}))
'''
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -c '+shlex.quote(code)],capture_output=True,check=True,timeout=60)
path=DEST/'ROOT_FAILURE_DIAGNOSIS_20261009.json'
with path.open('xb') as stream:stream.write(result.stdout)
report=json.loads(result.stdout)
for row in report['files']:
    if row['path'].endswith('.log') or row['path'].endswith('failure_receipt.json'):
        print(json.dumps(row))
print(json.dumps({'path':str(path),'sha256':hashlib.sha256(result.stdout).hexdigest(),'read_only':True,'files':len(report['files'])}))
