"""Start exactly the newly frozen scientific queue; historical queues untouched."""
from pathlib import Path
import hashlib
import json
import shutil
import subprocess
import time

REPO = Path('/workspace/GuardFed-celeba-expanded')
STAGE = REPO/'results/revision_20261009/celeba_mechanism_v1'
CHECKS = Path('/workspace/guardfed_checks/server_reactivation_20261009')
receipt = json.loads((STAGE/'dispatch_receipt.json').read_text())
assert receipt['pass'] and receipt['image_gates'] == 18 and len(receipt['full_regressions']) == 2
assert len(receipt['reused_full']) == 100 and receipt['torch_version'] == '2.11.0+cu128'
assert receipt['strict_prerequisites']['gate_actual_runtime'] == '2.11.0+cu128'
assert not (STAGE/'formal_queue_progress.json').exists()
sv = subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_preflight'],capture_output=True,text=True)
assert sv.returncode in (0,3) and 'EXITED' in sv.stdout
for proc in Path('/proc').iterdir():
    if not proc.name.isdigit(): continue
    try: cmd=(proc/'cmdline').read_bytes().replace(b'\0',b' ')
    except (PermissionError,FileNotFoundError,ProcessLookupError): continue
    assert not (str(REPO).encode() in cmd and (b'worker.py' in cmd or b'run_revision_ablation.py' in cmd)),proc.name
wrapper=Path('/opt/supervisor-scripts/guardfed_celeba_mechanism_formal.sh')
config=Path('/etc/supervisor/conf.d/guardfed_celeba_mechanism_formal.conf')
assert not wrapper.exists() and not config.exists()
shutil.copyfile(CHECKS/wrapper.name,wrapper); wrapper.chmod(0o755)
shutil.copyfile(CHECKS/config.name,config)
subprocess.run(['supervisorctl','reread'],check=True)
subprocess.run(['supervisorctl','update','guardfed_celeba_mechanism_formal'],check=True)
subprocess.run(['supervisorctl','start','guardfed_celeba_mechanism_formal'],check=True)
record={'started_unix':time.time(),'server':'89.22.197.55:60350','instance':52183675,
        'service':'guardfed_celeba_mechanism_formal','new_scientific_jobs_planned':800,'rounds':70,'concurrency':8,
        'reused_Full':100,'evaluation_split':'valid','old_queue_restarted':False,'test_started':False,
        'dispatch_receipt_sha256':hashlib.sha256((STAGE/'dispatch_receipt.json').read_bytes()).hexdigest(),
        'sglang_stopped_by_explicit_user_authorization':True}
(CHECKS/'formal_launch.json').write_text(json.dumps(record,indent=2)+'\n')
print(json.dumps(record))
