from pathlib import Path
import json,subprocess,datetime,os
p=Path('/workspace/guardfed_checks/celeba_native_mismatch_diagnostic_execution_20261009')
q=json.loads(Path('/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1/formal_queue_progress.json').read_text())
r={'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'diagnostic_pid_34608_exists':Path('/proc/34608').exists(),'formal_service':subprocess.check_output(['supervisorctl','status','guardfed_celeba_mechanism_formal'],text=True).strip(),'formal_queue':q,'gpu':subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],text=True).strip()}
with (p/'post_diagnostic_live.json').open('x') as f:json.dump(r,f,indent=2)
print(json.dumps({k:v for k,v in r.items() if k!='formal_queue'}))
