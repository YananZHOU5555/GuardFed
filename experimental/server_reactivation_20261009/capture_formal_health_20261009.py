"""Read-only live check; save to a fresh path before replacing the canonical JSON."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CHECKS = ROOT / 'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
code = """from pathlib import Path
import hashlib,runpy,sys
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
sys.argv=['check_mechanism_live_20261009.py','formal']
runpy.run_path('/workspace/guardfed_checks/server_reactivation_20261009/check_mechanism_live_20261009.py',run_name='__main__')
"""
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '60350',
                         'root@89.22.197.55', 'python -c ' + shlex.quote(code)],
                        capture_output=True, timeout=50, check=True)
data = json.loads(result.stdout)
assert data['phase'] == 'formal' and len(data['active']) <= 8
path = CHECKS / ('root_live_' + datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '.json')
with path.open('xb') as stream:
    stream.write(result.stdout)
assert json.loads(path.read_bytes()) == data
(CHECKS / 'latest_formal_live.json').write_bytes(result.stdout)
print(json.dumps(dict(path=str(path), sha256=hashlib.sha256(result.stdout).hexdigest(),
                      checked_utc=data['checked_utc'], service=data['service'],
                      completed_observed=data['queue_completed'], failed=data['failed'],
                      active_rounds=[r.get('progress', {}).get('round') for r in data['active']], pending=data['pending'],
                      gpu=data['gpu_csv'], recovery=data['gpu_recovery'],
                      cpu_cores=data['cpu_used_cores_2sec'], cpu_quota=data['cpu_quota_cores'],
                      memory=data['memory_used_bytes'], memory_events=data['memory_events'],
                      disk_free=data['disk_free_bytes'], errors=data['recent_active_log_errors'])))
