"""Capture one original-schema FLGMM snapshot, without acceptance or remote writes."""
from pathlib import Path
import datetime,hashlib,json,subprocess
ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch/observations/bounded_review_20261009T130206Z/observe_once.py'
BASE=ROOT/'tmp/celeba_flgmm_final6_closure_20261009'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -'],input=SOURCE.read_bytes(),capture_output=True,timeout=45)
out=BASE/('SNAPSHOT_'+stamp);out.mkdir(exist_ok=False)
(out/'COMMAND_STDOUT.txt').write_bytes(result.stdout);(out/'COMMAND_STDERR.txt').write_bytes(result.stderr)
result.check_returncode()
assert result.stdout.startswith(b'SNAPSHOT_JSON=')
raw=result.stdout[len(b'SNAPSHOT_JSON='):];data=json.loads(raw)
(out/'AUTHORIZED_SNAPSHOT.json').write_bytes(raw)
print(json.dumps(dict(path=str(out/'AUTHORIZED_SNAPSHOT.json'),sha256=hashlib.sha256(raw).hexdigest(),
 utc=data['utc'],queue=data['queue'],failures=data['failure_paths'],errors=data['recent_log_error_matches'],
 restricted_CPU106=data['restricted_CPU106_Python'])))
