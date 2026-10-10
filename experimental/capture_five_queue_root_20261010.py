"""Reuse the sealed stdlib-only observer once; counts remain observation only."""
from pathlib import Path
import datetime, hashlib, json, subprocess
R=Path(__file__).resolve().parents[1]
S=R/'tmp/celeba_five_stage_health_20261010'
O=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
source=(S/'observe_remote.py').read_bytes()
assert hashlib.sha256(source).hexdigest()=='6e49d3d6c6342d61f54c93155af5d1d0a26ac5d5fc6dfac5eecb18104d9a4802'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
cp=subprocess.run(json.loads((S/'COMMAND.json').read_bytes())['command'],input=source,capture_output=True,timeout=50)
p=O/('root_five_queue_'+stamp+'.raw.json')
with p.with_suffix('.stderr.txt').open('xb') as f:f.write(cp.stderr)
assert cp.returncode==0,cp.stderr.decode(errors='replace')
d=json.loads(cp.stdout)
assert d['status']=='SINGLE_FIVE_STAGE_READONLY_HEALTH_NOT_ACCEPTANCE' and d['new_accepted']==0 and d['no_remote_writes'] and d['no_CNN_or_fit_or_training']
with p.open('xb') as f:f.write(cp.stdout)
print(json.dumps(dict(path=p.relative_to(R).as_posix(),sha256=hashlib.sha256(cp.stdout).hexdigest(),utc=d['utc'],main=d['main_mechanism'],FL_complete=len(d['FLGMM']['terminal_ids']),FL_rounds=d['FLGMM']['active'],gradient_complete=len(d['gradient64']['terminal_ids']),gradient_rounds=d['gradient64']['active'],remaining_remote_closed=d['remaining620']['remote_closed_n'],Hybrid_complete=len(d['Hybrid96']['terminal_records']),Hybrid_queue=d['Hybrid96']['queue_progress'],resources=d['resources'],logs=d['logs']),ensure_ascii=False))
