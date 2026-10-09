"""Run the existing read-only screen observer and save source-bound current facts."""
from pathlib import Path
import datetime, hashlib, json, subprocess
ROOT=Path(__file__).resolve().parents[1]
CHECKS=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
source=ROOT/'tmp/celeba_hybrid_screen_execution_20261009/results_incremental_20261009T1133Z/observe.py'
guard="from pathlib import Path\nimport hashlib\nassert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'\n"
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -B -'],
    input=(guard+source.read_text(encoding='utf8')).encode(),capture_output=True,check=True,timeout=50)
data=json.loads(result.stdout)
assert data['guide_sha256']=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
stamp=datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
raw=CHECKS/f'auxiliary_screens_{stamp}.RAW.json'
with raw.open('xb') as stream:stream.write(result.stdout)
screens=[]
for name,pin in [('flgmm','aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'),('hybrid','2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f')]:
    s=data[name];assert s['source_seal_sha256']==pin and len(s['rows'])==32
    terminal=[r for r in s['rows'] if r['result_present'] and r['terminal_acceptance'] and r['progress']['round']==70]
    active=[r for r in s['rows'] if r['progress'] and not r['result_present']]
    failures=s['terminal_flags'] if name=='flgmm' else [p for p in s['terminal_flags'] if 'failure' in p.lower()]
    failures=failures+[p for r in s['rows'] for p in r['failure_files']]
    screens.append(dict(name=name,source_seal_sha256=pin,terminal_candidates=terminal,active_or_partial=active,failures=failures))
normalized=dict(utc=data['utc'],read_only=True,new_acceptances=0,screens=screens,
    raw_observation=raw.relative_to(ROOT).as_posix(),raw_sha256=hashlib.sha256(result.stdout).hexdigest(),
    observer_source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),services=data['services'])
path=CHECKS/f'auxiliary_screens_{stamp}.json'
with path.open('x',encoding='utf8') as stream:json.dump(normalized,stream,indent=2);stream.write('\n')
print(json.dumps(dict(utc=data['utc'],path=path.relative_to(ROOT).as_posix(),
    screens=[dict(name=s['name'],observed_complete=len(s['terminal_candidates']),active=len(s['active_or_partial']),
        rounds=[r['progress']['round'] for r in s['active_or_partial']],pending=32-len(s['terminal_candidates'])-len(s['active_or_partial']),failures=s['failures']) for s in screens])))
