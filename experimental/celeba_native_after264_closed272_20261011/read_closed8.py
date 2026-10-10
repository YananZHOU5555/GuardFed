"""Read and bind only the frozen eight outputs; not scientific acceptance or collection."""
from pathlib import Path
import hashlib, json, shlex, subprocess

HERE = Path(__file__).resolve().parent
scope = json.loads((HERE / 'FROZEN_SCOPE.json').read_bytes())
assert hashlib.sha256((HERE / 'FROZEN_SCOPE.json').read_bytes()).hexdigest() == 'c583b10314307fb8e7f55da77e8f9ed910575ea5ec9f5ee36a21e83fbeac4a4a'
code = '''from pathlib import Path
import hashlib,json,datetime
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
root=Path('/workspace/GuardFed-celeba-expanded')
stage=root/'results/revision_20261009/celeba_mechanism_v1'
canon=Path('/workspace/guardfed_checks/server_reactivation_20261009/mechanism_science_backups_20261009/verified_ledger.json')
assert sha(canon)==PARENT
manifest=read(stage/'manifest.json');entries={e['id']:e for e in manifest['jobs']}
tool=Path('/workspace/guardfed_checks/server_reactivation_20261009/evidence_v4.py')
assert sha(tool)=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
rows=[]
for identity in IDS:
 e=entries[identity];job=read(e['job']);out=Path(e['output']);r=read(out/'result.json');a=read(out/'mechanism_acceptance.json');p=read(out/'progress.json')
 assert sha(e['job'])==e['job_sha256'] and job['id']==identity==p['job_id']
 assert r['config']==job['config'] and job['config']['celeba_evaluation_split']=='valid'
 assert r['rounds']==p['round']==a['rounds']==70 and a['pass']
 assert not list(out.glob('failure*.json'))
 paths=[Path(e['job']),out/'result.json',out/'model.pt',out/'mechanism_acceptance.json',out/'candidate_mask_audit.json',out/'progress.json',stage/'logs'/(identity+'.log')]
 pins={str(p):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}
 assert pins[str(out/'model.pt')]['sha256']==a['checkpoint_sha256']==r['revision_job']['checkpoint_sha256']
 assert pins[str(out/'result.json')]['sha256']==a['result_sha256'] and pins[e['job']]['sha256']==a['job_sha256']
 rows.append({'id':identity,'files':pins,'round':70,'terminal_metadata_identity_checks':True})
assert sha(canon)==PARENT
print(json.dumps({'status':'READONLY_CLOSED8_METADATA_NOT_ORIGINAL_STRICT_ACCEPTANCE','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'ids':IDS,'rows':rows,'manifest_sha256':sha(stage/'manifest.json'),'canonical_ledger_sha256':sha(canon),'original_tool_sha256':sha(tool),'new_accepted':0,'new_archive':False,'Torch_import':False,'remote_writes':False,'source_images_rehashed':False}))
'''
bindings = 'IDS=' + repr(scope['delta8']) + '\nPARENT=' + repr(scope['parent_ledger_sha256']) + '\n'
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '60350', 'root@89.22.197.55', 'python -c ' + shlex.quote(bindings + code)], capture_output=True, timeout=60)
assert not (HERE / 'READONLY_CLOSED8.stdout.json').exists()
(HERE / 'READONLY_CLOSED8.stdout.json').write_bytes(result.stdout)
(HERE / 'READONLY_CLOSED8.stderr.txt').write_bytes(result.stderr)
result.check_returncode()
report = json.loads(result.stdout)
assert report['ids'] == scope['delta8'] and len(report['rows']) == 8 and report['new_accepted'] == 0
print(json.dumps({'status':report['status'],'ids':report['ids'],'report_sha256':hashlib.sha256(result.stdout).hexdigest()}))
