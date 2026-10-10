"""Read-only actual-process observation of the frozen exact10 replay."""
from pathlib import Path
import argparse, datetime, hashlib, json, subprocess, sys

if sys.flags.optimize:
    raise RuntimeError('Optimized Python is forbidden for root operation guards')
ROOT = Path(__file__).resolve().parents[2]
EX = ROOT / 'tmp/celeba_mechanism_valid_C_after60_20261010/execution_candidate'
REMOTE = '/workspace/guardfed_checks/celeba_mechanism_valid_C_after60_20261010/execution_candidate'
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--progress', action='store_true')
args = parser.parse_args()
deployment = read(EX / 'deployment_receipt.json')
assert deployment['remote_installation']['returncode'] == 0
scope = read(EX.parent / 'SCOPE.json')
assert len(scope['selected_ids']) == 10 and len(scope['excluded_prior_ids']) == 160
stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
code = """from pathlib import Path
import datetime,hashlib,json,os,subprocess
H=Path(%r)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(Path('/etc/vast-agents-guide.md'))=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha(H/'EXECUTION_SOURCE_SHA256.json')==%r and sha(H/'ROOT_APPROVED.json')==%r
for row in read(H/'EXECUTION_SOURCE_SHA256.json')['members']:assert sha(H/row['path'])==row['sha256']
processes=[]
for p in Path('/proc').iterdir():
 if not p.name.isdigit():continue
 try:
  argv=[s.decode(errors='replace') for s in (p/'cmdline').read_bytes().split(b'\\0') if s]
  if str(H/'batch.py') not in argv:continue
  stat=(p/'stat').read_text().rsplit(')',1)[1].split()
  if stat[0]=='Z':continue
  env=dict(x.split('=',1) for x in (p/'environ').read_text().split('\\0') if '=' in x)
  processes.append(dict(pid=int(p.name),argv=argv,cpus=sorted(os.sched_getaffinity(int(p.name))),
   nice=os.getpriority(os.PRIO_PROCESS,int(p.name)),io=subprocess.run(['ionice','-p',p.name],capture_output=True,text=True).stdout.strip(),
   thread_affinities={t.name:sorted(os.sched_getaffinity(int(t.name))) for t in (p/'task').iterdir()},
   user_seconds=int(stat[11])/os.sysconf('SC_CLK_TCK'),system_seconds=int(stat[12])/os.sysconf('SC_CLK_TCK'),
   environment={k:env.get(k) for k in ('CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS')}))
 except (FileNotFoundError,ProcessLookupError,PermissionError):continue
files={name:dict(sha256=sha(H/name),data=read(H/name)) for name in ('APPROVED.json','preflight.json','start_receipt.json')}
print(json.dumps(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
 service=subprocess.run(['supervisorctl','status','guardfed_celeba_mechanism_valid_C_after60'],capture_output=True,text=True).stdout.strip(),
 processes=processes,files=files,completed=[read(p) for p in sorted(H.glob('completed_*.json'))],
 approval_hash_file_sha256=sha(H/'APPROVED.sha256'),
 batch_failure=read(H/'batch_failure.json') if (H/'batch_failure.json').exists() else None,
 batch_complete=read(H/'batch_complete.json') if (H/'batch_complete.json').exists() else None,
 guide_sha256=sha(Path('/etc/vast-agents-guide.md')),no_registry_update=True)))
""" % (REMOTE, deployment['execution_seal_sha256'], deployment['root_approval_sha256'])
result = subprocess.run(['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '60350',
                         'root@89.22.197.55', 'python -B -'], input=code.encode(), capture_output=True, check=True, timeout=45)
raw = EX / (f'ROOT_PROGRESS_{stamp}.RAW.json' if args.progress else 'ROOT_STARTUP_OBSERVATION.RAW.json')
with raw.open('xb') as stream:
    stream.write(result.stdout)
data = json.loads(result.stdout)
assert not data['batch_failure']
assert data['files']['APPROVED.json']['data']['root_approval_sha256'] == deployment['root_approval_sha256']
preflight = data['files']['preflight.json']['data']
assert preflight['external_draft_sha256'] == deployment['external_draft_sha256']
assert preflight['empty_outputs'] and preflight['no_duplicate']
budget = preflight['resources_after']
assert budget['conservative_total_including_this8'] <= budget['actual_quota_cores']
assert data['files']['start_receipt.json']['data']['execution_seal_sha256'] == deployment['execution_seal_sha256']
assert len(data['processes']) <= 2
for process in data['processes']:
    assert process['cpus'] == list(range(112, 120)) and process['nice'] >= 10 and process['io'] == 'idle'
    assert all(cpus == list(range(112, 120)) for cpus in process['thread_affinities'].values())
    assert process['environment'] == dict(CUDA_VISIBLE_DEVICES='', OMP_NUM_THREADS='8', MKL_NUM_THREADS='8', OPENBLAS_NUM_THREADS='1')
workers = [p for p in data['processes'] if 'worker' in p['argv']]
assert ('RUNNING' in data['service'] and len(workers) == 1) or (data['batch_complete'] and len(data['completed']) == 10 and not data['processes'])
for name, row in data['files'].items():
    payload = (json.dumps(row['data'], indent=2, allow_nan=False) + '\n').encode()
    assert hashlib.sha256(payload).hexdigest() == row['sha256']
    source = EX / name
    if source.exists():
        assert sha(source) == row['sha256']
    else:
        with source.open('xb') as stream:
            stream.write(payload)
approved_sha = EX / 'APPROVED.sha256'
if args.progress:
    assert approved_sha.exists()
else:
    with approved_sha.open('xb') as stream:
        stream.write((data['files']['APPROVED.json']['sha256'] + '\n').encode())
assert sha(approved_sha) == data['approval_hash_file_sha256']
data.update(status='ROOT_C_AFTER60_REAL_LINUX_' + ('PROGRESS' if args.progress else 'STARTUP') + '_AND_ALLOCATION_PASS',
    deployment_receipt_sha256=sha(EX / 'deployment_receipt.json'), execution_seal_sha256=deployment['execution_seal_sha256'],
    original160_not_rerun=True, new_training=0, new_Full_inference=0, scientific_offserver_new_accepted=0, test_inference=False)
if args.progress:
    data['source_startup_proof_sha256'] = sha(EX / 'ROOT_STARTUP_OBSERVATION.json')
target = EX / (f'ROOT_PROGRESS_{stamp}.json' if args.progress else 'ROOT_STARTUP_OBSERVATION.json')
with target.open('x', encoding='utf8') as stream:
    json.dump(data, stream, indent=2)
    stream.write('\n')
print(json.dumps(dict(status=data['status'], service=data['service'], actual_processes=len(data['processes']),
    worker_pids=[p['pid'] for p in workers], remote_terminal_candidates=len(data['completed']), proof_sha256=sha(target))))
