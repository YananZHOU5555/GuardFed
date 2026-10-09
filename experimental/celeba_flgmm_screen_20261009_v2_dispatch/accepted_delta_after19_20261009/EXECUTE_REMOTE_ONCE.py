from pathlib import Path
import datetime,hashlib,json,os,subprocess,time
B=Path('/workspace/guardfed_checks/celeba_flgmm_screen_20261009/accepted_delta_after19_20261009')
expected={'PREVIOUS_CHAIN.json': '84be50b9c7dc44ceb3bf34e1bedd0c3ce36e5186b9020e1d6935b9cdc0c95dcd', 'AUTHORIZED_SNAPSHOT.json': 'f87eb797aef52016d1e9a77de871b5cd4942474545c2e628184f993a2ebfc404', 'EXACT_DELTA.json': '581af67d01acbed358b14851f09864aeb179e55a7a8e70ebd2b1b495732ac3f0', 'collect_delta.py': 'f3a2ad1af7a8b334e88c190e83cf1ebf1251b99d29a32097e22d7f9e466d8e20', 'COLLECTOR_DIFF.patch': '4a3fea3d2bf7f082838269c54eb87df226e82c3b20119586fe9e790377a718bf', 'SOURCE_RECEIPT.json': 'd56ff7fbf3cd11aa6ae3b8a4dcdee8ca034bbd89a3d7b8ca48bb38639cba2e03', 'PREVIOUS_LATEST.json': 'c6e70a975b0081cc44aab1c5d0999a419edb20fc8147f31f458fe9362bfbf1e0'}
for rel,want in expected.items():assert hashlib.sha256((B/rel).read_bytes()).hexdigest()==want,rel
assert not (B/'COLLECTOR_COMMAND_START.json').exists()
command=['taskset','-c','106','ionice','-c','3','nice','-n','10','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',str(B/'collect_delta.py')]
env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='',PYTHONDONTWRITEBYTECODE='1')
start=time.monotonic()
with (B/'COLLECTOR_COMMAND_START.json').open('x') as f:json.dump(dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),command=command,helper_cpu=106,threads=1,nice=10,IO='idle',fixed_snapshot=expected['AUTHORIZED_SNAPSHOT.json'],automatic_retry=False),f,indent=2)
r=subprocess.run(command,capture_output=True,env=env,timeout=240)
(B/'COLLECTOR_STDOUT.txt').write_bytes(r.stdout);(B/'COLLECTOR_STDERR.txt').write_bytes(r.stderr)
with (B/'COLLECTOR_COMMAND_RESULT.json').open('x') as f:json.dump(dict(returncode=r.returncode,elapsed_seconds=time.monotonic()-start,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),stdout_sha256=hashlib.sha256(r.stdout).hexdigest(),stderr_sha256=hashlib.sha256(r.stderr).hexdigest(),automatic_retry=False),f,indent=2)
print('COLLECTOR_RETURN='+str(r.returncode));print(r.stdout.decode());print(r.stderr.decode())
raise SystemExit(r.returncode)
