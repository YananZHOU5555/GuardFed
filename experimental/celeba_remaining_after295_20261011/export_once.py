"""One authorized original-CLI exact5 A export. Never retries or changes a source."""
from pathlib import Path
import datetime,hashlib,json,shlex,subprocess
HERE=Path(__file__).resolve().parent
pre=json.loads((HERE/'PREFLIGHT.stdout.json').read_bytes())
from verify_native_inputs import verify_native_inputs
execution=verify_native_inputs()
assert pre['selected_ids']==execution['exact_candidate_ids']
assert pre['candidate_ids']==execution['exact_candidate_ids']
assert pre['status']=='BOUNDED_CLOSED_INTERSECTION_NO_SELECTED_WORKER_CPU111_AVAILABLE' and len(pre['selected_ids'])==5
tag='after295_'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S%fZ')
remote='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010'
queue='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_v2_20261010'
cmd=['taskset','-c','111','nice','-n','10','ionice','-c','3','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B',remote+'/transport.py','export',
 '--source',queue,'--source-seal',pre['source_seal_sha256'],'--transport-seal',pre['transport_seal_sha256'],
 '--runtime',queue+'/attempt1','--review',queue+'/ROOT_APPROVED.json','--review-sha256','55e0c1a5c08fd00a33ff1caaa862b5b1e67c4328559b750a95b6d0a5d1aebb6c',
 '--tag',tag,'--previous',pre['previous_receipt']['receipt'],'--previous-sha256',pre['previous_receipt']['receipt_sha256'],'--ids',*pre['selected_ids']]
remote_argv=['env','CUDA_VISIBLE_DEVICES=','OMP_NUM_THREADS=1','MKL_NUM_THREADS=1','OPENBLAS_NUM_THREADS=1','NUMEXPR_NUM_THREADS=1','PYTHONDONTWRITEBYTECODE=1','taskset','-c','111','nice','-n','10','ionice','-c','3','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','-B','-']
argv=['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55',shlex.join(remote_argv)]
bridge=(HERE/'remote_cpu111_export.py').read_text('utf8')
input_code='import sys\nsys.argv='+repr(cmd[cmd.index(remote+'/transport.py'):])+'\n'+bridge
with (HERE/'EXPORT_COMMAND.json').open('x',encoding='utf8') as f:json.dump(dict(tag=tag,argv=argv,original_CLI_cpu111=cmd,CPU_affinity_memory_bridge_sha256=hashlib.sha256(bridge.encode()).hexdigest(),automatic_retry=False,accepted_offserver=0),f,indent=2);f.write('\n')
print(json.dumps(dict(step='original_export_started_once',tag=tag)),flush=True)
r=subprocess.run(argv,input=input_code.encode(),capture_output=True)
(HERE/'EXPORT.stdout.json').write_bytes(r.stdout);(HERE/'EXPORT.stderr.txt').write_bytes(r.stderr)
with (HERE/'EXPORT_EXIT.json').open('x',encoding='utf8') as f:json.dump(dict(returncode=r.returncode,tag=tag,automatic_retry=False),f);f.write('\n')
r.check_returncode();receipt=json.loads(r.stdout)
assert receipt['accepted_new_ids']==pre['selected_ids'] and receipt['all_transported_ids']==pre['previous_receipt']['all_transported_ids']+pre['selected_ids']
assert receipt['accepted_offserver']==receipt['models_repacked']==0
print(json.dumps(dict(step='original_export_complete',tag=tag,archive_sha256=receipt['archive_sha256'],members=receipt['members'],new=len(receipt['accepted_new_ids']),cumulative=len(receipt['all_transported_ids']))),flush=True)
