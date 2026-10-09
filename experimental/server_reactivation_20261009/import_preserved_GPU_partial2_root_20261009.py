"""Revalidate saved partial GPU outputs without CNN, preserving the full failure."""
from pathlib import Path
import datetime
import hashlib
import importlib.util
import json
import shlex
import subprocess
import sys

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_valid_gpu_remaining464_execution_20261009'
OUT=BASE/'partial2_explicit_import'
assert not OUT.exists();OUT.mkdir()
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
EVIDENCE=ROOT/'tmp/celeba_valid_gpu_remaining464_evidence_20261009'
prior_path=EVIDENCE/'chunk_001/cumulative_458_accepted.json'
prior_sha=sha(prior_path);prior=read(prior_path)
expected=['FairGuard_IID_Sp-DFA_seed91007','FairGuard_IID_Sp-DFA_seed91008']
assert prior['accepted_n']==458 and not set(expected)&set(prior['accepted_ids'])
STAGE='/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/remaining464_attempt1/chunk_002'
code=r'''from pathlib import Path
import hashlib,json,subprocess
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
assert sha('/workspace/guardfed_checks/celeba_valid_gpu_recovery_implementation_20261009/PACKAGE_SHA256.json')=='6ae15988b5d0b1ebe4371166afa99bceca015394cba8ecc6f987773621d55b56'
assert not Path(STAGE+'/strict_acceptance.json').exists()
args=['taskset','-c','105','nice','-n','10','ionice','-c','3','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python','/workspace/guardfed_checks/celeba_valid_gpu_recovery_implementation_20261009/recovery.py','accept','--batch',STAGE+'/batch','--output',STAGE+'/strict_acceptance.json','--review','/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009/ROOT_REVIEW_REMAINING464.json','--review-sha256','2b646f10bb47e2e94e1421fd045564b7392b69962b2b51d455c7c0e85420bfac','--package-sha256','6ae15988b5d0b1ebe4371166afa99bceca015394cba8ecc6f987773621d55b56']
r=subprocess.run(args,capture_output=True,text=True)
print(json.dumps({'returncode':r.returncode,'stdout':r.stdout,'stderr':r.stderr,'report_sha256':sha(STAGE+'/strict_acceptance.json') if Path(STAGE+'/strict_acceptance.json').exists() else None}))
raise SystemExit(0 if r.returncode==1 else 2)
'''
result=subprocess.run(['ssh','-o','BatchMode=yes','-o','ConnectTimeout=15','-p','60350','root@89.22.197.55','python -c '+shlex.quote('STAGE='+repr(STAGE)+'\n'+code)],capture_output=True,timeout=120)
(OUT/'REMOTE_STRICT_RUN.json').write_bytes(result.stdout);(OUT/'SSH_STDERR.log').write_bytes(result.stderr)
assert result.returncode==0, 'Preserve failed strict import; never blind retry'
remote=read(OUT/'REMOTE_STRICT_RUN.json');assert remote['returncode']==1
strict_path=OUT/'original_partial_strict_acceptance.json'
subprocess.run(['scp','-o','BatchMode=yes','-o','ConnectTimeout=15','-P','60350','root@89.22.197.55:'+STAGE+'/strict_acceptance.json',str(strict_path)],check=True,timeout=45)
assert sha(strict_path)==remote['report_sha256']
strict=read(strict_path)
assert strict['status']=='PARTIAL_OR_INVALID_VALID_REPLAY' and strict['requested_n']==11 and strict['accepted_n']==2 and strict['accepted_ids']==expected
assert len(strict['invalid'])==9 and strict['max_abs_native_metric_difference']==0
spec=importlib.util.spec_from_file_location('gpu_saved_evidence',EVIDENCE/'evidence.py')
evidence=importlib.util.module_from_spec(spec);spec.loader.exec_module(evidence)
assert sha(EVIDENCE/'PACKAGE_SHA256.json')=='52f46820fd3d01ea532af1ec70f7a8c731116662541c85307549730ac20f609b'
c=evidence.context()
failure=BASE/'failure_chunk002';inventory=read(failure/'failure_remote_archive_inventory.json')
assert sha(failure/'failure_chunk_evidence.tar.gz')=='bd6d4d165f826d5233b421162c59d2e4efe440c49d82701d0ee1d36b8cc0afee'
dest=OUT/'verified_extract';evidence.unpack(failure/'failure_chunk_evidence.tar.gz',inventory,dest)
batch=read(dest/'batch/batch_inputs.json');execution=read(dest/'batch/batch_execution.json')
assert sha(dest/'batch/batch_inputs.json')==strict['batch_inputs_sha256'] and execution['finished_zero_exit_ids']==expected
assert execution['source_unchanged'] and execution['source_after']==batch['source_before']
assert set(x['id'] for x in strict['invalid'])==set(batch['selected_ids'])-set(expected)
results=[evidence.check_run(dest/'batch/runs'/identity,c['records'][identity],c,strict,batch,STAGE) for identity in expected]
assert sha(prior_path)==prior_sha and all(r['native_max_abs_difference']==0 for r in results)
proof=dict(status='ROOT_EXPLICIT_GPU_PARTIAL2_ORIGINAL_STRICT_AND_SAVED_ARRAY_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    accepted_new_ids=expected,new_n=2,partial_strict_sha256=sha(strict_path),prior458_sha256=prior_sha,
    failure_archive_sha256=inventory['sha256'],failure_inventory_sha256=sha(failure/'failure_remote_archive_inventory.json'),
    evidence_package_sha256=sha(EVIDENCE/'PACKAGE_SHA256.json'),source_package_sha256=c['recovery_sha'],review_sha256=c['review_sha'],
    root_refit_original_remote_strict_only=True,saved_metrics_verified=18,saved_confusion_counts_verified=48,saved_prediction_rules_verified=6,
    results=results,new_CNN_inference=0,original_failure_preserved=True,missing9_not_accepted=True,test=False)
proof_path=OUT/'ROOT_OFFSERVER_IMPORT_VERIFICATION.json'
with proof_path.open('x',encoding='utf8',newline='\n') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
collector=dict(scope='NINE_METHOD900_VALID_REPLAY_CUMULATIVE_EXPLICIT_GPU_PARTIAL2_VERSION',status='PARTIAL_UNIQUE_VALID_REPLAY_ACCEPTED',
    accepted_n=460,expected_n=900,accepted_ids=prior['accepted_ids']+expected,added_ids=expected,
    previous_collector_path=str(prior_path),previous_collector_sha256=prior_sha,new_proof_path=str(proof_path),new_proof_sha256=sha(proof_path),
    CPU_provenance_n=434,GPU_provenance_n=26,all900_native_valid_replayed=False,test_evaluation_performed=False,uniform_device_comparison=False)
assert len(set(collector['accepted_ids']))==460
with (OUT/'cumulative_460_accepted.json').open('x',encoding='utf8',newline='\n') as stream:json.dump(collector,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=proof['status'],accepted_n=460,proof_sha256=sha(proof_path),collector_sha256=sha(OUT/'cumulative_460_accepted.json'))))
