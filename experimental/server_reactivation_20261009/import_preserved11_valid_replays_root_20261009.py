"""Explicitly revalidate and register preserved partial10+GPU1; no CNN replay."""
from pathlib import Path
import datetime
import hashlib
import json
import shlex
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_valid_gpu_recovery_execution_20261009'
LOCAL = BASE / 'preserved11_import'
REMOTE = '/workspace/guardfed_checks/celeba_valid_gpu_recovery_execution_20261009'
PKG = '/workspace/guardfed_checks/celeba_valid_gpu_recovery_implementation_20261009'
read = lambda p: json.loads(p.read_bytes())
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
save = lambda p, v: p.write_text(json.dumps(v, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
assert not LOCAL.exists()
first = read(BASE / 'first1_backup/ROOT_OFFSERVER_VERIFICATION.json')
assert first['status'] == 'ROOT_FIRST_GPU_VALID_REPLAY_OFFSERVER_AND_SAVED_ARRAY_PASS'
manifest = read(ROOT / 'tmp/celeba_valid_recovery_prepared_20261009/manifest.json')
ids = [r['id'] for r in manifest['records'] if r['classification'] != 'UNEXECUTED_465']
assert len(ids) == len(set(ids)) == 11
old_collector = BASE / 'cumulative_425_accepted.json'
old = read(old_collector)
assert old['accepted_n'] == 425 and not set(ids) & set(old['accepted_ids'])
review = read(BASE / 'ROOT_REVIEW_FIRST1.json')
review.update(approved_ids=ids, execute_new465=False, import_cpu_partial10=True,
              import_gpu_diagnostic1=True, output_parent=REMOTE + '/preserved11_import',
              approved_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              review_scope='Exact preserved CPU partial10 and one saved successful GPU diagnostic. Rebuild original strict and saved-array science; no CNN/training/test/retry. Original CPU failure stays invalid.')
LOCAL.mkdir()
review_path = LOCAL / 'ROOT_REVIEW_IMPORT11.json'
save(review_path, review)
review_sha = sha(review_path)
ssh = ['ssh', '-o', 'BatchMode=yes', '-o', 'ConnectTimeout=15', '-p', '60350', 'root@89.22.197.55']
code = """from pathlib import Path
import hashlib,json,importlib.util
assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
parent=Path(REVIEW['output_parent']); target=Path(TARGET)
assert not parent.exists() and not target.exists()
spec=importlib.util.spec_from_file_location('import11_preflight',Path(PKG)/'recovery.py'); m=importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
resource=m.resource_preflight(REVIEW)
parent.mkdir()
with target.open('xb') as out: out.write(BYTES)
assert hashlib.sha256(target.read_bytes()).hexdigest()==REVIEW_SHA
print(json.dumps({'status':'IMPORT11_EXPLICIT_REVIEW_DEPLOYED_NO_CNN','resource':resource}))
"""
bound = '\n'.join(k+'='+repr(v) for k,v in dict(REVIEW=review, TARGET=REMOTE+'/ROOT_REVIEW_IMPORT11.json',
                 PKG=PKG, BYTES=review_path.read_bytes(), REVIEW_SHA=review_sha).items())+'\n'+code
result = subprocess.run(ssh+['python -c '+shlex.quote(bound)],capture_output=True,check=True,timeout=60)
(LOCAL / 'PREDEPLOY_RESOURCES.json').write_bytes(result.stdout)
for command, name in [('import-cpu-partial','CPU10.json'), ('import-gpu-diagnostic','GPU1.json')]:
    argv = ['nice','-n','10','ionice','-c','3','/workspace/guardfed_envs/celeba-cu128-20261009/bin/python',
            PKG+'/recovery.py',command,'--review',REMOTE+'/ROOT_REVIEW_IMPORT11.json',
            '--review-sha256',review_sha,'--package-sha256',review['implementation_package_sha256'],
            '--output',review['output_parent']+'/'+name]
    with (LOCAL / (name+'.log')).open('xb') as log:
        result = subprocess.run(ssh+[shlex.join(argv)],stdout=log,stderr=subprocess.STDOUT)
    save(LOCAL / (name+'.EXIT.json'), {'returncode':result.returncode,'log_sha256':sha(LOCAL/(name+'.log'))})
    assert result.returncode == 0, 'Failstop: preserve import evidence; do not blindly retry'
members = {}
for name in ['CPU10.json','CPU10.original_v4_strict.json','GPU1.json']:
    result = subprocess.run(ssh+['cat '+shlex.quote(review['output_parent']+'/'+name)],capture_output=True,check=True,timeout=60)
    (LOCAL/name).write_bytes(result.stdout)
    members[name]={'sha256':sha(LOCAL/name),'bytes':len(result.stdout)}
cpu, gpu = read(LOCAL/'CPU10.json'), read(LOCAL/'GPU1.json')
for report in [cpu,gpu]:
    assert report['status']=='REVIEWED_IMPORT_STRICT_MATCH_PENDING_OFFSERVER_AND_REGISTRATION'
    assert report['review_sha256']==review_sha and report['new_CNN_inference']==0
    assert report['original_CPU_failure_still_invalid'] and not report['cohort_registered']
assert cpu['eligible_n']==10 and gpu['eligible_n']==1
assert set(cpu['eligible_ids']+gpu['eligible_ids'])==set(ids)
assert cpu['original_strict_or_diagnostic_check']==read(LOCAL/'CPU10.original_v4_strict.json')
assert cpu['original_strict_or_diagnostic_check']['max_abs_native_metric_difference']==0
assert gpu['original_strict_or_diagnostic_check']['native_comparison']['max_abs_difference']==0
verification={'status':'ROOT_PRESERVED_IMPORT11_STRICT_AND_OFFSERVER_REPORTS_PASS',
              'utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'eligible_ids':ids,
              'review_sha256':review_sha,'members':members,'CPU10_source_proof_sha256':'8e37410b4ec83f57d5550da962997f449cfd8a07934627327344a4d6eddc5bdc',
              'GPU1_source_proof_sha256':'26779b047b792b2b352153a5e16ba0a4acec413212fa9d3799c750cc9ce0ff92',
              'old425_collector_sha256':sha(old_collector),'new_CNN_inference':0,
              'original_CPU_failure_still_invalid':True,'no_old_arrays_or_models_repacked':True}
save(LOCAL/'ROOT_OFFSERVER_IMPORT_VERIFICATION.json',verification)
collector=dict(old)
collector.update(accepted_n=436,accepted_ids=old['accepted_ids']+ids,
                 previous_425_collector_sha256=sha(old_collector),
                 preserved11_import_verification_sha256=sha(LOCAL/'ROOT_OFFSERVER_IMPORT_VERIFICATION.json'),
                 preserved11_import_ids=ids,CPU_partial10_registered=True,diagnostic1_registered=True,
                 original_failed_CPU_prediction_still_invalid=True)
assert len(set(collector['accepted_ids']))==436 and sha(old_collector)==verification['old425_collector_sha256']
save(BASE/'cumulative_436_accepted.json',collector)
print(json.dumps({'status':verification['status'],'accepted_n':436,'new_CNN_inference':0,'remaining':464}))
