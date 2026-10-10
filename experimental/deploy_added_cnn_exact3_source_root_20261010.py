"""Deploy only the sealed small source package; never starts science."""
from pathlib import Path
import base64, datetime, hashlib, json, shlex, subprocess, sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / 'tmp/celeba_added_cnn_three_view_gate_preparation_20261010'
OUT = ROOT / 'tmp/celeba_added_cnn_exact3_root_execution_20261010'
BASE = '/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010'
SEAL = '49c42b90214eae57d456f8ded1d2a4ff2de2762a78d6bcf5c4ae1a5766de9434'

def sha(b): return hashlib.sha256(b).hexdigest()
def read(p): return json.loads(p.read_text(encoding='utf-8-sig'))
def save(p, v): p.write_text(json.dumps(v, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')

def main():
    OUT.mkdir(exist_ok=True)
    assert sha((SRC/'FILES_SHA256.json').read_bytes()) == SEAL
    pins = read(SRC/'FILES_SHA256.json')['files']
    members = {}
    for rel, pin in pins.items():
        p = (SRC/rel).resolve()
        assert p.is_relative_to(SRC.resolve())
        b = p.read_bytes()
        assert sha(b)==pin['sha256'] and len(b)==pin['bytes']
        members['source/'+rel] = b
    members['source/FILES_SHA256.json'] = (SRC/'FILES_SHA256.json').read_bytes()
    actual_p = SRC/'ROOT_ACTUAL_METADATA_REPLAY.json'
    independent_p = ROOT/'tmp/celeba_added_cnn_exact3_independent_review_20261010/REVIEW.json'
    actual, independent = read(actual_p), read(independent_p)
    assert sha(actual_p.read_bytes())=='c67d1078890866f04b782b93e0e5ad62ace6e1c2cc8bfbd81fac8985afa61e90'
    assert sha(independent_p.read_bytes())=='c4c8c1662e9f9e999e305e737652df5fa32caacb89e19d49ba4d0bdbae56ad21'
    assert actual['actual_exit']==0 and actual['actual_output_exact'] is True
    assert independent['source_adoptable'] and independent['no_source_blocker']
    review = dict(status='ROOT_EXACT3_SOURCE_REVIEW_PASS_NOT_RUNTIME_ACCEPTANCE',
        utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), source_adoptable=True,
        package_sha256=SEAL, manifest_sha256=sha((SRC/'MANIFEST.json').read_bytes()),
        root_actual_metadata_replay_sha256=sha(actual_p.read_bytes()),
        independent_source_review_sha256=sha(independent_p.read_bytes()),
        exact_ids=[r['id'] for r in read(SRC/'MANIFEST.json')['records']],
        scientific_function_sources_exact=17, original_runtime_helpers_AST_exact=7,
        inverse_replay_AST_exact=True, root_read_entire_candidate=True,
        identity_and_runtime_path_rebinding_only=True, tolerance=1e-12,
        no_test=True, no_training=True, no_source_math_changes=True,
        CPU_runtime_not_CUDA_equivalence=True, root_fit_only=True,
        runtime_still_requires_actual_fresh_preflight_and_exact3_authorization=True,
        broad_CPU_masks_are_not_exclusive_reservations=True,
        failure_stop_preserve_no_auto_retry=True, new_scientific_acceptances=0)
    save(OUT/'ROOT_SOURCE_REVIEW.json', review)
    members['ROOT_SOURCE_REVIEW.json']=(OUT/'ROOT_SOURCE_REVIEW.json').read_bytes()
    payload = {'base':BASE, 'members': {r:{'base64':base64.b64encode(b).decode(), 'sha256':sha(b), 'bytes':len(b)} for r,b in members.items()}}
    remote = r'''
import base64,datetime,hashlib,json,pathlib,sys
p=json.load(sys.stdin); base=pathlib.Path(p['base'])
assert str(base)=='/workspace/guardfed_checks/celeba_added_cnn_three_view_gate_20261010'
assert not base.exists(), 'Fresh deployment path required; preserve existing attempt'
checked={}
for rel,pin in p['members'].items():
 target=base/rel
 assert target.is_relative_to(base) and '..' not in pathlib.PurePosixPath(rel).parts
 b=base64.b64decode(pin['base64'],validate=True)
 assert hashlib.sha256(b).hexdigest()==pin['sha256'] and len(b)==pin['bytes']
 checked[rel]=(target,b,pin)
base.mkdir(parents=True,exist_ok=False)
for rel,(target,b,pin) in checked.items():
 target.parent.mkdir(parents=True,exist_ok=True); target.write_bytes(b)
 assert hashlib.sha256(target.read_bytes()).hexdigest()==pin['sha256']
r={'status':'ROOT_SMALL_SOURCE_DEPLOYED_BYTES_VERIFIED_NOT_STARTED','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'base':str(base),'members':{r:{'sha256':v[2]['sha256'],'bytes':v[2]['bytes']} for r,v in checked.items()},'new_inference':0,'new_fit':0,'new_training':0}
(base/'SOURCE_DEPLOYMENT.json').write_text(json.dumps(r,indent=2)+'\n')
print(json.dumps(r))
'''
    (OUT/'deploy_source_remote.py').write_text(remote,encoding='utf-8')
    command='env CUDA_VISIBLE_DEVICES= PYTHONDONTWRITEBYTECODE=1 taskset -c 110 ionice -c 3 nice -n 10 /workspace/guardfed_envs/celeba-cu128-20261009/bin/python -B -c '+shlex.quote(remote)
    cp=subprocess.run(['ssh','-p','60350','-o','BatchMode=yes','-o','ConnectTimeout=15','root@89.22.197.55',command],input=json.dumps(payload).encode(),capture_output=True,timeout=60)
    (OUT/'DEPLOY_STDOUT.txt').write_bytes(cp.stdout); (OUT/'DEPLOY_STDERR.txt').write_bytes(cp.stderr)
    if cp.returncode:
        save(OUT/'DEPLOY_FAILURE.json',{'status':'SOURCE_DEPLOY_FAILURE_PRESERVED_NO_RETRY','exit':cp.returncode})
        print(cp.stderr.decode(errors='replace')); raise SystemExit(cp.returncode)
    result=json.loads(cp.stdout)
    assert result['members']=={r:{'sha256':sha(b),'bytes':len(b)} for r,b in members.items()}
    save(OUT/'SOURCE_DEPLOYMENT.json',result)
    print(json.dumps({'status':result['status'],'members':len(members),'bytes':sum(map(len,members.values())),'review_sha256':sha(members['ROOT_SOURCE_REVIEW.json'])}))

if __name__=='__main__': main()
