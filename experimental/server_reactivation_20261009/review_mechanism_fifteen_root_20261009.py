"""Review an exact fifteen-terminal evaluation delta; no image inference."""
from pathlib import Path
import ast
import datetime
import hashlib
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / 'tmp/celeba_mechanism_valid_incremental_v2_20261009'
OUT = ROOT / 'tmp/celeba_mechanism_valid_incremental_v2_root_approval_20261009'
OUT.mkdir(exist_ok=False)
def read(p): return json.loads(p.read_text(encoding='utf8'))
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(PACK/'FILES_SHA256.json') == '70d0d920c4c5351c42efc9968fe3c38eed431d208b94bc8af486ba49d869a42d'
seal = read(PACK/'FILES_SHA256.json')
assert len(seal['members']) == 19
for row in seal['members']:
    p = PACK / row['path']
    assert sha(p) == row['sha256'] and p.stat().st_size == row['size']
old = ROOT/'tmp/celeba_mechanism_valid_replay_20261009/bridge.py'
def functions(p):
    text = p.read_text(encoding='utf8')
    return {n.name:(ast.get_source_segment(text,n),ast.dump(n,include_attributes=False))
            for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
original, current = functions(old), functions(PACK/'bridge.py')
for name in ('bind_runtime','full_reference','reference_baseline_full'):
    assert original[name] == current[name], name
subprocess.run([sys.executable,str(PACK/'selfcheck.py'),'--output-dir',str(OUT/'selfcheck')],check=True)
check = read(OUT/'selfcheck/selfcheck.json')
assert check['rejection_count'] == 36 and not check['image_inference_performed']
scope = read(PACK/'SCOPE.json')
assert len(scope['selected_ids']) == len(set(scope['selected_ids'])) == 15
assert not set(scope['selected_ids']).intersection(scope['already_closed_replay_ids'])
assert scope['compute_threads'] == 8 and scope['max_processes'] == 1
assert not scope['new_training'] and not scope['new_full_inference'] and not scope['final_test_dispatch']
proof = dict(status='ROOT_REVIEW_PASS_BOUNDED_FIFTEEN_VALID_REPLAY',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_seal_sha256=sha(PACK/'FILES_SHA256.json'), scope_sha256=sha(PACK/'SCOPE.json'),
    inventory_sha256=scope['inventory_sha256'],bridge_sha256=scope['bridge_sha256'],
    selected_ids=scope['selected_ids'],closed_eight_excluded=True,compute_threads=8,
    max_processes=1,allowed_cpus=list(range(112,120)),nice=10,idle_io=True,
    scientific_bind_runtime_source_and_ast_exact=True,root_local_rejection_checks=36,
    native_tolerance=1e-12,new_training=False,new_Full_inference=False,test=False,
    activation_requires_fresh_resource_identity_and_empty_outputs=True,
    execution_authorized_within_existing_user_request=True,automatic_retry_authorized=False)
(OUT/'APPROVED.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
(OUT/'APPROVED.sha256').write_text(sha(OUT/'APPROVED.json')+'  APPROVED.json\n',encoding='utf8')
print(json.dumps(dict(status=proof['status'],approved_sha256=sha(OUT/'APPROVED.json'),selected=15)))
