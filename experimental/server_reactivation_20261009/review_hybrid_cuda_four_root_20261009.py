"""Review a prepared CUDA gate; approve four canaries, not the draft32 search."""
from pathlib import Path
import ast
import datetime
import hashlib
import json

ROOT=Path(__file__).resolve().parents[1]
PACK=ROOT/'tmp/celeba_hybrid_gpu_prepared_20261009'
OUT=ROOT/'tmp/celeba_hybrid_cuda_four_root_approval_20261009'
def read(p):return json.loads(p.read_text(encoding='utf8'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
assert sha(PACK/'FILES_SHA256.json')=='bccf8332e86b2a87af14ee81d7f6a610757bb02bf3afa331d3948980873c52b2'
files=read(PACK/'FILES_SHA256.json')['files']
assert len(files)==64
for name,digest in files.items():assert sha(PACK/name)==digest,name
reuse=read(PACK/'source_reuse.json')
for name,digest in reuse['original_five_snapshot_hashes'].items():
    assert sha(PACK/'scientific_snapshot'/name)==digest
    assert sha(ROOT/'tmp/celeba_hybrid_realimage_gate_20261009/sealed_group_a_snapshot'/name)==digest
def tree(p):return {n.name:n for n in ast.parse(p.read_text(encoding='utf8')).body if isinstance(n,ast.FunctionDef)}
old,new=tree(PACK/'original_writer_policy.py'),tree(PACK/'writer_policy.py')
for name in ('check_sidecar','require'):
    assert ast.dump(old[name],include_attributes=False)==ast.dump(new[name],include_attributes=False)
assert [ast.dump(n,include_attributes=False) for n in old['sanitize_result'].body[4:]] == [
    ast.dump(n,include_attributes=False) for n in new['sanitize_result'].body[4:]]
checks=read(PACK/'selfcheck.json')
assert checks['check_count']==35 and checks['GPU_calls']==checks['image_loads']==checks['new_training']==0
gate,screen=read(PACK/'gate_scope.json'),read(PACK/'screen_scope.json')
assert len(gate['jobs'])==4 and len(screen['jobs'])==32
cpu=read(ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/HYBRID_REPAIRED_TWO_ROOT_VERIFICATION.json')
assert cpu['aggregate_four_canaries_accepted'] and cpu['scientific_table_records']==0
OUT.mkdir(exist_ok=False)
proof=dict(status='ROOT_REVIEW_APPROVES_FOUR_CUDA_CANARIES_ONLY',
    verified_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    prepared_seal_sha256=sha(PACK/'FILES_SHA256.json'),prepared_members_verified=64,
    cpu_four_root_proof_sha256=sha(ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/HYBRID_REPAIRED_TWO_ROOT_VERIFICATION.json'),
    exact_gate_ids=[r['id'] for r in gate['jobs']],max_processes=1,cpu_threads=1,
    allowed_cpus=[104],nice=10,idle_io=True,cuda_visible_device='0',
    gpu_uuid='GPU-da357477-30a7-fddc-344b-a20513b9a2d0',
    requires_fresh_resource_proof_within90s=True,requires_new_empty_execution_copy=True,
    original_scientific_five_unchanged=True,writer_visitor_bits_reason_AST_exact=True,
    real_CUDA_gate_not_CPU_equivalence=True,screen32_authorized=False,formal100_authorized=False,
    test_authorized=False,automatic_retry_authorized=False,
    execution_authorized_within_existing_user_request=True)
(OUT/'APPROVED.json').write_text(json.dumps(proof,indent=2)+'\n',encoding='utf8')
(OUT/'APPROVED.sha256').write_text(sha(OUT/'APPROVED.json')+'  APPROVED.json\n',encoding='utf8')
print(json.dumps(dict(status=proof['status'],approval_sha256=sha(OUT/'APPROVED.json'))))
