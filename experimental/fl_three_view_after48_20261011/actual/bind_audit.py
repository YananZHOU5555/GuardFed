"""Bind actual new13 Linux/F proofs before the single zero-fit audit; no fit."""
from pathlib import Path
import ast,hashlib,json,sys
H=Path(__file__).resolve().parent;C=H.parent;S=C/'saved';R=C.parents[1];A=H/'saved001'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
sys.path.insert(0,str(S));import contract as k
assert not (S/'FILES_SHA256.json').exists() and not (S/'SOURCE_PINS.json').exists()
for rel,pin in read(S/'STATIC_SOURCE_SHA256.json')['files'].items():
 assert sha(S/rel)==pin['sha256'] and (S/rel).stat().st_size==pin['bytes']
linux=read(A/'LINUX_SAVED_CHECK.json');k.linux_proof(linux)
transport=read(A/'TRANSPORT_VERIFICATION.json')
assert transport['status']=='FLGMM13_F_TRANSPORT_ALL_MEMBERS_SHA_PASS_NOT_SCIENTIFIC_ACCEPTANCE' and transport['member_count']==30
assert transport['linux_proof_sha256']==sha(A/'LINUX_SAVED_CHECK.json') and transport['gate_result_sha256']==linux['gate_result_sha256']
q=read(S/'AUDIT_TEST_PINS.json');q['linux_sha256']=sha(A/'LINUX_SAVED_CHECK.json');q['transport_sha256']=sha(A/'TRANSPORT_VERIFICATION.json');q['exact_ids']=k.IDS
diag=R/q['setup_extractor'];node=next(n for n in ast.parse(diag.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='setup_nodes')
ns={'ast':ast};exec(compile(ast.Module(body=[node],type_ignores=[]),'<original setup extractor only>','exec'),ns)
prefix=ns['setup_nodes']((S/'verify_arrays.py').read_text());q['setup_AST_sha256']=hashlib.sha256(ast.dump(ast.Module(body=prefix,type_ignores=[]),include_attributes=False).encode()).hexdigest()
for p in [S/'verify_arrays.py',S/'contract.py',S/'FIXED_IDS.json',S/'STATIC_SOURCE_SHA256.json',C/'MANIFEST.json',C/'FILES_SHA256.json',C/'originals/replay.py',C/'originals/core.py',A/'LINUX_SAVED_CHECK.json',A/'TRANSPORT_VERIFICATION.json',R/'tmp/fl_native_after54_20261011/ROOT_ADOPTION_REVIEW.json',R/'tmp/celeba_flgmm_closed47_root_execution_20261011/ROOT_SCIENTIFIC_ADOPTION.json']:
 q['files'][p.relative_to(R).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size}
assert q['files']['tmp/celeba_flgmm_closed47_root_execution_20261011/saved_acceptance_actual001/OFFSERVER_ARRAY_REFIT_CHECK.failure.json']['sha256']==q['original_failure_sha256']=='c93d11ee41400659f74597beb1ca0cab19a64b647c805ee0534051fb55f41812'
q['scope']='Only saved outputs for the actual new13. Original whole Linux check supplies root-refit evidence; previous Windows47 refit remains failed. No Windows recalibration claim.'
with (S/'SOURCE_PINS.json').open('x',encoding='utf8') as f:json.dump(q,f,indent=2);f.write('\n')
files={p.name:{'sha256':sha(p),'bytes':p.stat().st_size} for p in sorted(S.iterdir()) if p.is_file()}
with (S/'FILES_SHA256.json').open('x',encoding='utf8') as f:json.dump({'files':files,'scope':q['scope']},f,indent=2);f.write('\n')
print(json.dumps({'source_seal_sha256':sha(S/'FILES_SHA256.json'),'actual_linux_sha256':q['linux_sha256'],'actual_transport_sha256':q['transport_sha256'],'records':13,'new_fit':0}))
