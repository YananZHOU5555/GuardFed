"""Bind one frozen exact4 Hybrid delta to the accepted23 parent; preserve original strict loop."""
from pathlib import Path
import ast,difflib,hashlib,json
B=Path(__file__).resolve().parent;H=B.parent/'celeba_hybrid_screen_execution_20261009';P=H/'accepted_delta_after22_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,value):
    with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def write(name,text):
    with (B/name).open('x',encoding='utf8',newline='\n') as f:f.write(text)
parent=P/'collect_once.py';assert sha(parent)=='6679e4b8594f730ebe09abac81992e64c7d5796fca675557549f1ba755a47ccf'
previous=read(B/'PREVIOUS_CHAIN.json');delta=read(B/'EXACT_DELTA.json');ids=delta['selected_ids']
assert previous['accepted_total']==23 and len(ids)==4 and not set(ids)&set(previous['accepted_job_ids'])
assert delta['eligible_for_single_strict_batch'] and delta['new_completed']==4
assert sha(H/'LATEST_BACKUP.json')==sha(B/'PREVIOUS_LATEST.json')
assert sha(H/read(B/'PREVIOUS_LATEST.json')['chain_file'])==sha(B/'PREVIOUS_CHAIN.json')==delta['previous_chain_sha256']
assert sha(Path(previous['root_adoption_path']))==previous['root_adoption_sha256']
assert sha(P/'OFFSERVER_MEMBER_TENSOR_PROOF.json')==previous['offserver_proof_sha256']
for rel,pin in read(H/'FILES_SHA256.json')['files'].items():assert sha(H/rel)==pin
old=parent.read_text('utf8');replacements=[
('7abf250cacde9968b7c06fc883623d3897dd06ba7c5db338c5f931734ceb97c4',sha(B/'PREVIOUS_CHAIN.json')),
('dbca388d5384d482bfbe84f13592742df38041d4989579c8091c68e71c59213e',sha(B/'AUTHORIZED_SNAPSHOT.json')),
("wanted=['CosineFairness_lam5.0_tau0.2_lr0.001_non-IID_Benign_seed91001_screen']",'wanted='+repr(ids)),
("previous['accepted_total']==22","previous['accepted_total']==23"),
('old_accepted=22,accepted_total=22+','old_accepted=23,accepted_total=23+'),
('accepted_delta_after22_20261010','accepted_delta_after23_20261010'),
('hybrid_after22_delta','hybrid_after23_delta')]
text=old
for before,after in replacements:assert before in text;text=text.replace(before,after)
compile(text,str(B/'collect_once.py'),'exec');reversed_text=text
for before,after in reversed(replacements):assert after in reversed_text;reversed_text=reversed_text.replace(after,before)
assert reversed_text==old
def loop(text):
    return next(n for n in ast.walk(ast.parse(text)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id=='e' and any(isinstance(v,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='result' for t in v.targets) for v in n.body))
assert ast.dump(loop(text),include_attributes=False)==ast.dump(loop(old),include_attributes=False)
write('collect_once.py',text)
write('COLLECTOR_DIFF.patch',''.join(difflib.unified_diff(old.splitlines(True),text.splitlines(True),fromfile=str(parent),tofile='collect_once.py')))
save('SOURCE_RECEIPT.json',dict(status='BOUND_METADATA_ONLY_ORIGINAL_SCIENTIFIC_BODY_BYTE_EXACT',parent=str(parent),parent_sha256=sha(parent),collector_sha256=sha(B/'collect_once.py'),replacements=replacements,reverse_all_bindings_source_exact=True,scientific_loop_AST_exact=True,source_members_verified=69,CPU=106,no_CNN=True))
transport=(P/'transport_collect_once.py').read_text('utf8');assert 'accepted_delta_after22_20261010' in transport
transport=transport.replace('accepted_delta_after22_20261010','accepted_delta_after23_20261010')
compile(transport,str(B/'transport_collect_once.py'),'exec');write('transport_collect_once.py',transport)
print(json.dumps(dict(status='EXACT4_COLLECTOR_READY_NOT_EXECUTED',collector_sha256=sha(B/'collect_once.py'),selected_ids=ids,previous_chain_sha256=sha(B/'PREVIOUS_CHAIN.json'))))
