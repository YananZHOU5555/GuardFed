"""Local-only preparation of the original after288 pipeline; no scientific execution."""
from pathlib import Path
import ast, difflib, hashlib, json

H=Path(__file__).resolve().parent; R=H.parents[1]
O=R/'tmp/celeba_remaining_after288_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,obj):
    with (H/name).open('x',encoding='utf8',newline='\n') as f:
        f.write(json.dumps(obj,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def replace(s,a,b):
    assert a in s,a
    return s.replace(a,b)

assert not (H/'PREPARED.json').exists()
assert sha(O/'SOURCE_FILES_SHA256.json')=='a6e7f307c8bb0adea1bb7dc6bdc4af425bfaa6d336eaa928fadc0dab3195f6bf'
for n,v in read(O/'SOURCE_FILES_SHA256.json')['files'].items():
    assert sha(O/n)==v['sha256'] and (O/n).stat().st_size==v['bytes']
P=read(O/'PREPARED.json'); oldP=dict(P); raw=read(O/'RAW_STORAGE_INDEX.json')
parent=R/'tmp/celeba_mechanism_remaining_after288_root_adoption_20261011'
assert sha(parent/'ROOT_ADOPTION.json')=='70689e63467d8866caa3beb06d2be5f911defd0a4d5f7f061ab005d20bd1c1bb'
assert sha(parent/'MECHANISM295_INDEX.json')=='7c46fcb20c15c377b1df17378f14b6282ee394bdcaacecd8d4f92be344777bef'
root=read(parent/'ROOT_ADOPTION.json'); index=read(parent/'MECHANISM295_INDEX.json')
ids=[f'minus_A_non-IID_Sp-DFA_seed{s}' for s in range(91006,91011)]
assert len(index['all_ids'])==len(set(index['all_ids']))==root['cumulative_accepted']==295
assert set(ids).isdisjoint(index['all_ids']) and len(raw['all_transported_ids'])==115
native=R/root['native_root_path']; np=read(native)
assert sha(native)==root['native_root_sha256']=='b9bc8099013b6cc7da17f0b72e60306d384549a1bac2da6a73b5e04ed4fa2128'
for field in list(P):
    if field.startswith(('native288','known_native288','pending_native295','authorized_snapshot','future_native')): del P[field]
P.update(status='SOURCE_PREPARED_EXACT5_WAITING_ACTUAL_NATIVE_ROOT_NO_EXECUTION',candidate_ids=ids,
    prior_accepted=295,prior_transported=115,prior_root_path=str(parent/'ROOT_ADOPTION.json'),prior_root_sha256=sha(parent/'ROOT_ADOPTION.json'),
    prior_index_path=str(parent/'MECHANISM295_INDEX.json'),prior_index_sha256=sha(parent/'MECHANISM295_INDEX.json'),
    previous_local_receipt=raw['receipt'],previous_remote_receipt='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/'+Path(raw['directory']).name+'/backup_receipt.json',
    previous_receipt_sha256=raw['receipt_sha256'],previous_all_transported_ids=raw['all_transported_ids'],
    native295_root_path=str(native),native295_root_sha256=sha(native),
    native295_inspection_path=str(native.parent/'inspection/inspection.json'),native295_inspection_sha256=np['inspection_sha256'],
    native295_ledger_path=str(native.parent/'verified_ledger.json'),native295_ledger_sha256=np['ledger_sha256'],
    replay_target_total=300,native_target_total=None,native_extra_records_are_not_replayed=True)
assert len(read(P['native295_inspection_path'])['records'])==395
assert len(read(P['native295_ledger_path'])['entries'])==42
save('PREPARED.json',P)
names=['bind_native.py','verify_native_inputs.py','execute_once.py','remote_preflight.py','remote_cpu111_export.py','export_once.py','download_verify_once.py','join_saved.py','cpu_release_readonly.py']
diff=[]; checks={}
for n in names:
    original=(O/n).read_text('utf8'); s=original
    if n=='bind_native.py':
        s=replace(s,'native295 proof','native N>=300 proof (replay remains exact5)')
        s=replace(s,'root_native_total=295,transport_target_only=7,proposed_replay_total=295','root_native_total=proof[\'total_new_strict_and_offserver\'],transport_target_only=5,proposed_replay_total=300')
        s=replace(s,'ACTUAL_NATIVE295_BOUND_NOT_TRANSPORTED_NOT_ADOPTED','ACTUAL_NATIVE_PROOF_BOUND_EXACT5_NOT_TRANSPORTED_NOT_ADOPTED')
    elif n=='verify_native_inputs.py':
        s=replace(s,'native295/old288','native N>=300 / old295; replay exact5')
        s=replace(s,'native288','native295')
        s=replace(s,"proof['total_new_strict_and_offserver']==295","proof['total_new_strict_and_offserver']==e['root_native_total']>=300")
        s=replace(s,"assert len(proof['new_ids'])==7 and set(proof['new_ids'])==set(p['pending_native295_ids'])", "assert len(proof['new_ids'])==len(set(proof['new_ids']))==e['root_native_total']-295\n    assert set(p['candidate_ids'])<=set(proof['new_ids'])\n    assert p['candidate_ids']==[f'minus_A_non-IID_Sp-DFA_seed{s}' for s in range(91006,91011)]\n    assert e['transport_target_only']==5 and e['proposed_replay_total']==300")
        s=replace(s,"len(old['records'])==388 and len(inspection['records'])==395","len(old['records'])==395 and len(inspection['records'])==100+e['root_native_total']")
        s=replace(s,"len(oldledger['entries'])==41", "len(oldledger['entries'])==42")
        s=replace(s,"len(ledger['entries'])==42", "len(ledger['entries'])==43")
        s=replace(s,"len(prior['all_ids'])==288", "len(prior['all_ids'])==295")
        s=replace(s,"row['attack'] in ('S-DFA','Sp-DFA')", "row['attack']=='Sp-DFA'")
    elif n=='remote_preflight.py':
        s=replace(s,'exact7','exact5'); s=replace(s,'candidate_ids='+repr(oldP['candidate_ids']),'candidate_ids='+repr(ids))
        s=replace(s,'len(ids)==7','len(ids)==5')
        s=replace(s,oldP['previous_remote_receipt'],P['previous_remote_receipt'])
        s=replace(s,oldP['previous_receipt_sha256'],P['previous_receipt_sha256'])
        s=replace(s,repr(oldP['previous_all_transported_ids']),repr(P['previous_all_transported_ids']))
    elif n=='export_once.py':
        s=replace(s,'A4 export','exact5 A export'); s=replace(s,"len(pre['selected_ids'])==7","len(pre['selected_ids'])==5")
        s=replace(s,"tag='after288_'","tag='after295_'")
    elif n=='download_verify_once.py':
        s=replace(s,'assert N==7','assert N==5');s=replace(s,"STORAGE_ROOT/'celeba_remaining_after288_20261011'","STORAGE_ROOT/'celeba_remaining_after295_20261011'")
        s=replace(s,'tmp/celeba_remaining620_after280_transport_20261011/RAW_STORAGE_INDEX.json','tmp/celeba_remaining_after288_20261011/RAW_STORAGE_INDEX.json')
        s=replace(s,oldP['previous_receipt_sha256'],P['previous_receipt_sha256']);s=replace(s,'cumulative=108+N','cumulative=115+N')
    elif n=='join_saved.py':
        s=replace(s,'actual native295 archive','actual adopted native archive (N>=300; only exact5 A replayed)')
        s=replace(s,'assert N==7','assert N==5');s=replace(s,"len(prior['all_ids'])==288","len(prior['all_ids'])==295")
        s=replace(s,"currentproof['total_new_strict_and_offserver']==295","currentproof['total_new_strict_and_offserver']==E['root_native_total']>=300")
        s=replace(s,'native288','native295');s=replace(s,"len(oldinspection['records'])==388","len(oldinspection['records'])==395")
        s=replace(s,'==288+N','==295+N');s=replace(s,'allids[:288]','allids[:295]');s=replace(s,'cumulative=288+N','cumulative=295+N')
        s=replace(s,'EXACT7','EXACT5');s=replace(s,'prior288_objects','prior295_objects');s=replace(s,'old388_native','old395_native')
        s=replace(s,"E=read(H/'EXECUTION_INPUTS.json');P=", "from verify_native_inputs import verify_native_inputs\n E=verify_native_inputs();P=")
    (H/n).write_bytes((O/n).read_bytes() if s==original else s.encode('utf8'))
    compile(s,str(H/n),'exec')
    checks[n]=dict(original_sha256=sha(O/n),new_sha256=sha(H/n),source_text_exact=s==original,compile=True)
    diff.extend(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='after288/'+n,tofile='after295/'+n))
original=(O/'join_saved.py').read_text('utf8');current=(H/'join_saved.py').read_text('utf8')
def record_body(s):
    return next(ast.get_source_segment(s,n) for n in ast.walk(ast.parse(s)) if isinstance(n,ast.For) and ast.unparse(n.target)=='rec' and ast.unparse(n.iter)=='records')
assert record_body(original)==record_body(current)
assert ast.dump(ast.parse(record_body(original)),include_attributes=False)==ast.dump(ast.parse(record_body(current)),include_attributes=False)
(H/'SOURCE_DIFF.patch').write_text(''.join(diff),encoding='utf8',newline='\n')
save('SOURCE_CHECK.json',dict(status='LOCAL_SOURCE_PREPARATION_PASS_NOT_EXECUTED',checks=checks,original_per_record_source_and_AST_exact=True,parent295_index_sha256=P['prior_index_sha256'],parent295_root_sha256=P['prior_root_sha256'],old395_native_records=395,old_native_ledger_entries=42,replay_new_ids=ids,replay_target=300,native_target_unbound=True,science_or_SSH_execution=0))
print(json.dumps(dict(status='SOURCE_PREPARED_NOT_EXECUTED',files=len(names),exact_per_record=True)))
