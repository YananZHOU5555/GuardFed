"""One local source-only derivation; no runtime, arrays, SSH or scientific imports."""
from pathlib import Path
import ast,difflib,hashlib,io,json,re,tokenize
H=Path(__file__).resolve().parent; R=H.parents[1]
P=R/'tmp/celeba_remaining_F_FedSA10_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(name,value):
    with (H/name).open('x',encoding='utf8',newline='\n') as f:
        f.write(value if isinstance(value,str) else json.dumps(value,ensure_ascii=False,indent=2)+'\n')
old=read(P/'PREPARED.json'); raw=read(P/'RAW_STORAGE_INDEX.json')
root=R/'tmp/celeba_mechanism_remaining_F_FedSA10_root_adoption_20261011/ROOT_ADOPTION.json'
assert sha(root)=='62475bb89325d0b229977629bb879d99994c4d313cb7924db9a5efbce6c9ec25'
rp=read(root); index=R/rp['records_index_path']; assert sha(index)==rp['records_index_sha256']
assert rp['cumulative_accepted']==len(read(index)['all_ids'])==330 and len(raw['all_transported_ids'])==150
assert rp['archive_sha256']==raw['archive_sha256'] and rp['offserver_proof_sha256']==raw['offserver_verification_sha256']
native=Path(old['required_native_root_path']); np=read(native)
assert sha(native)==old['required_native_root_sha256']=='654ce595db6ab06bf63c1c673c2da48c6e04aaee3c327fa44f4d3c3cac0a1830'
inspection=native.parent/'inspection/inspection.json'; ledger=native.parent/'verified_ledger.json'
assert sha(inspection)==np['inspection_sha256'] and sha(ledger)==np['ledger_sha256']
ids=[f'minus_F_IID_S-DFA_seed{s}' for s in range(91001,91011)]
known=[f'minus_F_IID_S-DFA_seed{s}' for s in (91002,91003,91004)]
assert set(ids)&set(np['new_ids'])==set(known)
assert len(read(inspection)['records'])==433 and len(read(ledger)['entries'])==47
prevremote='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/'+Path(raw['directory']).name+'/backup_receipt.json'
p=dict(status='SOURCE_ONLY_EXACT10_F_IID_SDFA_NATIVE_AND_SNAPSHOT_UNBOUND',candidate_ids=ids,
       prior_accepted=330,prior_transported=150,prior_root_path=str(root),prior_root_sha256=sha(root),
       prior_index_path=str(index),prior_index_sha256=sha(index),previous_local_receipt=raw['receipt'],
       previous_remote_receipt=prevremote,previous_receipt_sha256=raw['receipt_sha256'],
       previous_all_transported_ids=raw['all_transported_ids'],known_native_archive_roots=[dict(path=str(native),sha256=sha(native))],
       already_native_candidate_ids=known,required_new_native_candidate_ids=[i for i in ids if i not in known],
       native333_root_path=str(native),native333_root_sha256=sha(native),
       native333_inspection_path=str(inspection),native333_inspection_sha256=sha(inspection),
       native333_ledger_path=str(ledger),native333_ledger_sha256=sha(ledger),
       replay_target_total=340,native_minimum_total=340,native_target_total=None,
       native_binding=None,authorized_snapshot_path=None,authorized_snapshot_sha256=None,
       required_native_root_path=None,required_native_root_sha256=None,
       native_extra_records_are_not_replayed=True,execution_performed=False)
put('PREPARED.json',p)
names=[n for n in read(P/'SOURCE_FILES_SHA256.json')['files'] if n.endswith('.py')]
for name in names:
    assert sha(P/name)==read(P/'SOURCE_FILES_SHA256.json')['files'][name]['sha256']
unchanged={'execute_once.py','remote_cpu111_export.py','cpu_release_readonly.py'}
numbermap={'320':'330','330':'340','420':'433','140':'150','150':'160','46':'47','47':'48'}
texts={}
for name in names:
    original=(P/name).read_text('utf8'); s=original
    if name not in unchanged:
        tokens=[]
        for token in tokenize.generate_tokens(io.StringIO(s).readline):
            if token.type==tokenize.NUMBER and token.string in numbermap: token=token._replace(string=numbermap[token.string])
            tokens.append(token)
        s=tokenize.untokenize(tokens)
        # Namespace and human scope substitutions are explicit, not number-prefix replacements.
        for a,b in [('celeba_remaining_F_FedSA10_20261011','celeba_remaining_F_SDFA10_prepared_20261011'),
                    ('celeba_mechanism_remaining_F_FedSA10_root_adoption_20261011','celeba_mechanism_remaining_F_SDFA10_root_adoption_20261011'),
                    ('minus_F_IID_FedSA_seed','minus_F_IID_S-DFA_seed'),('F_FedSA10_','F_SDFA10_'),
                    ('F FedSA','F S-DFA'),('F_FEDSA10','F_SDFA10'),('native320','native333'),
                    ('prior320','prior330'),('old320','old330'),('old420','old433'),('old46','old47'),
                    ('original320','original330'),('MECHANISM330_INDEX','MECHANISM340_INDEX'),
                    ('REPLAY330','REPLAY340'),('MIN333','MIN340'),('native >=330','native >=340'),
                    ('native total>=330','native total>=340'),('three-view330','three-view340'),
                    ('old320','old330')]:s=s.replace(a,b)
        s=s.replace(old['previous_receipt_sha256'],raw['receipt_sha256'])
        s=s.replace(old['prior_root_sha256'],sha(root)).replace(old['prior_index_sha256'],sha(index))
    if name=='remote_preflight.py':
        s=s.replace(old['previous_remote_receipt'],prevremote)
        s=s.replace(repr(old['previous_all_transported_ids']),repr(raw['all_transported_ids']))
    if name=='download_verify_once.py':
        s=s.replace("tmp/celeba_remaining_F_FFlip10_20261011/RAW_STORAGE_INDEX.json","tmp/celeba_remaining_F_FedSA10_20261011/RAW_STORAGE_INDEX.json")
    if name=='adopt_by_root.py':
        s=s.replace("row['attack']=='FedSA'","row['attack']=='S-DFA'")
        # Preserve all previously complete scenes; append only the newly closed S-DFA scene.
        s=s.replace("complete_F_scenes=[['IID','Benign'],['IID','F Flip'],['IID','FedSA']]", "complete_F_scenes=[['IID','Benign'],['IID','F Flip'],['IID','FedSA'],['IID','S-DFA']]")
        s=s.replace("parent_adopter_path='tmp/celeba_remaining_F_FFlip10_20261011/adopt_by_root.py',parent_adopter_sha256=sha(R/'tmp/celeba_remaining_F_FFlip10_20261011/adopt_by_root.py')", "parent_adopter_path='tmp/celeba_remaining_F_FedSA10_20261011/adopt_by_root.py',parent_adopter_sha256=sha(R/'tmp/celeba_remaining_F_FedSA10_20261011/adopt_by_root.py')")
        last="assert {int(i[-5:]) for i in index['all_ids'] if i.startswith('minus_F_IID_S-DFA_seed')}==set(range(91001,91011))"
        assert last in s
        s=s.replace(last,"assert {int(i[-5:]) for i in index['all_ids'] if i.startswith('minus_F_IID_FedSA_seed')}==set(range(91001,91011))\n"+last)
    if name=='verify_native_inputs.py':
        s=s.replace("assert e['native_root_path']==p['required_native_root_path'] and e['native_root_sha256']==p['required_native_root_sha256'] and e['root_native_total']==333", "assert e['root_native_total']>=340 and p['native_binding'] is None\n    assert e['native_root_path']!=p['native333_root_path']")
        s=s.replace("e['root_native_total']-330", "e['root_native_total']-333")
        s=s.replace("assert set(p['candidate_ids'])<=set(proof['new_ids'])\n    assert set(p['candidate_ids'])&{r['id'] for r in old['records']}==set()", "assert set(p['required_new_native_candidate_ids'])<=set(proof['new_ids'])\n    assert set(p['candidate_ids'])&{r['id'] for r in old['records']}==set(p['already_native_candidate_ids'])\n    assert p['already_native_candidate_ids']==[f'minus_F_IID_S-DFA_seed{s}' for s in (91002,91003,91004)]\n    assert p['required_new_native_candidate_ids']==[i for i in p['candidate_ids'] if i not in p['already_native_candidate_ids']]")
        s=s.replace("row['attack']=='FedSA'", "row['attack']=='S-DFA'")
        s=s.replace('assert len(roots)==1','assert len(roots)==2')
        s=s.replace('    return e\n', "    assert sha(e['authorized_snapshot_path'])==e['authorized_snapshot_sha256']\n    snapshot=read(e['authorized_snapshot_path'])\n    assert set(p['candidate_ids'])<=set(snapshot['remaining620']['remote_closed_ids'])\n    assert not snapshot['remaining620']['failures'] and not snapshot['source_data_binding']['changed_paths']\n    return e\n")
    if name=='bind_native.py':
        s=s.replace("p=read(H/'PREPARED.json')", "p=read(H/'PREPARED.json')\n    snapshot=a.snapshot.resolve();assert snapshot.is_file() and sha(snapshot)==a.snapshot_sha256\n    assert snapshot.is_relative_to(R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009')")
        s=s.replace("root_native_total=proof['total_new_strict_and_offserver']", "authorized_snapshot_path=str(snapshot),authorized_snapshot_sha256=a.snapshot_sha256,\n        root_native_total=proof['total_new_strict_and_offserver']")
        s=s.replace("p.add_argument('--prepared-seal-sha256',required=True);main(p.parse_args())", "p.add_argument('--prepared-seal-sha256',required=True)\n    p.add_argument('--snapshot',type=Path,required=True);p.add_argument('--snapshot-sha256',required=True);main(p.parse_args())")
    texts[name]=s
    if name in unchanged:
        with (H/name).open('xb') as f:f.write((P/name).read_bytes())
    else:put(name,s)
put('PARENT_SOURCE_PINS.json',dict(parent=str(P.relative_to(R)),parent_seal_sha256=sha(P/'SOURCE_FILES_SHA256.json'),
     files={n:dict(sha256=sha(P/n),bytes=(P/n).stat().st_size) for n in names},parent_root_sha256=sha(root),
     parent_index_sha256=sha(index),native333_sha256=sha(native),no_parent_science_reexecuted=True))
diff=''.join(''.join(difflib.unified_diff((P/n).read_text('utf8').splitlines(True),(H/n).read_text('utf8').splitlines(True),fromfile='FedSA10/'+n,tofile='SDFA10/'+n)) for n in names)
put('SOURCE_DIFF.patch',diff)
print(json.dumps(dict(status='SOURCE_DERIVED_NOT_BOUND_OR_EXECUTED',python_files=len(names),prior_root_sha256=sha(root),prior_index_sha256=sha(index))))
