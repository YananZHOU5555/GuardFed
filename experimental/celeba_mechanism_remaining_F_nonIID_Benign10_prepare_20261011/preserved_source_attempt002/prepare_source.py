"""Derive a source-only exact10 saved-output increment; never imports scientific code."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/celeba_remaining_F_SpDFA10_prepared_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,v):
    raw=v if isinstance(v,bytes) else (v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n').encode('utf8')
    with (H/n).open('xb') as f:f.write(raw)
def replace(s,a,b):
    assert a in s,repr(a)
    return s.replace(a,b)
assert not (H/'PREPARED.json').exists()
assert sha(P/'SOURCE_FILES_SHA256.json')=='ef2c4b306268ee6adad78ccdd1da90eeb67fddb576aedaf72fccdb7fd5fad5eb'
parentseal=read(P/'SOURCE_FILES_SHA256.json');old=read(P/'PREPARED.json');raw=read(P/'RAW_STORAGE_INDEX.json')
root=R/'tmp/celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011/ROOT_ADOPTION.json'
assert sha(root)=='3f7b436e0b2fec0401d7bf6117d6dca48aef21c185b6f170d37954e2cafee5a4'
rp=read(root);index=R/rp['records_index_path'];assert sha(index)==rp['records_index_sha256']
assert rp['cumulative_accepted']==len(read(index)['all_ids'])==350
assert len(raw['all_transported_ids'])==170 and raw['archive_sha256']==rp['archive_sha256']
assert raw['offserver_verification_sha256']==rp['offserver_proof_sha256']
native=R/rp['native_root_path'];assert sha(native)==rp['native_root_sha256']
np=read(native);inspection=native.parent/'inspection/inspection.json';ledger=native.parent/'verified_ledger.json'
assert np['root_adopted'] and np['total_new_strict_and_offserver']==350 and not np['test']
assert sha(inspection)==np['inspection_sha256'] and sha(ledger)==np['ledger_sha256']
assert len(read(inspection)['records'])==450 and len(read(ledger)['entries'])==49
ids=[f'minus_F_non-IID_Benign_seed{s}' for s in range(91001,91011)]
assert not set(ids)&set(read(index)['all_ids']) and not set(ids)&set(read(inspection)['accepted_new_ids'])
prevremote='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/'+Path(raw['directory']).name+'/backup_receipt.json'
p=dict(status='SOURCE_ONLY_EXACT10_F_NONIID_BENIGN_PENDING_NATIVE_ROOT_AND_SNAPSHOT',candidate_ids=ids,
 prior_accepted=350,prior_transported=170,prior_root_path=str(root),prior_root_sha256=sha(root),prior_index_path=str(index),prior_index_sha256=sha(index),
 previous_local_receipt=raw['receipt'],previous_remote_receipt=prevremote,previous_receipt_sha256=raw['receipt_sha256'],previous_all_transported_ids=raw['all_transported_ids'],
 known_native_archive_roots=[],already_native_candidate_ids=[],required_new_native_candidate_ids=ids,
 native350_root_path=str(native),native350_root_sha256=sha(native),native350_inspection_path=str(inspection),native350_inspection_sha256=sha(inspection),native350_ledger_path=str(ledger),native350_ledger_sha256=sha(ledger),
 replay_target_total=360,native_target_minimum=360,native_binding=None,authorized_snapshot_path=None,authorized_snapshot_sha256=None,
 required_native_root_path=None,required_native_root_sha256=None,native_extra_records_are_not_replayed=True,execution_performed=False)
put('PREPARED.json',p)
names=['bind_native.py','cpu_release_readonly.py','download_verify_once.py','execute_once.py','export_once.py','finalize_actual_handoff.py','join_saved.py','remote_cpu111_export.py','remote_preflight.py','verify_native_inputs.py']
unchanged={'execute_once.py','remote_cpu111_export.py','cpu_release_readonly.py'}
changes={}
for n in names:
    assert sha(P/n)==parentseal['files'][n]['sha256']
    original=(P/n).read_bytes();s=original.decode('utf8')
    if n not in unchanged:
        for a,b in [('celeba_remaining_F_SpDFA10_prepared_20261011',H.name),('F_SPDFA10_','F_NONIID_BENIGN10_'),('F Sp-DFA','F non-IID Benign'),('native =350','native >=360'),('native total=350','native total>=360'),('three-view350','three-view360'),('EXACT350','MIN360'),('native341','native350'),('prior340','prior350'),('old441','old450'),('old48','old49'),('old340','old350')]:
            s=s.replace(a,b)
    if n=='bind_native.py':
        s=replace(s,'proposed_replay_total=350','proposed_replay_total=360')
    elif n=='remote_preflight.py':
        s=replace(s,repr([f'minus_F_IID_Sp-DFA_seed{x}' for x in range(91001,91011)]),repr(ids))
        s=replace(s,old['previous_remote_receipt'],prevremote)
        s=replace(s,old['previous_receipt_sha256'],raw['receipt_sha256'])
        s=replace(s,repr(old['previous_all_transported_ids']),repr(raw['all_transported_ids']))
    elif n=='download_verify_once.py':
        s=replace(s,'tmp/celeba_remaining_F_SDFA10_prepared_20261011/RAW_STORAGE_INDEX.json','tmp/celeba_remaining_F_SpDFA10_prepared_20261011/RAW_STORAGE_INDEX.json')
        s=replace(s,old['previous_receipt_sha256'],raw['receipt_sha256'])
        s=replace(s,'cumulative=160+N','cumulative=170+N')
    elif n=='join_saved.py':
        s=replace(s,"len(prior['all_ids'])==340","len(prior['all_ids'])==350")
        s=replace(s,"E['root_native_total']==350","E['root_native_total']>=360")
        s=replace(s,"len(oldinspection['records'])==441","len(oldinspection['records'])==450")
        s=replace(s,'==340+N and allids[:340]','==350+N and allids[:350]')
        s=replace(s,'cumulative=340+N','cumulative=350+N')
    elif n=='finalize_actual_handoff.py':
        for a,b in [("idx['all_ids'][:340]","idx['all_ids'][:350]"),("len(set(idx['all_ids']))==350","len(set(idx['all_ids']))==360"),("len(raw['all_transported_ids'])==170","len(raw['all_transported_ids'])==180"),("raw['all_transported_ids'][:160]","raw['all_transported_ids'][:170]"),("E['root_native_total']==350","E['root_native_total']>=360"),('prior_root_accepted_replays=340','prior_root_accepted_replays=350'),('replay_total_if_root_adopts=350','replay_total_if_root_adopts=360'),('prior_transported=160,total_transported=170','prior_transported=170,total_transported=180'),('proposed_total\':350','proposed_total\':360')]:s=replace(s,a,b)
    if n=='verify_native_inputs.py':
        # Only the native/snapshot metadata binder changes. Original transport and saved science are not imported here.
        start=s.index('    assert e[\'root_native_total\']');end=s.index('    prior=read(',start)
        s=s[:start]+'''    assert e['root_native_total']>=p['native_target_minimum']==360 and p['native_binding'] is None
    assert p['required_native_root_path'] is None and p['required_native_root_sha256'] is None
    assert e['native_root_path']!=p['native350_root_path']
    assert proof['root_adopted'] is True and proof['total_new_strict_and_offserver']==e['root_native_total'] and proof['test'] is False
    assert proof['inspection_sha256']==e['native_inspection_sha256'] and proof['ledger_sha256']==e['native_ledger_sha256']
    assert proof['previous_ledger_sha256']==p['native350_ledger_sha256']
    assert len(proof['new_ids'])==len(set(proof['new_ids']))==e['root_native_total']-350
    assert set(p['required_new_native_candidate_ids'])<=set(proof['new_ids'])
    assert p['already_native_candidate_ids']==[] and p['known_native_archive_roots']==[]
    assert not set(p['candidate_ids'])&{r['id'] for r in old['records']}
    assert p['required_new_native_candidate_ids']==p['candidate_ids']
    assert not set(proof['new_ids'])&{r['id'] for r in old['records']}
    assert p['candidate_ids']==[f'minus_F_non-IID_Benign_seed{s}' for s in range(91001,91011)]
    assert e['transport_target_only']==10 and e['proposed_replay_total']==360
    assert len(old['records'])==450 and len(inspection['records'])==100+e['root_native_total']
    assert [row for row in inspection['records'] if row['id'] not in proof['new_ids']]==old['records']
    assert len(oldledger['entries'])==49 and ledger['entries'][:-1]==oldledger['entries'] and len(ledger['entries'])==50
''' +s[end:]
        s=replace(s,"len(prior['all_ids'])==340","len(prior['all_ids'])==350")
        s=replace(s,"('minus_F','IID') and row['attack']=='Sp-DFA'","('minus_F','non-IID') and row['attack']=='Benign'")
        s=replace(s,'len(roots)==2','len(roots)==1')
        s=replace(s,"    assert e['authorized_snapshot_path']==p['authorized_snapshot_path'] and e['authorized_snapshot_sha256']==p['authorized_snapshot_sha256']","    assert p['authorized_snapshot_path'] is None and p['authorized_snapshot_sha256'] is None")
    compile(s,str(H/n),'exec')
    result=original if n in unchanged else s.encode('utf8')
    put(n,result)
    changes[n]=''.join(difflib.unified_diff(original.decode('utf8').splitlines(True),result.decode('utf8').splitlines(True),fromfile='parent/'+n,tofile='nonIID_Benign10/'+n))
put('SOURCE_DIFF.patch',''.join(changes.values()))
transport=R/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010'
inputs=read(transport/'INPUTS.json')
put('PARENT_SOURCE_PINS.json',dict(parent_path=str(P.relative_to(R)),parent_source_seal_sha256=sha(P/'SOURCE_FILES_SHA256.json'),files={n:dict(sha256=sha(P/n),bytes=(P/n).stat().st_size) for n in names},prior_root_sha256=sha(root),prior_index_sha256=sha(index),native350_root_sha256=sha(native),native350_inspection_sha256=sha(inspection),native350_ledger_sha256=sha(ledger),
 original_transport_path=str((transport/'transport.py').relative_to(R)),original_transport_sha256=sha(transport/'transport.py'),transport_source_seal_sha256=sha(transport/'FILES_SHA256.json'),transport_inputs_sha256=sha(transport/'INPUTS.json'),original_dependencies=inputs['dependencies'],valid_cache_sha256=inputs['valid_cache_sha256'],scientific_execution=0))
put('ROOT_INPUT_TEMPLATE.json',dict(status='TEMPLATE_ONLY_NOT_EXECUTION_OR_ADOPTION',native_root_path=None,native_root_sha256=None,native_total=None,authorized_snapshot_path=None,authorized_snapshot_sha256=None,source_review_path=None,source_review_sha256=None,prior_root_path=str(root),prior_root_sha256=sha(root),prior_index_path=str(index),prior_index_sha256=sha(index),candidate_ids=ids,proposed_replay_total=360,proposed_remaining620_transported=180,archive_sha256=None,archive_members=None,offserver_sha256=None,delivery_sha256=None,execution_authorized=False))
def block(s,a,b):return s[s.index(a):s.index(b,s.index(a))]
oldjoin=(P/'join_saved.py').read_bytes();newjoin=(H/'join_saved.py').read_bytes()
science_old=block(oldjoin,b'  for rec in records:',b'  assert len(member_checks)')
science_new=block(newjoin,b'  for rec in records:',b'  assert len(member_checks)')
assert science_old==science_new
assert ast.dump(ast.parse('def f():\n'+science_old.decode('utf8')),include_attributes=False)==ast.dump(ast.parse('def f():\n'+science_new.decode('utf8')),include_attributes=False)
resource_marker=b'assert os.sched_getaffinity(0)=={111}'
assert (P/'remote_preflight.py').read_bytes().split(resource_marker,1)[1]==(H/'remote_preflight.py').read_bytes().split(resource_marker,1)[1]
assert all((P/n).read_bytes()==(H/n).read_bytes() for n in unchanged)
for dep in inputs['dependencies'].values():assert sha(R/dep['local_relative'])==dep['sha256']
put('SOURCE_CHECK.json',dict(status='SOURCE_ONLY_COMPILE_METADATA_DIFF_AND_RAW_REUSE_PASS_NO_EXECUTION',compiled=names+['prepare_source.py'],three_runtime_helpers_raw_exact=sorted(unchanged),original_per_record_join_loop_raw_exact=True,original_per_record_join_loop_AST_exact=True,join_loop_sha256=hashlib.sha256(science_new).hexdigest(),original_remote_resource_suffix_raw_exact=True,original_transport_and_saved_verifier_unchanged=True,all_original_dependency_source_pins_actual_match=True,prior350_index_whole_file_SHA_bound=True,prior350_objects_and_order_retained_by_original_immutable_parent_reference=True,old_native450_records_and_49_ledger_guard=True,exact10_ids_fixed=True,actual_native_root_and_snapshot_unfilled=True,archive_member_count_dynamic=True,scientific_execution=0,SSH=0,model_array_reads=0,F_write=0,root_adopted=False))
print(json.dumps(dict(status='SOURCE_ONLY_DERIVED_COMPILED_NOT_EXECUTED',selected=10,prior_replays=350,proposed_replays=360,native_root_sha256=None,archive_members=None)))
