"""Local source derivation only; never executes transport, arrays or science."""
from pathlib import Path
import difflib,hashlib,io,json,tokenize
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/celeba_remaining_F_SDFA10_prepared_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n')
old=read(P/'PREPARED.json');raw=read(P/'RAW_STORAGE_INDEX.json')
root=R/'tmp/celeba_mechanism_remaining_F_SDFA10_root_adoption_20261011/ROOT_ADOPTION.json'
assert sha(root)=='80a7f38890c2a7f8c75c0472489315bb2acb9e43c9e28d686a799cfea9709a02'
rp=read(root);index=R/rp['records_index_path'];assert sha(index)==rp['records_index_sha256']
assert rp['cumulative_accepted']==len(read(index)['all_ids'])==340 and len(raw['all_transported_ids'])==160
assert rp['archive_sha256']==raw['archive_sha256'] and rp['offserver_proof_sha256']==raw['offserver_verification_sha256']
base=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009'
native=base/'mechanism_science_backups_20261009/root_delta_20261010T222658Z/ROOT_DELTA_VERIFICATION.json'
current=base/'mechanism_science_backups_20261009/root_delta_20261010T232753Z/ROOT_DELTA_VERIFICATION.json'
assert sha(native)=='755f996594675c858b696ec1e0ac7038de9e8ad2a1fd011212fc78a38b736ba4'
assert sha(current)=='706b528831145c2a2402eb040f1e8a19279b09b29171e99082aa1c410db69770'
np=read(native);cp=read(current);inspection=native.parent/'inspection/inspection.json';ledger=native.parent/'verified_ledger.json'
assert sha(inspection)==np['inspection_sha256'] and sha(ledger)==np['ledger_sha256']
ids=[f'minus_F_IID_Sp-DFA_seed{s}' for s in range(91001,91011)];known=ids[:1]
assert set(ids)&set(np['new_ids'])==set(known) and cp['new_ids']==ids[1:]
assert len(read(inspection)['records'])==441 and len(read(ledger)['entries'])==48
assert cp['root_adopted'] and cp['total_new_strict_and_offserver']==350 and not cp['test']
snapshot=base/'root_five_queue_20261010T233639Z.raw.json'
assert sha(snapshot)=='92d48120f2dcdc45e473aabc9fa07b8f6a6f1055d20d7f3e36fb7cc5d2a74133'
snap=read(snapshot);assert set(ids)<=set(snap['remaining620']['remote_closed_ids'])
assert not snap['remaining620']['failures'] and not snap['source_data_binding']['changed_paths']
prevremote='/workspace/guardfed_checks/celeba_mechanism_remaining_evaluation_transport_20261010/exports/'+Path(raw['directory']).name+'/backup_receipt.json'
p=dict(status='SOURCE_ONLY_EXACT10_F_IID_SPDFA_KNOWN_NATIVE350_SNAPSHOT_NOT_EXECUTION_BOUND',candidate_ids=ids,
 prior_accepted=340,prior_transported=160,prior_root_path=str(root),prior_root_sha256=sha(root),prior_index_path=str(index),prior_index_sha256=sha(index),
 previous_local_receipt=raw['receipt'],previous_remote_receipt=prevremote,previous_receipt_sha256=raw['receipt_sha256'],previous_all_transported_ids=raw['all_transported_ids'],
 known_native_archive_roots=[dict(path=str(native),sha256=sha(native))],already_native_candidate_ids=known,required_new_native_candidate_ids=ids[1:],
 native341_root_path=str(native),native341_root_sha256=sha(native),native341_inspection_path=str(inspection),native341_inspection_sha256=sha(inspection),native341_ledger_path=str(ledger),native341_ledger_sha256=sha(ledger),
 replay_target_total=350,native_target_total=350,native_binding=None,authorized_snapshot_path=str(snapshot),authorized_snapshot_sha256=sha(snapshot),
 required_native_root_path=str(current),required_native_root_sha256=sha(current),native_extra_records_are_not_replayed=True,execution_performed=False)
put('PREPARED.json',p)
names=['adopt_by_root.py','bind_native.py','cpu_release_readonly.py','download_verify_once.py','execute_once.py','export_once.py','finalize_actual_handoff.py','join_saved.py','remote_cpu111_export.py','remote_preflight.py','verify_native_inputs.py']
parentseal=read(P/'SOURCE_FILES_SHA256.json')
assert sha(P/'SOURCE_FILES_SHA256.json')=='a8d582069825ba0943f0b68dab2bfca66d5e32e9e66e16f0d64ec23afe959d45'
for name in names:assert sha(P/name)==parentseal['files'][name]['sha256']
unchanged={'execute_once.py','remote_cpu111_export.py','cpu_release_readonly.py'}
numbermap={'330':'340','340':'350','433':'441','150':'160','160':'170','47':'48','48':'49'}
for name in names:
 original=(P/name).read_text('utf8');s=original
 if name not in unchanged:
  if name=='remote_preflight.py':
   assert repr(old['previous_all_transported_ids']) in s
   s=s.replace(repr(old['previous_all_transported_ids']),'PREVIOUS_IDS_PLACEHOLDER')
  tokens=[]
  for t in tokenize.generate_tokens(io.StringIO(s).readline):
   if t.type==tokenize.NUMBER and t.string in numbermap:t=t._replace(string=numbermap[t.string])
   tokens.append(t)
  s=tokenize.untokenize(tokens)
  for a,b in [('celeba_remaining_F_SDFA10_prepared_20261011','celeba_remaining_F_SpDFA10_prepared_20261011'),('celeba_mechanism_remaining_F_SDFA10_root_adoption_20261011','celeba_mechanism_remaining_F_SpDFA10_root_adoption_20261011'),('minus_F_IID_S-DFA_seed','minus_F_IID_Sp-DFA_seed'),('F_SDFA10_','F_SPDFA10_'),('F S-DFA','F Sp-DFA'),('native333','native341'),('prior330','prior340'),('old330','old340'),('old433','old441'),('old47','old48'),('original330','original340'),('MECHANISM340_INDEX','MECHANISM350_INDEX'),('REPLAY340','REPLAY350'),('MIN340','EXACT350'),('native >=340','native =350'),('native total>=340','native total=350'),('three-view340','three-view350')]:s=s.replace(a,b)
  s=s.replace(old['previous_receipt_sha256'],raw['receipt_sha256']).replace(old['prior_root_sha256'],sha(root)).replace(old['prior_index_sha256'],sha(index))
  s=s.replace('>=350','==350')
 if name=='remote_preflight.py':s=s.replace(old['previous_remote_receipt'],prevremote).replace('PREVIOUS_IDS_PLACEHOLDER',repr(raw['all_transported_ids']))
 if name=='download_verify_once.py':s=s.replace('tmp/celeba_remaining_F_FedSA10_20261011/RAW_STORAGE_INDEX.json','tmp/celeba_remaining_F_SDFA10_prepared_20261011/RAW_STORAGE_INDEX.json')
 if name=='verify_native_inputs.py':
  s=s.replace("assert e['root_native_total']==350 and p['native_binding'] is None", "assert e['root_native_total']==p['native_target_total']==350 and p['native_binding'] is None\n    assert e['native_root_path']==p['required_native_root_path'] and e['native_root_sha256']==p['required_native_root_sha256']")
  s=s.replace("e['root_native_total']-333","e['root_native_total']-341")
  s=s.replace("for s in (91002,91003,91004)","for s in (91001,)")
  s=s.replace("row['attack']=='S-DFA'","row['attack']=='Sp-DFA'")
  s=s.replace("assert set(p['required_new_native_candidate_ids'])<=set(proof['new_ids'])","assert p['required_new_native_candidate_ids']==proof['new_ids']")
  s=s.replace("assert sha(e['authorized_snapshot_path'])==e['authorized_snapshot_sha256']", "assert e['authorized_snapshot_path']==p['authorized_snapshot_path'] and e['authorized_snapshot_sha256']==p['authorized_snapshot_sha256']\n    assert sha(e['authorized_snapshot_path'])==e['authorized_snapshot_sha256']")
 if name=='finalize_actual_handoff.py':
  s=s.replace("raw['archive_members']==92", "type(raw['archive_members']) is int and raw['archive_members']>0\nassert sha(raw['offserver_verification'])==raw['offserver_verification_sha256']\nassert raw['archive_members']==read(raw['offserver_verification'])['archive_member_verification']['members_verified']")
 if name=='adopt_by_root.py':
  s=s.replace("row['attack']=='S-DFA'","row['attack']=='Sp-DFA'")
  s=s.replace("archive_check['members_verified']==92", "archive_check['members_verified']==h['archive_members']==off['archive_member_verification']['members_verified'] and type(h['archive_members']) is int and h['archive_members']>0")
  s=s.replace("archive_members=92", "archive_members=archive_check['members_verified']")
  s=s.replace("complete_F_scenes=[['IID','Benign'],['IID','F Flip'],['IID','FedSA'],['IID','S-DFA']]", "complete_F_scenes=[['IID','Benign'],['IID','F Flip'],['IID','FedSA'],['IID','S-DFA'],['IID','Sp-DFA']]")
  s=s.replace('tmp/celeba_remaining_F_FedSA10_20261011/adopt_by_root.py','tmp/celeba_remaining_F_SDFA10_prepared_20261011/adopt_by_root.py')
  last="assert {int(i[-5:]) for i in index['all_ids'] if i.startswith('minus_F_IID_Sp-DFA_seed')}==set(range(91001,91011))"
  assert last in s
  s=s.replace(last,"assert {int(i[-5:]) for i in index['all_ids'] if i.startswith('minus_F_IID_S-DFA_seed')}==set(range(91001,91011))\n"+last)
 if name in unchanged:
  with (H/name).open('xb') as f:f.write((P/name).read_bytes())
 else:put(name,s)
put('PARENT_SOURCE_PINS.json',dict(parent=str(P.relative_to(R)),parent_seal_sha256=sha(P/'SOURCE_FILES_SHA256.json'),files={n:dict(sha256=sha(P/n),bytes=(P/n).stat().st_size) for n in names},prior_root_sha256=sha(root),prior_index_sha256=sha(index),native341_sha256=sha(native),native350_sha256=sha(current),no_parent_science_reexecuted=True))
put('SOURCE_DIFF.patch',''.join(''.join(difflib.unified_diff((P/n).read_text('utf8').splitlines(True),(H/n).read_text('utf8').splitlines(True),fromfile='SDFA10/'+n,tofile='SpDFA10/'+n)) for n in names))
put('ROOT_INPUT_TEMPLATE.json',dict(status='TEMPLATE_ONLY_NOT_EXECUTION_AUTHORIZATION',native_root_path=str(current),native_root_sha256=sha(current),native_total=350,replay_total=350,new_replays=10,authorized_snapshot_path=str(snapshot),authorized_snapshot_sha256=sha(snapshot),delivery_sha256=None,delivery_members=None,handoff_sha256=None,proposed_index_sha256=None,archive_sha256=None,archive_members=None,offserver_sha256=None,execution_authorized=False))
print(json.dumps(dict(status='SOURCE_ONLY_DERIVED',native_total=350,prior_replay=340,proposed_replay=350,selected=10,remote_closed_observed=len(snap['remaining620']['remote_closed_ids']),execution_performed=False)))
