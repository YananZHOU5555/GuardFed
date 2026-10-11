"""Source/metadata checks only: no SSH, archive access, arrays, tensor code or science."""
from pathlib import Path
import ast,copy,hashlib,importlib.util,json
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/celeba_remaining_F_SDFA10_prepared_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def loop(path,target):
 s=path.read_text('utf8');nodes=[n for n in ast.walk(ast.parse(s)) if isinstance(n,ast.For) and ast.unparse(n.target)==target]
 assert len(nodes)==1
 return ast.get_source_segment(s,nodes[0])
def main():
 pins=read(H/'PARENT_SOURCE_PINS.json');assert sha(P/'SOURCE_FILES_SHA256.json')==pins['parent_seal_sha256']
 for n,pin in pins['files'].items():assert sha(P/n)==pin['sha256'] and (P/n).stat().st_size==pin['bytes']
 names=list(pins['files'])+['prepare_source.py','check_source.py']
 for n in names:compile((H/n).read_text('utf8'),str(H/n),'exec')
 unchanged=['execute_once.py','remote_cpu111_export.py','cpu_release_readonly.py']
 for n in unchanged:assert (H/n).read_bytes()==(P/n).read_bytes()
 science={}
 for n in ['join_saved.py','adopt_by_root.py']:
  a=loop(P/n,'rec');b=loop(H/n,'rec');assert a==b
  science[n+':for_rec']=hashlib.sha256(b.encode()).hexdigest()
 a=loop(P/'adopt_by_root.py','na');b=loop(H/'adopt_by_root.py','na')
 assert a.replace("row['attack']=='S-DFA'","row['attack']=='Sp-DFA'")==b
 science['adopt_by_root.py:for_na_only_attack_literal_changed']=True
 marker='assert os.sched_getaffinity(0)=={111}'
 old=(P/'remote_preflight.py').read_text('utf8');new=(H/'remote_preflight.py').read_text('utf8')
 assert old[old.index(marker):]==new[new.index(marker):]
 science['remote_preflight_resource_and_producer_guard_suffix_sha256']=hashlib.sha256(new[new.index(marker):].encode()).hexdigest()
 p=read(H/'PREPARED.json');assert p['native_binding'] is None and p['execution_performed'] is False
 assert p['prior_transported']==len(p['previous_all_transported_ids'])==160
 assert repr(p['previous_all_transported_ids']) in new and repr(p['candidate_ids']) in new
 assert not set(p['candidate_ids'])&set(p['previous_all_transported_ids'])
 assert read(H/'ROOT_INPUT_TEMPLATE.json')['archive_members'] is None
 spec=importlib.util.spec_from_file_location('metadata_validator_only',H/'verify_native_inputs.py');m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
 root=Path(p['required_native_root_path']);rp=read(root)
 e=dict(native_root_verified=True,native_root_path=str(root),native_root_sha256=p['required_native_root_sha256'],
  native_inspection_path=str(root.parent/'inspection/inspection.json'),native_inspection_sha256=rp['inspection_sha256'],
  native_ledger_path=str(root.parent/'verified_ledger.json'),native_ledger_sha256=rp['ledger_sha256'],
  exact_candidate_ids=p['candidate_ids'],root_native_total=350,transport_target_only=10,proposed_replay_total=350,
  native_archive_roots=p['known_native_archive_roots']+[dict(path=str(root),sha256=p['required_native_root_sha256'])],
  authorized_snapshot_path=p['authorized_snapshot_path'],authorized_snapshot_sha256=p['authorized_snapshot_sha256'])
 # Read compact JSON once; mutations below are memory-only and never become proof files.
 paths={e[k] for k in e if k.endswith('_path')}|{p[k] for k in p if k.endswith('_path') and p[k]}
 objects={str(x):read(x) for x in paths};hashes={str(x):sha(x) for x in paths}
 m.read=lambda path:objects[str(path)];m.sha=lambda path:hashes[str(path)]
 assert m.validate(e,p)==e
 cases=[]
 def reject(label,edit_e=None,edit_p=None,edit_objects=None):
  ee,pp=copy.deepcopy(e),copy.deepcopy(p);saved=copy.deepcopy(objects)
  try:
   if edit_e:edit_e(ee)
   if edit_p:edit_p(pp)
   if edit_objects:edit_objects(objects)
   try:m.validate(ee,pp)
   except (AssertionError,KeyError):cases.append(label)
   else:raise RuntimeError('accepted invalid metadata: '+label)
  finally:objects.clear();objects.update(saved)
 reject('native352_not350',edit_e=lambda x:x.update(root_native_total=352))
 reject('extra_Benign_ID',edit_e=lambda x:x['exact_candidate_ids'].append('minus_F_non-IID_Benign_seed91001'))
 reject('prior340_order',edit_objects=lambda x:x[p['prior_index_path']]['all_ids'].append(p['candidate_ids'][0]))
 reject('native_prefix_changed',edit_objects=lambda x:x[e['native_inspection_path']]['records'][0].update(seed=-1))
 reject('ledger48_prefix_changed',edit_objects=lambda x:x[e['native_ledger_path']]['entries'][0].update(test_fixture=True))
 reject('missing_SpDFA91010_closure',edit_objects=lambda x:x[e['authorized_snapshot_path']]['remaining620']['remote_closed_ids'].remove(p['candidate_ids'][-1]))
 reject('snapshot_failure',edit_objects=lambda x:x[e['authorized_snapshot_path']]['remaining620'].update(failures={'fixture':True}))
 reject('source_drift',edit_objects=lambda x:x[e['authorized_snapshot_path']]['source_data_binding'].update(changed_paths=['fixture']))
 reject('wrong_native_hash',edit_e=lambda x:x.update(native_root_sha256='0'*64))
 reject('wrong_snapshot_hash',edit_e=lambda x:x.update(authorized_snapshot_sha256='0'*64))
 reject('wrong_attack_record',edit_objects=lambda x:next(r for r in x[e['native_inspection_path']]['records'] if r['id']==p['candidate_ids'][-1]).update(attack='S-DFA'))
 reject('wrong_transport_scope',edit_e=lambda x:x.update(transport_target_only=12))
 for n in ['finalize_actual_handoff.py','adopt_by_root.py']:
  s=(H/n).read_text('utf8');assert '==92' not in s and 'archive_members=92' not in s
  assert "['archive_member_verification']['members_verified']" in s
 # Assert the exact unmodified original saved verifier/transport dependencies without executing them.
 dep=read(R/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010/INPUTS.json')['dependencies']['saved_verifier']
 assert sha(R/dep['local_relative'])==dep['sha256']=='939a3b47f04f4aba88f735b63e64354f9b8e879b6870a72fc86b5ae651c5edee'
 transport=R/'tmp/celeba_mechanism_remaining_evaluation_transport_20261010/transport.py'
 assert sha(transport)=='f71e6e4152625a5a0582a61ff9b3e55e4ccee65dfd70a4e657851c0247f9c9d7'
 assert not any((H/n).exists() for n in ['EXECUTION_INPUTS.json','PREFLIGHT_COMMAND.json','MECHANISM_INDEX.json','ACTUAL_HANDOFF.json','ROOT_ACTUAL_PINS.json'])
 return dict(status='SOURCE_AND_ACTUAL_METADATA_ONLY_PASS',compiled=len(names),unchanged_helpers=unchanged,science_and_resource_equivalence=science,
  metadata_rejection_cases=cases,native_root_sha256=p['required_native_root_sha256'],prior340_root_sha256=p['prior_root_sha256'],prior340_index_sha256=p['prior_index_sha256'],
  selected_ids=p['candidate_ids'],snapshot_sha256=p['authorized_snapshot_sha256'],snapshot_remote_closed=172,subset_only=10,
  native_total=350,prior_three_view=340,proposed_three_view=350,prior_transported=160,proposed_transported=170,
  original_transport_sha256=sha(transport),original_saved_verifier_sha256=dep['sha256'],archive_members='unbound; must match actual original archive verifier and offserver proof',
  SSH=0,science=0,fit=0,arrays_read=0,models_read=0,execution_bound=False,root_adopted=False)
if __name__=='__main__':print(json.dumps(main(),ensure_ascii=False,indent=2))
