"""Prepare exact actual46-minus-adopted42 metadata/source only; no SSH/Torch/F writes."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'tmp/gradient_native_after39_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def put(name,data):
 with (H/name).open('xb') as f:f.write(data)
def save(name,value):put(name,(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode())
def change(s,a,b,n=1):
 assert s.count(a)==n,(a,s.count(a),n)
 return s.replace(a,b)
raw=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_five_queue_20261010T180131Z.raw.json'
assert sha(raw)=='59f180458fd5fcf06dd2065a80b14150aa3ffc490928eb5bd989963ea7410884'
root=P/'ROOT_ADOPTION_REVIEW.json';off=P/'OFFSERVER_ACCEPTANCE.json'
assert sha(root)=='7eec0792793dab9777fda31ca55c779dd68827e9827186f45dbe706cb9fea1ea'
assert sha(off)=='abf8d3050973d7ab1dbc244cfd74a4e410b7e9b5748b2c7e12d93f62b74428bb'
p=read(root);o=read(off);s=read(raw);g=s['gradient64']
state_path=R/'docs/server_deployment_20260923/training_20260923/TRAINING_STATE.json';state=read(state_path)['gradient64_validation_search_20261010']
assert state['offserver_accepted']==p['accepted_total']==len(p['accepted_ids'])==42
assert state['root_adoption_sha256']==sha(root) and state['accepted_ids']==p['accepted_ids']==o['accepted_job_ids']
assert len(g['terminal_ids'])==len(set(g['terminal_ids']))==46 and set(p['accepted_ids'])<=set(g['terminal_ids'])
assert not g['failure_paths'] and not g['source']['changed_members']
assert g['source']['sha256']=='11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced'
ids=[x['id'] for x in g['rows'] if x['id'] in g['terminal_ids'] and x['id'] not in p['accepted_ids']]
expected=['Huber_eta0.03_T00.1_M0_non-IID_Benign_seed91001_screen','Huber_eta0.03_T00.1_M0_non-IID_S-DFA_seed91001_screen','Huber_eta0.03_T00.1_M1_IID_Benign_seed91001_screen','Huber_eta0.03_T00.1_M1_IID_S-DFA_seed91001_screen']
assert ids==expected and len(ids)==len(set(ids))==4
selected=[x for x in g['rows'] if x['id'] in ids]
for row in selected:assert row['round']==70 and row['observed_terminal'] and not row['active'] and row['acceptance_present'] and all(row['identity_checks'].values())
for name,pin in read(P/'SOURCE_FILES_SHA256.json')['files'].items():assert sha(P/name)==pin['sha256'] and (P/name).stat().st_size==pin['bytes']
put('PREVIOUS_ROOT_ADOPTION.json',root.read_bytes());put('PREVIOUS_OFFSERVER_ACCEPTANCE.json',off.read_bytes())
save('SNAPSHOT.json',dict(status='FIXED_PARENT_PROVIDED_ACTUAL_GRADIENT46_OBSERVATION_SUBSET',original_snapshot_path=raw.relative_to(R).as_posix(),original_snapshot_sha256=sha(raw),utc=s['utc'],guide_sha256=s['guide_sha256'],gradient64=g,new_observation_performed=False))
a=read(P/'AUTHORIZED_SNAPSHOT.json');a.update(status='PARENT_AUTHORIZED_FIXED_ACTUAL46_MINUS_ACCEPTED42_NOT_ADOPTION',snapshot_sha256=sha(H/'SNAPSHOT.json'),snapshot_utc=s['utc'],accepted_prior_ids=p['accepted_ids'],accepted_prior_count=42,authorized_ids=ids,terminal_count=46,prior_root_path=root.relative_to(R).as_posix(),prior_root_sha256=sha(root),prior_offserver_path=off.relative_to(R).as_posix(),prior_offserver_sha256=sha(off),parent_authorization='Root requested source-only preparation for actual46 minus adopted42; runtime requires a separate actual root source review and explicit execution authorization')
assert a['manifest_sha256']==g['manifest_sha256'];save('AUTHORIZED_SNAPSHOT.json',a)
diff=[];pins={};loop_checks={}
for n in ['run_once.py','collect_delta.py','verify_delta.py','finalize.py']:
 old=(P/n).read_text('utf8');new=old
 if 'gradient64_delta_after39_closed42' in new:new=new.replace('gradient64_delta_after39_closed42','gradient64_delta_after42_closed46')
 if 'old39_ordered_prefix_exact' in new:new=new.replace('old39_ordered_prefix_exact','old42_ordered_prefix_exact')
 if n=='run_once.py':
  new=change(new,'AFTER39_CLOSED42','AFTER42_CLOSED46')
  new=change(new,"len(auth['authorized_ids'])==3","len(auth['authorized_ids'])==4")
  new=change(new,' auth=guard_review(args)\n payload='," auth=guard_review(args)\n storage=load(R/'tmp/guardfed_local_storage.py','bulk_storage_precollect')\n save('F_VOLUME_BEFORE_COLLECT.json',storage.check_bulk_storage(0))\n payload=")
 elif n=='collect_delta.py':
  new=change(new,'850b91b68e01f98f53a82f26897a43a486857b69fd4335c2e8e8e79fba9e1e3b',sha(root))
  new=change(new,'82c2912ee80118d8f6c9aa9daf317cb671f4f435801cb6cae467d80fa9968e1a',sha(off))
  new=change(new,"prior['accepted_total']==39","prior['accepted_total']==42")
  new=change(new,'accepted39 binding','accepted42 binding')
  new=change(new,'accepted_before=39,strict_cumulative=39+len(ids)','accepted_before=42,strict_cumulative=42+len(ids)')
 elif n=='verify_delta.py':
  new=change(new,"prior['accepted_total']==39","prior['accepted_total']==42")
  new=change(new,'Actual39 parent mismatch','Actual42 parent mismatch')
  new=change(new,'accepted_before=39,accepted_total=39+len(ids)','accepted_before=42,accepted_total=42+len(ids)')
 else:
  for x,y in [("len(a['authorized_ids'])==3","len(a['authorized_ids'])==4"),("p['accepted_before']==39 and p['accepted_total']==42","p['accepted_before']==42 and p['accepted_total']==46"),('EXACT3','EXACT4'),('accepted_before=39','accepted_before=42'),('strict_offserver_cumulative=42','strict_offserver_cumulative=46'),('root_accepted_cumulative_before_this_delivery=39','root_accepted_cumulative_before_this_delivery=42'),('accepted_total=42','accepted_total=46'),('dict(new=n,total=42','dict(new=n,total=46'),('exact3','exact4'),('fixed 42 terminal jobs minus the root-adopted accepted39','fixed 46 terminal jobs minus the root-adopted accepted42'),('three Huber','four Huber'),('new3','new4'),('strict/offserver42/64','strict/offserver46/64'),('Previous39','Previous42')]:
   new=change(new,x,y,new.count(x))
 compile(new,str(H/n),'exec');put(n,new.encode())
 diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=(P/n).relative_to(R).as_posix(),tofile=(H/n).relative_to(R).as_posix()))
 pins[n]=dict(path=(P/n).relative_to(R).as_posix(),sha256=sha(P/n))
 if n in ('collect_delta.py','verify_delta.py'):
  def loops(text):return [ast.get_source_segment(text,node) for node in ast.walk(ast.parse(text)) if isinstance(node,ast.For) and isinstance(node.target,ast.Name) and node.target.id=='ID']
  assert loops(old)==loops(new)
  loop_checks[n]=dict(per_ID_loops=len(loops(old)),source_segments_byte_exact=True,AST_exact=True)
put('pinned_collect_one.py',(P/'pinned_collect_one.py').read_bytes())
assert sha(H/'pinned_collect_one.py')=='985f04d65afd8867cd606ac54b31b081ebeb3626b03b615245abe31a1d5504a7'
put('SOURCE_DIFF.patch',''.join(diff).encode())
save('SOURCE_REUSE.json',dict(status='SOURCE_ONLY_READY_EXACT4_PARENT42_FIXED46_NOT_EXECUTED_NOT_ADOPTED',parent_sources=pins,parent_source_seal_sha256=sha(P/'SOURCE_FILES_SHA256.json'),pinned_collect_one_sha256=sha(H/'pinned_collect_one.py'),scientific_loop_checks=loop_checks,changes=['actual adopted42 root/offserver and observed46 exact4 scope/archive/output namespace','F-volume fresh guard moved from old preparation into runtime collect, before SSH; resource/guide/owner/CUDA policy unchanged'],original_checked_result_sha256='2c5d7699c6e9967d32c9080fb56b4672beb821cee0b20187c68195fb37d204e9',original_archive_verifier_sha256='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef',prior42_root_and_offserver_copied_byte_exact=True,new_science=False,scope=ids,snapshot_source_path=raw.relative_to(R).as_posix(),snapshot_source_sha256=sha(raw),state_read_sha256=sha(state_path),metadata_constant_negative_ids=[x['id'] for x in selected if x.get('constant_prediction_metadata')],negative_zero_disparity_is_not_fairness_success=True,actual_server_strict_executed=False,actual_offserver_executed=False,actual_root_adopted_new=0))
print(json.dumps(dict(exact_ids=ids,loops=loop_checks,root42_sha256=sha(root),offserver42_sha256=sha(off),source_only=True)))
