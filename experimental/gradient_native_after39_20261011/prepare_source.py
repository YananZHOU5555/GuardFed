"""Bind the actual parent39 and fixed42 observation; no scientific execution."""
from pathlib import Path
import ast, difflib, hashlib, importlib.util, json
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/gradient_native_after32_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False,allow_nan=False);f.write('\n')
def put(n,b):
 with (H/n).open('xb') as f:f.write(b)
def replace(s,a,b,n=None):
 count=s.count(a);assert count and (n is None or count==n),(a,count,n)
 return s.replace(a,b)

snapshot=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_five_queue_20261010T164728Z.raw.json'
assert sha(snapshot)=='cf27044d202579b17a7ee185207e7ff1f999c486d975260fbeda7af855b6c3e3'
root=P/'ROOT_ADOPTION_REVIEW.json';off=P/'OFFSERVER_ACCEPTANCE.json'
assert sha(root)=='850b91b68e01f98f53a82f26897a43a486857b69fd4335c2e8e8e79fba9e1e3b'
assert sha(off)=='82c2912ee80118d8f6c9aa9daf317cb671f4f435801cb6cae467d80fa9968e1a'
s=read(snapshot);g=s['gradient64'];p=read(root);o=read(off)
assert p['accepted_total']==39 and p['accepted_ids']==o['accepted_job_ids'] and len(set(p['accepted_ids']))==39
assert len(g['terminal_ids'])==len(set(g['terminal_ids']))==42 and not g['failure_paths']
assert not g['source']['changed_members']
ids=[x['id'] for x in g['rows'] if x['id'] in g['terminal_ids'] and x['id'] not in p['accepted_ids']]
expected=['Huber_eta0.03_T00.01_M1_non-IID_S-DFA_seed91001_screen','Huber_eta0.03_T00.1_M0_IID_Benign_seed91001_screen','Huber_eta0.03_T00.1_M0_IID_S-DFA_seed91001_screen']
assert ids==expected and set(p['accepted_ids'])<=set(g['terminal_ids'])
for row in g['rows']:
 if row['id'] in ids:assert row['round']==70 and row['observed_terminal'] and not row['active'] and row['acceptance_present'] and all(row['identity_checks'].values())
put('PREVIOUS_ROOT_ADOPTION.json',root.read_bytes());put('PREVIOUS_OFFSERVER_ACCEPTANCE.json',off.read_bytes())
save('SNAPSHOT.json',dict(status='FIXED_PARENT_PROVIDED_ACTUAL_GRADIENT42_OBSERVATION_SUBSET',original_snapshot_path=snapshot.relative_to(R).as_posix(),original_snapshot_sha256=sha(snapshot),utc=s['utc'],guide_sha256=s['guide_sha256'],gradient64=g,new_observation_performed=False))
a=read(P/'AUTHORIZED_SNAPSHOT.json')
a.update(status='PARENT_AUTHORIZED_FIXED_ACTUAL42_MINUS_ACCEPTED39_NOT_ADOPTION',snapshot_sha256=sha(H/'SNAPSHOT.json'),snapshot_utc=s['utc'],accepted_prior_ids=p['accepted_ids'],accepted_prior_count=39,authorized_ids=ids,terminal_count=42,prior_root_path=root.relative_to(R).as_posix(),prior_root_sha256=sha(root),prior_offserver_path=off.relative_to(R).as_posix(),prior_offserver_sha256=sha(off),parent_authorization='Explicit root exact3 strict/archive/offserver collection task; fixed actual42 minus accepted39 only, all negative results retained; no recipe or shared writes')
assert a['manifest_sha256']==g['manifest_sha256'] and a['source_seal_sha256']=='11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced'
save('AUTHORIZED_SNAPSHOT.json',a)
diff=[];pins={};loops={}
for n in ['run_once.py','collect_delta.py','verify_delta.py','finalize.py']:
 old=(P/n).read_text('utf8');new=old
 for x,y in [('gradient64_delta_after32_closed39','gradient64_delta_after39_closed42'),('old32_ordered_prefix_exact','old39_ordered_prefix_exact')]:
  if x in new:new=replace(new,x,y)
 if n=='run_once.py':
  new=replace(new,'AFTER32_CLOSED39','AFTER39_CLOSED42',1)
  new=replace(new,"len(auth['authorized_ids'])==7","len(auth['authorized_ids'])==3",1)
  new=replace(new,"if any('celeba_gradient64_delta' in x and 'collect_delta.py' in x for x in argv)","if any(('celeba_gradient64_delta' in x or 'gradient_native_after' in x) and 'collect_delta.py' in x for x in argv)",1)
  new=replace(new,"busy=[];helpers=[]\nfor p in Path('/proc').iterdir():","assert hashlib.sha256(Path('/etc/vast-agents-guide.md').read_bytes()).hexdigest()=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'\nbusy=[];helpers=[]\nfor p in Path('/proc').iterdir():",2)
 elif n=='collect_delta.py':
  new=replace(new,'185f0fa292be27b6921776818d13871cb8a5b787a765867b802362991d1f9a81',sha(root),1)
  new=replace(new,'40a8e3f3697c4a9dc32250b3b51b4c7afb56af9a412048c560f128d3d28d77b0',sha(off),1)
  new=replace(new,"prior['accepted_total']==32","prior['accepted_total']==39",1)
  new=replace(new,'accepted32 binding','accepted39 binding',1)
  new=replace(new,'accepted_before=32,strict_cumulative=32+len(ids)','accepted_before=39,strict_cumulative=39+len(ids)',1)
 elif n=='verify_delta.py':
  new=replace(new,"prior['accepted_total']==32","prior['accepted_total']==39",1)
  new=replace(new,'Actual32 parent mismatch','Actual39 parent mismatch',1)
  new=replace(new,'accepted_before=32,accepted_total=32+len(ids)','accepted_before=39,accepted_total=39+len(ids)',1)
 else:
  for x,y in [("len(a['authorized_ids'])==7","len(a['authorized_ids'])==3"),("p['accepted_before']==32 and p['accepted_total']==39","p['accepted_before']==39 and p['accepted_total']==42"),('EXACT7','EXACT3'),('accepted_before=32','accepted_before=39'),('strict_offserver_cumulative=39','strict_offserver_cumulative=42'),('root_accepted_cumulative_before_this_delivery=32','root_accepted_cumulative_before_this_delivery=39'),('accepted_total=39','accepted_total=42'),('dict(new=n,total=39','dict(new=n,total=42'),('exact7','exact3'),('fixed 39 terminal jobs minus the root-adopted accepted32','fixed 42 terminal jobs minus the root-adopted accepted39'),('seven Huber','three Huber'),('new7','new3'),('strict/offserver39/64','strict/offserver42/64'),('Previous32','Previous39')]:new=replace(new,x,y)
 compile(new,str(H/n),'exec')
 put(n,new.encode('utf8'))
 diff.extend(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile=str(P/n),tofile=str(H/n)))
 pins[n]=dict(path=(P/n).relative_to(R).as_posix(),sha256=sha(P/n))
 if n in ['collect_delta.py','verify_delta.py']:
  def scientific_loops(t):return [ast.get_source_segment(t,x) for x in ast.walk(ast.parse(t)) if isinstance(x,ast.For) and isinstance(x.target,ast.Name) and x.target.id=='ID']
  assert scientific_loops(old)==scientific_loops(new)
  loops[n]=dict(per_ID_loops=len(scientific_loops(old)),source_segments_byte_exact=True,AST_exact=True)
put('pinned_collect_one.py',(P/'pinned_collect_one.py').read_bytes())
assert sha(H/'pinned_collect_one.py')=='985f04d65afd8867cd606ac54b31b081ebeb3626b03b615245abe31a1d5504a7'
put('SOURCE_DIFF.patch',''.join(diff).encode('utf8'))
save('SOURCE_REUSE.json',dict(status='SOURCE_READY_EXACT3_PARENT39_FIXED42_NOT_ACCEPTED',parent_sources=pins,pinned_collect_one_sha256=sha(H/'pinned_collect_one.py'),scientific_loop_checks=loops,changes=['parent39 actual root/offserver pins and exact3 scope/archive/namespace','CPU110 resource policy unchanged; add guide check before remote source writes and recognize gradient_native_after collector namespace in duplicate guard'],original_checked_result_sha256='2c5d7699c6e9967d32c9080fb56b4672beb821cee0b20187c68195fb37d204e9',original_archive_verifier_sha256='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef',new_science=False,old39_proof_bytes_exact=True,old7_constant_negative_records_retained=True,scope=ids,compiled=['prepare_source.py','run_once.py','collect_delta.py','verify_delta.py','finalize.py','pinned_collect_one.py']))
names=['prepare_source.py','run_once.py','collect_delta.py','verify_delta.py','finalize.py','pinned_collect_one.py','SNAPSHOT.json','AUTHORIZED_SNAPSHOT.json','PREVIOUS_ROOT_ADOPTION.json','PREVIOUS_OFFSERVER_ACCEPTANCE.json','SOURCE_DIFF.patch','SOURCE_REUSE.json']
for n in names:
 if n.endswith('.py'):compile((H/n).read_text('utf8'),str(H/n),'exec')
save('SOURCE_FILES_SHA256.json',dict(status='PREPARED_ORIGINAL_GRADIENT_EXACT3_PARENT39_SOURCE_ONLY',files={n:dict(sha256=sha(H/n),bytes=(H/n).stat().st_size) for n in names}))
save('EXECUTION_SOURCE_REVIEW.json',dict(status='PARENT_AUTHORIZED_FIXED_GRADIENT64_DELTA_AFTER39_CLOSED42',root_adoption_performed=False,reviewer='baseline_delta_10_17 under explicit root exact3 actual-collection authorization; not an independent scientific adoption',source_files_sha256=sha(H/'SOURCE_FILES_SHA256.json'),authorization_sha256=sha(H/'AUTHORIZED_SNAPSHOT.json'),exact_new_ids=ids,source_diff_sha256=sha(H/'SOURCE_DIFF.patch')))
spec=importlib.util.spec_from_file_location('storage',R/'tmp/guardfed_local_storage.py');storage=importlib.util.module_from_spec(spec);spec.loader.exec_module(storage)
save('F_VOLUME_BEFORE_COLLECT.json',storage.check_bulk_storage(0))
print(json.dumps(dict(exact_ids=ids,source_seal=sha(H/'SOURCE_FILES_SHA256.json'),authorization_review_sha256=sha(H/'EXECUTION_SOURCE_REVIEW.json'),loops=loops)))
