"""Freeze source-only delivery once; no remote access or scientific execution."""
from pathlib import Path
import ast,hashlib,json,tarfile
H=Path(__file__).resolve().parent;O=H.with_name('celeba_mechanism_valid_C_after40_20261010');E=H/'execution_candidate'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def write(name,data):
 with (H/name).open('x',encoding='utf-8',newline='\n') as f:json.dump(data,f,indent=2,allow_nan=False);f.write('\n')
def funcs(p):
 s=Path(p).read_text(encoding='utf-8-sig');return {n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
old,new=funcs(O/'bridge.py'),funcs(H/'bridge.py');names=read(H/'SOURCE_REUSE.json')['exact_bridge_functions']
assert len(names)==11 and all(old[n]==new[n] for n in names)
assert funcs(O/'execution_candidate/batch.py')['runtime_policy']==funcs(E/'batch.py')['runtime_policy']
assert sha(O/'execution_candidate/resource_extra.py')==sha(E/'resource_extra.py')
inv=read(H/'inventory_actual150_Full100refs.json');prior=read(O/'inventory_actual147_Full100refs.json')
assert [r for r in inv['records'] if r['id'] in set(inv['excluded_prior_replay_ids'])]==prior['records']
assert inv['full_references']==prior['full_references']
def spans(p,key):
 s=Path(p).read_text(encoding='utf-8-sig');pos=s.index('[',s.index('"'+key+'"'))+1;out=[];dec=json.JSONDecoder()
 while True:
  while s[pos].isspace() or s[pos]==',':pos+=1
  if s[pos]==']':return out
  value,end=dec.raw_decode(s,pos);out.append((value['id'],s[pos:end]));pos=end
prior_ids=set(inv['excluded_prior_replay_ids'])
assert [(i,v) for i,v in spans(H/'inventory_actual150_Full100refs.json','records') if i in prior_ids]==spans(O/'inventory_actual147_Full100refs.json','records')
assert spans(H/'inventory_actual150_Full100refs.json','full_references')==spans(O/'inventory_actual147_Full100refs.json','full_references')
for seal in [H/'FILES_SHA256.json',E/'EXECUTION_SOURCE_SHA256.json']:
 for row in read(seal)['members']:
  p=seal.parent/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['size']
for name,pin in read(H/'INPUT_PINS.json').items():assert sha(H.parents[1]/name)==pin['sha256']
write('LINEAGE.json',dict(status='SOURCE_METADATA_IDENTITY_PASS_NOT_EXECUTION_APPROVAL',parent_science_seal_sha256=sha(O/'FILES_SHA256.json'),parent_execution_seal_sha256=sha(O/'execution_candidate/EXECUTION_SOURCE_SHA256.json'),parent_bridge_sha256=sha(O/'bridge.py'),science_seal_sha256=sha(H/'FILES_SHA256.json'),execution_seal_sha256=sha(E/'EXECUTION_SOURCE_SHA256.json'),exact_function_source_sha256={n:hashlib.sha256(new[n].encode()).hexdigest() for n in names},whole_bind_runtime_source_exact=True,runtime_policy_source_exact=True,resource_module_byte_exact=True,old147_records_exact=True,old147_raw_json_bytes_and_order_exact=True,Full100_refs_exact=True,Full100_raw_json_bytes_and_order_exact=True,native_tolerance=1e-12,actual_native150_root_review_sha256=sha(H/'ROOT_NATIVE_INCREMENT_REVIEW.json'),prior147_root_adoption_sha256='64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea',actual_CNN=0,actual_execution_approval=False))
science=read(H/'FILES_SHA256.json')['members'];execution=read(E/'EXECUTION_SOURCE_SHA256.json')['members']
archive_names=[r['path'] for r in science]+['FILES_SHA256.json']+['execution_candidate/'+r['path'] for r in execution]+['execution_candidate/EXECUTION_SOURCE_SHA256.json','COMMANDS.md','README.md','LINEAGE.json','MINIMAL_SOURCE_DIFF.patch','SELF_CHECK.json','check_prepared.py']
assert len(archive_names)==len(set(archive_names))==29
with tarfile.open(H/'source_prepared.tar.gz','x:gz') as t:
 for n in sorted(archive_names):t.add(H/n,arcname=n,recursive=False)
with tarfile.open(H/'source_prepared.tar.gz') as t:
 assert t.getnames()==sorted(archive_names)
 for m in t.getmembers():assert m.isfile() and hashlib.sha256(t.extractfile(m).read()).hexdigest()==sha(H/m.name)
write('PACKAGE_RECEIPT.json',dict(status='PREPARED_NOT_APPROVED_SOURCE_PACKAGE_VERIFIED',selected_ids=inv['selected_replay_ids'],native_global=150,inventory_records=150,prior_three_views_excluded=147,new_image_replays_performed=0,science_seal_sha256=sha(H/'FILES_SHA256.json'),execution_seal_sha256=sha(E/'EXECUTION_SOURCE_SHA256.json'),inventory_sha256=sha(H/'inventory_actual150_Full100refs.json'),bridge_sha256=sha(H/'bridge.py'),source_archive='source_prepared.tar.gz',source_archive_sha256=sha(H/'source_prepared.tar.gz'),archive_members=len(archive_names),members={n:{'sha256':sha(H/n),'bytes':(H/n).stat().st_size} for n in sorted(archive_names)}))
write('HANDOFF.json',dict(status='PREPARED_ONLY_EXACT3_SOURCE_AND_METADATA_GATES_PASS_NO_EXECUTION',selected_ids=inv['selected_replay_ids'],native_accepted_snapshot=150,three_view_accepted_unchanged=147,new_three_view_accepted=0,Full_reference_only=100,C_IID_SpDFA_native_n=10,C_IID_SpDFA_scene_complete=True,package_receipt_sha256=sha(H/'PACKAGE_RECEIPT.json'),science_seal_sha256=sha(H/'FILES_SHA256.json'),execution_seal_sha256=sha(E/'EXECUTION_SOURCE_SHA256.json'),inventory_sha256=sha(H/'inventory_actual150_Full100refs.json'),source_archive_sha256=sha(H/'source_prepared.tar.gz'),native_review_sha256=sha(H/'ROOT_NATIVE_INCREMENT_REVIEW.json'),selfcheck_sha256=sha(H/'SELF_CHECK.json'),positive_worker_prebind_ids=read(H/'SELF_CHECK.json')['worker_original_pre_bind_path_reached'],metadata_refusals=read(H/'SELF_CHECK.json')['refusal_count'],old147_records_exact=True,Full100_references_exact=True,exact_whole_bridge_functions=names,resource_module_byte_exact=True,namespace='/workspace/guardfed_checks/'+H.name,service='guardfed_celeba_mechanism_valid_C_after47',prior_service_required_exited='guardfed_celeba_mechanism_valid_C_after40',runtime_resources_measured=False,external_approval_created=False,root_independent_review_pending=True,SSH=False,CNN=0,training=0,test=False,canonical_changed=False,STATE_changed=False,Git_changed=False,source_sealed_do_not_edit=True,source_notes='MINIMAL_SOURCE_DIFF.patch and LINEAGE.json; only scope/count/pin/namespace metadata changes. Read COMMANDS.md for actual root approval and Linux installer CLI.'))
members=[{'path':p.relative_to(H).as_posix(),'sha256':sha(p),'size':p.stat().st_size} for p in sorted(H.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.name!='PACKAGE_SHA256.json']
write('PACKAGE_SHA256.json',dict(status='PREPARED_NOT_APPROVED_COMPLETE_DELIVERY',members=members))
for row in read(H/'PACKAGE_SHA256.json')['members']:assert sha(H/row['path'])==row['sha256']
print(json.dumps({'package_sha256':sha(H/'PACKAGE_SHA256.json'),'package_members':len(members),'package_receipt_sha256':sha(H/'PACKAGE_RECEIPT.json'),'source_archive_sha256':sha(H/'source_prepared.tar.gz'),'handoff_sha256':sha(H/'HANDOFF.json'),'science_seal_sha256':sha(H/'FILES_SHA256.json'),'execution_seal_sha256':sha(E/'EXECUTION_SOURCE_SHA256.json')}))
