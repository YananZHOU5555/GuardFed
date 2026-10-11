"""Source-only exact9 derivation; compact parent metadata only, no scientific execution."""
from pathlib import Path
import ast,difflib,hashlib,io,json,tokenize
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'tmp/fl_native_after67_FedSA10_prepared_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(n,v):
 with (H/n).open('x',encoding='utf8',newline='') as f:f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n')
latest=R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json';l=read(latest);root=R/l['root_adoption_path'];off=Path(l['next_collector_previous_path'])
assert sha(latest)=='7cb5a47061abf2a311a0bc98427f67587116c893a230fd001f050604d7fa1c82'
assert sha(root)==l['root_adoption_sha256']=='5fdf135fd13665df6d528818db81765ad3567d5d754b18c538b1772f3994c80d'
assert sha(off)==l['next_collector_previous_sha256']=='7c79ee6a2434b1a1c1df1977a9ac7091b386a78412c7ad838ed44c907c448ab0'
assert read(root)['accepted_total']==read(off)['accepted_total']==77 and read(root)['accepted_job_ids']==read(off)['accepted_job_ids']
for n,path in [('PREVIOUS_LATEST.json',latest),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',off)]:
 with (H/n).open('xb') as f:f.write(path.read_bytes())
oldscope=read(P/'FROZEN_SCOPE.json');mp=R/oldscope['original_manifest_path'];assert sha(mp)==oldscope['original_manifest_sha256']
manifest=read(mp);jobs=[j for j in manifest['jobs'] if j['distribution']=='non-IID' and j['attack']=='S-DFA']
assert [j['seed'] for j in jobs]==list(range(91002,91011));ids=[j['id'] for j in jobs]
reused=[j for j in manifest['reused_jobs'] if j['distribution']=='non-IID' and j['attack']=='S-DFA'];assert len(reused)==1 and reused[0]['accepted_record']['seed']==91001
assert not set(ids)&set(read(off)['accepted_job_ids']) and len(manifest['reused_jobs'])==4
for j in jobs:assert sha(mp.parent/'jobs'/j['job'])==j['job_sha256']
scope=dict(status='PREPARED_EXACT9_IDENTITIES_NOT_TERMINAL_OR_ACCEPTED_PROOF',ids=ids,jobs=jobs,original_manifest_path=oldscope['original_manifest_path'],original_manifest_sha256=sha(mp),source_package_sha256=oldscope['source_package_sha256'],prior_native_new=77,proposed_native_new_if_root_adopts=86,screen_reused_separately=4,screen91001_excluded_from_new_native=dict(id=reused[0]['id'],checkpoint_sha256=reused[0]['accepted_record']['checkpoint_sha256'],original_job_sha256=reused[0]['job_sha256']),new_native_accepted_by_preparation=0,new_three_view_accepted=0)
put('FROZEN_SCOPE.json',scope)
pending=dict(status='PENDING_ACTUAL_TERMINAL_SNAPSHOT_AND_ROOT_EXECUTION_REVIEW',authorized_snapshot_file='AUTHORIZED_SNAPSHOT.json',required_status='ROOT_AUTHORIZED_FIXED_EXACT9_AFTER_NATIVE77_NOT_ACCEPTANCE',snapshot_path=None,snapshot_sha256=None,snapshot_utc=None,terminal_count=None,prior_count=77,authorized_ids=ids,prior_root_path=l['root_adoption_path'],prior_root_sha256=sha(root),prior_offserver_sha256=sha(off),required_root_review_fields=['source_adoptable','exact_selected_ids','prepared_seal_sha256','authorization_sha256'],actual_root_execution_review_path=None,actual_root_execution_review_sha256=None,execution_authorized=False,scientific_execution_performed=False,actual_output_hashes=None)
put('PENDING_BINDING.json',pending)
parentseal=sha(P/'PREPARED_FILES_SHA256.json');assert parentseal=='386b98d867284aa45011340c0962f06bb5ece78790fc1461812b00a3274631ea'
names=['run_once.py','collect_delta.py','preflight.py','close_observation.py','root_adopt.py'];pins=read(P/'PREPARED_FILES_SHA256.json')['files']
for n in names:assert sha(P/n)==pins[n]['sha256']
changes={}
for n in names:
 raw=(P/n).read_bytes();s=raw.decode('utf8');original=s
 # Same-length NUMBER tokens only; CPU110, nice10, round70 and all scientific numbers are unaffected.
 tokens=[]
 for t in tokenize.generate_tokens(io.StringIO(s).readline):
  if t.type==tokenize.NUMBER and t.string in {'67','77'}:t=t._replace(string={'67':'77','77':'86'}[t.string])
  tokens.append(t)
 s=tokenize.untokenize(tokens)
 mappings=[(repr(oldscope['ids']),repr(ids)),('fl_native_after67_FedSA10_prepared_20261011','fl_native_after77_SDFA9_prepared_20261011'),
 ('21f9b5d3cb07405d4100d28c83b1ce578e27fdb24382f9631d28eb3c7029f637',sha(off)),('a11cd9ed94ab136b49dd41975d45f684d88236b6918c295461089a98adbc7db0',sha(root)),
 ('exact10 after67','exact9 after77'),('exact10 only','exact9 only'),('All ten authorized','All nine authorized'),('frozen ten-ID','frozen nine-ID'),('EXACT10_AFTER_NATIVE67','EXACT9_AFTER_NATIVE77'),('after67','after77'),('old67','old77'),('prior67','prior77'),('Parent67','Parent77'),('prior_accepted_new=67','prior_accepted_new=77'),('prior67_chain','prior77_chain'),('root77.tmp','root86.tmp')]
 for a,b in mappings:s=s.replace(a,b)
 if n=='run_once.py':
  s=s.replace("receipt['accepted_new']==10","receipt['accepted_new']==9")
  s=s.replace('{67+len(ids)}','{77+len(ids)}')
 if n=='root_adopt.py':
  s=s.replace('(10,86,96,4)','(9,86,96,4)').replace('accepted_new=10','accepted_new=9')
  s=s.replace("h['archive_members'] == 107", "h['archive_members'] == receipt['archived_member_count'] == off['archived_member_count']")
  s=s.replace('archive_members_verified=107','archive_members_verified=len(listed)')
  s=s.replace("==80 and sum(r['elements'] for r in tensor['records'])==935060", "==8*len(off['new_ids']) and sum(r['elements'] for r in tensor['records'])==93506*len(off['new_ids'])")
 # Keep the parent's newline bytes and unchanged function/loop formatting.
 with (H/n).open('xb') as f:f.write(s.encode('utf8'))
 compile(s,str(H/n),'exec')
 changes[n]=dict(parent_sha256=sha(P/n),source_sha256=sha(H/n))
put('SOURCE_DIFF.patch',''.join(''.join(difflib.unified_diff((P/n).read_text('utf8').splitlines(True),(H/n).read_text('utf8').splitlines(True),fromfile='FedSA10/'+n,tofile='SDFA9/'+n)) for n in names))
reuse=read(P/'SOURCE_REUSE.json');reuse.update(thin_entry_sha256=sha(H/'collect_delta.py'),effective_collector_sha256=sha(H/'collect_delta.py'),sole_effective_change='Exact9 non-IID S-DFA seeds91002..10 after native77; screen91001 remains separate. Original science/strict/archive bodies retained.',predecessor_root_sha256=sha(root),predecessor_offserver_sha256=sha(off),direct_parent_transport_path=str((P/'run_once.py').relative_to(R)).replace('\\','/'),direct_parent_transport_sha256=sha(P/'run_once.py'),source_snapshot_sha256=None,source_snapshot_pending=True)
put('SOURCE_REUSE.json',reuse)
put('SOURCE_PARENT_PINS.json',dict(parent_directory=str(P.relative_to(R)).replace('\\','/'),parent_prepared_seal_sha256=parentseal,files={str((P/n).relative_to(R)).replace('\\','/'):sha(P/n) for n in names},previous_latest_sha256=sha(latest),previous_root_path=l['root_adoption_path'],previous_root_sha256=sha(root),previous_offserver_sha256=sha(off),source_changes=changes))
put('ROOT_INPUT_TEMPLATE.json',dict(status='TEMPLATE_ONLY_NO_EXECUTION_AUTHORIZATION',prior_native_new=77,new_native_count=9,proposed_native_total=86,screen_reused_separately=4,proposed_native_plus_reused=90,three_view_accepted_change=0,exact_ids=ids,parent_root_path=l['root_adoption_path'],parent_root_sha256=sha(root),parent_latest_sha256=sha(latest),parent_offserver_sha256=sha(off),actual_snapshot_path=None,actual_snapshot_sha256=None,root_execution_review_path=None,root_execution_review_sha256=None,handoff_sha256=None,delivery_seal_sha256=None,delivery_files=None,archive_sha256=None,archive_members=None,new_offserver_sha256=None,execution_authorized=False,scientific_execution_performed=False))
print(json.dumps(dict(status='EXACT9_SOURCE_DERIVED_NOT_EXECUTED',new_native_count=9,prior_native=77,proposed_native=86,reused_screen=4)))
