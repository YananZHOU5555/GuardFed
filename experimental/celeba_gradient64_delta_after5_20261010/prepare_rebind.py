"""Exact transport metadata rebind of accepted after1 helpers; no scientific change."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1];OLD=R/'tmp/celeba_gradient64_delta_after1_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
a=read(H/'AUTHORIZED_SNAPSHOT.json');assert a['accepted_prior_count']==5 and len(a['authorized_ids'])==5
def save(name,value):
 with (H/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2);f.write('\n')
def replace(text,old,new,count=1):
 assert text.count(old)==count,(old,text.count(old),count)
 return text.replace(old,new)
diff=[];original_pins={}
for name in ['collect_delta.py','verify_delta.py','run_once.py','finalize.py']:
 original=(OLD/name).read_text('utf8');text=original;original_pins[name]=sha(OLD/name)
 text=text.replace('gradient64_delta_after1.tar.gz','gradient64_delta_after5.tar.gz')
 if name in ('collect_delta.py','verify_delta.py'):
  text=replace(text,"previous['accepted_new_ids']","previous['accepted_job_ids']")
  text=replace(text,"prior['accepted_count']==1","prior['accepted_total']==5")
 if name=='collect_delta.py':
  text=replace(text,'ca4a38076e5080d94069cdabd50a032d84a96a339914761ec568db44a43db978',a['prior_root_sha256'])
  text=replace(text,'35089298546d72ef6b6503b63c7887912c8cd88e97a87d297d766245193e7cfa',a['prior_offserver_sha256'])
  text=replace(text,'Actual original accepted1 binding required','Actual root-adopted accepted5 binding required')
  text=replace(text,'accepted_before=1,strict_cumulative=1+len(ids)','accepted_before=5,strict_cumulative=5+len(ids)')
 elif name=='verify_delta.py':
  text=replace(text,'Original1 parent mismatch','Actual5 parent mismatch')
  text=replace(text,'accepted_before=1,accepted_total=1+len(ids)','accepted_before=5,accepted_total=5+len(ids)')
  text=replace(text,'old1_ordered_prefix_exact=True','old5_ordered_prefix_exact=True')
 elif name=='run_once.py':
  text=replace(text,"review['status']=='PASS_GRADIENT64_EXACT4_DELTA_SOURCE_READY' and review['source_adoptable'] is True",
   "review['status']=='PARENT_AUTHORIZED_FIXED_GRADIENT64_DELTA_AFTER5' and review['root_adoption_performed'] is False")
  text=replace(text,"len(review['exact_new_ids'])==4","len(review['exact_new_ids'])==len(auth['authorized_ids'])==5")
 elif name=='finalize.py':
  text=replace(text,"assert n==4 and","assert n==len(a['authorized_ids'])==5 and")
  text=replace(text,"p['accepted_before']==1 and p['accepted_total']==5","p['accepted_before']==5 and p['accepted_total']==10")
  text=text.replace('old1_ordered_prefix_exact','old5_ordered_prefix_exact')
  text=replace(text,'ROOT_READY_GRADIENT64_EXACT4_ORIGINAL_STRICT_OFFSERVER_PASS_NOT_ADOPTED','ROOT_READY_GRADIENT64_EXACT5_ORIGINAL_STRICT_OFFSERVER_PASS_NOT_ADOPTED')
  text=replace(text,'accepted_before=1,new_strict_offserver_count=n,strict_offserver_cumulative=5,root_accepted_cumulative_before_this_delivery=1',
   'accepted_before=5,new_strict_offserver_count=n,strict_offserver_cumulative=10,root_accepted_cumulative_before_this_delivery=5')
  text=replace(text,'auxiliary_snapshot_schema_failure_preserved=True','auxiliary_snapshot_schema_failure_preserved=False')
  text=replace(text,'accepted_before=1,accepted_new=n,accepted_total=5','accepted_before=5,accepted_new=n,accepted_total=10')
  text=replace(text,"new=n,total=5","new=n,total=10")
  start=text.index("with (H/'README.md').open")
  stop=text.index('\nfiles=',start)
  text=text[:start]+'''with (H/'README.md').open('x',encoding='utf8',newline='\\n') as f:f.write('''+repr('''# Actual gradient64 exact5 delta — root adoption pending

One saved v2a snapshot fixed 10 terminal jobs minus the root-adopted accepted5. The exact new5 passed original checked_result remotely and locally, original v4 archive/member verification and unchanged saved fullstate layout/dtype/finiteness. Actual cumulative strict/offserver10/64 is not root adoption or completed search; no recipe was selected.

Only predecessor schema/pins/counts, a fresh namespace/archive and explicit parent-task authorization were rebound in the after1 transport. Original scientific acceptor, original shared input binding, original model checks and archive verifier are unchanged. Previous5 models were not repackaged; all outcomes, including constant predictions, are preserved. Shared live logs are bounded prefixes, not closed per-job logs. All raw/model/archive/restored bulk went directly to freshly checked healthy Yanan 2TB F with required+reserve guard; no internal fallback.

Local CPU Torch differs from server cu128; no runtime equivalence is claimed. No CNN forward, optimizer, fitting, training, data loading, test or prediction recomputation occurred. CPU106 released. Later queue completions excluded. LATEST/STATE/Git/source/service unchanged. Git47 cutoff does not include this independent new batch; root must separately adopt.
''')+''')'''+text[stop:]
 ast.parse(text)
 with (H/name).open('x',encoding='utf8',newline='\n') as f:f.write(text)
 diff.extend(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='accepted_after1/'+name,tofile='new_after5/'+name))
with (H/'pinned_collect_one.py').open('xb') as f:f.write((OLD/'pinned_collect_one.py').read_bytes())
assert sha(H/'pinned_collect_one.py')=='985f04d65afd8867cd606ac54b31b081ebeb3626b03b615245abe31a1d5504a7'
with (H/'SOURCE_DIFF.patch').open('x',encoding='utf8') as f:f.write(''.join(diff))
save('SOURCE_REUSE.json',dict(status='PARENT_AUTHORIZED_TRANSPORT_METADATA_ONLY_REBIND_NOT_ROOT_ADOPTION',
 original_helper_pins=original_pins,pinned_helper_sha256=sha(H/'pinned_collect_one.py'),
 scientific_package_sha256=a['source_seal_sha256'],original_acceptor_sha256='2c5d7699c6e9967d32c9080fb56b4672beb821cee0b20187c68195fb37d204e9',
 original_archive_verifier_sha256='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef',
 previous_root_sha256=a['prior_root_sha256'],previous_offserver_sha256=a['prior_offserver_sha256'],
 differences=['actual predecessor5 schema/pins instead of original1','one fixed snapshot exact new5','new transport namespace/archive','parent explicit task authorization in place of original separate source-review CLI'],
 original_scientific_acceptor_modified=False,original_archive_verifier_modified=False,science_or_tensor_checks_modified=False,
 original_package_unchanged=True,root_adoption_performed=False,automatic_retry=False))
# The producer body, original strict call, archive/member writer and saved tensor checks are exact.
for name,start,end in [('collect_delta.py',' for ID in ids:'," for n in read(PKG/'FILES_SHA256.json')"),
 ('verify_delta.py','  records=[];tensor_rows=[]',"  require(not torch.cuda.is_initialized()")]:
 old=(OLD/name).read_text('utf8');new=(H/name).read_text('utf8')
 assert old[old.index(start):old.index(end)]==new[new.index(start):new.index(end)]
names=['observe_once.py','prepare_rebind.py','collect_delta.py','verify_delta.py','run_once.py','finalize.py','pinned_collect_one.py','SOURCE_REUSE.json','SOURCE_DIFF.patch']
save('SOURCE_FILES_SHA256.json',dict(status='SOURCE_TRANSPORT_REBIND_ONLY',files={n:dict(sha256=sha(H/n),bytes=(H/n).stat().st_size) for n in names}))
save('PARENT_AUTHORIZATION.json',dict(status='PARENT_AUTHORIZED_FIXED_GRADIENT64_DELTA_AFTER5',
 source_files_sha256=sha(H/'SOURCE_FILES_SHA256.json'),authorization_sha256=sha(H/'AUTHORIZED_SNAPSHOT.json'),
 exact_new_ids=a['authorized_ids'],root_adoption_performed=False,
 parent_task='NEW_TASK /root: collect one latest fixed terminal snapshot minus accepted5 using original strict/transport; no shared writes or recipe selection',
 scope=['collect','download','original strict and member/saved-state checks','CPU106 release','compact handoff'],
 future_completions_excluded=True))
print(json.dumps(dict(source_seal_sha256=sha(H/'SOURCE_FILES_SHA256.json'),parent_authorization_sha256=sha(H/'PARENT_AUTHORIZATION.json'),new_ids=a['authorized_ids'],prior=5)))
