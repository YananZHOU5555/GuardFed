"""Bind reviewed original readers to this one actual snapshot; no remote execution."""
from pathlib import Path
import ast,difflib,hashlib,json,runpy
H=Path(__file__).resolve().parent;R=H.parents[1];G=R/'tmp/celeba_gradient64_delta_after18_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(p,v):
 with p.open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\n')
def write(p,s):
 with p.open('x',encoding='utf8',newline='\n') as f:f.write(s)
def repl(s,a,b,count=1):
 assert s.count(a)==count,(a,s.count(a),count)
 return s.replace(a,b)
def scientific_loops(s):
 return [ast.get_source_segment(s,n) for n in ast.walk(ast.parse(s)) if isinstance(n,ast.For) and isinstance(n.target,ast.Name) and n.target.id in ('ID','identity')]
f=read(H/'FINDINGS.json');pins=read(H/'INPUT_PINS.json');snap=read(H/'SNAPSHOT.json');flids=f['FL_new_ids'];gids=f['gradient_new_ids']
assert len(flids)==6 and len(gids)==5 and f['FL_accepted_before']==38 and f['gradient_accepted_before']==18
assert not snap['CPU110_owners'] and not snap['existing_collectors']
latest=R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json'
for n,p in [('PREVIOUS_LATEST.json',latest),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',Path(pins['FL_offserver_path']))]:
 with (H/n).open('xb') as out:out.write(p.read_bytes())
save(H/'AUTHORIZED_SNAPSHOT.json',dict(status='PREPARED_FIXED_ONE_SNAPSHOT_EXACT6_NOT_ACCEPTANCE',snapshot_path=(H/'SNAPSHOT.json').relative_to(R).as_posix(),snapshot_sha256=sha(H/'SNAPSHOT.json'),snapshot_utc=snap['utc'],prior_count=38,terminal_count=44,authorized_ids=flids,source_hashes={'PACKAGE_SHA256.json':snap['FLGMM']['source']['sha256'],'manifest.json':snap['FLGMM']['manifest_sha256']},prior_root_path=pins['FL_root_path'],prior_root_sha256=pins['FL_root_sha256'],prior_offserver_sha256=pins['FL_offserver_sha256'],no_future_terminal_ids_allowed=True,root_adoption_required=True,new_accepted=0))
old=R/'tmp/celeba_flgmm_fullcoverage_delta_after32_20261010'
t=(old/'collect_delta.py').read_text('utf8');loop=next(n for n in ast.parse(t).body if isinstance(n,ast.For));edits=ast.literal_eval(loop.iter)
edits[0]=(edits[0][0],'    authorized_ids='+repr(flids)+'\n')
original=R/'tmp/celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py';s=original.read_text('utf8');effective=s
for a,b in edits:effective=repl(effective,a,b)
assert scientific_loops(s)==scientific_loops(effective)
assert effective[effective.index('    before=repo_identity'):]==s[s.index('    before=repo_identity'):]
wrapper="from pathlib import Path\nimport hashlib\np=Path(%r)\ns=p.read_text('utf8')\nassert hashlib.sha256(p.read_bytes()).hexdigest()==%r\nfor a,b in %r:\n    assert s.count(a)==1\n    s=s.replace(a,b)\nexec(compile(s,__file__,'exec'),globals())\n"%('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py',sha(original),edits)
write(H/'collect_delta.py',wrapper)
reuse=read(old/'SOURCE_REUSE.json');reuse.update(thin_entry_sha256=sha(H/'collect_delta.py'),effective_collector_sha256=hashlib.sha256(effective.encode()).hexdigest(),sole_effective_change='Frozen snapshot exact6 IDs; approved queue.failed and exact-completeness guards unchanged',resource_metadata_change='CPU110/one thread/nice10/idleIO/CUDA-hidden unchanged from after32',reused_after32_transport_sha256=sha(old/'run_once.py'))
save(H/'SOURCE_REUSE.json',reuse)
write(H/'SOURCE_DIFF.patch',''.join(difflib.unified_diff(s.splitlines(True),effective.splitlines(True),fromfile='original_after14_collector',tofile='effective_frozen_after38_collector')))
# Keep the already accepted after32 transport, changing only predecessor counts/pins.
src=(old/'run_once.py').read_text('utf8');tree=ast.parse(src)
body=ast.get_source_segment(src,next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='reused'))
body=repl(body,"PREVIOUS =",'UNUSED') if False else body
body=body.replace('32+len(ids)','38+len(ids)').replace('32+n','38+n').replace("==32\"","==38\"").replace('[:32]','[:38]').replace('=32\'','=38\'').replace('old32_ordered','old38_ordered').replace('after32:','after38:').replace('prior32','prior38')
# Verify every count rebinding in the generated transport specification explicitly.
assert "'total=38+n'" in body and 'accepted_before=38' in body and "prior['accepted_total']==38" in body and "p['accepted_job_ids'][:38]" in body
header="""from pathlib import Path
import argparse,ast,hashlib,json,runpy,sys,traceback
sys.dont_write_bytecode=True
H=Path(__file__).resolve().parent;R=H.parents[1]
PRIOR=R/'tmp/celeba_flgmm_fullcoverage_delta_after28_20261010'
ORIGINAL=R/'tmp/celeba_flgmm_fullcoverage_delta_after22_20261010'
S=Path('F:/YananResearchStorage/GuardFed')/H.name/'batch'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8',newline='\\n') as f:json.dump(v,f,indent=2,ensure_ascii=False);f.write('\\n')
"""
header+='PREVIOUS='+repr(pins['FL_offserver_sha256'])+'\nROOTPROOF='+repr(pins['FL_root_sha256'])+'\nIDS='+repr(flids)+'\n'
tail="""
if __name__=='__main__':
 try:
  p=argparse.ArgumentParser();p.add_argument('phase',choices=['collect','verify','finalize']);p.add_argument('--review',type=Path,required=True);p.add_argument('--review-sha256',required=True);a=p.parse_args()
  assert sha(a.review)==a.review_sha256
  review=read(a.review);assert review['source_adoptable'] is True and review['exact_selected_ids']==IDS and review['prepared_seal_sha256']==sha(H/'PREPARED_FILES_SHA256.json')
  for n,pin in read(H/'PREPARED_FILES_SHA256.json')['files'].items():assert sha(H/n)==pin['sha256']
  g=reused();g['source_check']()
  assert sha(g['B']/'LATEST_BACKUP.json')==sha(H/'PREVIOUS_LATEST.json')
  if a.phase=='collect':
   save('F_VOLUME_BEFORE_COLLECT.json',g['storage'](0));g['parent'](IDS).collect()
  else:g[a.phase]()
 except BaseException as e:
  if not isinstance(e,SystemExit):save('FAILURE_'+str(__import__('time').time_ns())+'.json',dict(error=repr(e),traceback=traceback.format_exc(),automatic_retry=False))
  raise
"""
write(H/'run_once.py',header+body+'\n'+tail)
ns=runpy.run_path(str(H/'run_once.py'),run_name='metadata_prepare_only');g=ns['reused']();g['source_check']()
assert g['H']==H and g['S'].resolve().drive.upper()=='F:'
save(H/'PREPARE_CHECKS.json',dict(status='SOURCE_METADATA_ONLY_READY_NOT_EXECUTED',exact_selected_ids=flids,prior_count=38,old38_excluded=True,per_ID_scientific_loop_bytes_exact=True,archive_strict_body_bytes_exact=True,previous_schema_package_sha256_present=read(Path(pins['FL_offserver_path']))['package_sha256']==snap['FLGMM']['source']['sha256'],volume=g['storage'](0),source_review_required=True))
# Gradient uses the actual adopted after10 original readers. Only metadata counts,
# predecessor pins and namespace/archive labels change; per-result loops stay exact.
gp=R/'tmp/celeba_gradient64_delta_after10_20261010'
for n,p in [('PREVIOUS_ROOT_ADOPTION.json',R/pins['gradient_root_path']),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',R/pins['gradient_offserver_path']),('pinned_collect_one.py',gp/'pinned_collect_one.py')]:
 with (G/n).open('xb') as out:out.write(p.read_bytes())
a=read(gp/'AUTHORIZED_SNAPSHOT.json');a.update(status='PARENT_AUTHORIZED_FIXED_ONE_SNAPSHOT_TERMINAL_MINUS_ACCEPTED18_NOT_ADOPTION',snapshot_sha256=sha(G/'SNAPSHOT.json'),snapshot_utc=snap['utc'],accepted_prior_ids=read(R/pins['gradient_root_path'])['accepted_ids'],accepted_prior_count=18,authorized_ids=gids,terminal_count=23,prior_root_path=pins['gradient_root_path'],prior_root_sha256=pins['gradient_root_sha256'],prior_offserver_path=pins['gradient_offserver_path'],prior_offserver_sha256=pins['gradient_offserver_sha256'],parent_authorization='Explicit root bounded task: one actual frozen terminal-minus18 snapshot; original strict; CPU110 serial; no shared adoption/STATE/Git')
save(G/'AUTHORIZED_SNAPSHOT.json',a)
diff=[];checks=[]
for name in ['collect_delta.py','verify_delta.py','run_once.py','finalize.py']:
 original=(gp/name).read_text('utf8');new=original
 pairs=[('gradient64_delta_after10.tar.gz','gradient64_delta_after18.tar.gz'),('old10_ordered_prefix_exact','old18_ordered_prefix_exact'),('accepted_before=10','accepted_before=18'),("prior['accepted_total']==10","prior['accepted_total']==18"),('accepted_total=10+len(ids)','accepted_total=18+len(ids)'),('strict_cumulative=10+len(ids)','strict_cumulative=18+len(ids)')]
 # These exact metadata lexemes never occur inside any scientific per-ID loop.
 pairs += [("p['accepted_before']==10","p['accepted_before']==18"),("p['accepted_total']==18","p['accepted_total']==23"),('strict_offserver_cumulative=18','strict_offserver_cumulative=23'),('root_accepted_cumulative_before_this_delivery=10','root_accepted_cumulative_before_this_delivery=18'),('accepted_total=18,','accepted_total=23,'),('previous10','previous18'),('Previous10','Previous18'),('accepted10','accepted18'),('Actual10','Actual18')]
 pairs += [('PARENT_AUTHORIZED_FIXED_GRADIENT64_DELTA_AFTER10','PARENT_AUTHORIZED_FIXED_GRADIENT64_DELTA_AFTER18'),("len(auth['authorized_ids'])==8","len(auth['authorized_ids'])==5"),("n==len(a['authorized_ids'])==8","n==len(a['authorized_ids'])==5"),('EXACT8','EXACT5'),('exact8','exact5'),('new8','new5'),('fixed 18 terminal jobs minus the root-adopted accepted18','fixed 23 terminal jobs minus the root-adopted accepted18'),('cumulative strict/offserver18/64','cumulative strict/offserver23/64'),('total=18,','total=23,')]
 for before,after in pairs:new=new.replace(before,after)
 new=new.replace('99a07fefad41e0f441288f86c5335e175eb5992983fec047ab3a1c9f1eacea57',pins['gradient_root_sha256']).replace('b07ce31865d243d0ac7447cbaa860197046e33e5323c3f078b966a36e80970ac',pins['gradient_offserver_sha256'])
 assert scientific_loops(original)==scientific_loops(new),(name,'scientific loop drift')
 compile(new,str(G/name),'exec');write(G/name,new)
 diff.extend(difflib.unified_diff(original.splitlines(True),new.splitlines(True),fromfile='adopted_after10/'+name,tofile='frozen_after18/'+name))
 checks.append(dict(file=name,parent_sha256=sha(gp/name),new_sha256=sha(G/name),scientific_per_ID_loop_bytes_exact=True))
write(G/'SOURCE_DIFF.patch',''.join(diff))
save(G/'SOURCE_REUSE.json',dict(status='METADATA_REBIND_ONLY_SOURCE_REVIEW_REQUIRED',parents=checks,old18_ids_excluded=True,new_ids=gids,source_package_sha256=a['source_seal_sha256'],original_acceptor_sha256='2c5d7699c6e9967d32c9080fb56b4672beb821cee0b20187c68195fb37d204e9',original_archive_verifier_sha256='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef',no_CNN_or_science_execution=True,volume=g['storage'](0)))
for home,filename in [(H,'PREPARED_FILES_SHA256.json'),(G,'SOURCE_FILES_SHA256.json')]:
 files={p.name:dict(sha256=sha(p),bytes=p.stat().st_size) for p in sorted(home.iterdir()) if p.is_file()}
 save(home/filename,dict(status='PREPARED_SOURCE_ONLY_NOT_ACCEPTANCE',files=files))
print(json.dumps(dict(FL_prepared_seal_sha256=sha(H/'PREPARED_FILES_SHA256.json'),gradient_source_seal_sha256=sha(G/'SOURCE_FILES_SHA256.json'),FL_ids=flids,gradient_ids=gids)))
