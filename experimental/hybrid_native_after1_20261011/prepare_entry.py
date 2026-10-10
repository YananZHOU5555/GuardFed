from pathlib import Path
import ast,difflib,hashlib,json
R=Path(__file__).resolve().parents[2]; H=Path(__file__).resolve().parent
O=R/'tmp/celeba_hybrid_fullcoverage_first_delta_20261010'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def put(n,v):
 with (H/n).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n')
def copy(src,n):
 with (H/n).open('xb') as f:f.write(src.read_bytes())
IDS=[f'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed{s}_fullcoverage' for s in range(91003,91011)]
L=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/root_five_queue_20261010T155515Z.raw.json'
assert sha(L)=='8884a432ecfd71b7d183357ba2663e336a3a7b89c3be4dc00ab7abed87f9b0c7'
live=read(L)['Hybrid96'];rows=live['terminal_records'];selected=[x for x in rows if x['id'] in IDS]
assert [x['id'] for x in selected]==IDS and len(rows)==9 and all(x['acceptance']['status']=='PASS' for x in selected)
P=R/'tmp/celeba_hybrid_first1_root_adoption_20261010/ROOT_ADOPTION.json'
PS=sha(P);assert PS=='04d62c367609d1c6d079ff538f45a575bff368944d4833f47fd1cf28159c5fa4'
assert read(P)['cumulative_accepted']==1 and read(P)['accepted_new_ids']==[IDS[0].replace('91003','91002')]
copy(P,'PRIOR_ROOT_ADOPTION.json');copy(O/'OFFSERVER_VERIFICATION.json','PRIOR_OFFSERVER.json')
assert sha(H/'PRIOR_OFFSERVER.json')=='16dc834a07c7b54e439a1c2b99d0dc528eec0d631b50a9e8fecd3c06735f2755'
for n in ['ROOT_BOUND_ADOPTION.json','ROOT_SEVEN_CANARY_CLOSURE.json','ROOT_COVERAGE_STARTUP.json','GUIDE.md','verify_saved.py','storage.py']:copy(O/n,n)
remote='/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/native_after1_20261011'
bulk='F:/YananResearchStorage/GuardFed/hybrid_native_after1_20261011';assert not Path(bulk).exists()
put('DELTA_SCOPE.json',dict(status='FIXED_CLOSED_EXACT8_AUTHORIZED_INCREMENT_NOT_EXECUTED',ids=IDS,accepted_before=1,prior_offserver_path='PRIOR_OFFSERVER.json',prior_offserver_sha256=sha(H/'PRIOR_OFFSERVER.json'),remote_output=remote+'/output',local_bulk_root=bulk,no_later_completions=True))
changes={}
for n in ['collect_once.py','restore_verify.py']:
 b=(O/n).read_bytes();c=b.replace(b'107',b'108');assert c!=b and c.replace(b'108',b'107')==b
 with (H/n).open('xb') as f:f.write(c)
 changes[n]={'original_sha256':sha(O/n),'new_sha256':sha(H/n),'only_CPU107_to_CPU108':True}
n='execute_transport.py';b=(O/n).read_text('utf8');s=b.replace('hybrid_fullcoverage_first_delta_20261010','hybrid_native_after1_20261011').replace("/first_delta'","/native_after1_20261011'").replace('107','108')
s=s.replace("assert sha(H/'ACTUAL_ROOT_APPROVAL.json')=='32cf2c69c55d855975114c5c9a78cc8f9f3ade7f57a0808e1f54820406a35e67'", "assert sha(H/'PRIOR_ROOT_ADOPTION.json')=='"+PS+"'\n assert read(H/'PRIOR_ROOT_ADOPTION.json')['cumulative_accepted']==1")
s=s.replace("assert sha(H/'FILES_SHA256.json')=='a098241183b2c6ca4041e19a5844c7f281d4494e220cea5488d1e91f9832ceb6'", "assert len(sys.argv)==3 and len(sys.argv[2])==64 and all(c in '0123456789abcdef' for c in sys.argv[2])\n assert sha(H/'FILES_SHA256.json')==sys.argv[2]")
s=s.replace("assert read(H/'DELTA_SCOPE.json')['ids']==['CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage']", "assert read(H/'DELTA_SCOPE.json')['ids']=="+repr(IDS)+"\n assert read(H/'DELTA_SCOPE.json')['accepted_before']==1\n assert read(H/'PRIOR_OFFSERVER.json')['accepted_ids']==read(H/'PRIOR_ROOT_ADOPTION.json')['accepted_new_ids']")
s=s.replace("+['FILES_SHA256.json','ACTUAL_ROOT_APPROVAL.json']", "+['FILES_SHA256.json']")
s=s.replace("'--source-seal-sha256','a098241183b2c6ca4041e19a5844c7f281d4494e220cea5488d1e91f9832ceb6'", "'--source-seal-sha256',pins['FILES_SHA256.json']")
assert 'ACTUAL_ROOT_APPROVAL' not in s and 'a098241183b2c6ca' not in s
with (H/n).open('x',encoding='utf8',newline='\n') as f:f.write(s)
changes[n]={'original_sha256':sha(O/n),'new_sha256':sha(H/n),'changes':'E/F/remote namespace;CPU108;exact8;actual parent1;external CLI source seal'}
put('FROZEN_OBSERVATION.json',dict(source_path=L.relative_to(R).as_posix(),sha256=sha(L),observation_utc=live['checked_utc'],terminal_count=9,selected_records=selected,selected_ids=IDS,not_new_acceptance=True))
put('INPUT_PINS.json',dict(prior_root_path=P.relative_to(R).as_posix(),prior_root_sha256=PS,prior_offserver_sha256=sha(H/'PRIOR_OFFSERVER.json'),original_first1_source_seal_sha256=sha(O/'FILES_SHA256.json'),original_tool_hashes={n:sha(O/n) for n in ['collect_once.py','verify_saved.py','storage.py','restore_verify.py','execute_transport.py']},package_sha256='a87a050b497a184efbe18b4649ad6bde40b9ea29a16f9e3efddd0ef1156e3b04',guide_sha256=sha(H/'GUIDE.md'),new_ids=IDS,reused_separate=4))
diff=[]
for n in ['collect_once.py','restore_verify.py','execute_transport.py']:diff.extend(difflib.unified_diff((O/n).read_text('utf8').splitlines(True),(H/n).read_text('utf8').splitlines(True),fromfile='first1/'+n,tofile='after1/'+n))
(H/'SOURCE_DIFF.patch').write_text(''.join(diff),encoding='utf8',newline='\n')
compiled=[]
for f in H.glob('*.py'):compile(f.read_text('utf8'),str(f),'exec');compiled.append(f.name)
assert (H/'verify_saved.py').read_bytes()==(O/'verify_saved.py').read_bytes() and (H/'storage.py').read_bytes()==(O/'storage.py').read_bytes()
put('SOURCE_CHECK.json',dict(status='PASS_LOCAL_METADATA_REBIND_AND_COMPILE_NOT_SCIENTIFIC_ACCEPTANCE',compiled=sorted(compiled),verify_saved_byte_exact=True,storage_byte_exact=True,collect_once_and_restore_inverse_CPU_binding_exact=True,changes=changes,exact8=True,accepted_before=1,root_adopted_by_this_package=False,execution_authority='Parent bounded explicit task:exact8 collection once,CPU108,no shared writes'))
(H/'README.md').write_text('Exact8 Hybrid70 native saved-record closure:IID Benign seeds91003–91010 only. Prior1(seed91002) and4 reuse models excluded from backup. Original verify_saved.py/storage.py byte exact; collector/restore CPU108 metadata only, transport exact-delta/namespace/actual-parent and external seal binding. Run execute_transport.py collect,then download,then verify,with actual FILES_SHA256.json SHA as second CLI argument. Fail-stop;no CNN/fit/train/test,queue/STATE/LATEST/Git writes. Bulk only F after fresh volume/health/capacity guards. Runtime bridge is record/tensor identity,not numerical-environment equivalence. No prediction arrays supplied or recomputed. Preparation001 KeyError occurred before source-copy or SSH and is retained.\n',encoding='utf8',newline='\n')
put('FILES_SHA256.json',dict(status='FROZEN_EXACT8_AUTHORIZED_NATIVE_CLOSURE_SOURCE_NOT_EXECUTED',files={f.name:{'sha256':sha(f),'bytes':f.stat().st_size} for f in sorted(H.iterdir()) if f.is_file()}))
print(json.dumps(dict(source_seal_sha256=sha(H/'FILES_SHA256.json'),members=len(read(H/'FILES_SHA256.json')['files']),exact8_ids=IDS),indent=2))
