"""Compact source/identity preparation only. Never opens models/arrays or runs science."""
from pathlib import Path,PurePosixPath
import ast,copy,difflib,hashlib,importlib.util,json,re,sys
H=Path(__file__).resolve().parent;R=H.parents[1]
P=R/'tmp/hybrid_missing8_pool32_runtime_prepared_20261011'
S=R/'tmp/hybrid_missing8_saved_runtime_prepared_20261011'
B=R/'tmp/celeba_hybrid_three_view_bridge_20261011'
N=R/'tmp/hybrid_after12_FFlip7_prepare_20261011'
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
pin=lambda p:dict(path=Path(p).as_posix(),sha256=sha(p),bytes=Path(p).stat().st_size)
pending={};mapping={};sources={};edits={};diffs=[]
def put(n,v):
 raw=v if isinstance(v,bytes) else (v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n').encode()
 assert n not in pending and not (H/n).exists(),n
 pending[n]=raw
def copy_pin(v,rel):
 p=Path(v['path']);assert p.suffix in {'.json','.py'} and p.stat().st_size<2_000_000
 raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest()==v['sha256'] and len(raw)==v['bytes']
 if rel in pending:assert pending[rel]==raw
 else:put(rel,raw)
 mapping[p.as_posix()]=dict(kind='package',relative=rel,sha256=v['sha256'],bytes=v['bytes']);sources[rel]=v
def derive(p,rel,changes):
 raw=p.read_bytes();text=raw.decode();spans=[]
 for old,new,role in changes:
  assert old in text,(rel,role,old);positions=[m.start() for m in re.finditer(re.escape(old),text)];n=len(positions);text=text.replace(old,new);spans.append(dict(old=old,new=new,count=n,positions=[pos+i*(len(new)-len(old)) for i,pos in enumerate(positions)],role=role))
 inverse=text
 for x in reversed(spans):
  for pos in reversed(x['positions']):
   assert inverse[pos:pos+len(x['new'])]==x['new'],(rel,x['role'],pos);inverse=inverse[:pos]+x['old']+inverse[pos+len(x['new']):]
 assert inverse.encode()==raw
 compile(text,str(H/rel),'exec');put(rel,text);edits[rel]=dict(parent=pin(p),spans=spans,inverse_raw_exact=True)
 diffs.append(''.join(difflib.unified_diff(raw.decode().splitlines(True),text.splitlines(True),fromfile=str(p),tofile=str(H/rel))))
 return text
assert sha(P/'FILES_SHA256.json')=='f3b197ce786a3f391f820baf8767fa25fba2d2387002acf654bced7dc8e95239'
assert sha(S/'FILES_SHA256.json')=='89bf8df9e575a627c726cfadbc449bafcca841832a9f1e49e09a6a375c818ce8'
native_path=R/'tmp/celeba_hybrid_native19_FFlip10_root_adoption_20261011/ROOT_ADOPTION.json'
assert sha(native_path)=='ba51c5308427a512d13f3fc06ebd5dc006ec1e0002fecb9708228f2f4fa867b2'
native=read(native_path);old=read(B/'INPUTS.json');registry=copy.deepcopy(old)
prior_view=R/'tmp/hybrid_missing8_root_adoption_prepared_20261011/ROOT_SCIENTIFIC_ADOPTION.json'
assert sha(prior_view)=='02b5f2c8fd26808f62a9e94d1901f4587d09d13dcce29b118113ccc0e1374e06'
view=read(prior_view);assert view['root_adoption'] and view['Hybrid_total_three_view_records']==10
ids=[f'CosineFairness_lam20.0_tau0.1_lr0.001_IID_F Flip_seed{s}_fullcoverage' for s in range(91001,91011)]
assert native['accepted_ids']==old['exact_ids']+ids[3:] and native['cumulative_accepted']==19
assert native['accepted_new_ids']==ids[3:] and native['previous_root_sha256']==old['batches']['after9']['root']['sha256']
assert not set(ids)&set(view['accepted_ids'])
volume=json.loads(__import__('subprocess').check_output(['powershell','-NoProfile','-Command','Get-Volume -DriveLetter F | Select-Object FileSystemLabel,HealthStatus,SizeRemaining | ConvertTo-Json -Compress'],text=True))
assert volume['FileSystemLabel']=='Yanan 2TB' and volume['HealthStatus']=='Healthy' and volume['SizeRemaining']>1024**3
f=Path('F:/YananResearchStorage/GuardFed/hybrid_after12_FFlip7_prepare_20261011/verified')
batch=dict(root=pin(native_path),root_status=native['status'],strict=pin(N/'REMOTE_STRICT.json'),offserver=pin(N/'OFFSERVER_VERIFICATION.json'),receipt=pin(N/'BACKUP_RECEIPT.json'),members=pin(N/'MEMBERS.json'),scope=pin(f/'stage/full_scope.json'),body=pin(f/'stage/body.py'),prior=dict(previous_root_sha256=old['batches']['after9']['root'],previous_offserver_sha256=pin(N/'PRIOR_OFFSERVER.json')))
for k,key in [('strict','remote_strict_sha256'),('offserver','offserver_sha256'),('receipt','receipt_sha256')]:assert batch[k]['sha256']==native[key]
inventory=read(N/'MEMBERS.json')['files'];stage='/workspace/guardfed_checks/celeba_hybrid_fullcoverage_v3_20261010/stage'
for rid in ids[3:]:
 metadata={}
 for kind in ['job','result','provenance','acceptance','native_replay']:
  member='stage/jobs/'+rid+'.json' if kind=='job' else 'stage/runs/'+rid+'/'+kind+'.json'
  metadata[kind]=dict(path=(f/member).as_posix(),**inventory[member],member=member)
 member='stage/runs/'+rid+'/model.pt'
 registry['records'][rid]=dict(batch='after12',metadata=metadata,checkpoint=dict(**inventory[member],member=member,path=(f/member).as_posix(),server_path=stage+'/runs/'+rid+'/model.pt'),server_output=stage+'/runs/'+rid)
registry['batches']['after12']=batch;registry['exact_ids']=native['accepted_ids'];registry['scope']='PRIVATE_ACCEPTED_NATIVE19_IDENTITY_ONLY_FIXED_FFLIP10_REPLAY_SCOPE'
assert all(registry['records'][rid]==old['records'][rid] for rid in old['exact_ids'])
put('originals/HYBRID_INPUTS.json',registry)
for name in ['fl_candidate.py','hybrid_bridge.py','bridge.py','SOURCE_REUSE.json','evaluator.py','replay.py','core.py','saved_science.py','saved_output_audit.py','receipt_diff.py']:
 copy_pin(pin(P/'originals'/name),'originals/'+name)
reuse=read(P/'originals/SOURCE_REUSE.json')
for src,name in zip(reuse['function_sources'],['evaluator.py','replay.py']):
 assert src['file']['sha256']==sha(P/'originals'/name)
 mapping[src['file']['path']]=dict(kind='package',relative='originals/'+name,sha256=src['file']['sha256'],bytes=src['file']['bytes'])
mapping[reuse['original_core']['path']]=dict(kind='package',relative='originals/core.py',sha256=sha(P/'originals/core.py'),bytes=(P/'originals/core.py').stat().st_size)
for key,name in [('original_bridge','bridge.py'),('original_reuse','SOURCE_REUSE.json')]:mapping[registry[key]['path']]=dict(kind='package',relative='originals/'+name,sha256=registry[key]['sha256'],bytes=registry[key]['bytes'])
copy_pin(registry['bound_root'],'proofs/bound_root.json');copy_pin(pin(prior_view),'proofs/prior_Benign10_root.json')
for name in ['after9','after12']:
 for key in ['root','strict','offserver','receipt','members','scope','body']:
  copy_pin(registry['batches'][name][key],'proofs/'+name+'_'+key+Path(registry['batches'][name][key]['path']).suffix)
 for key,v in registry['batches'][name].get('prior',{}).items():copy_pin(v,'proofs/'+name+'_'+key+'.json')
private_spec=importlib.util.spec_from_file_location('_hybrid_fFlip_metadata_identity',B/'bridge.py');private=importlib.util.module_from_spec(private_spec);private_spec.loader.exec_module(private);private.inputs=lambda:registry
records=[];repo_hashes={}
for rid in ids:
 identity=private.identity_record(rid);selected=registry['records'][rid];artifacts={}
 assert identity['attack']=='F Flip' and identity['distribution']=='IID' and identity['seed']==91001+ids.index(rid)
 for kind,v in selected['metadata'].items():
  server=stage+'/jobs/'+rid+'.json' if kind=='job' else stage+'/runs/'+rid+'/'+Path(v['path']).name
  mapping[v['path']]=dict(kind='server',server_path=server,sha256=v['sha256'],bytes=v['bytes']);artifacts[kind]=dict(v,server_path=server)
 artifacts['model']=selected['checkpoint'];model=selected['checkpoint'];mapping[model['path']]=dict(kind='server',server_path=model['server_path'],sha256=model['sha256'],bytes=model['bytes'])
 for rel,h in identity['source_hashes'].items():
  if rel.startswith(('scripts/','src/','data/celeba/')):
   assert rel not in repo_hashes or repo_hashes[rel]==h;repo_hashes[rel]=h
 records.append(dict(id=rid,method='CosineFairnessHybrid',distribution='IID',attack='F Flip',seed=identity['seed'],terminal_round=70,split='valid',n_eval=19867,actual_alpha=5000.0,identity=identity,runtime_output=selected['server_output'],runtime_artifacts=artifacts,original_training_torch=identity['training_torch'],original_training_device=identity['config']['device']))
assert 'torch' not in sys.modules
input_bytes=pending['originals/HYBRID_INPUTS.json'];input_pin=dict(path=(H/'originals/HYBRID_INPUTS.json').as_posix(),sha256=hashlib.sha256(input_bytes).hexdigest(),bytes=len(input_bytes))
mapping[input_pin['path']]=dict(kind='package',relative='originals/HYBRID_INPUTS.json',sha256=input_pin['sha256'],bytes=input_pin['bytes'])
base='/workspace/guardfed_checks/hybrid_FFlip10_three_view_prepare_20261011';program='guardfed_hybrid_FFlip10_valid'
common=[('exact8','exact10','scope label'),('Exact8','Exact10','scope label'),('celeba_hybrid_three_view_missing8_20261011',H.name,'namespace'),('hybrid_missing8','hybrid_FFlip10','namespace'),('HYBRID_MISSING8','HYBRID_FFLIP10','scope status'),('HYBRID_ADOPTED_BENIGN_MISSING8_CANARY_FIRST_SOURCE_ONLY','HYBRID_ADOPTED_FFLIP10_CANARY_FIRST_SOURCE_ONLY','scope'),('Exactly8 adopted Hybrid IID Benign checkpoints; first91003 canary then seven sequentially;','Exactly10 adopted Hybrid IID F Flip checkpoints; first91001 canary then nine sequentially;','scope text'),('IID_Benign_seed%d_fullcoverage','IID_F Flip_seed%d_fullcoverage','fixed attack'),('range(91003, 91011)','range(91001, 91011)','fixed seeds'),('previously_accepted_skip_ids\'] == [IDS[0].replace(\'91003\', \'91002\')]','previously_accepted_skip_ids\'] == []','no FFlip prior replay'),('1434b40de5116bf3d53bc5a6ae2bd3b90f4e54222f23ad099ff72b1fd3ef1775',sha(native_path),'actual native19 root'),('631d7ee2acf1cbe3453d849523e262f29ff5b354b5ddb56c60e5779e94364456',sha(prior_view),'actual prior Benign10 replay'),("(METHOD, 'IID', 'Benign', 91003 + IDS.index(row['id']), 70, 19867)","(METHOD, 'IID', 'F Flip', 91001 + IDS.index(row['id']), 70, 19867)",'accepted identity'),('First91003 must pass before the remaining seven.','First91001 must pass before the remaining nine.','scope text'),('Canary failed; remaining seven forbidden','Canary failed; remaining nine forbidden','scope text')]
candidate_changes=common+[("CPUS = None  # Root binds an 8-CPU mask or a same-socket 32-CPU pool; computation stays 8 threads.","CPUS = None; SMT_SIBLINGS = None  # Actual same-socket32 pool/siblings late-bound by root; computation stays8.",'external resource binding'),('global CPUS','global CPUS, SMT_SIBLINGS','external resource binding'),("CPUS == list(range(64, 96))","CPUS == sorted(set(CPUS)) and len(CPUS) == 32 and min(CPUS) >= 0",'pool late binding'),('Fixed64..95 pool required; computation remains8 threads','Explicit same-socket32 pool required; computation remains8 threads','pool late binding'),("auth, p, review = read(a.authorization), read(a.preflight), read(a.source_review)","auth, p, review = read(a.authorization), read(a.preflight), read(a.source_review)\n    SMT_SIBLINGS = p['smt_sibling_cpus']\n    require(SMT_SIBLINGS == sorted(set(SMT_SIBLINGS)) and len(SMT_SIBLINGS) == 32 and not set(SMT_SIBLINGS).intersection(CPUS) and auth['smt_sibling_cpus'] == SMT_SIBLINGS, 'Actual root-bound32 SMT siblings required')",'actual topology late binding'),('list(range(320, 352))','SMT_SIBLINGS','actual siblings'),('set(range(320, 352))','set(SMT_SIBLINGS)','actual siblings')]
candidate=derive(P/'candidate.py','candidate.py',candidate_changes)
manifest=dict(scope='HYBRID_ADOPTED_FFLIP10_CANARY_FIRST_SOURCE_ONLY',status='PREPARED_NOT_EXECUTED_NOT_DISPATCH_AUTHORIZED',exact_ids=ids,canary_id=ids[0],previously_accepted_skip_ids=[],excluded_screen91001_identity_gap=False,private_inputs_pin=input_pin,native_root_pin=pin(native_path),existing_replay_root_pin=pin(prior_view),preserved_Benign10_ids=view['accepted_ids'],preserved_Benign10_root_sha256=sha(prior_view),records=records,path_map=mapping,server_repo='/workspace/GuardFed-celeba-expanded',server_python='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python',server_namespace=base,runtime_repo_hashes=repo_hashes,cpu_affinity=None,smt_sibling_cpus=None,threads=8,device='cpu',dtype='float32',views=['native','raw','shared_calibration'],native_tolerance=1e-12,new_three_view_accepted=0,dispatch_authorized=False,sources=sources,pending=['actual source review','actual same-socket32 pool and32 SMT sibling binding and fresh eligible-capacity preflight','root authorization','new10 original Linux CPU replay/root-only calibration','Linux original whole saved check','F exact10 saved-array transport','Windows original zero-fit saved-output audit','root complementary evidence adoption'])
validation=next(n for n in ast.parse(candidate).body if isinstance(n,ast.FunctionDef) and n.name=='validate_manifest')
ns=dict(IDS=ids,METHOD='CosineFairnessHybrid',PurePosixPath=PurePosixPath,require=private.require);exec(compile(ast.Module(body=[validation],type_ignores=[]),'metadata_validate_manifest','exec'),ns);ns['validate_manifest'](manifest)
put('MANIFEST.json',manifest)
saved_changes=[('exact8','exact10','scope label'),('Exact8','Exact10','scope label'),('HYBRID_MISSING8','HYBRID_FFLIP10','scope status'),('guardfed_hybrid_missing8_valid',program,'service namespace'),('hybrid_three_view_missing8','hybrid_FFlip10_three_view_prepare','namespace'),("attack='Benign'","attack='F Flip'",'accepted header'),("linux['cached_root_refits'] == 8","linux['cached_root_refits'] == 10",'exact10 metadata count'),('cached_root_refits=8 if linux_mode else 0, fit_calls=8 if linux_mode else 0','cached_root_refits=10 if linux_mode else 0, fit_calls=10 if linux_mode else 0','exact10 metadata count'),('records=8, fit_calls=','records=10, fit_calls=','exact10 metadata count'),('Exact19 saved members','Exact23 saved members','exact10 inventory label')]
saved=derive(P/'check_saved.py','check_saved.py',saved_changes)
put('SOURCE_PINS.json',dict(native_root=pin(native_path),prior_Benign10_root=pin(prior_view),native_prior3_batch=registry['batches']['after9']['root'],native_new7_batch=batch['root'],original_function_sources=reuse['function_sources'],original_hybrid_bridge=pin(B/'bridge.py'),private_identity_record_raw_exact=True,scientific_functions_changed=0,original_fullcoverage_package_sha256=native['package_sha256'],metadata_only_F_volume=volume,models_or_arrays_opened=0,scientific_execution=0))
# The runtime package seal is separate from later helper/prepare-delivery seal to avoid hash cycles.
package_files={n:dict(sha256=hashlib.sha256(v).hexdigest(),bytes=len(v)) for n,v in pending.items()}
package=dict(status='HYBRID_FFLIP10_SOURCE_ONLY_NOT_AUTHORIZED',files=package_files)
put('FILES_SHA256.json',package);package_sha=hashlib.sha256(pending['FILES_SHA256.json']).hexdigest();check_sha=hashlib.sha256(pending['check_saved.py']).hexdigest()
oldpackage='f3b197ce786a3f391f820baf8767fa25fba2d2387002acf654bced7dc8e95239';oldcheck='1410e344f07cb7d5987dcf773de4dab968bb3f03d21b378f59704e67074bd924'
for name in ['contract.py','check_linux.py','linux_saved_remote.py','transport.py','transport_remote.py']:
 raw=(S/name).read_text();changes=[]
 for oldv,newv,role in [('exact8','exact10','scope label'),('Exact8','Exact10','scope label'),('celeba_hybrid_three_view_missing8_20261011',H.name,'namespace'),('hybrid_missing8_pool32_runtime_prepared_20261011',H.name,'local namespace'),('HYBRID_MISSING8','HYBRID_FFLIP10','scope status'),('hybrid_missing8','hybrid_FFlip10','namespace'),(oldpackage,package_sha,'actual prepared package'),(oldcheck,check_sha,'actual prepared consumer')]:
  if oldv in raw:changes.append((oldv,newv,role));raw=raw.replace(oldv,newv)
 if name=='contract.py':changes+=[('ROOT = HERE.parents[1]','ROOT = HERE.parents[2]','helper subdirectory'),("CANDIDATE = ROOT/'tmp/"+H.name+"'","CANDIDATE = HERE.parent",'helper subdirectory'),('len(set(IDS))==8','len(set(IDS))==10','exact10 count'),("value['cached_root_refits']==8","value['cached_root_refits']==10",'exact10 count'),("value['fit_calls']==8","value['fit_calls']==10",'exact10 count')]
 if name=='check_linux.py':changes+=[("'records':8","'records':10",'exact10 count')]
 if name=='transport.py':changes+=[('==len(expected)+1==20','==len(expected)+1==24','exact10 archive layout'),('member_count=20','member_count=24','exact10 archive layout'),("'members':20","'members':24",'exact10 archive layout')]
 if name=='transport_remote.py':changes+=[(repr(read(P/'MANIFEST.json')['exact_ids']),repr(ids),'exact10 fixed IDs'),("len(linux['records'])==8","len(linux['records'])==10",'exact10 count'),("linux['cached_root_refits']==8","linux['cached_root_refits']==10",'exact10 count')]
 derive(S/name,'saved/'+name,changes)
put('saved/FIXED_IDS.json',dict(exact_ids=ids))
wrapper=R/'tmp/hybrid_missing8_saved_root_execution_20261011/windows_runtime_wrapper.py'
derive(wrapper,'windows_runtime_wrapper.py',[("tmp/hybrid_missing8_pool32_runtime_prepared_20261011","tmp/"+H.name,'local package namespace'),(oldpackage,package_sha,'actual prepared package'),(oldcheck,check_sha,'actual prepared consumer')])
put('SOURCE_ADAPTATIONS.json',edits);put('SOURCE_DIFF.patch',''.join(diffs))
compiled=[]
for n,v in pending.items():
 if n.endswith('.py'):compile(v,str(H/n),'exec');compiled.append(n)
def functions(raw):
 text=raw.decode();return {n.name:ast.get_source_segment(text,n) for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
prior_funcs=functions((P/'candidate.py').read_bytes());new_funcs=functions(pending['candidate.py'])
unchanged=[n for n in prior_funcs if prior_funcs[n]==new_funcs[n]]
assert set(prior_funcs)-set(unchanged)=={'exclusive_cpus','runtime_nodes','validate_manifest','authorize','run'}
for name in ['parent_functions','resolve_origin','bound_bridge','runtime_record','check_inputs']:assert name in unchanged
for name in ['selected_originals','f_volume','contained_f','save']:assert functions((P/'check_saved.py').read_bytes())[name]==functions(pending['check_saved.py'])[name]
science={}
for src,name in zip(reuse['function_sources'],['evaluator.py','replay.py']):
 text=pending['originals/'+name].decode();nodes={n.name:n for n in ast.parse(text).body if isinstance(n,ast.FunctionDef)}
 for fn,digest in src['functions'].items():assert hashlib.sha256(ast.get_source_segment(text,nodes[fn]).encode()).hexdigest()==digest;science[fn]=digest
assert len(science)==17 and 'torch' not in sys.modules
put('SOURCE_CHECK.json',dict(status='SOURCE_ONLY_COMPILE_METADATA_JOIN_RAW_INVERSE_ORIGINAL_SCIENCE_REUSE_PASS_NOT_EXECUTED',compiled=compiled,private_identity_record_source_byte_exact=True,science17_functions= science,original_saved_whole_source_sha256='d512e5b2b6614b762d921dd94b2b5162687c0bbddde4caaf8b584b3b22dba745',candidate_unchanged_functions=unchanged,changed_candidate_functions_metadata_resource_only=['exclusive_cpus','runtime_nodes','validate_manifest','authorize','run'],full_source_inverse_raw_exact=list(edits),parent_Benign10_root_byte_exact=True,parent_native12_records_unchanged=True,FFlip_native10_identity_join=True,FFlip_native_batches=[3,7],reused_FFlip_three_view_arrays=0,new_Linux_replay_and_root_only_calibration_required=True,Windows_zero_fit_only=True,archive_expected_payload_members=23,archive_expected_ZIP_members=24,actual_gate_SHA=None,actual_Linux_SHA=None,actual_transport_SHA=None,actual_Windows_SHA=None,SSH=0,F_write=0,model_array_reads=0,forward=0,fit=0,training=0,statistics=0,dispatch_authorized=False,new_three_view_accepted=0))
put('ROOT_INPUTS.template.json',dict(status='TEMPLATE_ONLY_NO_EXECUTION_OR_ACCEPTANCE',package_sha256=package_sha,manifest_sha256=hashlib.sha256(pending['MANIFEST.json']).hexdigest(),native19_root=pin(native_path),prior_Benign10_root=pin(prior_view),exact_ids=ids,cpu_affinity=None,smt_sibling_cpus=None,source_review=None,preflight=None,authorization=None,start_receipt=None,gate=None,Linux_whole=None,F_transport=None,Windows_zero_fit=None,Windows_command=None,Windows_exit=None,exact_diagnostic_map=None,root_adoption_authorized=False,Windows_whole_refit_pass=False,full100_complete=False,final_test=False))
for n,v in pending.items():
 p=H/n;p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(v)
print(json.dumps(dict(status='SOURCE_ONLY_PREPARED_NOT_EXECUTED',package_sha256=package_sha,manifest_sha256=sha(H/'MANIFEST.json'),native_records=10,prior_three_view_preserved=10,root_calibration_execution=0,files_written=len(pending)),indent=2))
