"""Prepare exact native150 minus closed147 C3; metadata/archive reads only."""
import ast,copy,difflib,hashlib,importlib.util,json,re,sys,tarfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=HERE.with_name('celeba_mechanism_valid_C_after40_20261010');EX=OLD/'execution_candidate';OUTEX=HERE/'execution_candidate'
SCOPE='MECHANISM_TERMINAL_VALID_REPLAY_C_AFTER47'
SELECTED = ['minus_C_IID_Sp-DFA_seed91008', 'minus_C_IID_Sp-DFA_seed91009', 'minus_C_IID_Sp-DFA_seed91010']
PINS={}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,expected=None):
 p=Path(p);h=sha(p);assert expected is None or h==expected,str(p)
 PINS[p.relative_to(ROOT).as_posix()]={'sha256':h,'bytes':p.stat().st_size};return json.loads(p.read_text(encoding='utf-8-sig'))
def text(p):
 p=Path(p);PINS[p.relative_to(ROOT).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size};return p.read_text(encoding='utf-8-sig')
def write(name,value):
 p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
 payload=value if isinstance(value,str) else json.dumps(value,indent=2,allow_nan=False)+'\n'
 with p.open('x',encoding='utf-8',newline='\n') as f:f.write(payload)
def load(name,p):
 spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def rewrite_assign(source,name,value):
 n=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
 return source.replace(ast.get_source_segment(source,n),name+' = '+repr(value),1)
def seal(names,**meta):return dict(meta,members=[{'path':n,'sha256':sha(HERE/n),'size':(HERE/n).stat().st_size} for n in names])

for folder,name,pin in [(OLD,'FILES_SHA256.json','22da16b734f2c6041d494e30c200d586ff79626ac0a5ec6b438bea405c7dc48b'),(EX,'EXECUTION_SOURCE_SHA256.json','1ac84b16f00c30ed3effd230f539824bb426540e66ec4981a53fdee676c50ec8')]:
 for row in read(folder/name,pin)['members']:
  p=folder/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['size'];text(p)
b=load('closed_C_after40_metadata',OLD/'bridge.py')
prior=read(OLD/'inventory_actual147_Full100refs.json','c65a6e8a88ef1e97a687bbf259840582c3d1f4c26e8e09deb604e94a4cc5b10f')
adoption_path=EX/'backups/incremental_20261009T233028Z/ROOT_ADOPTION_REVIEW.json'
adoption=read(adoption_path,'64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea')
assert adoption['cumulative_three_view_models']==147 and adoption['accepted_new']==7 and adoption['original140_unchanged'] and adoption['all_native_differences_zero']
ni=read(HERE/'NATIVE_INPUTS.json');assert ni['status']=='ACTUAL_STRICT_OFFSERVER_NATIVE150_C_AFTER47_INPUTS' and ni['selected_ids']==SELECTED
loaded={}
for name,pin in ni['files'].items():
 p=ROOT/pin['path'];assert sha(p)==pin['sha256'];PINS[pin['path']]={'sha256':sha(p),'bytes':p.stat().st_size}
 if p.suffix=='.json':loaded[name]=read(p,pin['sha256'])
inspection,receipt,proof=loaded['inspection'],loaded['receipt'],loaded['proof']
assert loaded['root_delta']['total_new_strict_and_offserver']==150 and loaded['root_delta']['new_ids']==SELECTED
review=loaded['root_review'];assert review['native_accepted']==review['ledger_accepted_unique_ids']==150 and review['original247_raw_json_record_bytes_exact']
assert review['added_ids']==SELECTED and review['inspection_sha256']==ni['files']['inspection']['sha256'] and review['ledger_sha256']==ni['files']['ledger']['sha256']
assert review['source_root_proof_sha256']==ni['files']['root_delta']['sha256']
assert inspection['new_count']==150 and inspection['reused_count']==100 and not inspection['invalid']
assert inspection['manifest_sha256']==b.MANIFEST_SHA and inspection['source_script_sha256']==b.EVIDENCE_V4_SHA and inspection['full_inventory_sha256']==b.BASELINE_INVENTORY_SHA
assert receipt['accepted_new_ids']==proof['accepted_new_ids']==SELECTED and proof['pass'] and proof['different_host_observed']
assert proof['archive_sha256']==receipt['archive_sha256']==ni['files']['archive']['sha256']
assert loaded['ledger']['entries'][-1]['receipt_sha256']==ni['files']['receipt']['sha256']
assert receipt['previous_receipt_sha256']==prior['backup_chain'][-1]['receipt_sha256'] and receipt['reused_full_weights_repacked']==0 and not receipt['failure_identities']
rows={r['id']:r for r in inspection['records'] if r['role']=='new'};closed=[r['id'] for r in prior['records']]
assert len(rows)==150 and len(closed)==147 and set(rows)-set(closed)==set(SELECTED)
assert all(rows[r['id']]==r['accepted_v4_row'] for r in prior['records'])
assert len([i for i in rows if i.startswith('minus_U_')])==100 and len([i for i in rows if i.startswith('minus_C_')])==50
baseline=read(ROOT/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json',b.BASELINE_INVENTORY_SHA)
manifest=read(ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/manifest.json',b.MANIFEST_SHA)
entries={r['id']:r for r in manifest['jobs']};full={b.cell(r):r for r in baseline['records'] if r['method']=='GuardFed-AD2+'}
ep=ROOT/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py';text(ep);assert sha(ep)==b.EVIDENCE_V4_SHA;evidence=load('native_v4_C_after47_metadata_only',ep)
archive=ROOT/ni['files']['archive']['path'];verified=evidence.verify_archive(archive,receipt);assert verified['members_verified']==proof['members_verified']==54
constructor_source=text(ROOT/'tmp/celeba_mechanism_valid_incremental_next11_20261009/prepare.py')
nodes=[n for n in ast.walk(ast.parse(constructor_source)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='record' for t in n.targets)]
assert len(nodes)==1;constructor=compile(ast.Module(body=nodes,type_ignores=[]),'original_record_constructor','exec')
artifacts={r['id']:copy.deepcopy(r) for r in prior['records']}
with tarfile.open(archive) as tar:
 members=json.load(tar.extractfile('backup_inventory.json'))['members']
 for model_id in SELECTED:
  result=json.load(tar.extractfile('runs/'+model_id+'/result.json'));job=json.load(tar.extractfile('jobs/'+model_id+'.json'))
  row,entry=rows[model_id],entries[model_id];control=full[job['distribution'],job['attack'],job['config']['seed']]
  evidence.terminal_checks(result,job,manifest,job['variant']);evidence.partition_identity(result,control)
  assert result['config']==job['config'] and result['revision_job']['variant']==job['variant']=='minus_C'
  assert result['revision_job']['checkpoint_sha256']==row['checkpoint_sha256'] and result['revision_job']['torch_version']=='2.11.0+cu128'
  assert result['revision_job']['adapter_hashes']==job['adapter_hashes']==manifest['adapter_hashes']
  refs={k:{'archive':archive.relative_to(ROOT).as_posix(),'archive_sha256':sha(archive),'member':name,**members[name]} for k,name in [('checkpoint','runs/'+model_id+'/model.pt'),('result','runs/'+model_id+'/result.json'),('raw_job','jobs/'+model_id+'.json')]}
  ns=dict(model_id=model_id,result=result,job=job,row=row,entry=entry,control=control,refs=refs,b=b,copy=copy);exec(constructor,ns);artifacts[model_id]=ns['record']
inv=copy.deepcopy(prior)
inv.update(schema='celeba_mechanism_valid_replay_inventory_C_after47',scope=SCOPE,status='PREPARED_NOT_APPROVED',mechanism_inspection_sha256=ni['files']['inspection']['sha256'],records=[artifacts[k] for k in sorted(artifacts)],excluded_prior_replay_ids=closed,selected_replay_ids=SELECTED,pending_new_ids_no_checkpoint=sorted(set(entries)-set(artifacts)),counts={'planned_new':800,'actual_accepted_new':150,'Full_reference_only':100,'pending_new_without_checkpoint':650},activation='PREPARED ONLY; exact3 native C minus closed U100+C47; no execution approval.')
inv['global_native_status']={'actual_strict_offserver':150,'scope_bound_prior_and_new':150,'known_accepted_outside_scope_ids':[],'actual_global_pending_native':650,'legacy_pending_field_semantics':'650 not yet native accepted in this pinned snapshot; no subsequent live progress inferred.'}
inv['prior_replay_boundary']={'status':'PRIOR147_EXCLUDED_ROOT_ADOPTION_RECEIPT_BOUND','root_adoption_receipt':adoption_path.relative_to(ROOT).as_posix(),'root_adoption_receipt_sha256':sha(adoption_path),'prior140_root_adoption_sha256':adoption['prior140_root_adoption_sha256'],'prior_inventory_sha256':sha(OLD/'inventory_actual147_Full100refs.json'),'next_execution_approved':False}
inv['increment_archive_sources']={i:archive.relative_to(ROOT).as_posix() for i in SELECTED}
inv['backup_chain'].append({'archive':ni['files']['archive']['path'],'archive_sha256':ni['files']['archive']['sha256'],'receipt':ni['files']['receipt']['path'],'receipt_sha256':ni['files']['receipt']['sha256'],'verification':ni['files']['proof']['path'],'verification_sha256':ni['files']['proof']['sha256'],'accepted_new_ids':SELECTED,'members_verified_now_without_inference':54})
inv['input_pins'].update({k:v['sha256'] for k,v in PINS.items()})
before=text(OLD/'bridge.py');after=before.replace(b.SCOPE,SCOPE).replace(b.INSPECTION_SHA,ni['files']['inspection']['sha256'])
for name,value in [('ACCEPTED_IDS',sorted(rows)),('EXCLUDED_PRIOR_IDS',closed),('REPLAY_IDS',SELECTED)]:after=rewrite_assign(after,name,value)
after=after.replace("len(records) == 147 and len({r['id'] for r in records}) == 147","len(records) == 150 and len({r['id'] for r in records}) == 150")
after=after.replace('exactly147 actual accepted terminals','exactly150 actual accepted terminals').replace('Actual accepted147 snapshot','Actual accepted150 snapshot').replace('Excluded-prior140/selected7','Excluded-prior147/selected3').replace('== 653','== 650')
after=after.replace('Only adopted U100+C40 plus exact seven C terminals; no other scope','Only adopted U100+C47 plus exact three C terminals; no other scope')
assert after.count('len(ids) == len(set(ids)) == 7')==1;after=after.replace('len(ids) == len(set(ids)) == 7','len(ids) == len(set(ids)) == 3')
write('bridge.py',after);write('inventory_actual150_Full100refs.json',inv)
nb=load('C_after47_metadata_only',HERE/'bridge.py');assert len(nb.validate_inventory(inv,baseline))==150
assert all(artifacts[r['id']]==r for r in prior['records']) and inv['full_references']==prior['full_references']
functions=lambda s:{n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
of,nf=functions(before),functions(after);exact=[n for n in of if n not in ['validate_inventory','require_approval']]
assert len(exact)==11 and all(of[n]==nf[n] for n in exact)
assert of['require_approval'].replace('len(ids) == len(set(ids)) == 7','len(ids) == len(set(ids)) == 3')==nf['require_approval']
f=read(OLD/'FULL100_ACTUAL_SOURCE_BINDINGS.json');f.pop('selected7_pairs');f['selected3_pairs']={i:artifacts[i]['paired_full']['id'] for i in SELECTED};write('FULL100_ACTUAL_SOURCE_BINDINGS.json',f)
scope=read(OLD/'SCOPE.json');scope.pop('prior140_root_adoption')
scope.update(scope=SCOPE,status='PREPARED_NOT_APPROVED',inventory_sha256=sha(HERE/'inventory_actual150_Full100refs.json'),bridge_sha256=sha(HERE/'bridge.py'),actual_native_accepted_records=150,selected_ids=SELECTED,excluded_prior_ids=closed,prior147_root_adoption=inv['prior_replay_boundary'],selected_records=[{'id':i,'checkpoint_sha256':artifacts[i]['checkpoint']['sha256'],'original_result_sha256':artifacts[i]['result']['sha256'],'raw_job_sha256':artifacts[i]['raw_job']['sha256'],'inventory_record_canonical_sha256':b.canonical(artifacts[i]),'paired_full':artifacts[i]['paired_full'],'Full_three_views_status':'ACTUAL900_SOURCE_BOUND_REFERENCE_ONLY_NO_NEW_FULL_INFERENCE'} for i in SELECTED],pending_native_count_without_model=650,global_native_status=inv['global_native_status'])
write('SCOPE.json',scope);write('SELECTED_3.txt','\n'.join(SELECTED)+'\n')
a=read(OLD/'APPROVAL_TEMPLATE.json');a.update(status='PREPARED_NOT_APPROVED',scope=SCOPE,inventory_sha256=scope['inventory_sha256'],bridge_sha256=scope['bridge_sha256'],selected_ids=SELECTED);write('APPROVAL_TEMPLATE.json',a)
write('SOURCE_REUSE.json',{'status':'PREPARED_SOURCE_IDENTITY_PASS_NOT_RUNTIME_APPROVAL','parent_science_seal_sha256':sha(OLD/'FILES_SHA256.json'),'parent_execution_seal_sha256':sha(EX/'EXECUTION_SOURCE_SHA256.json'),'exact_bridge_functions':exact,'ten_original_scientific_functions_and_variant_helper_exact':True,'bind_runtime_replay_accept_source_byte_exact':True,'require_approval_exact3_change':True,'original147_records_exact':True,'Full100_refs_exact':True,'original_record_constructor_AST':True,'native_archive_members_verified':54,'science_imports':False})
write('INPUT_PINS.json',PINS)
science=['prepare.py','bridge.py','inventory_actual150_Full100refs.json','SCOPE.json','SELECTED_3.txt','APPROVAL_TEMPLATE.json','FULL100_ACTUAL_SOURCE_BINDINGS.json','SOURCE_REUSE.json','NATIVE_INPUTS.json','INPUT_PINS.json']
write('FILES_SHA256.json',seal(science,status='PREPARED_NOT_APPROVED',scope=SCOPE,CNN=False,dispatch=False))
OUTEX.mkdir(exist_ok=False);remote='/workspace/guardfed_checks/'+HERE.name
bindings=read(EX/'RUNTIME_BINDINGS.json');bindings['outputs']={i:remote+'/runs/'+i for i in SELECTED};write('execution_candidate/RUNTIME_BINDINGS.json',bindings)
changes={OLD.name:HERE.name,'C_AFTER40':'C_AFTER47','valid_C_after40':'valid_C_after47','inventory_actual147_Full100refs.json':'inventory_actual150_Full100refs.json','closed140':'closed147','Prior140':'Prior147','prior140':'prior147','prepared7 frozen':'prepared3 frozen','reviewed7 root':'reviewed3 root','Only7 IDs':'Only3 IDs','exact7-scope':'exact3-scope',sha(OLD/'bridge.py'):sha(HERE/'bridge.py'),sha(OLD/'inventory_actual147_Full100refs.json'):sha(HERE/'inventory_actual150_Full100refs.json'),sha(OLD/'SCOPE.json'):sha(HERE/'SCOPE.json'),sha(OLD/'FILES_SHA256.json'):sha(HERE/'FILES_SHA256.json'),sha(EX/'RUNTIME_BINDINGS.json'):sha(OUTEX/'RUNTIME_BINDINGS.json')}
def rebind(s):return re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group(0)],s)
diff=list(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='sealed_C_after40/bridge.py',tofile='C_after47/bridge.py'))
for name in ['batch.py','install_once.py','resource_extra.py','verify_saved_increment.py','verify_backup.py','backup_completed.py','service.sh.template','supervisor.conf.template']:
 source=text(EX/name);s=rebind(source)
 if name=='batch.py':s=rewrite_assign(s,'SELECTED',SELECTED).replace("len(scope['excluded_prior_ids']) == 140","len(scope['excluded_prior_ids']) == 147").replace('Prepared exact7-terminal','Prepared exact3-terminal')
 if name=='install_once.py':
  s=s.replace('len(chosen)==7','len(chosen)==3').replace('reviewed7 CPU','reviewed3 CPU')
  s=s.replace("service = 'guardfed_celeba_mechanism_valid_C_after36'","service = 'guardfed_celeba_mechanism_valid_C_after40'")
  s=s.replace('/celeba_mechanism_valid_C_after36_20261010/execution_candidate/batch.py','/celeba_mechanism_valid_C_after40_20261010/execution_candidate/batch.py')
  s=s.replace('Previously accepted C after36','Previously accepted C after40').replace('eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a','64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea')
 if name=='backup_completed.py':s=s.replace('len(prior|set(ids))<7','len(prior|set(ids))<3').replace('ALL7_STRICT','ALL3_STRICT')
 if name=='resource_extra.py':assert s==source
 write('execution_candidate/'+name,s);diff.extend(difflib.unified_diff(source.splitlines(True),s.splitlines(True),fromfile='sealed_C_after40/'+name,tofile='C_after47/'+name))
write('MINIMAL_SOURCE_DIFF.patch',''.join(diff))
for name in ['ROOT_REVIEW_TEMPLATE.json','APPROVED_TEMPLATE.json']:
 t=json.loads(rebind(json.dumps(read(EX/name))))
 t.update(selected_ids=SELECTED,scope_sha256=sha(HERE/'SCOPE.json'),inventory_sha256=scope['inventory_sha256'],bridge_sha256=scope['bridge_sha256'],execution_seal_sha256=None,status='PREPARED_NOT_APPROVED')
 if name=='ROOT_REVIEW_TEMPLATE.json':t.update(source_seal_sha256=sha(HERE/'FILES_SHA256.json'),execution_authorized_within_existing_user_request=False,closed147_must_not_replay=closed)
 else:t.update(root_approval_sha256=None,outputs={i:remote+'/execution_candidate/runs/'+i for i in SELECTED},original_scope_outputs=bindings['outputs'])
 write('execution_candidate/'+name,t)
files=sorted(p.name for p in OUTEX.iterdir() if p.is_file())
write('execution_candidate/EXECUTION_SOURCE_SHA256.json',{'status':'PREPARED_ONLY_REQUIRES_EXTERNAL_ROOT_EXECUTION_APPROVAL','parent_science_seal_sha256':sha(HERE/'FILES_SHA256.json'),'successful_parent_execution_seal_sha256':sha(EX/'EXECUTION_SOURCE_SHA256.json'),'members':[{'path':n,'sha256':sha(OUTEX/n),'size':(OUTEX/n).stat().st_size} for n in files]})
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
for name,pin in PINS.items():assert sha(ROOT/name)==pin['sha256']
print(json.dumps({'status':'PREPARED_ONLY','selected':3,'science_seal':sha(HERE/'FILES_SHA256.json'),'execution_seal':sha(OUTEX/'EXECUTION_SOURCE_SHA256.json')}))
