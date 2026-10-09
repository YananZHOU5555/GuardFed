"""Prepare exact8 only after native100 strict/offserver pins are supplied; no runtime."""
import ast,copy,difflib,hashlib,importlib.util,json,re,sys,tarfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=HERE.with_name('celeba_mechanism_valid_incremental_after82_v2_20261009')
EX=OLD/'execution_candidate';OUTEX=HERE/'execution_candidate'
SCOPE='MECHANISM_TERMINAL_VALID_REPLAY_INCREMENTAL_AFTER92'
SELECTED=[f'minus_U_non-IID_Sp-DFA_seed{s}' for s in range(91003,91011)]
PINS={}

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p,expected=None):
    p=Path(p);h=sha(p);assert expected is None or h==expected,str(p)
    PINS[p.relative_to(ROOT).as_posix()]={'sha256':h,'bytes':p.stat().st_size}
    return json.loads(p.read_text(encoding='utf-8-sig'))
def text(p):
    p=Path(p);PINS[p.relative_to(ROOT).as_posix()]={'sha256':sha(p),'bytes':p.stat().st_size};return p.read_text(encoding='utf-8-sig')
def write(name,value):
    p=HERE/name;p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(value if isinstance(value,str) else json.dumps(value,indent=2,allow_nan=False)+'\n')
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def rewrite_assign(source,name,value):
    n=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    return source.replace(ast.get_source_segment(source,n),name+' = '+repr(value),1)
def seal(names,**meta):return dict(meta,members=[{'path':n,'sha256':sha(HERE/n),'size':(HERE/n).stat().st_size} for n in names])

for directory,name,expected in [(OLD,'FILES_SHA256.json','b95801ac039bb39276e79012393792adfde290b92091ff45c0a4323bd4fdd8f0'),(EX,'EXECUTION_SOURCE_SHA256.json','94d99842b334346ae8a6715b84f7fddea35574a77fbf51f9460b25301c6ccae6')]:
    for row in read(directory/name,expected)['members']:
        p=directory/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['size'];text(p)
b=load('sealed_parent92_bridge',OLD/'bridge.py')
prior=read(OLD/'inventory_actual92_Full100refs.json','bedb867ae9bdf965ef7143a5b4f077620911ec26d93346337307046e64b5c9da')
adoption_path=EX/'backups/incremental_20261009T174922Z/ROOT_ADOPTION_REVIEW.json'
adoption=read(adoption_path,'b9e40d1ca565c0bcf146058433ff3e037ab4e824aa6972d1a3f3f47a088e8683')
assert adoption['cumulative_three_view_models']==92 and adoption['accepted_new']==10 and adoption['original82_unchanged'] and adoption['all_native_differences_zero']
ni=read(HERE/'NATIVE_INPUTS.json')
assert ni['status']=='ACTUAL_STRICT_OFFSERVER_NATIVE100_INPUTS' and ni['selected_ids']==SELECTED
loaded={}
for name,pin in ni['files'].items():
    p=ROOT/pin['path'];assert sha(p)==pin['sha256'];PINS[pin['path']]={'sha256':pin['sha256'],'bytes':p.stat().st_size}
    if p.suffix=='.json':loaded[name]=read(p,pin['sha256'])
inspection,receipt,proof=loaded['inspection'],loaded['receipt'],loaded['proof']
assert inspection['new_count']==104 and inspection['reused_count']==100 and not inspection['invalid']
assert inspection['manifest_sha256']==b.MANIFEST_SHA and inspection['source_script_sha256']==b.EVIDENCE_V4_SHA and inspection['full_inventory_sha256']==b.BASELINE_INVENTORY_SHA
assert receipt['accepted_new_ids']==proof['accepted_new_ids'] and set(SELECTED)<set(receipt['accepted_new_ids']) and len(receipt['accepted_new_ids'])==12 and proof['pass'] and proof['different_host_observed']
assert proof['archive_sha256']==receipt['archive_sha256']==ni['files']['archive']['sha256']
assert receipt['previous_receipt_sha256']==prior['backup_chain'][-1]['receipt_sha256'] and receipt['reused_full_weights_repacked']==0 and not receipt['failure_identities']
all_rows={r['id']:r for r in inspection['records'] if r['role']=='new'}
rows={i:r for i,r in all_rows.items() if i.startswith('minus_U_')};outside=sorted(set(all_rows)-set(rows))
assert outside==[f'minus_C_IID_Benign_seed{s}' for s in range(91001,91005)]
closed=[r['id'] for r in prior['records']]
assert len(rows)==100 and set(rows)-set(closed)==set(SELECTED) and all(rows[r['id']]==r['accepted_v4_row'] for r in prior['records'])
baseline=read(ROOT/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json',b.BASELINE_INVENTORY_SHA)
manifest=read(ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/manifest.json',b.MANIFEST_SHA)
entries={r['id']:r for r in manifest['jobs']};full={b.cell(r):r for r in baseline['records'] if r['method']=='GuardFed-AD2+'}
ep=ROOT/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py';text(ep);assert sha(ep)==b.EVIDENCE_V4_SHA;evidence=load('original_native_evidence',ep)
archive=ROOT/ni['files']['archive']['path'];verified=evidence.verify_archive(archive,receipt);assert verified['members_verified']==proof['members_verified']
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
        assert result['config']==job['config'] and result['revision_job']['variant']==job['variant']=='minus_U'
        assert result['revision_job']['checkpoint_sha256']==row['checkpoint_sha256'] and result['revision_job']['torch_version']=='2.11.0+cu128'
        assert result['revision_job']['adapter_hashes']==job['adapter_hashes']==manifest['adapter_hashes']
        refs={k:{'archive':archive.relative_to(ROOT).as_posix(),'archive_sha256':sha(archive),'member':name,**members[name]} for k,name in [('checkpoint','runs/'+model_id+'/model.pt'),('result','runs/'+model_id+'/result.json'),('raw_job','jobs/'+model_id+'.json')]}
        ns=dict(model_id=model_id,result=result,job=job,row=row,entry=entry,control=control,refs=refs,b=b,copy=copy);exec(constructor,ns);artifacts[model_id]=ns['record']
inv=copy.deepcopy(prior);inv.update(schema='celeba_mechanism_valid_replay_inventory_after92',scope=SCOPE,status='PREPARED_NOT_APPROVED',mechanism_inspection_sha256=ni['files']['inspection']['sha256'],records=[artifacts[k] for k in sorted(artifacts)],excluded_prior_replay_ids=closed,selected_replay_ids=SELECTED,pending_new_ids_no_checkpoint=sorted(set(entries)-set(artifacts)),counts={'planned_new':800,'actual_accepted_new':100,'Full_reference_only':100,'pending_new_without_checkpoint':700},activation='PREPARED ONLY; strict/offserver native100 minus adopted three-view92 exact8; no new execution approval.')
inv['global_native_status']={'actual_strict_offserver':104,'scope_bound_U':100,'known_accepted_outside_scope_ids':outside,'actual_global_pending_native':696,'legacy_pending_field_semantics':'700 IDs not checkpoint-bound in this U-only inventory, including four accepted C deliberately excluded; not a global no-checkpoint assertion.'}
inv['prior_replay_boundary']={'status':'PRIOR92_EXCLUDED_ROOT_ADOPTION_RECEIPT_BOUND','root_adoption_receipt':adoption_path.relative_to(ROOT).as_posix(),'root_adoption_receipt_sha256':sha(adoption_path),'prior82_root_adoption_sha256':adoption['prior82_root_adoption_sha256'],'prior_inventory_sha256':sha(OLD/'inventory_actual92_Full100refs.json'),'next_execution_approved':False}
inv['backup_chain'].append({'archive':archive.relative_to(ROOT).as_posix(),'archive_sha256':sha(archive),'receipt':ni['files']['receipt']['path'],'receipt_sha256':ni['files']['receipt']['sha256'],'verification':ni['files']['proof']['path'],'verification_sha256':ni['files']['proof']['sha256'],'accepted_new_ids':receipt['accepted_new_ids'],'selected_scope_ids':SELECTED,'excluded_outside_scope_ids':outside,'members_verified_now_without_inference':verified['members_verified']})
inv['input_pins'].update({k:v['sha256'] for k,v in PINS.items()})
before=text(OLD/'bridge.py');after=before.replace(b.SCOPE,SCOPE).replace(b.INSPECTION_SHA,ni['files']['inspection']['sha256'])
for name,value in [('ACCEPTED_IDS',sorted(rows)),('EXCLUDED_PRIOR_IDS',closed),('REPLAY_IDS',SELECTED)]:after=rewrite_assign(after,name,value)
after=after.replace('== 92','== 100').replace('exactly92','exactly100').replace('accepted92','accepted100').replace('prior82','prior92').replace('selected10','selected8').replace('== 708','== 700')
assert after.count('len(ids) == len(set(ids)) == 10')==1
after=after.replace('len(ids) == len(set(ids)) == 10','len(ids) == len(set(ids)) == 8')
write('bridge.py',after);write('inventory_actual100_Full100refs.json',inv)
nb=load('new_after92_metadata_only',HERE/'bridge.py');assert len(nb.validate_inventory(inv,baseline))==100
assert all(artifacts[r['id']]==r for r in prior['records']) and inv['full_references']==prior['full_references']
functions=lambda s:{n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
oldfunc,newfunc=functions(before),functions(after);exact=[n for n in oldfunc if n not in ['validate_inventory','require_approval']]
assert len(exact)==10 and all(oldfunc[n]==newfunc[n] for n in exact)
assert oldfunc['require_approval'].replace('== 10','== 8')==newfunc['require_approval']
fullbindings=read(OLD/'FULL100_ACTUAL_SOURCE_BINDINGS.json');fullbindings.pop('selected10_pairs');fullbindings['selected8_pairs']={i:artifacts[i]['paired_full']['id'] for i in SELECTED};write('FULL100_ACTUAL_SOURCE_BINDINGS.json',fullbindings)
scope=read(OLD/'SCOPE.json');scope['global_native_status']=inv['global_native_status'];scope.pop('prior82_root_adoption');scope.update(scope=SCOPE,status='PREPARED_NOT_APPROVED',inventory_sha256=sha(HERE/'inventory_actual100_Full100refs.json'),bridge_sha256=sha(HERE/'bridge.py'),actual_native_accepted_records=100,selected_ids=SELECTED,excluded_prior_ids=closed,prior92_root_adoption=inv['prior_replay_boundary'],selected_records=[{'id':i,'checkpoint_sha256':artifacts[i]['checkpoint']['sha256'],'original_result_sha256':artifacts[i]['result']['sha256'],'raw_job_sha256':artifacts[i]['raw_job']['sha256'],'inventory_record_canonical_sha256':b.canonical(artifacts[i]),'paired_full':artifacts[i]['paired_full'],'Full_three_views_status':'ACTUAL900_SOURCE_BOUND_REFERENCE_ONLY_NO_NEW_FULL_INFERENCE'} for i in SELECTED],pending_native_count_without_model=700)
write('SCOPE.json',scope);write('SELECTED_8.txt','\n'.join(SELECTED)+'\n')
a=read(OLD/'APPROVAL_TEMPLATE.json');a.update(status='PREPARED_NOT_APPROVED',scope=SCOPE,inventory_sha256=scope['inventory_sha256'],bridge_sha256=scope['bridge_sha256'],selected_ids=SELECTED);write('APPROVAL_TEMPLATE.json',a)
write('SOURCE_REUSE.json',{'status':'PREPARED_SOURCE_IDENTITY_PASS_NOT_RUNTIME_APPROVAL','parent_science_seal_sha256':sha(OLD/'FILES_SHA256.json'),'parent_execution_seal_sha256':sha(EX/'EXECUTION_SOURCE_SHA256.json'),'exact_bridge_functions':exact,'bind_runtime_replay_accept_source_byte_exact':True,'require_approval_exact8_change':True,'original92_records_exact':True,'Full100_refs_exact':True,'original_record_constructor_AST':True,'native_archive_members_verified':verified['members_verified'],'science_imports':False})
write('INPUT_PINS.json',PINS)
science=['prepare.py','bridge.py','inventory_actual100_Full100refs.json','SCOPE.json','SELECTED_8.txt','APPROVAL_TEMPLATE.json','FULL100_ACTUAL_SOURCE_BINDINGS.json','SOURCE_REUSE.json','NATIVE_INPUTS.json','INPUT_PINS.json']
write('FILES_SHA256.json',seal(science,status='PREPARED_NOT_APPROVED',scope=SCOPE,CNN=False,dispatch=False))
OUTEX.mkdir(exist_ok=False)
remote='/workspace/guardfed_checks/'+HERE.name
bindings=read(EX/'RUNTIME_BINDINGS.json');bindings['outputs']={i:remote+'/runs/'+i for i in SELECTED};write('execution_candidate/RUNTIME_BINDINGS.json',bindings)
changes={OLD.name:HERE.name,'INCREMENTAL_AFTER82_V2':'INCREMENTAL_AFTER92','BOUNDED_AFTER82_V2':'BOUNDED_AFTER92','APPROVED_AFTER82_V2':'APPROVED_AFTER92','AFTER82_V2_VALID':'AFTER92_VALID','PREPARED_AFTER82_V2':'PREPARED_AFTER92','valid_after82_v2':'valid_after92','inventory_actual92_Full100refs.json':'inventory_actual100_Full100refs.json','closed82':'closed92','Prior82':'Prior92','prior82':'prior92','actual_native92':'actual_native100','prepared10':'prepared8','reviewed10':'reviewed8','Only10 IDs':'Only8 IDs','exact10-scope':'exact8-scope',sha(OLD/'bridge.py'):sha(HERE/'bridge.py'),sha(OLD/'inventory_actual92_Full100refs.json'):sha(HERE/'inventory_actual100_Full100refs.json'),sha(OLD/'SCOPE.json'):sha(HERE/'SCOPE.json'),sha(OLD/'FILES_SHA256.json'):sha(HERE/'FILES_SHA256.json'),sha(EX/'RUNTIME_BINDINGS.json'):sha(OUTEX/'RUNTIME_BINDINGS.json'),b.INSPECTION_SHA:ni['files']['inspection']['sha256']}
def rebind(s):
    for x,y in changes.items():s=s.replace(x,y)
    return s
diff=list(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='sealed_parent/bridge.py',tofile='after92/bridge.py'))
for name in ['batch.py','install_once.py','resource_extra.py','verify_saved_increment.py','verify_backup.py','backup_completed.py','service.sh.template','supervisor.conf.template']:
    source=text(EX/name);s=rebind(source)
    if name=='batch.py':s=rewrite_assign(s,'SELECTED',SELECTED).replace("len(scope['excluded_prior_ids']) == 82","len(scope['excluded_prior_ids']) == 92").replace('Prepared next11-terminal','Prepared exact8-terminal')
    if name=='install_once.py':
        s=s.replace('len(chosen)==10','len(chosen)==8').replace('prior_failed_service_stopped','prior_completed_service_stopped').replace('previous_failed_service','previous_completed_service')
        s=s.replace("service = 'guardfed_celeba_mechanism_valid_after82'","service = 'guardfed_celeba_mechanism_valid_after82_v2'")
        s=s.replace('/celeba_mechanism_valid_incremental_after82_20261009/execution_candidate/batch.py','/celeba_mechanism_valid_incremental_after82_v2_20261009/execution_candidate/batch.py')
        s=s.replace('Preserved original after82 service','Previously accepted after82_v2 service').replace('Original after82 worker','Previously accepted after82_v2 worker')
        s=s.replace("preserved_failure_sha256='b63558fb33e5dd7c576ff3a45f76e334e4d40ab5ca3e884618aec6a862852415'","prior_success_root_adoption_sha256='b9e40d1ca565c0bcf146058433ff3e037ab4e824aa6972d1a3f3f47a088e8683'")
    if name=='backup_completed.py':s=s.replace('len(prior|set(ids))<10','len(prior|set(ids))<8').replace('ALL10_STRICT','ALL8_STRICT')
    if name=='resource_extra.py':assert s==source
    write('execution_candidate/'+name,s);diff.extend(difflib.unified_diff(source.splitlines(True),s.splitlines(True),fromfile='sealed_parent/'+name,tofile='after92/'+name))
write('MINIMAL_SOURCE_DIFF.patch',''.join(diff))
for name in ['ROOT_REVIEW_TEMPLATE.json','APPROVED_TEMPLATE.json']:
    t=json.loads(rebind(json.dumps(read(EX/name))))
    t.update(selected_ids=SELECTED,scope_sha256=sha(HERE/'SCOPE.json'),inventory_sha256=scope['inventory_sha256'],bridge_sha256=scope['bridge_sha256'],execution_seal_sha256=None,status='PREPARED_NOT_APPROVED')
    if name=='ROOT_REVIEW_TEMPLATE.json':t.update(source_seal_sha256=sha(HERE/'FILES_SHA256.json'),execution_authorized_within_existing_user_request=False,closed92_must_not_replay=closed)
    else:t.update(root_approval_sha256=None,outputs={i:remote+'/execution_candidate/runs/'+i for i in SELECTED},original_scope_outputs=bindings['outputs'])
    write('execution_candidate/'+name,t)
files=sorted(p.name for p in OUTEX.iterdir() if p.is_file())
write('execution_candidate/EXECUTION_SOURCE_SHA256.json',{'status':'PREPARED_ONLY_REQUIRES_EXTERNAL_ROOT_EXECUTION_APPROVAL','parent_science_seal_sha256':sha(HERE/'FILES_SHA256.json'),'successful_parent_execution_seal_sha256':sha(EX/'EXECUTION_SOURCE_SHA256.json'),'members':[{'path':n,'sha256':sha(OUTEX/n),'size':(OUTEX/n).stat().st_size} for n in files]})
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
for name,pin in PINS.items():assert sha(ROOT/name)==pin['sha256']
print(json.dumps({'status':'PREPARED_ONLY','selected':8,'science_seal':sha(HERE/'FILES_SHA256.json'),'execution_seal':sha(OUTEX/'EXECUTION_SOURCE_SHA256.json')}))
