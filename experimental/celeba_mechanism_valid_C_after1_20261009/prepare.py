"""Prepare exact eleven native-accepted C models after accepted C1; no runtime."""
import ast,copy,difflib,hashlib,importlib.util,json,sys,tarfile
from pathlib import Path
sys.dont_write_bytecode=True
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
OLD=HERE.with_name('celeba_mechanism_valid_C1_gate_20261009');EX=OLD/'execution_candidate';OUTEX=HERE/'execution_candidate'
SCOPE='MECHANISM_TERMINAL_VALID_REPLAY_C_AFTER1'
SELECTED=[f'minus_C_IID_Benign_seed{s}' for s in range(91002,91011)]+[f'minus_C_IID_F Flip_seed{s}' for s in (91001,91002)]
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
    payload=value if isinstance(value,str) else json.dumps(value,indent=2,allow_nan=False)+'\n'
    if p.exists():
        assert p.read_bytes()==payload.encode('utf-8'),str(p);return
    with p.open('x',encoding='utf-8',newline='\n') as f:f.write(payload)
def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p);m=importlib.util.module_from_spec(spec);sys.modules[name]=m;spec.loader.exec_module(m);return m
def rewrite_assign(source,name,value):
    n=next(n for n in ast.parse(source).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in n.targets))
    return source.replace(ast.get_source_segment(source,n),name+' = '+repr(value),1)
def seal(names,**meta):return dict(meta,members=[{'path':n,'sha256':sha(HERE/n),'size':(HERE/n).stat().st_size} for n in names])

for directory,name,pin in [(OLD,'FILES_SHA256.json','1fac2f4e1cad1068cbf8421c885c16b4ffe0f52066f1889192c8b7f7b1b82e42'),(EX,'EXECUTION_SOURCE_SHA256.json','7a580b765cbc6fab1aa1ab592d830ee6e063e27fba86fd6449b42a11faf14fa1')]:
    for row in read(directory/name,pin)['members']:
        p=directory/row['path'];assert sha(p)==row['sha256'] and p.stat().st_size==row['size'];text(p)
b=load('accepted_C1_bridge_metadata',OLD/'bridge.py')
prior=read(OLD/'inventory_actual101_Full100refs.json','7c87f1bbdf3c46d9f991fa391a423245c87327244c1d1d3ea2efec7de3fa8184')
adoption_path=EX/'backups/incremental_20261009T185829Z/ROOT_ADOPTION_REVIEW.json'
adoption=read(adoption_path,'d045665b066dafc25f9970adfdffef9c9a8a388575ec87b9b54d5dcabfa65cab')
assert adoption['cumulative_three_view_models']==101 and adoption['accepted_new']==1 and adoption['original100_unchanged'] and adoption['all_native_differences_zero']
ni=read(HERE/'NATIVE_INPUTS.json');assert ni['status']=='ACTUAL_STRICT_OFFSERVER_NATIVE112_C_AFTER1_INPUTS' and ni['selected_ids']==SELECTED
loaded={}
for name,pin in ni['files'].items():
    p=ROOT/pin['path'];assert sha(p)==pin['sha256'];PINS[pin['path']]={'sha256':sha(p),'bytes':p.stat().st_size}
    if p.suffix=='.json':loaded[name]=read(p,pin['sha256'])
inspection,receipt,proof=loaded['inspection'],loaded['receipt'],loaded['proof']
assert loaded['root_review']['native_accepted']==112 and loaded['root_review']['ledger_accepted_unique_ids']==112
assert loaded['root_review']['inspection_sha256']==ni['files']['inspection']['sha256'] and loaded['root_review']['ledger_sha256']==ni['files']['ledger']['sha256']
assert inspection['new_count']==112 and inspection['reused_count']==100 and not inspection['invalid']
assert inspection['manifest_sha256']==b.MANIFEST_SHA and inspection['source_script_sha256']==b.EVIDENCE_V4_SHA and inspection['full_inventory_sha256']==b.BASELINE_INVENTORY_SHA
assert receipt['accepted_new_ids']==proof['accepted_new_ids'] and len(receipt['accepted_new_ids'])==8 and proof['pass'] and proof['different_host_observed']
assert proof['archive_sha256']==receipt['archive_sha256']==ni['files']['archive']['sha256']
assert loaded['ledger']['entries'][-1]['receipt_sha256']==ni['files']['receipt']['sha256']
assert receipt['previous_receipt_sha256']==prior['backup_chain'][-1]['receipt_sha256'] and receipt['reused_full_weights_repacked']==0 and not receipt['failure_identities']
rows={r['id']:r for r in inspection['records'] if r['role']=='new'};closed=[r['id'] for r in prior['records']]
assert len(rows)==112 and set(rows)-set(closed)==set(SELECTED) and all(rows[r['id']]==r['accepted_v4_row'] for r in prior['records'])
assert len([i for i in rows if i.startswith('minus_U_')])==100 and len([i for i in rows if i.startswith('minus_C_')])==12
baseline=read(ROOT/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json',b.BASELINE_INVENTORY_SHA)
manifest=read(ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/manifest.json',b.MANIFEST_SHA)
entries={r['id']:r for r in manifest['jobs']};full={b.cell(r):r for r in baseline['records'] if r['method']=='GuardFed-AD2+'}
ep=ROOT/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py';text(ep);assert sha(ep)==b.EVIDENCE_V4_SHA;evidence=load('native_v4_metadata_only',ep)
archives={};verified={}
for prefix in ('','previous_'):
    ap=ROOT/ni['files'][prefix+'archive']['path'];rc=loaded[prefix+'receipt'];pf=loaded[prefix+'proof']
    assert sha(ap)==rc['archive_sha256']==pf['archive_sha256'] and rc['accepted_new_ids']==pf['accepted_new_ids'] and pf['pass']
    assert not rc['failure_identities'] and rc['reused_full_weights_repacked']==0
    ver=evidence.verify_archive(ap,rc);assert ver['members_verified']==pf['members_verified'];verified[prefix or 'new']=ver['members_verified']
    for i in rc['accepted_new_ids']:
        if i in SELECTED:assert i not in archives;archives[i]=(ap,rc)
assert set(archives)==set(SELECTED)
constructor_source=text(ROOT/'tmp/celeba_mechanism_valid_incremental_next11_20261009/prepare.py')
nodes=[n for n in ast.walk(ast.parse(constructor_source)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='record' for t in n.targets)]
assert len(nodes)==1;constructor=compile(ast.Module(body=nodes,type_ignores=[]),'original_record_constructor','exec')
artifacts={r['id']:copy.deepcopy(r) for r in prior['records']}
for model_id in SELECTED:
    archive,_=archives[model_id]
    with tarfile.open(archive) as tar:
        members=json.load(tar.extractfile('backup_inventory.json'))['members'];result=json.load(tar.extractfile('runs/'+model_id+'/result.json'));job=json.load(tar.extractfile('jobs/'+model_id+'.json'))
    row,entry=rows[model_id],entries[model_id];control=full[job['distribution'],job['attack'],job['config']['seed']]
    evidence.terminal_checks(result,job,manifest,job['variant']);evidence.partition_identity(result,control)
    assert result['config']==job['config'] and result['revision_job']['variant']==job['variant']=='minus_C'
    assert result['revision_job']['checkpoint_sha256']==row['checkpoint_sha256'] and result['revision_job']['torch_version']=='2.11.0+cu128'
    assert result['revision_job']['adapter_hashes']==job['adapter_hashes']==manifest['adapter_hashes']
    refs={k:{'archive':archive.relative_to(ROOT).as_posix(),'archive_sha256':sha(archive),'member':name,**members[name]} for k,name in [('checkpoint','runs/'+model_id+'/model.pt'),('result','runs/'+model_id+'/result.json'),('raw_job','jobs/'+model_id+'.json')]}
    ns=dict(model_id=model_id,result=result,job=job,row=row,entry=entry,control=control,refs=refs,b=b,copy=copy);exec(constructor,ns);artifacts[model_id]=ns['record']
inv=copy.deepcopy(prior);inv.update(schema='celeba_mechanism_valid_replay_inventory_C_after1',scope=SCOPE,status='PREPARED_NOT_APPROVED',mechanism_inspection_sha256=ni['files']['inspection']['sha256'],records=[artifacts[k] for k in sorted(artifacts)],excluded_prior_replay_ids=closed,selected_replay_ids=SELECTED,pending_new_ids_no_checkpoint=sorted(set(entries)-set(artifacts)),counts={'planned_new':800,'actual_accepted_new':112,'Full_reference_only':100,'pending_new_without_checkpoint':688},activation='PREPARED ONLY; exact11 accepted C minus closed U100+C1; no new execution approval.')
inv['global_native_status']={'actual_strict_offserver':112,'scope_bound_prior_and_new':112,'known_accepted_outside_scope_ids':[],'actual_global_pending_native':688,'legacy_pending_field_semantics':'688 not yet native accepted in this pinned snapshot; do not infer subsequent live progress.'}
inv['prior_replay_boundary']={'status':'PRIOR101_EXCLUDED_ROOT_ADOPTION_RECEIPT_BOUND','root_adoption_receipt':adoption_path.relative_to(ROOT).as_posix(),'root_adoption_receipt_sha256':sha(adoption_path),'prior100_root_adoption_sha256':adoption['prior100_root_adoption_sha256'],'prior_inventory_sha256':sha(OLD/'inventory_actual101_Full100refs.json'),'next_execution_approved':False}
inv.pop('gate_archive_reuse',None)
inv['increment_archive_sources']={i:archives[i][0].relative_to(ROOT).as_posix() for i in SELECTED}
inv['backup_chain'].append({'archive':ni['files']['archive']['path'],'archive_sha256':ni['files']['archive']['sha256'],'receipt':ni['files']['receipt']['path'],'receipt_sha256':ni['files']['receipt']['sha256'],'verification':ni['files']['proof']['path'],'verification_sha256':ni['files']['proof']['sha256'],'accepted_new_ids':receipt['accepted_new_ids'],'members_verified_now_without_inference':verified['new']})
inv['input_pins'].update({k:v['sha256'] for k,v in PINS.items()})
before=text(OLD/'bridge.py');after=before.replace(b.SCOPE,SCOPE).replace(b.INSPECTION_SHA,ni['files']['inspection']['sha256'])
for name,value in [('ACCEPTED_IDS',sorted(rows)),('EXCLUDED_PRIOR_IDS',closed),('REPLAY_IDS',SELECTED)]:after=rewrite_assign(after,name,value)
after=after.replace("len(records) == 101 and len({r['id'] for r in records}) == 101", "len(records) == 112 and len({r['id'] for r in records}) == 112")
after=after.replace('exactly101 actual accepted terminals','exactly112 actual accepted terminals').replace('Actual accepted101 snapshot','Actual accepted112 snapshot').replace('Excluded-prior100/selected1','Excluded-prior101/selected11').replace('== 699','== 688')
after=after.replace('Only adopted U100 plus frozen C1 metadata; no other gate scope','Only adopted U100+C1 plus exact eleven C terminals; no other scope')
assert after.count('len(ids) == len(set(ids)) == 1')==1
after=after.replace('len(ids) == len(set(ids)) == 1','len(ids) == len(set(ids)) == 11')
write('bridge.py',after);write('inventory_actual112_Full100refs.json',inv)
nb=load('C_after1_metadata_only',HERE/'bridge.py');assert len(nb.validate_inventory(inv,baseline))==112
assert all(artifacts[r['id']]==r for r in prior['records']) and inv['full_references']==prior['full_references']
functions=lambda s:{n.name:ast.get_source_segment(s,n) for n in ast.parse(s).body if isinstance(n,ast.FunctionDef)}
of,nf=functions(before),functions(after);exact=[n for n in of if n not in ['validate_inventory','require_approval']]
assert len(exact)==11 and all(of[n]==nf[n] for n in exact)
assert of['require_approval'].replace('len(ids) == len(set(ids)) == 1','len(ids) == len(set(ids)) == 11')==nf['require_approval']
f=read(OLD/'FULL100_ACTUAL_SOURCE_BINDINGS.json');f.pop('selected1_pairs');f['selected11_pairs']={i:artifacts[i]['paired_full']['id'] for i in SELECTED};write('FULL100_ACTUAL_SOURCE_BINDINGS.json',f)
scope=read(OLD/'SCOPE.json');scope.pop('prior100_root_adoption');scope.update(scope=SCOPE,status='PREPARED_NOT_APPROVED',inventory_sha256=sha(HERE/'inventory_actual112_Full100refs.json'),bridge_sha256=sha(HERE/'bridge.py'),actual_native_accepted_records=112,selected_ids=SELECTED,excluded_prior_ids=closed,prior101_root_adoption=inv['prior_replay_boundary'],selected_records=[{'id':i,'checkpoint_sha256':artifacts[i]['checkpoint']['sha256'],'original_result_sha256':artifacts[i]['result']['sha256'],'raw_job_sha256':artifacts[i]['raw_job']['sha256'],'inventory_record_canonical_sha256':b.canonical(artifacts[i]),'paired_full':artifacts[i]['paired_full'],'Full_three_views_status':'ACTUAL900_SOURCE_BOUND_REFERENCE_ONLY_NO_NEW_FULL_INFERENCE'} for i in SELECTED],pending_native_count_without_model=688,global_native_status=inv['global_native_status'])
write('SCOPE.json',scope);write('SELECTED_11.txt','\n'.join(SELECTED)+'\n')
a=read(OLD/'APPROVAL_TEMPLATE.json');a.update(status='PREPARED_NOT_APPROVED',scope=SCOPE,inventory_sha256=scope['inventory_sha256'],bridge_sha256=scope['bridge_sha256'],selected_ids=SELECTED);write('APPROVAL_TEMPLATE.json',a)
write('SOURCE_REUSE.json',{'status':'PREPARED_SOURCE_IDENTITY_PASS_NOT_RUNTIME_APPROVAL','parent_science_seal_sha256':sha(OLD/'FILES_SHA256.json'),'parent_execution_seal_sha256':sha(EX/'EXECUTION_SOURCE_SHA256.json'),'exact_bridge_functions':exact,'ten_original_scientific_functions_and_variant_helper_exact':True,'bind_runtime_replay_accept_source_byte_exact':True,'require_approval_exact11_change':True,'original101_records_exact':True,'Full100_refs_exact':True,'original_record_constructor_AST':True,'native_archive_members_verified':verified,'science_imports':False})
write('INPUT_PINS.json',PINS)
science=['prepare.py','bridge.py','inventory_actual112_Full100refs.json','SCOPE.json','SELECTED_11.txt','APPROVAL_TEMPLATE.json','FULL100_ACTUAL_SOURCE_BINDINGS.json','SOURCE_REUSE.json','NATIVE_INPUTS.json','INPUT_PINS.json']
write('FILES_SHA256.json',seal(science,status='PREPARED_NOT_APPROVED',scope=SCOPE,CNN=False,dispatch=False))
OUTEX.mkdir(exist_ok=False);remote='/workspace/guardfed_checks/'+HERE.name
bindings=read(EX/'RUNTIME_BINDINGS.json');bindings['outputs']={i:remote+'/runs/'+i for i in SELECTED};write('execution_candidate/RUNTIME_BINDINGS.json',bindings)
changes={OLD.name:HERE.name,'C1_GATE':'C_AFTER1','valid_C1_gate':'valid_C_after1','inventory_actual101_Full100refs.json':'inventory_actual112_Full100refs.json','closed100':'closed101','Prior100':'Prior101','prior100':'prior101','prepared1 frozen':'prepared11 frozen','reviewed1 root':'reviewed11 root','Only1 ID':'Only11 IDs','exact1-scope':'exact11-scope',sha(OLD/'bridge.py'):sha(HERE/'bridge.py'),sha(OLD/'inventory_actual101_Full100refs.json'):sha(HERE/'inventory_actual112_Full100refs.json'),sha(OLD/'SCOPE.json'):sha(HERE/'SCOPE.json'),sha(OLD/'FILES_SHA256.json'):sha(HERE/'FILES_SHA256.json'),sha(EX/'RUNTIME_BINDINGS.json'):sha(OUTEX/'RUNTIME_BINDINGS.json')}
def rebind(s):
    for x,y in changes.items():s=s.replace(x,y)
    return s
diff=list(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='sealed_C1/bridge.py',tofile='C_after1/bridge.py'))
for name in ['batch.py','install_once.py','resource_extra.py','verify_saved_increment.py','verify_backup.py','backup_completed.py','service.sh.template','supervisor.conf.template']:
    source=text(EX/name);s=rebind(source)
    if name=='batch.py':s=rewrite_assign(s,'SELECTED',SELECTED).replace("len(scope['excluded_prior_ids']) == 100","len(scope['excluded_prior_ids']) == 101").replace('Prepared exact1-terminal','Prepared exact11-terminal')
    if name=='install_once.py':
        s=s.replace('len(chosen)==1','len(chosen)==11').replace('reviewed1 CPU','reviewed11 CPU')
        s=s.replace("service = 'guardfed_celeba_mechanism_valid_after92'","service = 'guardfed_celeba_mechanism_valid_C1_gate'")
        s=s.replace('/celeba_mechanism_valid_incremental_after92_20261009/execution_candidate/batch.py','/celeba_mechanism_valid_C1_gate_20261009/execution_candidate/batch.py')
        s=s.replace('Previously accepted after92','Previously accepted C1 gate').replace('9050eb059a797c70f0ca977294989b5ae5757286dbc85b36d529012cb5ab72ee','d045665b066dafc25f9970adfdffef9c9a8a388575ec87b9b54d5dcabfa65cab')
    if name=='backup_completed.py':s=s.replace('len(prior|set(ids))<1','len(prior|set(ids))<11').replace('ALL1_STRICT','ALL11_STRICT')
    if name=='resource_extra.py':assert s==source
    write('execution_candidate/'+name,s);diff.extend(difflib.unified_diff(source.splitlines(True),s.splitlines(True),fromfile='sealed_C1/'+name,tofile='C_after1/'+name))
write('MINIMAL_SOURCE_DIFF.patch',''.join(diff))
for name in ['ROOT_REVIEW_TEMPLATE.json','APPROVED_TEMPLATE.json']:
    t=json.loads(rebind(json.dumps(read(EX/name))))
    t.update(selected_ids=SELECTED,scope_sha256=sha(HERE/'SCOPE.json'),inventory_sha256=scope['inventory_sha256'],bridge_sha256=scope['bridge_sha256'],execution_seal_sha256=None,status='PREPARED_NOT_APPROVED')
    if name=='ROOT_REVIEW_TEMPLATE.json':t.update(source_seal_sha256=sha(HERE/'FILES_SHA256.json'),execution_authorized_within_existing_user_request=False,closed101_must_not_replay=closed)
    else:t.update(root_approval_sha256=None,outputs={i:remote+'/execution_candidate/runs/'+i for i in SELECTED},original_scope_outputs=bindings['outputs'])
    write('execution_candidate/'+name,t)
files=sorted(p.name for p in OUTEX.iterdir() if p.is_file())
write('execution_candidate/EXECUTION_SOURCE_SHA256.json',{'status':'PREPARED_ONLY_REQUIRES_EXTERNAL_ROOT_EXECUTION_APPROVAL','parent_science_seal_sha256':sha(HERE/'FILES_SHA256.json'),'successful_parent_execution_seal_sha256':sha(EX/'EXECUTION_SOURCE_SHA256.json'),'members':[{'path':n,'sha256':sha(OUTEX/n),'size':(OUTEX/n).stat().st_size} for n in files]})
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
for name,pin in PINS.items():assert sha(ROOT/name)==pin['sha256']
print(json.dumps({'status':'PREPARED_ONLY','selected':11,'science_seal':sha(HERE/'FILES_SHA256.json'),'execution_seal':sha(OUTEX/'EXECUTION_SOURCE_SHA256.json')}))
