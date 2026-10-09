"""Prepare only the frozen native82 minus adopted three-view71 delta; no runtime."""
from pathlib import Path
import ast
import copy
import difflib
import hashlib
import importlib.util
import json
import sys
import tarfile
sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT/'tmp/celeba_mechanism_valid_incremental_next11_20261009'
EX = OLD/'execution_candidate'
OUTEX = HERE/'execution_candidate'
BACK = ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
TAG = 'root_delta_20261009T154227Z'
SCOPE = 'MECHANISM_TERMINAL_VALID_REPLAY_INCREMENTAL_AFTER71'
SELECTED = [f'minus_U_non-IID_FedSA_seed{s}' for s in range(91002,91010)] + [f'minus_U_non-IID_S-DFA_seed{s}' for s in range(91001,91004)]
PINS = {}


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def read(p, expected=None):
    p = Path(p)
    h = sha(p)
    assert expected is None or h == expected, str(p)
    PINS[str(p.relative_to(ROOT)).replace('\\','/')] = {'sha256':h,'bytes':p.stat().st_size}
    return json.loads(p.read_text(encoding='utf-8-sig'))


def save(name, obj):
    p = HERE/name
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:
        f.write(json.dumps(obj,indent=2,allow_nan=False)+'\n')


def source(p):
    p = Path(p)
    PINS[str(p.relative_to(ROOT)).replace('\\','/')] = {'sha256':sha(p),'bytes':p.stat().st_size}
    return p.read_text(encoding='utf-8-sig')


def write_source(name, text):
    p = HERE/name
    p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8',newline='\n') as f:
        f.write(text)


def load(name,p):
    spec=importlib.util.spec_from_file_location(name,p)
    module=importlib.util.module_from_spec(spec)
    sys.modules[name]=module
    spec.loader.exec_module(module)
    return module


science_seal = read(OLD/'FILES_SHA256.json','65f706e8c7c7d7e18e76c8a300dd845297c97b3bd5c6c182bbbb8aad0103b5ff')
execution_seal = read(EX/'EXECUTION_SOURCE_SHA256.json','9f1252dd7c11abfe7cee297b2028ca9c58f179d964efb39ee6008ea860508b15')
for folder, seal in [(OLD,science_seal),(EX,execution_seal)]:
    for row in seal['members']:
        p=folder/row['path']
        assert sha(p)==row['sha256'] and p.stat().st_size==row['size'],str(p)
        source(p)
b=load('original_next11_bridge_for_preparation',OLD/'bridge.py')
prior=read(OLD/'inventory_actual71_Full100refs.json', 'ee2ccf040f677e97c2f03c01e760e36cdd51a21f1c53b7a0122d8c218af34ef0')
adoption_path=EX/'backups/incremental_20261009T152532Z/ROOT_ADOPTION_REVIEW.json'
adoption=read(adoption_path,'692ecd168ecab0b9c960965decb68424ce2ad80a5cf7ca452d2739da6b0a768a')
assert adoption['cumulative_three_view_models']==71 and adoption['accepted_new']==11 and adoption['prior_three_view_models']==60 and adoption['all_native_differences_zero']
assert set(prior['excluded_prior_replay_ids'])|set(adoption['accepted_new_ids']) == {r['id'] for r in prior['records']}
prior60_path=ROOT/prior['prior_replay_boundary']['root_adoption_receipt']
prior60=read(prior60_path,adoption['prior60_root_adoption_sha256'])
assert prior60['cumulative_three_view_models']==60
ledger=read(BACK/'verified_ledger.json','f013da17296ad095000075c47309ef13d361428b82d636b543086fcf21f1dcde')
root=read(BACK/TAG/'ROOT_DELTA_VERIFICATION.json','62e2145a136035bc0a0fd85fa75c083dbbc12834f782bb6b508610983e063a40')
receipt_path=BACK/(TAG+'.tar.gz.receipt.json')
receipt=read(receipt_path,ledger['entries'][-1]['receipt_sha256'])
proof_path=BACK/(TAG+'_offserver_verification.json')
proof=read(proof_path,root['offserver_proof_sha256'])
inspection_path=BACK/('mechanism_inspection_v4_'+TAG)/'inspection.json'
inspection=read(inspection_path,root['inspection_sha256'])
assert root['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and root['total_new_strict_and_offserver']==82
assert root['ledger_sha256']==sha(BACK/'verified_ledger.json') and root['receipt_sha256']==sha(receipt_path)
assert receipt['accepted_new_ids']==root['new_ids']==proof['accepted_new_ids']==SELECTED
assert proof['pass'] and proof['different_host_observed'] and proof['archive_sha256']==receipt['archive_sha256']==root['archive_sha256']
assert inspection['new_count']==82 and inspection['reused_count']==100 and not inspection['invalid']
assert inspection['manifest_sha256']==b.MANIFEST_SHA and inspection['source_script_sha256']==b.EVIDENCE_V4_SHA and inspection['full_inventory_sha256']==b.BASELINE_INVENTORY_SHA
rows={r['id']:r for r in inspection['records'] if r['role']=='new'}
closed=[r['id'] for r in prior['records']]
assert len(rows)==82 and set(rows)-set(closed)==set(SELECTED) and not set(closed)&set(SELECTED)
assert all(r['accepted_v4_row']==rows[r['id']] for r in prior['records'])
baseline=read(ROOT/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json',b.BASELINE_INVENTORY_SHA)
manifest=read(ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/manifest.json',b.MANIFEST_SHA)
full={b.cell(r):r for r in baseline['records'] if r['method']=='GuardFed-AD2+'}
entries={r['id']:r for r in manifest['jobs']}
archive=BACK/(TAG+'.tar.gz')
assert sha(archive)==receipt['archive_sha256']
PINS[str(archive.relative_to(ROOT)).replace('\\','/')]={'sha256':sha(archive),'bytes':archive.stat().st_size}
assert receipt['previous_receipt_sha256']==prior['backup_chain'][-1]['receipt_sha256'] and receipt['reused_full_weights_repacked']==0 and not receipt['failure_identities']
evidence_path=ROOT/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
assert sha(evidence_path)==b.EVIDENCE_V4_SHA
source(evidence_path)
evidence=b.load('original_v4_evidence_for_after71',evidence_path,b.EVIDENCE_V4_SHA)
verified=evidence.verify_archive(archive,receipt)
assert verified['members_verified']==proof['members_verified']==118
artifacts={r['id']:copy.deepcopy(r) for r in prior['records']}
# Reuse the original next11 inventory constructor itself, without recoding it.
prepare_source=source(OLD/'prepare.py')
constructors=[n for n in ast.walk(ast.parse(prepare_source)) if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='record' for t in n.targets)]
assert len(constructors)==1
constructor=compile(ast.Module(body=constructors,type_ignores=[]),'original_next11/record_constructor','exec')
archive_rel=str(archive.relative_to(ROOT)).replace('\\','/')
with tarfile.open(archive,'r:gz') as tar:
    backup_inventory=json.load(tar.extractfile('backup_inventory.json'))
    assert backup_inventory['reused_full_weights_repacked']==0
    for model_id in SELECTED:
        result=json.load(tar.extractfile('runs/'+model_id+'/result.json'))
        job=json.load(tar.extractfile('jobs/'+model_id+'.json'))
        row,entry=rows[model_id],entries[model_id]
        control=full[job['distribution'],job['attack'],job['config']['seed']]
        evidence.terminal_checks(result,job,manifest,job['variant'])
        evidence.partition_identity(result,control)
        assert result['revision_job']['variant']==job['variant']=='minus_U' and result['config']==job['config']
        assert result['revision_job']['checkpoint_sha256']==row['checkpoint_sha256'] and result['revision_job']['torch_version']=='2.11.0+cu128'
        assert job['adapter_hashes']==result['revision_job']['adapter_hashes']==manifest['adapter_hashes']
        refs={}
        for kind,member in [('checkpoint','runs/'+model_id+'/model.pt'),('result','runs/'+model_id+'/result.json'),('raw_job','jobs/'+model_id+'.json')]:
            refs[kind]={'archive':archive_rel,'archive_sha256':receipt['archive_sha256'],'member':member,**backup_inventory['members'][member]}
        ns=dict(model_id=model_id,result=result,job=job,row=row,entry=entry,control=control,refs=refs,b=b,copy=copy)
        exec(constructor,ns)
        artifacts[model_id]=ns['record']
inv=copy.deepcopy(prior)
inv.update(schema='celeba_mechanism_valid_replay_inventory_after71',scope=SCOPE,status='PREPARED_NOT_APPROVED',mechanism_inspection_sha256=root['inspection_sha256'],records=[artifacts[k] for k in sorted(artifacts)],excluded_prior_replay_ids=closed,selected_replay_ids=SELECTED,pending_new_ids_no_checkpoint=sorted(set(entries)-set(artifacts)),counts={'planned_new':800,'actual_accepted_new':82,'Full_reference_only':100,'pending_new_without_checkpoint':718},activation='PREPARED ONLY; native82 minus adopted three-view71 exact11. No resource preflight, execution or acceptance for these11 has occurred here.')
inv['prior_replay_boundary']={'status':'PRIOR71_EXCLUDED_ROOT_ADOPTION_RECEIPT_BOUND','root_adoption_receipt':str(adoption_path.relative_to(ROOT)).replace('\\','/'),'root_adoption_receipt_sha256':sha(adoption_path),'prior60_root_adoption_sha256':sha(prior60_path),'prior_inventory_sha256':sha(OLD/'inventory_actual71_Full100refs.json'),'next_execution_approved':False}
inv['backup_chain'].append({'archive':archive_rel,'archive_sha256':receipt['archive_sha256'],'receipt':str(receipt_path.relative_to(ROOT)).replace('\\','/'),'receipt_sha256':sha(receipt_path),'verification':str(proof_path.relative_to(ROOT)).replace('\\','/'),'verification_sha256':sha(proof_path),'accepted_new_ids':SELECTED,'members_verified_now_without_inference':118})
inv['input_pins'].update({k:v['sha256'] for k,v in PINS.items()})
bridge_source=source(OLD/'bridge.py')
bridge_new=bridge_source.replace('INCREMENTAL_NEXT11','INCREMENTAL_AFTER71').replace("INSPECTION_SHA = '"+b.INSPECTION_SHA+"'","INSPECTION_SHA = '"+root['inspection_sha256']+"'")
for key,value in [('ACCEPTED_IDS',sorted(rows)),('EXCLUDED_PRIOR_IDS',closed),('REPLAY_IDS',SELECTED)]:
    n=next(n for n in ast.parse(bridge_new).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets))
    bridge_new=bridge_new.replace(ast.get_source_segment(bridge_new,n),key+' = '+repr(value),1)
bridge_new=bridge_new.replace('== 71','== 82').replace('exactly71','exactly82').replace('accepted71','accepted82').replace('prior60','prior71').replace('== 729','== 718')
write_source('bridge.py',bridge_new)
new_bridge=load('after71_prepared_bridge',HERE/'bridge.py')
assert inv['full_references']==prior['full_references'] and all(artifacts[r['id']]==r for r in prior['records'])
new_bridge.validate_inventory(inv,baseline)
save('inventory_actual82_Full100refs.json',inv)
inventory_sha=sha(HERE/'inventory_actual82_Full100refs.json')
bridge_sha=sha(HERE/'bridge.py')
table_path=ROOT/'tmp/celeba_nine_method_three_view_tables_20261009/records_three_views_900.json'
actual900=read(table_path,'983bca43dff7e94dd79a312f273c65ed3ea1d146fff3ebfbf3bd2a854e124529')
collector=read(ROOT/actual900['final_collector_path'],actual900['final_collector_sha256'])
assert collector['accepted_n']==900
byid={r['id']:r for r in actual900['records']}
full_bindings=[]
for ref in inv['full_references']:
    r=byid[ref['id']]
    assert r['checkpoint_sha256']==ref['checkpoint_sha256'] and r['model_inventory_record_sha256']==ref['baseline_record_canonical_sha256']
    full_bindings.append({'full_reference':ref,'actual900_collector_sha256':actual900['final_collector_sha256'],'receipt_sha256':r['receipt_sha256'],'prediction_arrays_sha256':r['prediction_arrays_sha256'],'source_binding':r['source_binding'],'runtime':r['runtime'],'new_Full_inference':False,'weights_repacked':False})
save('FULL100_ACTUAL_SOURCE_BINDINGS.json',{'status':'REFERENCE_ONLY_ALL100_ACTUAL900_SOURCE_BOUND_NO_NEW_FULL_INFERENCE','source_records_path':str(table_path.relative_to(ROOT)).replace('\\','/'),'source_records_sha256':sha(table_path),'records':full_bindings,'selected11_pairs':{i:artifacts[i]['paired_full']['id'] for i in SELECTED}})
scope={'status':'PREPARED_NOT_APPROVED','scope':SCOPE,'inventory_sha256':inventory_sha,'bridge_sha256':bridge_sha,'actual_native_accepted_records':82,'selected_ids':SELECTED,'excluded_prior_ids':closed,'prior71_root_adoption':inv['prior_replay_boundary'],'selected_records':[{'id':i,'checkpoint_sha256':artifacts[i]['checkpoint']['sha256'],'original_result_sha256':artifacts[i]['result']['sha256'],'raw_job_sha256':artifacts[i]['raw_job']['sha256'],'inventory_record_canonical_sha256':b.canonical(artifacts[i]),'paired_full':artifacts[i]['paired_full'],'Full_three_views_status':'ACTUAL900_SOURCE_BOUND_REFERENCE_ONLY_NO_NEW_FULL_INFERENCE'} for i in SELECTED],'native_tolerance':1e-12,'views':b.VIEWS,'train_count':162770,'valid_count':19867,'rounds':70,'Full_reference_only_count':100,'pending_native_count_without_model':718,'runtime_proposal':{'fresh_child':True,'device':'cpu','CUDA_VISIBLE_DEVICES':'','compute_threads':8,'max_processes':1,'allowed_cpus':list(range(112,120)),'nice':10,'io_priority':'idle','resource_allocation_measured_here':False},'no_training':True,'no_test':True,'automatic_retry':False,'requires_root_source_review_and_external_exact_approval':True}
save('SCOPE.json',scope)
write_source('SELECTED_11.txt','\n'.join(SELECTED)+'\n')
approval=read(OLD/'APPROVAL_TEMPLATE.json')
approval.update(scope=SCOPE,inventory_sha256=inventory_sha,bridge_sha256=bridge_sha,selected_ids=SELECTED)
save('APPROVAL_TEMPLATE.json',approval)
bindings=read(EX/'RUNTIME_BINDINGS.json')
old_name='celeba_mechanism_valid_incremental_next11_20261009'
new_name=HERE.name
remote='/workspace/guardfed_checks/'+new_name
bindings['outputs']={i:remote+'/runs/'+i for i in SELECTED}
save('execution_candidate/RUNTIME_BINDINGS.json',bindings)
source_review=read(OLD/'root_source_review/ROOT_REVIEW.json','b1ff1fad6f5fe08cebec3008b6df861396c2e11c289b8f05815debb78dc3ed14')
assert source_review['bind_runtime_and_other_scientific_functions_source_exact'] and not source_review['runtime_approval']
reuse={}
for name in ['require','digest','canonical','read','save_new','load','cell','full_reference','require_approval','bind_runtime','reference_baseline_full']:
    a=next(n for n in ast.parse(bridge_source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    z=next(n for n in ast.parse(bridge_new).body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.get_source_segment(bridge_source,a)==ast.get_source_segment(bridge_new,z),name
    reuse[name]={'source_exact':True,'AST_exact':True,'source_sha256':hashlib.sha256(ast.get_source_segment(bridge_new,z).encode()).hexdigest()}
oldval=next(n for n in ast.parse(bridge_source).body if isinstance(n,ast.FunctionDef) and n.name=='validate_inventory')
newval=next(n for n in ast.parse(bridge_new).body if isinstance(n,ast.FunctionDef) and n.name=='validate_inventory')
reverse=ast.get_source_segment(bridge_new,newval).replace('== 82','== 71').replace('exactly82','exactly71').replace('accepted82','accepted71').replace('prior71','prior60').replace('== 718','== 729')
assert reverse==ast.get_source_segment(bridge_source,oldval)
save('SOURCE_REUSE.json',{'status':'LOCAL_PREPARATION_IDENTITY_AND_SOURCE_CHECK_PASS_NOT_RUNTIME_APPROVAL','original_science_seal_sha256':sha(OLD/'FILES_SHA256.json'),'original_execution_seal_sha256':sha(EX/'EXECUTION_SOURCE_SHA256.json'),'original_root_source_review_sha256':sha(OLD/'root_source_review/ROOT_REVIEW.json'),'original_root_execution_review_sha256':sha(EX/'ROOT_EXECUTION_REVIEW.json'),'bridge_functions':reuse,'validate_inventory_only_snapshot_count_and_diagnostic_strings_changed':True,'record_constructor_original_AST':True,'original_prior71_records_exact':True,'Full100_references_exact':True,'new_native_archive_members_verified':118,'selected11_identity_exact':True,'scientific_runtime_imported':False})
save('NATIVE82_MINUS_THREEVIEW71.json',{'status':'EXACT11_PREPARED_NOT_EXECUTED','native82_inspection_sha256':sha(inspection_path),'ledger_sha256':sha(BACK/'verified_ledger.json'),'adopted71_receipt_sha256':sha(adoption_path),'prior60_receipt_sha256':sha(prior60_path),'native82_ids':sorted(rows),'closed_three_view71_ids':closed,'selected11_ids':SELECTED,'selected11_records':scope['selected_records'],'resources':'ACTUAL_LINUX_PREFLIGHT_PENDING; original preflight must run at root installation; no current allocation/health claim','new_three_view_acceptance':False,'new_CNN_inference':0,'Full_weights_repacked':0,'test':False})
# Science seal is independent of later execution seal. No old runtime artifacts
# or approvals are copied into the new namespace.
save('INPUT_PINS.json',PINS)
science_files=['prepare.py','bridge.py','inventory_actual82_Full100refs.json','SCOPE.json','SELECTED_11.txt','APPROVAL_TEMPLATE.json','FULL100_ACTUAL_SOURCE_BINDINGS.json','SOURCE_REUSE.json','NATIVE82_MINUS_THREEVIEW71.json','INPUT_PINS.json']
save('FILES_SHA256.json',{'status':'PREPARED_NOT_APPROVED','scope':SCOPE,'members':[{'path':n,'sha256':sha(HERE/n),'size':(HERE/n).stat().st_size} for n in science_files],'excludes_this_seal_itself':True,'CNN':False,'dispatch':False})
new_seal_sha=sha(HERE/'FILES_SHA256.json')
changes={old_name:new_name,'INCREMENTAL_NEXT11':'INCREMENTAL_AFTER71','BOUNDED_NEXT11':'BOUNDED_AFTER71','APPROVED_NEXT11':'APPROVED_AFTER71','NEXT11_VALID':'AFTER71_VALID','PREPARED_NEXT11':'PREPARED_AFTER71','valid_next11':'valid_after71','inventory_actual71_Full100refs.json':'inventory_actual82_Full100refs.json','closed60':'closed71','Prior60':'Prior71','prior60':'prior71','exactly71':'exactly82','actual_native71':'actual_native82',b.INSPECTION_SHA:root['inspection_sha256'],'a8237e434b6e4d3a92042fc05b44adecdace03d1a2f105c4f68f2e47780ec220':bridge_sha,'ee2ccf040f677e97c2f03c01e760e36cdd51a21f1c53b7a0122d8c218af34ef0':inventory_sha,'65f706e8c7c7d7e18e76c8a300dd845297c97b3bd5c6c182bbbb8aad0103b5ff':new_seal_sha,'7ecd04d71ffea033d0fc56d693894b70337fc8cbf0f1d6f621ab27f7c4fa196d':sha(OUTEX/'RUNTIME_BINDINGS.json')}
diff=[]
for name in ['batch.py','install_once.py','resource_extra.py','verify_saved_increment.py','verify_backup.py','backup_completed.py','service.sh.template','supervisor.conf.template']:
    before=source(EX/name)
    after=before
    for old,new in changes.items():
        after=after.replace(old,new)
    if name=='batch.py':
        n=next(n for n in ast.parse(after).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SELECTED' for t in n.targets))
        after=after.replace(ast.get_source_segment(after,n),'SELECTED = '+repr(SELECTED),1).replace("len(scope['excluded_prior_ids']) == 60","len(scope['excluded_prior_ids']) == 71")
    if name=='resource_extra.py':
        assert after==before
    write_source('execution_candidate/'+name,after)
    diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='next11/'+name,tofile='after71/'+name))
diff.extend(difflib.unified_diff(bridge_source.splitlines(True),bridge_new.splitlines(True),fromfile='next11/bridge.py',tofile='after71/bridge.py'))
write_source('MINIMAL_SOURCE_DIFF.patch',''.join(diff))
for name in ['ROOT_REVIEW_TEMPLATE.json','APPROVED_TEMPLATE.json']:
    j=read(EX/name)
    text=json.dumps(j)
    for old,new in changes.items():text=text.replace(old,new)
    j=json.loads(text)
    j.update(selected_ids=SELECTED,scope_sha256=sha(HERE/'SCOPE.json'),inventory_sha256=inventory_sha,bridge_sha256=bridge_sha,execution_seal_sha256=None,status='PREPARED_NOT_APPROVED')
    if name=='ROOT_REVIEW_TEMPLATE.json':
        j.update(source_seal_sha256=new_seal_sha,execution_authorized_within_existing_user_request=False,closed71_must_not_replay=closed)
    else:
        j.update(root_approval_sha256=None,outputs={i:remote+'/execution_candidate/runs/'+i for i in SELECTED},original_scope_outputs=bindings['outputs'])
    save('execution_candidate/'+name,j)
execution_files=[p.name for p in OUTEX.iterdir() if p.is_file()]
save('execution_candidate/EXECUTION_SOURCE_SHA256.json',{'status':'PREPARED_ONLY_REQUIRES_EXTERNAL_ROOT_EXECUTION_APPROVAL','parent_science_seal_sha256':new_seal_sha,'original_next11_execution_seal_sha256':sha(EX/'EXECUTION_SOURCE_SHA256.json'),'members':[{'path':n,'sha256':sha(OUTEX/n),'size':(OUTEX/n).stat().st_size} for n in sorted(execution_files)]})
for rel,pin in PINS.items():assert sha(ROOT/rel)==pin['sha256'],rel
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
save('PREPARATION_CHECK.json',{'status':'PREPARED_ONLY_LOCAL_IDENTITY_AND_SOURCE_CHECK_PASS','native_actual':82,'prior_three_view_actual':71,'selected':11,'IDs':SELECTED,'new_native_archive_members_verified':118,'prior71_records_unchanged':True,'Full100_references_unchanged':True,'original_bridge_scientific_functions_source_exact':True,'resource_extra_source_byte_exact':True,'cpu_threads':8,'compute_processes':1,'actual_Linux_preflight_performed':False,'server_operations':0,'CNN_inference':0,'tensor_load':0,'training':0,'test':0,'Full_model_copies':0,'science_seal_sha256':new_seal_sha,'execution_seal_sha256':sha(OUTEX/'EXECUTION_SOURCE_SHA256.json'),'original_inputs_rehashed_unchanged':len(PINS)})
print(json.dumps({'status':'PREPARED_ONLY','science_seal_sha256':new_seal_sha,'execution_seal_sha256':sha(OUTEX/'EXECUTION_SOURCE_SHA256.json'),'selected':SELECTED}),flush=True)
