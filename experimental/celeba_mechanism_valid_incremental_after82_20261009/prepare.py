"""Prepare only the frozen native92 minus adopted three-view82 delta; no runtime."""
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
OLD = ROOT/'tmp/celeba_mechanism_valid_incremental_after71_20261009'
EX = OLD/'execution_candidate'
OUTEX = HERE/'execution_candidate'
BACK = ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
TAG = 'root_delta_20261009T171247Z'
SCOPE = 'MECHANISM_TERMINAL_VALID_REPLAY_INCREMENTAL_AFTER82'
SELECTED = ['minus_U_non-IID_FedSA_seed91010'] + [f'minus_U_non-IID_S-DFA_seed{s}' for s in range(91004,91011)] + [f'minus_U_non-IID_Sp-DFA_seed{s}' for s in range(91001,91003)]
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


science_seal = read(OLD/'FILES_SHA256.json','d05a0b81620d1791858a9f71405252443e593600f390359175ff961dde0e2bec')
execution_seal = read(EX/'EXECUTION_SOURCE_SHA256.json','0fe232aab75a870b7840fd6ef0bba85b3d12c541ca49a0c0976f3f8a7095352f')
for folder, seal in [(OLD,science_seal),(EX,execution_seal)]:
    for row in seal['members']:
        p=folder/row['path']
        assert sha(p)==row['sha256'] and p.stat().st_size==row['size'],str(p)
        source(p)
b=load('original_after71_bridge_for_preparation',OLD/'bridge.py')
prior=read(OLD/'inventory_actual82_Full100refs.json', '72aca76f626580faf465c61f35b2274068e76782eb1bcef543e98da23c2e6d1e')
adoption_path=EX/'backups/incremental_20261009T163050Z/ROOT_ADOPTION_REVIEW.json'
adoption=read(adoption_path,'fb2bc745e7a6b6aa3d7d4cb898aefa31c6acbf73bee7a8e1a30ab667e03c204c')
assert adoption['cumulative_three_view_models']==82 and adoption['accepted_new']==11 and adoption['prior_three_view_models']==71 and adoption['all_native_differences_zero']
assert set(prior['excluded_prior_replay_ids'])|set(adoption['accepted_new_ids']) == {r['id'] for r in prior['records']}
prior71_path=ROOT/prior['prior_replay_boundary']['root_adoption_receipt']
prior71=read(prior71_path,adoption['prior71_root_adoption_sha256'])
assert prior71['cumulative_three_view_models']==71
ledger=read(BACK/'verified_ledger.json','dda48b06f25f529bb8688acc5f6979d5bd27adf3964068e204451f43d43072a2')
root=read(BACK/TAG/'ROOT_DELTA_VERIFICATION.json','a4a8c506668db73b16931a37fa3fe843c5590b0bf5049846967bb9a86be6bd56')
receipt_path=BACK/(TAG+'.tar.gz.receipt.json')
receipt=read(receipt_path,ledger['entries'][-1]['receipt_sha256'])
proof_path=BACK/(TAG+'_offserver_verification.json')
proof=read(proof_path,root['offserver_proof_sha256'])
inspection_path=BACK/('mechanism_inspection_v4_'+TAG)/'inspection.json'
inspection=read(inspection_path,root['inspection_sha256'])
assert root['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and root['total_new_strict_and_offserver']==92
assert root['ledger_sha256']==sha(BACK/'verified_ledger.json') and root['receipt_sha256']==sha(receipt_path)
assert receipt['accepted_new_ids']==root['new_ids']==proof['accepted_new_ids']==SELECTED
assert proof['pass'] and proof['different_host_observed'] and proof['archive_sha256']==receipt['archive_sha256']==root['archive_sha256']
assert inspection['new_count']==92 and inspection['reused_count']==100 and not inspection['invalid']
assert inspection['manifest_sha256']==b.MANIFEST_SHA and inspection['source_script_sha256']==b.EVIDENCE_V4_SHA and inspection['full_inventory_sha256']==b.BASELINE_INVENTORY_SHA
rows={r['id']:r for r in inspection['records'] if r['role']=='new'}
closed=[r['id'] for r in prior['records']]
assert len(rows)==92 and set(rows)-set(closed)==set(SELECTED) and not set(closed)&set(SELECTED)
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
evidence=b.load('original_v4_evidence_for_after82',evidence_path,b.EVIDENCE_V4_SHA)
verified=evidence.verify_archive(archive,receipt)
assert verified['members_verified']==proof['members_verified']==110
artifacts={r['id']:copy.deepcopy(r) for r in prior['records']}
# Reuse the original next11 inventory constructor itself, without recoding it.
prepare_source=source(ROOT/'tmp/celeba_mechanism_valid_incremental_next11_20261009/prepare.py')
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
inv.update(schema='celeba_mechanism_valid_replay_inventory_after82',scope=SCOPE,status='PREPARED_NOT_APPROVED',mechanism_inspection_sha256=root['inspection_sha256'],records=[artifacts[k] for k in sorted(artifacts)],excluded_prior_replay_ids=closed,selected_replay_ids=SELECTED,pending_new_ids_no_checkpoint=sorted(set(entries)-set(artifacts)),counts={'planned_new':800,'actual_accepted_new':92,'Full_reference_only':100,'pending_new_without_checkpoint':708},activation='PREPARED ONLY; native92 minus adopted three-view82 exact10. No resource preflight, execution or acceptance for these10 has occurred here.')
inv['prior_replay_boundary']={'status':'PRIOR82_EXCLUDED_ROOT_ADOPTION_RECEIPT_BOUND','root_adoption_receipt':str(adoption_path.relative_to(ROOT)).replace('\\','/'),'root_adoption_receipt_sha256':sha(adoption_path),'prior71_root_adoption_sha256':sha(prior71_path),'prior_inventory_sha256':sha(OLD/'inventory_actual82_Full100refs.json'),'next_execution_approved':False}
inv['backup_chain'].append({'archive':archive_rel,'archive_sha256':receipt['archive_sha256'],'receipt':str(receipt_path.relative_to(ROOT)).replace('\\','/'),'receipt_sha256':sha(receipt_path),'verification':str(proof_path.relative_to(ROOT)).replace('\\','/'),'verification_sha256':sha(proof_path),'accepted_new_ids':SELECTED,'members_verified_now_without_inference':110})
inv['input_pins'].update({k:v['sha256'] for k,v in PINS.items()})
bridge_source=source(OLD/'bridge.py')
bridge_new=bridge_source.replace('INCREMENTAL_AFTER71','INCREMENTAL_AFTER82').replace("INSPECTION_SHA = '"+b.INSPECTION_SHA+"'","INSPECTION_SHA = '"+root['inspection_sha256']+"'")
for key,value in [('ACCEPTED_IDS',sorted(rows)),('EXCLUDED_PRIOR_IDS',closed),('REPLAY_IDS',SELECTED)]:
    n=next(n for n in ast.parse(bridge_new).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id==key for t in n.targets))
    bridge_new=bridge_new.replace(ast.get_source_segment(bridge_new,n),key+' = '+repr(value),1)
bridge_new=bridge_new.replace('== 82','== 92').replace('exactly82','exactly92').replace('accepted82','accepted92').replace('prior71','prior82').replace('== 718','== 708')
write_source('bridge.py',bridge_new)
new_bridge=load('after82_prepared_bridge',HERE/'bridge.py')
assert inv['full_references']==prior['full_references'] and all(artifacts[r['id']]==r for r in prior['records'])
new_bridge.validate_inventory(inv,baseline)
save('inventory_actual92_Full100refs.json',inv)
inventory_sha=sha(HERE/'inventory_actual92_Full100refs.json')
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
save('FULL100_ACTUAL_SOURCE_BINDINGS.json',{'status':'REFERENCE_ONLY_ALL100_ACTUAL900_SOURCE_BOUND_NO_NEW_FULL_INFERENCE','source_records_path':str(table_path.relative_to(ROOT)).replace('\\','/'),'source_records_sha256':sha(table_path),'records':full_bindings,'selected10_pairs':{i:artifacts[i]['paired_full']['id'] for i in SELECTED}})
scope={'status':'PREPARED_NOT_APPROVED','scope':SCOPE,'inventory_sha256':inventory_sha,'bridge_sha256':bridge_sha,'actual_native_accepted_records':92,'selected_ids':SELECTED,'excluded_prior_ids':closed,'prior82_root_adoption':inv['prior_replay_boundary'],'selected_records':[{'id':i,'checkpoint_sha256':artifacts[i]['checkpoint']['sha256'],'original_result_sha256':artifacts[i]['result']['sha256'],'raw_job_sha256':artifacts[i]['raw_job']['sha256'],'inventory_record_canonical_sha256':b.canonical(artifacts[i]),'paired_full':artifacts[i]['paired_full'],'Full_three_views_status':'ACTUAL900_SOURCE_BOUND_REFERENCE_ONLY_NO_NEW_FULL_INFERENCE'} for i in SELECTED],'native_tolerance':1e-12,'views':b.VIEWS,'train_count':162770,'valid_count':19867,'rounds':70,'Full_reference_only_count':100,'pending_native_count_without_model':708,'runtime_proposal':{'fresh_child':True,'device':'cpu','CUDA_VISIBLE_DEVICES':'','compute_threads':8,'max_processes':1,'allowed_cpus':list(range(112,120)),'nice':10,'io_priority':'idle','resource_allocation_measured_here':False},'no_training':True,'no_test':True,'automatic_retry':False,'requires_root_source_review_and_external_exact_approval':True}
save('SCOPE.json',scope)
write_source('SELECTED_10.txt','\n'.join(SELECTED)+'\n')
approval=read(OLD/'APPROVAL_TEMPLATE.json')
approval.update(scope=SCOPE,inventory_sha256=inventory_sha,bridge_sha256=bridge_sha,selected_ids=SELECTED)
save('APPROVAL_TEMPLATE.json',approval)
bindings=read(EX/'RUNTIME_BINDINGS.json')
old_name='celeba_mechanism_valid_incremental_after71_20261009'
new_name=HERE.name
remote='/workspace/guardfed_checks/'+new_name
bindings['outputs']={i:remote+'/runs/'+i for i in SELECTED}
save('execution_candidate/RUNTIME_BINDINGS.json',bindings)
source_review=read(OLD/'root_independent_review/ROOT_READY_REVIEW.json','3c2aedafad42e7e5787ad6f58e42fedb21932a13a5b641fc1bb168ddba8250ea')
assert source_review['scientific_functions_unchanged'] and not source_review['CNN_executed'] and not source_review['dispatch_performed']
reuse={}
for name in ['require','digest','canonical','read','save_new','load','cell','full_reference','require_approval','bind_runtime','reference_baseline_full']:
    a=next(n for n in ast.parse(bridge_source).body if isinstance(n,ast.FunctionDef) and n.name==name)
    z=next(n for n in ast.parse(bridge_new).body if isinstance(n,ast.FunctionDef) and n.name==name)
    assert ast.get_source_segment(bridge_source,a)==ast.get_source_segment(bridge_new,z),name
    reuse[name]={'source_exact':True,'AST_exact':True,'source_sha256':hashlib.sha256(ast.get_source_segment(bridge_new,z).encode()).hexdigest()}
oldval=next(n for n in ast.parse(bridge_source).body if isinstance(n,ast.FunctionDef) and n.name=='validate_inventory')
newval=next(n for n in ast.parse(bridge_new).body if isinstance(n,ast.FunctionDef) and n.name=='validate_inventory')
reverse=ast.get_source_segment(bridge_new,newval).replace('== 92','== 82').replace('exactly92','exactly82').replace('accepted92','accepted82').replace('prior82','prior71').replace('== 708','== 718')
assert reverse==ast.get_source_segment(bridge_source,oldval)
save('SOURCE_REUSE.json',{'status':'LOCAL_PREPARATION_IDENTITY_AND_SOURCE_CHECK_PASS_NOT_RUNTIME_APPROVAL','original_science_seal_sha256':sha(OLD/'FILES_SHA256.json'),'original_execution_seal_sha256':sha(EX/'EXECUTION_SOURCE_SHA256.json'),'original_root_source_review_sha256':sha(OLD/'root_independent_review/ROOT_READY_REVIEW.json'),'original_root_execution_approval_sha256':sha(EX/'ROOT_APPROVED.json'),'bridge_functions':reuse,'validate_inventory_only_snapshot_count_and_diagnostic_strings_changed':True,'record_constructor_original_AST':True,'original_prior82_records_exact':True,'Full100_references_exact':True,'new_native_archive_members_verified':110,'selected10_identity_exact':True,'scientific_runtime_imported':False})
save('NATIVE92_MINUS_THREEVIEW82.json',{'status':'EXACT10_PREPARED_NOT_EXECUTED','native92_inspection_sha256':sha(inspection_path),'ledger_sha256':sha(BACK/'verified_ledger.json'),'adopted82_receipt_sha256':sha(adoption_path),'prior71_receipt_sha256':sha(prior71_path),'native92_ids':sorted(rows),'closed_three_view82_ids':closed,'selected10_ids':SELECTED,'selected10_records':scope['selected_records'],'resources':'ACTUAL_LINUX_PREFLIGHT_PENDING; original preflight must run at root installation; no current allocation/health claim','new_three_view_acceptance':False,'new_CNN_inference':0,'Full_weights_repacked':0,'test':False})
# Science seal is independent of later execution seal. No old runtime artifacts
# or approvals are copied into the new namespace.
save('INPUT_PINS.json',PINS)
science_files=['prepare.py','bridge.py','inventory_actual92_Full100refs.json','SCOPE.json','SELECTED_10.txt','APPROVAL_TEMPLATE.json','FULL100_ACTUAL_SOURCE_BINDINGS.json','SOURCE_REUSE.json','NATIVE92_MINUS_THREEVIEW82.json','INPUT_PINS.json']
save('FILES_SHA256.json',{'status':'PREPARED_NOT_APPROVED','scope':SCOPE,'members':[{'path':n,'sha256':sha(HERE/n),'size':(HERE/n).stat().st_size} for n in science_files],'excludes_this_seal_itself':True,'CNN':False,'dispatch':False})
new_seal_sha=sha(HERE/'FILES_SHA256.json')
changes={old_name:new_name,'INCREMENTAL_AFTER71':'INCREMENTAL_AFTER82','BOUNDED_AFTER71':'BOUNDED_AFTER82','APPROVED_AFTER71':'APPROVED_AFTER82','AFTER71_VALID':'AFTER82_VALID','PREPARED_AFTER71':'PREPARED_AFTER82','valid_after71':'valid_after82','inventory_actual82_Full100refs.json':'inventory_actual92_Full100refs.json','closed71':'closed82','Prior71':'Prior82','prior71':'prior82','exactly82':'exactly92','actual_native82':'actual_native92',b.INSPECTION_SHA:root['inspection_sha256'],'5ef8398b77dcba3f69948b4873ba25b9c4bfd5f2a4acd1ab56c2127e0b6fa4ce':bridge_sha,'72aca76f626580faf465c61f35b2274068e76782eb1bcef543e98da23c2e6d1e':inventory_sha,'d05a0b81620d1791858a9f71405252443e593600f390359175ff961dde0e2bec':new_seal_sha,'c92c3c984300eeb7bb5db3c6d7745d611d9e587ba352cc0c7c37efb3170258d5':sha(OUTEX/'RUNTIME_BINDINGS.json')}
diff=[]
for name in ['batch.py','install_once.py','resource_extra.py','verify_saved_increment.py','verify_backup.py','backup_completed.py','service.sh.template','supervisor.conf.template']:
    before=source(EX/name)
    after=before
    for old,new in changes.items():
        after=after.replace(old,new)
    if name=='batch.py':
        n=next(n for n in ast.parse(after).body if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='SELECTED' for t in n.targets))
        after=after.replace(ast.get_source_segment(after,n),'SELECTED = '+repr(SELECTED),1).replace("len(scope['excluded_prior_ids']) == 71","len(scope['excluded_prior_ids']) == 82")
    if name=='install_once.py':
        after=after.replace('len(chosen)==11','len(chosen)==10').replace('reviewed11','reviewed10')
    if name=='backup_completed.py':
        after=after.replace('len(prior|set(ids))<11','len(prior|set(ids))<10').replace('ALL11_STRICT','ALL10_STRICT')
    after=after.replace('prepared11','prepared10').replace('reviewed11','reviewed10').replace('Only11 IDs','Only10 IDs').replace('exact11-scope','exact10-scope')
    if name=='resource_extra.py':
        assert after==before
    write_source('execution_candidate/'+name,after)
    diff.extend(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='after71/'+name,tofile='after82/'+name))
diff.extend(difflib.unified_diff(bridge_source.splitlines(True),bridge_new.splitlines(True),fromfile='after71/bridge.py',tofile='after82/bridge.py'))
write_source('MINIMAL_SOURCE_DIFF.patch',''.join(diff))
for name in ['ROOT_REVIEW_TEMPLATE.json','APPROVED_TEMPLATE.json']:
    j=read(EX/name)
    text=json.dumps(j)
    for old,new in changes.items():text=text.replace(old,new)
    j=json.loads(text)
    j.update(selected_ids=SELECTED,scope_sha256=sha(HERE/'SCOPE.json'),inventory_sha256=inventory_sha,bridge_sha256=bridge_sha,execution_seal_sha256=None,status='PREPARED_NOT_APPROVED')
    if name=='ROOT_REVIEW_TEMPLATE.json':
        j.update(source_seal_sha256=new_seal_sha,execution_authorized_within_existing_user_request=False,closed82_must_not_replay=closed)
    else:
        j.update(root_approval_sha256=None,outputs={i:remote+'/execution_candidate/runs/'+i for i in SELECTED},original_scope_outputs=bindings['outputs'])
    save('execution_candidate/'+name,j)
execution_files=[p.name for p in OUTEX.iterdir() if p.is_file()]
save('execution_candidate/EXECUTION_SOURCE_SHA256.json',{'status':'PREPARED_ONLY_REQUIRES_EXTERNAL_ROOT_EXECUTION_APPROVAL','parent_science_seal_sha256':new_seal_sha,'original_after71_execution_seal_sha256':sha(EX/'EXECUTION_SOURCE_SHA256.json'),'members':[{'path':n,'sha256':sha(OUTEX/n),'size':(OUTEX/n).stat().st_size} for n in sorted(execution_files)]})
for rel,pin in PINS.items():assert sha(ROOT/rel)==pin['sha256'],rel
assert 'torch' not in sys.modules and 'numpy' not in sys.modules
save('PREPARATION_CHECK.json',{'status':'PREPARED_ONLY_LOCAL_IDENTITY_AND_SOURCE_CHECK_PASS','native_actual':92,'prior_three_view_actual':82,'selected':10,'IDs':SELECTED,'new_native_archive_members_verified':110,'prior82_records_unchanged':True,'Full100_references_unchanged':True,'original_bridge_scientific_functions_source_exact':True,'resource_extra_source_byte_exact':True,'cpu_threads':8,'compute_processes':1,'actual_Linux_preflight_performed':False,'server_operations':0,'CNN_inference':0,'tensor_load':0,'training':0,'test':0,'Full_model_copies':0,'science_seal_sha256':new_seal_sha,'execution_seal_sha256':sha(OUTEX/'EXECUTION_SOURCE_SHA256.json'),'original_inputs_rehashed_unchanged':len(PINS)})
print(json.dumps({'status':'PREPARED_ONLY','science_seal_sha256':new_seal_sha,'execution_seal_sha256':sha(OUTEX/'EXECUTION_SOURCE_SHA256.json'),'selected':SELECTED}),flush=True)
