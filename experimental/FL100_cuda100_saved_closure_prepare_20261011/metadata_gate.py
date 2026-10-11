"""FL metadata closure only. Scientific saved-output functions remain original."""
from pathlib import Path
import copy,hashlib,json
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,msg):
    if not ok:raise ValueError(msg)
def canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()
def original_receipt_canonical(v):return hashlib.sha256(json.dumps(v,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def projected(row):
    identity=row['identity'];record=copy.deepcopy(identity)
    record.update(distribution=row['distribution'],attack=row['attack'],seed=row['seed'],result=identity['original_artifact_pins']['result'],raw_job=identity['original_artifact_pins']['job'],config_canonical_sha256=canonical(identity['config']),training_torch=row['original_training_torch'])
    return record

def verify_metadata(inp,extract,members,frozen):
    x=extract;base=inp['base'];ids=frozen['ids'];need(base==frozen['planned_namespace'] and inp['exact_ids']==ids,'Unbound/wrong fixed3 namespace')
    def member(rel):
        need(rel in members and sha(x/rel)==members[rel]['sha256'],'Missing/changed archive member: '+rel);return x/rel
    def remote(pin):
        need(set(pin)=={'path','sha256'} and pin['path'].startswith(base+'/'),'Wrong producer pin')
        rel='base/'+pin['path'][len(base)+1:];p=member(rel);need(sha(p)==pin['sha256'],'Wrong actual producer member');return p
    selection=read(member('base/COHORT_SELECTION.json'));need(selection['frozen_selection']==frozen,'Foreign selected original identities')
    need(selection['full_cohort_sha256']==inp['cohort_sha256']==frozen['cohort_sha256'],'Wrong formal FL100 cohort')
    deployment=read(member('base/DEPLOYMENT_SEAL.json'));need(sha(x/'base/DEPLOYMENT_SEAL.json')==inp['deployment_sha256'],'Wrong deployment')
    for rel,p in deployment['files'].items():
        if 'base/'+rel in members:need(sha(x/'base'/rel)==p['sha256'],'Deployed source changed')
    for rel in ['guard.py','run_queue.py','worker.py','ROOT_QUEUE_AUTH.json']:
        need(sha(member('base/'+rel))==deployment['files'][rel]['sha256'],'Missing deployed source/auth')
    need(sha(member('base/adapter/FILES_SHA256.json'))==frozen['formal_seal'],'Wrong FL formal source seal')
    for rel,p in read(x/'base/adapter/FILES_SHA256.json')['files'].items():need(sha(member('base/adapter/'+rel))==p['sha256'],'Formal source incomplete')
    auth=read(member('base/ROOT_QUEUE_AUTH.json'));need(auth['execution_authorized'] is True and auth['canary_ids']==frozen['canary_ids'] and auth['ordered_ids']==ids and len(ids)==100,'Wrong actual FL scope')
    need(auth['exact_cohort']=={'path':base+'/adapter/EXACT100.json','sha256':frozen['cohort_sha256']},'Wrong actual cohort pin')
    complete=read(member('base/canary/COMPLETE.json'));need(sha(x/'base/canary/COMPLETE.json')==inp['canary_complete_sha256'] and complete['ids']==frozen['canary_ids'] and complete['server_saved_accepted'] is True,'Incomplete fixed3')
    need(complete['cohort']==auth['exact_cohort'] and complete['root_offserver_adopted'] is False,'Wrong completion link')
    remaining=read(member('base/remaining/COMPLETE.json'));need(sha(x/'base/remaining/COMPLETE.json')==inp['remaining_complete_sha256'] and remaining['ids']==frozen['remaining_ids'] and remaining['server_saved_accepted'] is True and remaining['cohort']==auth['exact_cohort'] and remaining['root_offserver_adopted'] is False,'Incomplete remaining97')
    need(set(complete['ids']).isdisjoint(remaining['ids']) and set(complete['ids']+remaining['ids'])==set(ids),'Wrong3+97 partition')
    need(set(inp['accepted_sha256'])==set(ids),'Missing actual task pin')
    verified={}
    for selected in frozen['rows']:
        row=selected['original_row'];route=selected['route'];identity=row['id'];record=projected(row);task='base/tasks/'+identity;run='base/runs/'+identity;package=selected['package']
        need(row['terminal_round']==70 and row['split']=='valid' and row['n_eval']==19867 and row['method']=='FLGMM' and canonical(row['identity'])==route['identity_canonical_sha256'],'Changed original endpoint/identity')
        accepted_path=member(task+'/ACCEPTED.json');accepted=read(accepted_path);need(sha(accepted_path)==inp['accepted_sha256'][identity] and accepted['id']==identity and accepted['root_offserver_adopted'] is False,'Wrong actual ACCEPTED')
        approval_path=remote(accepted['approval']);approval=read(approval_path);receipt_path=remote(accepted['receipt']);array_path=remote(accepted['arrays']);binding_path=remote(accepted['binding']);acceptance_path=remote(accepted['acceptance'])
        for p,name in [(approval_path,task+'/approval.json'),(receipt_path,run+'/receipt.json'),(array_path,run+'/validation_predictions.npz'),(binding_path,run+'/formal_binding.json'),(acceptance_path,run+'/formal_saved_acceptance.json')]:need(p==member(name),'Wrong task/run link')
        r=read(receipt_path);binding=read(binding_path);acceptance=read(acceptance_path)
        need(approval['status']=='ROOT_AUTHORIZED_UNIFORM_CUDA_FL100_FORMAL_V1' and approval['execution_authorized'] is True and approval['id']==identity and approval['identity_canonical_sha256']==route['identity_canonical_sha256'] and approval['native_index']==route['native_index'],'Wrong FL approval identity')
        need(approval['author_decision_sha256']=='a1f1be152a0aa5668b91d73d7297ea3e9275efaac8d82c3b22cfbe97b7b08fce' and approval['decision']=='B' and approval['exact_cohort']==auth['exact_cohort'] and approval['ordered_input_ids']==auth['ordered_ids'],'Wrong authorization lineage')
        need(approval['native_tolerance']==1e-12 and approval['target_split']=='valid' and approval['device']=='cuda:0' and approval['allowed_cpus']==[111] and approval['compute_threads']==approval['max_processes']==1,'Changed runtime/metric protocol')
        need(all(approval[k] is False for k in ('training','test','automatic_retry','diagnostic_reuse','old_CPU_reuse_as_CUDA')),'Forbidden execution')
        pre=read(remote(approval['Linux_preflight']));need(pre['pass'] is True and pre['input_scope_sha256']==frozen['cohort_sha256'] and pre['gpu_uuid']==approval['gpu_uuid'] and pre['allowed_cpus']==[111],'Original preflight mismatch')
        deps=approval['runtime_dependencies'];need(set(deps)==set(package),'Missing original package roles')
        for role,pin in deps.items():
            need(pin['sha256']==package[role]['sha256'] and binding['source_before'][pin['path']]['sha256']==pin['sha256'],'Changed frozen package role: '+role)
            if role not in {'manifest','proof_registry'}:need(sha(member('packages/'+route['identity_source']+'/'+role))==pin['sha256'],'Transported source role changed')
        need(sha(remote(approval['author_decision']))=='a1f1be152a0aa5668b91d73d7297ea3e9275efaac8d82c3b22cfbe97b7b08fce','Wrong author bytes')
        need(approval['gpu_uuid']==auth['gpu_uuid'],'Wrong actual authorized GPU')
        need(sha(remote(approval['source_seal']))==frozen['formal_seal'] and sha(remote(approval['formal_GPU_body']))==deployment['files']['adapter/gpu_body.py']['sha256'] and sha(remote(approval['resource_gate_source']))==deployment['files']['guard.py']['sha256'],'Wrong actual formal source')
        need(approval['adapter_sha256']==deployment['files']['adapter/adapter.py']['sha256'] and pre['adapter_sha256']==approval['adapter_sha256'],'Adapter/preflight differs')
        need(binding['scope']=='UNIFORM_CUDA_FL100_FORMAL_V1' and binding['id']==identity and binding['approval_sha256']==sha(approval_path) and binding['receipt_sha256']==sha(receipt_path) and binding['input_scope_sha256']==frozen['cohort_sha256'],'Mixed formal binding')
        need(binding['source_before']==binding['source_after'] and binding['artifact_before']==binding['artifact_after'],'Original source/artifact mutation')
        for kind in ['checkpoint','result','raw_job']:need(record[kind]['sha256'] in {p['sha256'] for p in binding['artifact_before'].values()},'Terminal artifact absent')
        need(r['scope']=='FORMAL_FL100_CUDA_PER_RECORD_V1' and r['status']=='NATIVE_VALID_REPLAY_PASS' and r['id']==identity and r['method']=='FLGMM' and r['external_identity_record_sha256']==original_receipt_canonical(record),'Wrong actual CUDA record')
        need(r['config_canonical_sha256']==record['config_canonical_sha256'] and all(r[k]==record[k] for k in ['distribution','attack','seed']) and r['original_training_torch']==record['training_torch'],'Changed config/seed/environment')
        for rk,k in [('checkpoint_sha256','checkpoint'),('original_result_sha256','result'),('original_job_sha256','raw_job')]:need(r[rk]==record[k]['sha256'],'Mixed checkpoint/result/job')
        need(r['valid_n']==19867 and r['valid_image_ids_sha256']=='64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf' and r['prediction_arrays_sha256']==sha(array_path),'Wrong valid array')
        need(r['runtime']['device']=='cuda:0' and r['runtime']['torch']=='2.11.0+cu128' and r['runtime']['cuda_device_count']==r['runtime']['torch_threads']==r['runtime']['interop_threads']==1,'Not unified CUDA runtime')
        need(r['weights_before']==r['weights_after'] and all(r[k] is False for k in ['optimizer_created','gradients_created','test_labels_accessed','test_inference_performed']),'Inference contract changed')
        need(acceptance['status']=='PER_RECORD_FL100_CUDA_SAVED_WHOLE_ACCEPTED' and acceptance['scope']==binding['scope'] and acceptance['id']==identity and acceptance['binding_sha256']==sha(binding_path),'Original Linux whole proof missing')
        need(acceptance['original_saved_science_sha256']==deps['saved_science']['sha256'] and acceptance['views']==r['views'] and acceptance['native_comparison']==r['native_comparison'] and r['native_comparison']['accepted'] is True and r['native_comparison']['max_abs_difference']<=1e-12,'Linux native/whole saved gate failed')
        verified[identity]=dict(record=record,r=r,receipt_path=receipt_path,array_path=array_path,accepted_path=accepted_path,run=run)
    return verified
