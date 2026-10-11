"""Only the fixed LoGo two saved-output identity chain; stdlib, no scientific imports."""
from pathlib import Path
import hashlib,json
HERE=Path(__file__).resolve().parent
IDS=['LoGoFair-DP_07_FedAvg_IID_Benign_seed91001','LoGoFair-DP_07_FedAvg_IID_F Flip_seed91001']
COHORT_SHA='52d6a15e077eb50b1d81d2aa0e481c6e2e5fb5dfeea24e84ba6036535dffed9b'
CORE_SHA='cdd5586517d6a5d51a455d01384dd3093f5bd79947f6ba829bfab42520cb5eed'
ROOT_SHA='83a047fdd4d9c997c3ca31ff713de134575f1bcc6c75f6da579031f604efb083'
INDEX_SHA='6a21265f08fb26da130d95c67bf4130039b25916f44a543cd695e15665a25b64'
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def pin(p):return {'path':str(p),'sha256':sha(p)}
def input_gate(x):
    need(x['exact_ids']==IDS and set(x['task_accepted'])==set(IDS),'Only original fixed2')
    for p in [x['root_queue_auth'],x['deployment_seal'],*x['task_accepted'].values()]:
        need(set(p)=={'path','sha256'} and isinstance(p['path'],str) and p['path'].startswith('/workspace/') and '..' not in Path(p['path']).parts and isinstance(p['sha256'],str) and len(p['sha256'])==64,'Unbound actual proof')
    need(x['FedAvg_CUDA_root']['sha256']==ROOT_SHA and x['FedAvg_CUDA_index']['sha256']==INDEX_SHA,'Different adopted FedAvg CUDA cache')
    need(isinstance(x['server_base'],str) and x['server_base'].startswith('/workspace/guardfed_checks/uniform_cuda_LoGo100_') and '..' not in Path(x['server_base']).parts,'Unbound actual namespace')
    need(isinstance(x['service'],str) and x['service'].startswith('guardfed_') and all(c.isalnum() or c=='_' for c in x['service']),'Unbound actual service')
    need(x['root_queue_auth']['path']==x['server_base']+'/ROOT_QUEUE_AUTH.json' and x['deployment_seal']['path']==x['server_base']+'/DEPLOYMENT_SEAL.json','Mixed producer')
    for i,p in x['task_accepted'].items():need(p['path']==x['server_base']+'/tasks/'+i+'/ACCEPTED.json','Mixed task namespace')
def verify_record(x,row,load):
    """load(pin) authenticates SHA and returns JSON; no freshness/time reinterpretation."""
    i=row['post_id'];b=x['server_base'];task=b+'/tasks/'+i;out=b+'/runs/'+i
    auth=load(x['root_queue_auth']);aa=load(x['task_accepted'][i])
    need(auth['status']=='ROOT_AUTHORIZED_LOGO100_CALLER_V1' and auth['execution_authorized'] is True and auth['canary_ids']==IDS,'Wrong root queue authority')
    need(auth['exact_cohort']['sha256']==COHORT_SHA and auth['FedAvg_CUDA_root']==x['FedAvg_CUDA_root'] and auth['FedAvg_CUDA_index']==x['FedAvg_CUDA_index'],'Wrong cohort/cache authority')
    need(aa['post_id']==i and aa['baseline_id']==row['baseline_id'] and aa['root_adopted'] is False,'Wrong original Linux task')
    for role,p in [('acceptance',out+'/formal_saved_acceptance.json'),('binding',out+'/formal_binding.json'),('approval',task+'/approval.json')]:need(aa[role]['path']==p,'Task link namespace differs')
    accepted=load(aa['acceptance']);formal=load(aa['binding']);approval=load(aa['approval']);result=accepted['result']
    need(accepted['status']=='PER_RECORD_LOGO_FROZEN_POST_CUDA_CACHE_SAVED_ACCEPTED' and accepted['new_fit']==accepted['new_CNN']==0 and accepted['root_adopted'] is False and accepted['test'] is False,'No original Linux whole acceptance')
    need(approval['status']=='ROOT_AUTHORIZED_LOGO100_FROZEN_POST_CUDA_CACHE_REEVALUATION' and approval['execution_authorized'] is True and approval['post_id']==i and approval['baseline_id']==row['baseline_id'] and approval['output']==out,'Mixed actual worker approval')
    need(approval['root_queue_auth']==x['root_queue_auth'] and approval['FedAvg_CUDA_root']==x['FedAvg_CUDA_root'] and approval['FedAvg_CUDA_index']==x['FedAvg_CUDA_index'],'Unbound worker authority')
    for k in ['new_CNN','refit','reselect_recipe','test','automatic_retry','diagnostic_reuse','CPU_reference_equality_required']:need(approval[k] is False,'Changed postprocessing scope')
    need(approval['netcal']=='1.3.6' and approval['post_device']=='cpu' and approval['compute_threads']==approval['max_processes']==1,'Changed frozen runtime')
    need(approval['adapter_sha256']==approval['adapter']['sha256']=='4121b32dbaafaa04dcb2d212882e3e6af04ce06cdd4f0c32f949e1e0453bb3f1','Changed formal API')
    need(approval['runtime_dependencies']['core']['sha256']==CORE_SHA,'Changed frozen metric core')
    expected_bridge='c1dfa5b5881407c16b0c7c37cfa17cb27c62f9d9fe125a51bdca0c77a0865e81' if row['reused_screen_record'] else '0527f26e9c22da94f35d647d96ab06b227568c9febd9850c6ac5f251a408c16d'
    need(approval['runtime_dependencies']['bridge']['sha256']==expected_bridge,'Mixed screen/fullcoverage bridge')
    need(set(approval['original_artifacts'])==set(row['original_artifacts']),'Missing original post identity')
    for role,p in approval['original_artifacts'].items():need(p['sha256']==row['original_artifacts'][role]['sha256'],'Changed original '+role)
    need(formal['status']=='FROZEN_LOGO_POST_CUDA_CACHE_WHOLE_CHECK_PASSED_NOT_ROOT_ADOPTED' and formal['post_id']==i and formal['baseline_id']==row['baseline_id'],'Wrong formal binding')
    need(formal['approval_sha256']==aa['approval']['sha256'] and formal['FedAvg_CUDA_root_sha256']==ROOT_SHA and formal['frozen_post_state_sha256']==row['original_artifacts']['post_state.pkl']['sha256'],'Mixed binding identity')
    need(formal['original_saved_consistency_tolerance']==1e-12 and formal['new_fit']==formal['new_CNN']==0 and formal['root_adopted'] is False and formal['test'] is False and formal['historical_CPU_equality_required'] is False,'Changed original saved gate')
    rp={'path':out+'/result.json','sha256':formal['result_sha256']};ap={'path':out+'/acceptance.json','sha256':formal['acceptance_sha256']}
    need(load(rp)==result,'Original Linux accepted result differs');artifact=load(ap)
    job=load(approval['original_artifacts']['job'])
    need(artifact['status']=='PASS' and artifact['job_sha256']==approval['original_artifacts']['job']['sha256'] and set(artifact['artifact_hashes'])=={'post_state.pkl','scores_predictions.npz','result.json','mapping.npz','mapping_metadata.json'},'Incomplete original artifact receipt')
    need(artifact['artifact_hashes']['result.json']==rp['sha256'],'Result receipt differs')
    for role in ['post_state.pkl','mapping.npz','mapping_metadata.json']:need(artifact['artifact_hashes'][role]==row['original_artifacts'][role]['sha256'],'Original frozen post/mapping changed')
    need(result['job']==job and job['id']==i and job['candidate']=='LoGoFair-DP_07' and job['fit_seed']==result['fit_seed']==1719 and job['evaluation_split']=='valid' and job['seed']==91001 and job['settings']['post_rounds']==30,'Changed recipe/seed/endpoint')
    need(result['status']=='complete' and result['settings']==job['settings'] and [r['round'] for r in result['history']]==list(range(1,31)),'Incomplete original frozen post history')
    need(result['checkpoint_sha256']==row['checkpoint_sha256'] and result['original_result_sha256']==row['original_FedAvg_result_sha256'] and result['accepted_margin_cache_sha256']==formal['CUDA_cache_sha256'],'Wrong checkpoint/result/cache')
    need(result['root_image_ids_sha256']==row['root_image_ids_sha256'] and result['valid_image_ids_sha256']==row['valid_image_ids_sha256'],'Different sample identity')
    need(result['mapping_sha256']==job['mapping_sha256']==row['original_artifacts']['mapping.npz']['sha256'] and result['mapping_metadata_sha256']==job['mapping_metadata_sha256']==row['original_artifacts']['mapping_metadata.json']['sha256'],'Different virtual cohort mapping')
    need(accepted['historical_CPU_difference']==formal['historical_CPU_difference'],'Mixed diagnostic')
    return dict(result=result,job=job,approval=approval,formal=formal,artifact=artifact,scores={'path':out+'/scores_predictions.npz','sha256':artifact['artifact_hashes']['scores_predictions.npz']},accepted=accepted)
