"""Apply frozen LoGo post_state to already adopted FedAvg CUDA cache; no CNN/refit."""
from pathlib import Path
import ast, copy, hashlib, inspect, json, os, sys, time

HERE=Path(__file__).resolve().parent
MODE='apply_original_frozen_post_state_to_root_adopted_FedAvg_CUDA_cache'
AUTHOR='a1f1be152a0aa5668b91d73d7297ea3e9275efaac8d82c3b22cfbe97b7b08fce'


def need(ok,message):
    if not ok:raise ValueError(message)


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
    return h.hexdigest()


def pinned(pin):
    need(isinstance(pin,dict) and set(pin)=={'path','sha256'},'Actual path/SHA required')
    p=Path(pin['path'])
    need(p.is_absolute() and '..' not in p.parts and p.resolve()==p and p.is_file()
         and not p.is_symlink() and sha(p)==pin['sha256'],'Changed/unbound dependency')
    return p


def canonical(x):
    return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode()).hexdigest()


def metadata_gate(a,cohort,row,now):
    need(a.get('status')=='ROOT_AUTHORIZED_LOGO100_FROZEN_POST_CUDA_CACHE_REEVALUATION'
         and a.get('execution_authorized') is True and a.get('mode')==MODE,'No actual frozen-model reevaluation approval')
    need(a.get('author_decision_sha256')==AUTHOR and a.get('post_id')==row['post_id']
         and a.get('baseline_id')==row['baseline_id'],'Foreign author/post/model identity')
    need(len(cohort['records'])==100 and len({r['baseline_id'] for r in cohort['records']})==100
         and all(r['baseline_id'].startswith('FedAvg_') for r in cohort['records']), 'Full/duplicate/partial input refused')
    need(a.get('netcal')=='1.3.6' and a.get('post_device')=='cpu'
         and a.get('compute_threads')==a.get('max_processes')==1,'Original CPU postprocessor budget required')
    for k in ['new_CNN','refit','reselect_recipe','test','automatic_retry','diagnostic_reuse','CPU_reference_equality_required']:
        need(a.get(k) is False,'Forbidden or non-original gate: '+k)
    cpus=a.get('allowed_cpus')
    need(isinstance(cpus,list) and cpus and all(type(c)is int and c>=0 for c in cpus)
         and len(set(cpus))==len(cpus),'Actual CPU affinity required')
    need(type(a.get('deadline_unix')) in (int,float) and now<a['deadline_unix'],'Expired root authorization')
    output=a.get('output')
    need(isinstance(output,str) and output.startswith('/workspace/guardfed_checks/uniform_cuda_LoGo100_')
         and '..' not in Path(output).parts and Path(output).name==row['post_id'],'New bound server namespace required')


def bind_logo(bridge,core,cache_verifier,resource_gate,actualapprovalpin):
    """Late-bound API; cache_verifier's real root/index schema is still missing.

    That reviewed module must verify_root_index(root_pin,index_pin,ordered_FedAvg_ids)
    and load_cache(record). It supplies original-metadata labels, never a new CNN.
    """
    need(sys.platform=='linux' and not sys.flags.optimize,'Strict original Linux runtime required')
    ap=pinned(actualapprovalpin);a=json.loads(ap.read_bytes())
    cp=pinned(a['cohort']);need(sha(cp)==sha(HERE/'EXACT100.json'),'Different accepted100 scope')
    cohort=json.loads(cp.read_bytes());row=next((r for r in cohort['records'] if r['post_id']==a.get('post_id')),None)
    need(row is not None,'Foreign LoGo post_state');metadata_gate(a,cohort,row,time.time())
    pins=json.loads((HERE/'SOURCE_PINS.json').read_bytes())
    need(a['adapter_sha256']==sha(__file__),'Adapter changed')
    sealpath=pinned(a['source_seal']);seal=json.loads(sealpath.read_bytes())
    for n,p in seal['files'].items():need(Path(n).name==n and sha(HERE/n)==p['sha256'],'Source seal changed')
    need(sha(pinned(a['author_decision']))==AUTHOR,'Author approval bytes changed')
    deps=a.get('runtime_dependencies');need(isinstance(deps,dict) and set(deps)=={'bridge','core','cache_verifier','resource_gate'},'Actual dependency roles required')
    dp={k:pinned(v) for k,v in deps.items()}
    for name,module in [('bridge',bridge),('core',core),('cache_verifier',cache_verifier)]:
        need(Path(inspect.getsourcefile(module)).resolve()==dp[name],'Wrong actual module: '+name)
    need(Path(inspect.getsourcefile(resource_gate)).resolve()==dp['resource_gate'],'Unreviewed resource callback')
    expected_bridge=pins['bridge']['sha256'] if row['reused_screen_record'] else pins['accepted_fullcoverage_bridge_sha256']
    need(sha(dp['bridge'])==expected_bridge and sha(dp['core'])==bridge.CORE_SHA,'Original source lineage changed')
    need(callable(cache_verifier.verify_root_index) and callable(cache_verifier.load_cache),'Actual upstream root verifier missing')
    artifacts=a.get('original_artifacts')
    need(isinstance(artifacts,dict) and set(artifacts)==set(row['original_artifacts']),'Frozen post/job/mapping identities required')
    oldpaths={k:pinned(v) for k,v in artifacts.items()}
    for k,p in oldpaths.items():need(sha(p)==row['original_artifacts'][k]['sha256'],'Accepted old artifact differs: '+k)
    job=json.loads(oldpaths['job'].read_bytes());old=json.loads(oldpaths['result.json'].read_bytes());old_accept=json.loads(oldpaths['acceptance'].read_bytes())
    need(old['job']==job and job['id']==row['post_id'] and job['baseline_id']==row['baseline_id']
         and job['fit_seed']==1719 and job['candidate']=='LoGoFair-DP_07'
         and job['settings']==cohort['selected_recipe']['settings'],'Original fitted recipe/job differs')
    need(old_accept['status']=='PASS' and old_accept['job_sha256']==sha(oldpaths['job']),'Broken original acceptance')
    for n,h in old_accept['artifact_hashes'].items():need(n in oldpaths and sha(oldpaths[n])==h,'Incomplete old artifact identity')
    need(old['metrics']==row['historical_CPU_native_metrics'] and old['checkpoint_sha256']==row['checkpoint_sha256'],'Historical native reference changed')
    bridge.validate_job(job,json.loads((bridge.HERE/'protocol.json').read_text(encoding='utf8')))
    for n,h in job['local_hashes'].items():need(sha(bridge.HERE/n)==h,'Original protocol/source dependency changed')
    np,torch=bridge.np,bridge.torch

    def fresh():
        need(os.environ.get('CUDA_VISIBLE_DEVICES')=='' and torch.cuda.device_count()==0,'Postprocessor must remain CPU-only')
        need(torch.get_num_threads()==torch.get_num_interop_threads()==1
             and set(os.sched_getaffinity(0))==set(a['allowed_cpus']) and time.time()<a['deadline_unix'],'CPU/deadline drift')
        resource_gate(a)

    fresh()
    # A future real root verifier is mandatory; a server ACCEPTED.json alone is insufficient.
    rp=pinned(a['FedAvg_CUDA_root']);ip=pinned(a['FedAvg_CUDA_index'])
    verified=cache_verifier.verify_root_index(a['FedAvg_CUDA_root'],a['FedAvg_CUDA_index'],[r['baseline_id'] for r in cohort['records']])
    need(verified['root_adopted'] is True and verified['root_sha256']==sha(rp)
         and verified['index_sha256']==sha(ip) and len(verified['records'])==100,'Actual adopted FedAvg100 required')
    selected=next((r for r in verified['records'] if r['id']==row['baseline_id']),None)
    need(selected is not None and [r['id'] for r in verified['records']]==[r['baseline_id'] for r in cohort['records']], 'Changed root-adopted FedAvg order')
    need(canonical(selected['inventory_record'])==row['route']['identity_canonical_sha256'],'Wrong adopted FedAvg model object')
    member_names={'receipt','predictions','formal_binding','saved_acceptance','original_result'}
    need(set(selected['members'])==member_names,'Complete actual cache member pins required')
    members={k:pinned(v) for k,v in selected['members'].items()}
    receipt=json.loads(members['receipt'].read_bytes());binding=json.loads(members['formal_binding'].read_bytes())
    saved=json.loads(members['saved_acceptance'].read_bytes())
    need(receipt['id']==row['baseline_id'] and receipt['method']=='FedAvg' and receipt['runtime']['device']=='cuda:0'
         and receipt['prediction_arrays_sha256']==sha(members['predictions'])
         and receipt['checkpoint_sha256']==row['checkpoint_sha256']
         and receipt['original_result_sha256']==row['original_FedAvg_result_sha256']
         and receipt['original_job_sha256']==row['original_FedAvg_job_sha256'],'Wrong formal FedAvg cache lineage')
    need(binding['scope']=='UNIFORM_CUDA_ORIGINAL800_FORMAL_V1' and binding['receipt_sha256']==sha(members['receipt'])
         and saved['status']=='PER_RECORD_ORIGINAL800_CUDA_SAVED_WHOLE_ACCEPTED' and saved['id']==row['baseline_id'], 'Server saved proof/receipt mismatch')
    need(sha(members['original_result'])==row['original_FedAvg_result_sha256'],'Original FedAvg result changed')
    original_result=json.loads(members['original_result'].read_bytes())
    original_entry=next(r for r in json.loads((bridge.HERE/'reuse_manifest.json').read_text(encoding='utf8'))['entries'] if r['id']==row['baseline_id'])
    need(original_entry['accepted_record']['checkpoint_sha256']==row['checkpoint_sha256']
         and original_entry['accepted_record']['original_result_sha256']==row['original_FedAvg_result_sha256'],'Original reuse identity changed')
    projected=copy.deepcopy(original_entry);projected['accepted_record']['cache_sha256']=sha(members['predictions'])
    source=(HERE/'original_functions.py.txt').read_text();need(sha(HERE/'original_functions.py.txt')==pins['original_functions_sha256'],'Private function source changed')
    runtime_source=dp['bridge'].read_text(encoding='utf8');reuse=json.loads((HERE/'SOURCE_REUSE.json').read_bytes())
    found=set()
    for n in ast.parse(runtime_source).body:
        if isinstance(n,ast.FunctionDef) and n.name in reuse['original_function_sha256']:
            need(hashlib.sha256(ast.get_source_segment(runtime_source,n).encode()).hexdigest()==reuse['original_function_sha256'][n.name],'Original scientific function changed')
            found.add(n.name)
    need(found==set(reuse['original_function_sha256']),'Missing original scientific function')
    original_files={p:sha(p) for p in list(oldpaths.values())+list(members.values())+list(dp.values())+[ap,cp,rp,ip,sealpath]}
    original_files.update({bridge.HERE/n:h for n,h in job['local_hashes'].items()})

    def unchanged():
        for p,h in original_files.items():need(sha(p)==h,'Input changed: '+str(p))
        for n,p in seal['files'].items():need(sha(HERE/n)==p['sha256'],'Adapter source changed')

    def external(reference,entry,_core):
        need(Path(reference)==members['predictions'] and entry==projected and _core is core,'Wrong bound CUDA cache')
        unchanged()
        cache=cache_verifier.load_cache(selected)
        with np.load(members['predictions'],allow_pickle=False) as z:
            need(np.array_equal(cache['root_margins'],z['root_margins']) and np.array_equal(cache['valid_margins'],z['valid_margins']),'CUDA margins changed')
            need(bridge.array_digest(z['root_image_ids'])==row['root_image_ids_sha256']
                 and bridge.array_digest(z['valid_image_ids'])==row['valid_image_ids_sha256'],'CUDA sample identity changed')
        need(cache['root_margins'].dtype==cache['valid_margins'].dtype==np.float32,'Original float32 score conversion required')
        return original_result,cache,None

    ns=dict(bridge.__dict__,verify_external=external,bound_cuda_entries=[projected])
    exec(compile(source,str(HERE/'original_functions.py.txt'),'exec'),ns)
    output=Path(a['output']);failure=output.with_name(output.name+'.formal_failure.json')
    need(output.resolve()==output,'Output path alias/traversal refused')

    def preserve(exc):
        need(not failure.exists(),'Preserve first failure')
        failure.parent.mkdir(parents=True,exist_ok=True)
        with failure.open('x',encoding='utf8') as f:json.dump(dict(error=repr(exc),automatic_retry=False,refit=False,new_CNN=False),f)

    def apply_saved_model():
        need(not output.exists() and not failure.exists(),'Single fresh attempt; preserve old output')
        try:
            fresh();unchanged()
            import netcal
            need(netcal.__version__=='1.3.6','Changed netcal runtime')
            _,cache,_=external(members['predictions'],projected,core)
            with np.load(oldpaths['mapping.npz'],allow_pickle=False) as z:mapping={k:z[k].copy() for k in z.files}
            metadata=json.loads(oldpaths['mapping_metadata.json'].read_bytes())
            ns['validate_mapping'](mapping,metadata,row['root_image_ids_sha256'],row['valid_image_ids_sha256'])
            probability=ns['scores_from_accepted_margins'](cache)
            post=ns['load_post'](oldpaths['post_state.pkl'],row['original_artifacts']['post_state.pkl']['sha256'])
            prediction=post.predict(probability['valid_probability'],cache['valid_sensitive'],mapping['valid_client_id'])
            output.mkdir(parents=True,exist_ok=False)
            for name in ['post_state.pkl','mapping.npz','mapping_metadata.json']:(output/name).write_bytes(oldpaths[name].read_bytes())
            np.savez_compressed(output/'scores_predictions.npz',**probability,**mapping,
                root_sensitive=cache['root_sensitive'],valid_sensitive=cache['valid_sensitive'],root_y=cache['root_y'],valid_y=cache['valid_y'],
                prediction=prediction,valid_native_prediction=(cache['valid_margins']>0).astype(int))
            result=copy.deepcopy(old)
            result.update(accepted_margin_cache_sha256=sha(members['predictions']),
                metrics=core.compute_metrics(cache['valid_y'],prediction,cache['valid_sensitive']),
                score_path='CPU float32 sigmoid of root-adopted FedAvg CUDA margins; original frozen LoGo post_state; no CNN or post refit',
                environment=dict(python=sys.version,torch=torch.__version__,numpy=np.__version__,netcal=netcal.__version__))
            bridge.write_json(output/'result.json',result)
            bridge.write_json(output/'acceptance.json',dict(status='PASS',job_sha256=sha(oldpaths['job']),
                artifact_hashes={n:sha(output/n) for n in ['post_state.pkl','scores_predictions.npz','result.json','mapping.npz','mapping_metadata.json']}))
            ns['checked_output'](oldpaths['job'],output,core,members['predictions'])
            need(sha(output/'post_state.pkl')==row['original_artifacts']['post_state.pkl']['sha256'],'Frozen classifier changed')
            # Historical cross-route difference is diagnostic, never a new acceptance predicate.
            difference={k:result['metrics'][k]-row['historical_CPU_native_metrics'][k] for k in ['accuracy','aeod','aspd']}
            formal=dict(status='FROZEN_LOGO_POST_CUDA_CACHE_WHOLE_CHECK_PASSED_NOT_ROOT_ADOPTED',post_id=row['post_id'],
                baseline_id=row['baseline_id'],approval_sha256=sha(ap),FedAvg_CUDA_root_sha256=sha(rp),
                CUDA_cache_sha256=sha(members['predictions']),frozen_post_state_sha256=sha(output/'post_state.pkl'),
                result_sha256=sha(output/'result.json'),acceptance_sha256=sha(output/'acceptance.json'),
                historical_CPU_difference=difference,historical_CPU_equality_required=False,
                original_saved_consistency_tolerance=1e-12,new_CNN=0,new_fit=0,root_adopted=False,test=False)
            unchanged();bridge.write_json(output/'formal_binding.json',formal);return formal
        except BaseException as exc:preserve(exc);raise

    def accept_saved_body():
        fresh();unchanged();formal=json.loads((output/'formal_binding.json').read_bytes())
        need(formal['approval_sha256']==sha(ap) and formal['FedAvg_CUDA_root_sha256']==sha(rp)
             and formal['CUDA_cache_sha256']==sha(members['predictions'])
             and formal['result_sha256']==sha(output/'result.json') and formal['acceptance_sha256']==sha(output/'acceptance.json')
             and formal['frozen_post_state_sha256']==sha(output/'post_state.pkl')==row['original_artifacts']['post_state.pkl']['sha256'],'Mixed formal saved output')
        result=ns['checked_output'](oldpaths['job'],output,core,members['predictions']);unchanged()
        return dict(status='PER_RECORD_LOGO_FROZEN_POST_CUDA_CACHE_SAVED_ACCEPTED',result=result,
                    historical_CPU_difference=formal['historical_CPU_difference'],root_adopted=False,new_fit=0,new_CNN=0,test=False)

    def accept_saved_predictions():
        need(not failure.exists(),'Preserved failure blocks acceptance')
        try:return accept_saved_body()
        except BaseException as exc:preserve(exc);raise

    return dict(apply_saved_model=apply_saved_model,accept_saved_predictions=accept_saved_predictions,
                original_post_id=row['post_id'],FedAvg_id=row['baseline_id'],refit_API=None)
