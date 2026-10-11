"""Future Windows offserver audit of exactly three original800 CUDA saved outputs; zero fit/CNN."""
from pathlib import Path,PurePosixPath
import argparse,ast,copy,datetime,hashlib,importlib.util,json,os,sys,tarfile,textwrap,traceback,types,zipfile
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE='/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002'
REPO='/workspace/GuardFed-celeba-expanded'
IDS=['FedAvg_IID_Benign_seed91001', 'FedAA-DDPG_policy0.001_keep16_local0.001_IID_Benign_seed91001', 'LASA_s0.3_l2_lr0.001_IID_Benign_seed91001_screen']
VIEWS=['native','raw','shared_calibration']
AUTHOR_SHA='a1f1be152a0aa5668b91d73d7297ea3e9275efaac8d82c3b22cfbe97b7b08fce'
FORMAL_API_SEAL_SHA='3a2d6300ff521ce92c75cc566045ba3d468bd4d426e5eb5ac8cac8dc42de1b31'
METADATA_SHA='161f8028f1c29ba470afa60cbd9fb54d7bf61b3cec5c525830ad7a3ef7ab2091'
VALID_SHA='64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf'
read=lambda p:json.loads(Path(p).read_bytes())
def need(ok,message):
    if not ok:raise ValueError(message)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
    return h.hexdigest()
def canonical(x):return hashlib.sha256(json.dumps(x,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()
def save(p,v):
    with p.open('x',encoding='utf8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def load_functions(path,names,namespace):
    tree=ast.parse(path.read_text('utf8'));nodes=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in names]
    need({n.name for n in nodes}==set(names),'Missing original science function')
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace
def source_check(expected):
    need(sha(HERE/'FILES_SHA256.json')==expected,'Wrong audit source seal')
    for rel,p in read(HERE/'FILES_SHA256.json')['files'].items():
        q=(HERE/rel).resolve();need(q.is_relative_to(HERE) and sha(q)==p['sha256'] and q.stat().st_size==p['bytes'],'Audit source changed')
    pins=read(HERE/'SOURCE_PINS.json')
    for name in ('original_fl_audit.py.txt','original_evaluator.py.txt','original_replay.py.txt','original_full_saved_science.py.txt'):
        need(sha(HERE/name)==pins[name]['sha256'],'Original source changed')
    return pins
def verified_inputs(a):
    need(sys.platform=='win32' and __debug__,'Windows without -O required')
    pins=source_check(a.source_seal_sha256)
    need(sha(a.inputs)==a.inputs_sha256,'Actual root input SHA differs');inp=read(a.inputs)
    need(inp['root_authorized_audit'] is True and inp['exact_ids']==IDS,'Root audit binding absent or wrong fixed3')
    review=Path(inp['source_review']['path']);need(sha(review)==inp['source_review']['sha256'] and read(review)['source_adoptable'] is True,'Actual root source review missing')
    need(read(review)['source_seal_sha256']==a.source_seal_sha256,'Root source review is for another audit')
    guard=ROOT/pins['storage_guard']['path'];need(sha(guard)==pins['storage_guard']['sha256'],'Storage guard changed')
    spec=importlib.util.spec_from_file_location('_old800_cuda3_storage',guard);g=importlib.util.module_from_spec(spec);spec.loader.exec_module(g)
    volume=g.check_bulk_storage()
    tp=Path(inp['transport']['path']);need(sha(tp)==inp['transport']['sha256'],'Transport proof changed');t=read(tp)
    need(t['status']=='PASS_ARCHIVE_AND_ALL_MEMBER_SHA' and t['all_members_verified'] is True,'Original transport incomplete')
    archive=Path(t['archive_path']).resolve();extract=Path(t['verified_extract']).resolve()
    if 'extract_root' in t:need(Path(t['extract_root']).resolve()==extract,'Conflicting extract path')
    need(all(p.drive.upper()=='F:' and p.is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()) for p in (archive,extract)),'Bulk outside guarded F')
    need(sha(archive)==t['archive_sha256']==inp['archive_sha256'],'Archive changed')
    index=extract/'ROOT_MEMBERS.json';need(sha(index)==t['member_index_sha256'],'Member index changed')
    members=read(index)['files'];need(members==t['members'] and len(members)==t['payload_members'],'Transport/index member disagreement')
    need(t['archive_members']==len(members)+1,'Archive member count mismatch')
    with tarfile.open(archive,'r:gz') as tar:
        mm=tar.getmembers();names=[m.name for m in mm]
        need(len(names)==len(set(names))==len(members)+1 and set(names)==set(members)|{'ROOT_MEMBERS.json'},'Missing/extra/duplicate archive member')
        for member in mm:
            rel=member.name;p=PurePosixPath(rel)
            need(member.isfile() and not p.is_absolute() and '..' not in p.parts and '\\' not in rel and ':' not in rel,'Unsafe archive member')
            local=extract.joinpath(*p.parts);need(local.resolve().is_relative_to(extract) and not local.is_symlink(),'Unsafe extracted file')
            h=hashlib.sha256();f=tar.extractfile(member)
            for b in iter(lambda:f.read(4*1024**2),b''):h.update(b)
            expected=members.get(rel,{'sha256':t['member_index_sha256'],'bytes':index.stat().st_size})
            need(h.hexdigest()==sha(local)==expected['sha256'] and member.size==local.stat().st_size==expected['bytes'],'Archive/extract/member SHA mismatch: '+rel)
            if rel.endswith('.npz'):
                allowed={'source/metadata.npz'}|{'base/runs/'+i+'/validation_predictions.npz' for i in IDS}
                need(rel in allowed,'Foreign prediction array')
            need(not rel.endswith(('.pt','.pth','.npy')),'Model/image bulk not part of fixed3 audit')
    return inp,t,extract,members,volume
def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--inputs',type=Path,required=True)
    for n in ('inputs-sha256','source-seal-sha256'):p.add_argument('--'+n,required=True)
    a=p.parse_args();out=HERE/'actual';out.mkdir(exist_ok=True)
    need(not any((out/n).exists() for n in ('STARTED.json','SAVED_OUTPUTS_AUDIT.json','FAILURE.json')),'Preserve previous attempt; no retry')
    inp,t,x,members,volume=verified_inputs(a)
    save(out/'STARTED.json',dict(inputs_sha256=a.inputs_sha256,source_seal_sha256=a.source_seal_sha256,exact_ids=IDS,fit_calls=0))
    rows=[]
    try:
        def member(rel):
            need(rel in members,'Missing required transport member: '+rel);return x/rel
        def remote_pin(pin):
            original=pin['path']
            need(original.startswith(BASE+'/'),'Unexpected original800 runtime path')
            q=member('base/'+original[len(BASE)+1:]);need(sha(q)==pin['sha256'],'Pinned formal member changed');return q
        deployment=read(member('base/DEPLOYMENT_SEAL.json'))
        need(sha(x/'base/DEPLOYMENT_SEAL.json')==inp['deployment_sha256'],'Wrong actual deployed source version')
        for rel,v in deployment['files'].items():
            if 'base/'+rel in members:need(sha(x/'base'/rel)==v['sha256'],'Deployed source/member differs')
        for rel in ('guard.py','run_queue.py','worker.py','adapter/adapter.py','adapter/gpu_body.py','adapter/old800_saved_science.py.txt','adapter/SOURCE_PINS.json','adapter/FILES_SHA256.json'):
            need(sha(member('base/'+rel))==deployment['files'][rel]['sha256'],'Required scientific source missing')
        need(sha(member('base/adapter/FILES_SHA256.json'))==FORMAL_API_SEAL_SHA,'Wrong formal API seal')
        for rel,v in read(x/'base/adapter/FILES_SHA256.json')['files'].items():
            need(sha(member('base/adapter/'+rel))==v['sha256'],'Formal API source missing or changed')
        selection=read(member('base/COHORT_SELECTION.json'));cohort_sha=selection['full_cohort']['sha256']
        need(selection['scope']=='ORIGINAL800_FIXED3_SAVED_OUTPUT_SELECTION_V1' and selection['exact_ids']==IDS,'Wrong fixed selection')
        need(selection['full_cohort']=={'path':BASE+'/EXACT800_INPUTS.json','sha256':inp['cohort_sha256']}
             and selection['original_count']==800 and selection['Full_excluded_and_reused_separately'] is True
             and selection['selection_basis']=='all_original900_records_except_GuardFed-AD2+'
             and selection['outcome_independent'] is True,'Wrong original800 lineage')
        need([r['id'] for r in selection['records']]==IDS and {r['method'] for r in selection['records']}=={'FedAvg','FedAA','LASA'},'Wrong three original records')
        author=read(member('base/author.json'));need(sha(x/'base/author.json')==AUTHOR_SHA and author['approved']['uniform_CUDA_evaluation'] and author['approved']['corresponding_Full_included'] and not author['existing_diagnostic_retroactively_adopted'],'Author scope differs')
        qauth=read(member('base/ROOT_QUEUE_AUTH.json'));complete=read(member('base/canary/COMPLETE.json'))
        need(qauth['execution_authorized'] is True and qauth['canary_ids']==IDS
             and qauth['exact_cohort']==selection['full_cohort']
             and canonical(qauth['ordered_ids'])==selection['ordered_full_ids_sha256'],'Wrong actual authorized cohort')
        need(sha(x/'base/canary/COMPLETE.json')==inp['canary_complete_sha256'] and complete['ids']==IDS
             and complete['cohort']==selection['full_cohort'] and complete['server_saved_accepted'] is True
             and complete['root_offserver_adopted'] is False,'All three original Linux saved whole checks required')
        need(set(inp['accepted_sha256'])==set(IDS),'Exactly three actual ACCEPTED pins required')
        metadata=member('source/metadata.npz');need(sha(metadata)==METADATA_SHA,'Original metadata SHA differs')
        os.environ.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
        import numpy as np
        evns=load_functions(HERE/'original_evaluator.py.txt',['canonical_sha','check_binary','group_metrics','_seal_fit','predict_views','evaluate_frozen_predictions'],dict(np=np,hashlib=hashlib,json=json,VIEWS=set(VIEWS),METHODS={'FedAvg','FedAA','LASA'}))
        ev=types.SimpleNamespace(**evns)
        vns=load_functions(HERE/'original_replay.py.txt',['array_sha','read_prefix','check_native'],dict(np=np,hashlib=hashlib,zipfile=zipfile,require=need,TOLERANCE=1e-12))
        v2=types.SimpleNamespace(np=np,require=need,digest=sha,VIEWS=VIEWS,check_native=vns['check_native'])
        with np.load(metadata,allow_pickle=False) as z:ids,split=z['image_id'],z['split']
        need(np.array_equal(ids,np.arange(1,202600)) and np.array_equal(np.flatnonzero(split==1),np.arange(162770,182637)),'Original metadata order/split differs')
        y,_=vns['read_prefix'](metadata,'Smiling',202599,182637);s,_=vns['read_prefix'](metadata,'Male',202599,182637)
        fn=load_functions(HERE/'original_fl_audit.py.txt',['saved_output_block'],dict(ast=ast,copy=copy,need=need))['saved_output_block']
        original=ast.parse(textwrap.dedent((HERE/'original_full_saved_science.py.txt').read_text('utf8'))).body[0]
        block=fn(original)
        from saved_fit_binding import saved_fits_for_original_predict
        for identity in IDS:
            record=next(r for r in selection['records'] if r['id']==identity)
            task='base/tasks/'+identity;run='base/runs/'+identity
            accepted_path=member(task+'/ACCEPTED.json');accepted=read(accepted_path)
            need(sha(accepted_path)==inp['accepted_sha256'][identity] and accepted['id']==identity and accepted['root_offserver_adopted'] is False,'Wrong actual Linux acceptance')
            approval_path=remote_pin(accepted['approval']);approval=read(approval_path)
            need(approval_path==member(task+'/approval.json'),'Wrong approval task')
            receipt_path=remote_pin(accepted['receipt']);array_path=remote_pin(accepted['arrays']);r=read(receipt_path)
            acceptance=read(remote_pin(accepted['acceptance']));binding=read(remote_pin(accepted['binding']))
            need(receipt_path==member(run+'/receipt.json') and array_path==member(run+'/validation_predictions.npz')
                 and remote_pin(accepted['binding'])==member(run+'/formal_binding.json')
                 and remote_pin(accepted['acceptance'])==member(run+'/formal_saved_acceptance.json'),'Wrong saved paths')
            need(approval['status']=='ROOT_AUTHORIZED_UNIFORM_CUDA_ORIGINAL800_FORMAL_V1' and approval['execution_authorized'] is True
                 and approval['author_decision_sha256']==AUTHOR_SHA and approval['decision']=='B'
                 and approval['record_canonical_sha256']==canonical(record),'Wrong approved original record')
            need(approval['exact_cohort']==selection['full_cohort'] and canonical(approval['ordered_input_ids'])==selection['ordered_full_ids_sha256']
                 and approval['native_tolerance']==1e-12 and approval['allowed_cpus']==[109] and approval['gpu_uuid']==qauth['gpu_uuid'],'Wrong scope/tolerance/resource identity')
            need(approval['target_split']=='valid' and approval['device']=='cuda:0' and approval['compute_threads']==approval['max_processes']==1
                 and approval['CNN_dtype']=='float32' and approval['postprocessing']=='ORIGINAL_FROZEN_CPU_NUMPY'
                 and all(approval[k] is False for k in ('training','test','automatic_retry','diagnostic_reuse')),'Changed protocol')
            need(approval['resource_gate_source']=={'path':BASE+'/guard.py','sha256':deployment['files']['guard.py']['sha256']},'Wrong actual caller source')
            pre=read(remote_pin(approval['Linux_preflight']));need(pre['pass'] is True and pre['gpu_uuid']==approval['gpu_uuid']
                 and pre['allowed_cpus']==[109] and pre['adapter_sha256']==approval['adapter_sha256']
                 and pre['input_scope_sha256']==cohort_sha,'Preflight/source mismatch')
            need(r['id']==identity and r['status']=='NATIVE_VALID_REPLAY_PASS' and r['scope']=='FORMAL_ORIGINAL800_CUDA_PER_RECORD_V1','Not a formal original800 CUDA receipt')
            need(r['model_inventory_record_sha256']==canonical(record) and r['config_canonical_sha256']==record['config_canonical_sha256']==canonical(record['config']),'Mixed frozen config')
            need(record['terminal_round']==70 and record['original_split']=='valid' and record['original_n_eval']==19867
                 and record['data_contract']['actual_train_rows']==162770 and record['method'] in {'FedAvg','FedAA','LASA'},'Wrong terminal/model interface')
            need(all(r[k]==record[k] for k in ('method','distribution','attack','seed')) and r['original_training_torch']==record['training_torch'],'Relabeled model/runtime')
            for k,f in [('checkpoint_sha256','checkpoint'),('original_result_sha256','result'),('original_job_sha256','raw_job')]:need(r[k]==record[f]['sha256'],'Mixed terminal artifacts')
            need(r['valid_n']==19867 and r['valid_image_ids_sha256']==VALID_SHA and r['prediction_arrays_sha256']==sha(array_path),'Wrong valid/array identity')
            need(r['runtime']['device']=='cuda:0' and r['runtime']['torch']=='2.11.0+cu128' and r['runtime']['cuda_device_count']==1 and r['runtime']['torch_threads']==r['runtime']['interop_threads']==1,'Wrong actual CUDA runtime')
            need(r['weights_before']==r['weights_after'] and all(r[k] is False for k in ('optimizer_created','gradients_created','test_labels_accessed','test_inference_performed')),'Inference-only/weight contract differs')
            need(binding['scope']=='UNIFORM_CUDA_ORIGINAL800_FORMAL_V1' and binding['id']==identity
                 and binding['approval_sha256']==sha(approval_path) and binding['receipt_sha256']==sha(receipt_path)
                 and binding['input_scope_sha256']==cohort_sha and binding['source_before']==binding['source_after']
                 and binding['artifact_before']==binding['artifact_after'],'Linux whole identity changed')
            need(all(record[k]['sha256'] in {v['sha256'] for v in binding['artifact_before'].values()} for k in ('checkpoint','result','raw_job')),'Linux whole did not bind terminal artifacts')
            for originalpath,meta in binding['source_before'].items():
                if originalpath.startswith(BASE+'/') and 'base/'+originalpath[len(BASE)+1:] in members:need(sha(x/'base'/originalpath[len(BASE)+1:])==meta['sha256'],'Whole proof/source member differs')
            need(approval['receipt_version']=='UNIFORM_CUDA_ORIGINAL800_FORMAL_V1'
                 and approval['adapter_sha256']==deployment['files']['adapter/adapter.py']['sha256']
                 and sha(remote_pin(approval['source_seal']))==FORMAL_API_SEAL_SHA
                 and sha(remote_pin(approval['author_decision']))==AUTHOR_SHA
                 and sha(remote_pin(approval['formal_GPU_body']))==deployment['files']['adapter/gpu_body.py']['sha256'],'Wrong original800 approval/source')
            need(acceptance['status']=='PER_RECORD_ORIGINAL800_CUDA_SAVED_WHOLE_ACCEPTED'
                 and acceptance['scope']==binding['scope'] and acceptance['binding_sha256']==sha(x/run/'formal_binding.json')
                 and acceptance['id']==identity and acceptance['views']==r['views']
                 and acceptance['native_comparison']==r['native_comparison'] and r['native_comparison']['accepted'] is True,'Linux saved whole/receipt differs')
            fits=saved_fits_for_original_predict(r['fits'],ev,record['method'])
            with np.load(array_path,allow_pickle=False) as z:
                need(set(z.files)=={'root_image_ids','valid_image_ids','root_margins','valid_margins'}|{'prediction_'+v for v in VIEWS},'Foreign/incomplete saved array keys')
                root_ids=z['root_image_ids'].copy()
                need(root_ids.ndim==1 and len(root_ids)==r['root_reconstruction']['root_n'] and len(np.unique(root_ids))==len(root_ids) and np.issubdtype(root_ids.dtype,np.integer) and ((root_ids>=1)&(root_ids<=162770)).all(),'Wrong saved root IDs')
                need(vns['array_sha'](root_ids)==record['data_contract']['root_image_ids_sha256']==r['root_reconstruction']['root_image_ids_sha256'],'Saved root ID identity differs')
                need(z['root_margins'].shape==root_ids.shape and np.isfinite(z['root_margins']).all(),'Invalid saved root margins')
            context=dict(v2=v2,path=x/run,r=r,root_ids=root_ids,ids=ids,s=s,y=y,evaluator=ev,fits=fits,original_result={'metrics':record['prior_validation_metrics']})
            exec(compile(ast.Module(body=[block],type_ignores=[]),'<same seven original saved-output statements; fixed3; no fit>','exec'),context)
            rows.append(dict(id=identity,receipt_sha256=sha(receipt_path),arrays_sha256=sha(array_path),accepted_sha256=sha(accepted_path),checkpoint_sha256=r['checkpoint_sha256'],views=context['scored'],native_comparison=context['comparison'],original_Linux_whole_saved_accepted=True,saved_fit_hashes={k:v['fit_sha256'] for k,v in r['fits'].items()},fit_calls=0))
        for rel,pin in members.items():need(sha(x/rel)==pin['sha256'],'Member changed during audit')
        source_check(a.source_seal_sha256)
        report=dict(status='OLD800_FIXED3_SAVED_OUTPUTS_AUDIT_NO_REFIT_PASS_NOT_ROOT_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),exact_ids=IDS,records=rows,transport_sha256=inp['transport']['sha256'],archive_sha256=inp['archive_sha256'],cohort_sha256=cohort_sha,source_seal_sha256=a.source_seal_sha256,inputs_sha256=a.inputs_sha256,fit_calls=0,new_CNN=0,new_training=0,test=False,root_adopted=False,uniform800_complete=False,Windows_whole_refit_executed=False,Windows_exact_recalibration_claimed=False,old_CPU_failure_preserved=True,metric_values_checked=27,integer_base_counts_checked=72,prediction_rules_checked=9,F_volume=volume,scope='Fixed3 only; original Linux whole plus SHA transport and saved-output consistency. No fit/model/CNN/test/cohort completion.')
        save(out/'SAVED_OUTPUTS_AUDIT.json',report);print(json.dumps(dict(status=report['status'],records=3,sha256=sha(out/'SAVED_OUTPUTS_AUDIT.json'))))
    except BaseException:
        save(out/'FAILURE.json',dict(status='OLD800_FIXED3_SAVED_AUDIT_FAILED_PRESERVED',completed=len(rows),traceback=traceback.format_exc(),root_adopted=False,automatic_retry=False));raise
if __name__=='__main__':main()
