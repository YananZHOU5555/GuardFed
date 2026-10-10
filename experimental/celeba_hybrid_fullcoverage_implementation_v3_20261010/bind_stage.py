"""Future one-time metadata binding only. Requires actual adopted32; never starts science."""
import argparse,importlib.util,json,shutil,sys
from pathlib import Path
from common import digest,read,write_json
from identity import require,make_job,validate_grid
HERE=Path(__file__).resolve().parent
def bind(args):
    require(not sys.flags.optimize,'Optimized Python refused')
    for p,h in [(args.summary,args.summary_sha256),(args.root32,args.root32_sha256),(args.approval,args.approval_sha256)]:require(digest(p)==h,'Actual external binding SHA mismatch')
    seal=read(HERE/'FILES_SHA256.json')
    for name,row in seal['files'].items():require(digest(HERE/name)==row['sha256'],'Prepared source changed')
    old=args.old_screen.resolve();protocol=read(old/'runtime_protocol.json');scope=read(old/'screen_scope.json')
    pins=read(HERE/'SOURCE_PINS.json')
    for name,h in pins['old_screen_files'].items():require(digest(old/name)==h,'Original screen source drift')
    for name,h in read(old/'FILES_SHA256.json')['files'].items():require(digest(old/name)==h,'Original full source seal drift')
    summary=read(args.summary);root=read(args.root32);approval=read(args.approval)
    require(root['status']=='ROOT_HYBRID32_SUMMARY_ADOPTED' and root['accepted_total']==32 and root['summary_sha256']==args.summary_sha256 and root['all32_offserver_verified'] is True and root['final_test'] is False,'Actual32 summary root adoption required; final partial delta is insufficient')
    require(root['source_seal_sha256']==digest(old/'FILES_SHA256.json'),'Root summary source identity changed')
    require(approval['status']=='ROOT_APPROVED_HYBRID100_METADATA_BINDING' and approval['summary_sha256']==args.summary_sha256 and approval['root32_sha256']==args.root32_sha256 and approval['source_seal_sha256']==digest(HERE/'FILES_SHA256.json'),'Exact independently reviewed metadata binding approval required')
    require(approval['execute_authorized'] is False and approval['final_test'] is False and Path(approval['output']).resolve()==args.output.resolve(),'Binding never authorizes execution')
    require(approval['gpu_index']==0 and approval['allowed_cpus']==[104] and approval['max_workers']==1,'No new resource direction')
    entry_by_id={e['id']:e for e in scope['jobs']}
    for group in summary['all_candidates']:
        for row in group['records']:
            e=entry_by_id[row['id']]
            require((row['candidate'],row['distribution'],row['attack'])==(e['tuning_candidate'],e['distribution'],e['attack']),'Every summary record must retain its original candidate/condition identity')
    # Metadata-only source pins; no worker or torch import here.
    helper=HERE/'metadata_contract.py';spec=importlib.util.spec_from_file_location('hybrid100_metadata',helper);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);m.SCREEN=old
    chosen=m.check_complete_summary(summary,protocol,scope,summary_path=args.summary,root_path=args.root32,root_sha256=args.root32_sha256);require(chosen==approval['selected_recipe'],'Approved winner differs from original frozen ranking')
    require(root['selected_recipe']==chosen,'Root-adopted winner differs from original frozen ranking')
    require(summary['source_seal_sha256']==digest(old/'FILES_SHA256.json'),'Summary source identity changed')
    jobs={e['id']:read(old/e['job']) for e in scope['jobs']};layout=m.coverage_layout(chosen,protocol,scope,jobs)
    candidate=next(c for c in protocol['candidates'] if c['id']==chosen)
    summary_records={r['id']:r for group in summary['all_candidates'] for r in group['records']}
    refs=[]
    for e in layout['reused_references']:
        j=jobs[e['id']];record=summary_records[e['id']];out=old/e['output']
        require(digest(old/e['job'])==e['job_sha256'] and digest(out/'model.pt')==record['checkpoint_sha256'] and digest(out/'acceptance.json')==record['acceptance_sha256'],'Original four artifacts changed')
        result=read(out/'result.json');require(result['config']==j['config'] and result['metrics']==record['metrics'],'Original summary/result/config mismatch')
        accepted=dict(record,rounds=70,job_sha256=e['job_sha256'],source_hashes=j['source_hashes'])
        refs.append(dict(e,legacy_release=str(old),accepted_record=accepted))
    new=[make_job(protocol,candidate,r['distribution'],r['attack'],r['seed'],'fullcoverage') for r in layout['new_grid']]
    gates=[make_job(protocol,candidate,'non-IID',a,91002,'canary',reference) for a,reference in [('Benign',False),('Benign',True),('S-DFA',False),('S-DFA',True),('F Flip',False),('FedSA',False),('Sp-DFA',False)]]
    validate_grid(new,gates,refs)
    output=args.output.resolve();require(not output.exists(),'Existing/failed output preserved; no automatic retry');output.mkdir()
    for name in seal['files']:
        target=output/name;target.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(HERE/name,target)
    shutil.copyfile(HERE/'FILES_SHA256.json',output/'PREPARED_SOURCE_SEAL.json')
    shutil.copyfile(old/'body.py',output/'body.py')
    require(digest(args.fl_runner)==pins['fl_queue_sha256'],'Unchanged FL lifecycle source required');shutil.copyfile(args.fl_runner,output/'fl_queue_original.py')
    for name,p in [('SELECTED_SUMMARY.json',args.summary),('ROOT32_ADOPTION.json',args.root32),('BIND_APPROVAL.json',args.approval),('ORIGINAL_PROTOCOL.json',old/'runtime_protocol.json')]:shutil.copyfile(p,output/name)
    entries=[];ge=[];(output/'jobs').mkdir()
    for job in new+gates:
        path=output/'jobs'/(job['id']+'.json');write_json(path,job)
        row=dict(id=job['id'],job=path.relative_to(output).as_posix(),job_sha256=digest(path),output=('runs/' if job['phase']=='fullcoverage' else 'gate_runs/')+job['id'],tuning_candidate=chosen,distribution=job['distribution'],attack=job['attack'],seed=job['config']['seed'])
        (entries if job['phase']=='fullcoverage' else ge).append(row)
    local={n:digest(output/n) for n in seal['files']};local.update({'body.py':digest(output/'body.py'),'fl_queue_original.py':digest(output/'fl_queue_original.py')})
    for name,items,horizon,stage in [('full_scope.json',entries,70,'validation_fullcoverage'),('gate_scope.json',ge,3,'real_image_cuda_pipeline_gate_only')]:
        write_json(output/name,dict(status='FROZEN_HYBRID100_PENDING_EXECUTION',scope_file=name,kind='cuda_pipeline_gate' if horizon==3 else 'validation_fullcoverage',rounds=horizon,jobs=items,evidence_stage=stage,protected_source_hashes=scope['protected_source_hashes'],local_hashes=local,guide_sha256=scope['guide_sha256'],claim_limit='Exposed valid-only; terminal native argmax, no test or resume guarantee; canaries are not scientific70 records.'))
    write_json(output/'manifest.json',dict(jobs=entries,reused_jobs=refs,preflight_jobs=ge,planned_new=96,planned_total=100))
    write_json(output/'BINDINGS.json',dict(status='BOUND_HYBRID100_NOT_EXECUTION_AUTHORITY',original_screen=str(old),repo=str(args.repo.resolve()),actual_summary32_sha256=args.summary_sha256,actual_root32_sha256=args.root32_sha256,old_screen_seal_sha256=digest(old/'FILES_SHA256.json'),source_seal_sha256=digest(HERE/'FILES_SHA256.json'),selected_recipe=candidate,bind_approval_sha256=args.approval_sha256,gpu_uuid=approval['gpu_uuid'],execution_authorized=False))
    write_json(output/'PACKAGE_SHA256.json',dict(status='FROZEN_PENDING_EXECUTION',files={p.relative_to(output).as_posix():digest(p) for p in output.rglob('*') if p.is_file()}))
    print(json.dumps(dict(status='BOUND_METADATA_ONLY',new=96,reused=4,canaries=7,package_sha256=digest(output/'PACKAGE_SHA256.json'),execution_authorized=False)))
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('old-screen','repo','fl-runner','summary','root32','approval','output'):p.add_argument('--'+name,type=Path,required=True)
    for name in ('summary','root32','approval'):p.add_argument('--'+name+'-sha256',required=True)
    bind(p.parse_args())
