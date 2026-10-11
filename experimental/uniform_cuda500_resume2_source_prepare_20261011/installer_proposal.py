"""Root-only Linux installer proposal. Requires actual external review and authorization.
No SSH transport. Never executed during source preparation.
"""
from pathlib import Path
import argparse,copy,datetime,hashlib,json,os,subprocess,sys,time

def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()
def read(p):return json.loads(Path(p).read_bytes())
def pin(p):return {'path':str(p),'sha256':sha(p)}
def put(p,v):
    with Path(p).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
def need(v,m):
    if not v:raise ValueError(m)
def check_pin(p):
    need(set(p)=={'path','sha256'} and Path(p['path']).is_absolute() and sha(p['path'])==p['sha256'],'Unbound actual pin')
    return Path(p['path'])
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--root-inputs',type=Path,required=True);ap.add_argument('--root-inputs-sha256',required=True);ap.add_argument('--source-dir',type=Path,required=True);a=ap.parse_args()
    need(sys.platform=='linux' and __debug__,'Actual strict Linux only')
    need(sha(a.root_inputs)==a.root_inputs_sha256,'Actual root authorization SHA required')
    root=read(a.root_inputs);need(root['execution_authorized'] is True and root['recovery_once'] is True and root['expected_prior']==57 and root['expected_remaining']==443,'No explicit once-only root443 recovery')
    review=read(check_pin(root['independent_review']));need(review['status'].startswith('PASS_') and review['blocking_findings']==[],'Independent source review not passed')
    src=a.source_dir.resolve();need(src.is_dir() and not src.is_symlink(),'Actual source staging required')
    prepared=read(src/'PREPARED_SOURCES.json')
    for name,item in prepared['sources'].items():
        need(Path(name).name==name and sha(src/name)==item['sha256'] and (src/name).stat().st_size==item['bytes'],'Prepared source changed')
        need(review['sources'][name]['sha256']==item['sha256'],'Source outside independent review')
    need(sha(__file__)==review['installer_sha256'],'Installer differs from independent review')
    guide=Path('/etc/vast-agents-guide.md');guide.read_text();need(sha(guide)=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa','Guide changed')
    fixed=read(src/'PINNED_PRIOR.json')
    O=Path(fixed['groups'][0]['prior_base']);P=Path(fixed['groups'][1]['prior_base'])
    B=Path('/workspace/guardfed_checks/uniform_cuda_formal_20261011_resume2');M=Path('/workspace/guardfed_checks/mechanism_cuda_formal_20261011_resume2')
    service='guardfed_uniform_cuda500_resume2_valid';conf=Path('/etc/supervisor/conf.d/'+service+'.conf')
    need(not B.exists() and not M.exists() and not conf.exists(),'Preserve existing namespaces/service; no retry')
    for old_service in ['guardfed_uniform_cuda500_valid','guardfed_uniform_cuda500_resume1_valid']:
        p=subprocess.run(['supervisorctl','status',old_service],capture_output=True,text=True,timeout=20)
        need(p.stdout.split()[:2]==[old_service,'EXITED'],'Prior producer still live')
    groups=copy.deepcopy(fixed['groups'])
    for group in groups:
        old=Path(group['prior_base'])
        for field in ['prior_deployment_seal','prior_cohort','prior_queue_failure']:
            check_pin(group[field])
        seal=read(group['prior_deployment_seal']['path'])
        for rel,item in seal['files'].items():need(sha(old/rel)==item['sha256'] and (old/rel).stat().st_size==item['bytes'],'Prior sealed source changed')
        for path,h in seal['external_code_pins'].items():need(sha(path)==h,'Prior external source changed')
        failure=read(group['prior_queue_failure']['path']);need(failure[group['completed_field']]==group['reused_cuda_ids'] and failure['id']==group['failed_id'],'Prior accepted order changed')
        for row in group['reused_cuda_receipts']:
            accepted=read(check_pin(row['accepted']))
            field='binding_sha256' if row['id'].startswith('GuardFed-AD2+_') else 'bridge_receipt_sha256'
            actual={'path':row['binding']['path'],'sha256':accepted['acceptance'][field]}
            if row['binding']['sha256'] is not None:need(actual==row['binding'],'Original binding pin differs')
            check_pin(actual);row['binding']=actual
    check_pin(fixed['old_resume_auth']);check_pin(fixed['resume1_started'])
    for item in fixed['preserved_partial_pins'].values():check_pin(item)
    failed=fixed['recover_once_id'];need(not (P/'tasks'/failed/'ACCEPTED.json').exists(),'Failed partial unexpectedly has task acceptance')
    priorauth=read(check_pin(groups[1]['prior_queue_auth']));need(time.time()<priorauth['deadline_unix'],'Inherited root deadline expired')
    oldcohort=read(P/'cohort.json');need(len(oldcohort['records'])==500 and oldcohort['closed_skip_ids']==[],'Exact original500 required')
    oldseal=read(P/'DEPLOYMENT_SEAL.json')
    B.mkdir();M.mkdir();(B/'logs').mkdir();(B/'runs').mkdir();(M/'runs').mkdir()
    for rel,item in oldseal['files'].items():
        if rel in ['dispatch.py','resume_queue.py']:continue
        dest=B/rel;dest.parent.mkdir(parents=True,exist_ok=True)
        with dest.open('xb') as f:f.write((P/rel).read_bytes())
    for source,dest in [('dispatch_resume2.py','dispatch.py'),('resume_queue.py','resume_queue.py'),('PINNED_PRIOR.json','PINNED_PRIOR.json')]:
        with (B/dest).open('xb') as f:f.write((src/source).read_bytes())
    put(B/'ROOT_SOURCE_REVIEW.json',review)
    seal=copy.deepcopy(oldseal)
    for name in ['dispatch.py','resume_queue.py','PINNED_PRIOR.json','ROOT_SOURCE_REVIEW.json']:
        seal['files'][name]={'sha256':sha(B/name),'bytes':(B/name).stat().st_size}
    put(B/'DEPLOYMENT_SEAL.json',seal)
    cohort=copy.deepcopy(oldcohort);cohort['source_seal']=pin(B/'DEPLOYMENT_SEAL.json');put(B/'cohort.json',cohort)
    need(cohort['records']==oldcohort['records'] and cohort['closed_skip_ids']==[],'Changed500 scientific scope')
    with (B/'formal_author.json').open('xb') as f:f.write((P/'formal_author.json').read_bytes())
    ns={'__file__':str(B/'dispatch.py'),'__name__':'resume2_metadata_preflight'};exec(compile((B/'dispatch.py').read_bytes(),str(B/'dispatch.py'),'exec'),ns)
    prior_affinity=os.sched_getaffinity(0);os.sched_setaffinity(0,[107]);os.setpriority(os.PRIO_PROCESS,0,10);subprocess.run(['ionice','-c','3','-p',str(os.getpid())],check=True)
    policies=dict(priorauth['protected_services'],guardfed_uniform_cuda500_valid=['EXITED'],guardfed_uniform_cuda500_resume1_valid=['EXITED'])
    ns['ID']=failed
    fresh=ns['resource_gate'](None,{'gpu_uuid':priorauth['gpu_uuid'],'protected_services':policies});put(B/'ROOT_FRESH_PREFLIGHT.json',fresh)
    qauth=dict(priorauth,cohort=pin(B/'cohort.json'),deployment_seal=pin(B/'DEPLOYMENT_SEAL.json'),root_source_review_sha256=sha(B/'ROOT_SOURCE_REVIEW.json'),fresh_preflight_sha256=sha(B/'ROOT_FRESH_PREFLIGHT.json'),protected_services=policies,prior_root_auth=groups[1]['prior_queue_auth'],explicit_infrastructure_resume=True)
    put(B/'ROOT_QUEUE_AUTH.json',qauth)
    auth=dict(status='ROOT_EXPLICITLY_AUTHORIZED_INFRASTRUCTURE_RESUME443_ONLY',root_explicit_recovery_authorized=True,automatic_retry=False,producer_groups=groups,recover_once_id=failed,one_failed_record_replay_budget=1,remaining_order=fixed['remaining_order'],resume_queue_source=pin(B/'resume_queue.py'),queue_auth=pin(B/'ROOT_QUEUE_AUTH.json'),source_review_sha256=root['independent_review']['sha256'],scientific_implementation_changed=False,parameters_changed=False,tolerance_changed=False,training=False,test=False,failed_partial_not_adopted=True,old_outputs_failures_all_preserved=True)
    put(B/'ROOT_RESUME_AUTH.json',auth)
    # Avoid any duplicated scientific bootstrap: this imports only stdlib queue/dispatch code.
    sys.path.insert(0,str(B));import resume_queue
    need(resume_queue.prior_cuda_receipts(auth)==fixed['reused_cuda_ids'],'Original57 reuse gate failed')
    put(B/'ROOT_PRIOR57_IDENTITY_GATE.json',{'status':'PASS57_PRIOR_CUDA_TASK_ACCEPTANCES_AND_CURRENT_IDENTITIES','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'reused_ids':fixed['reused_cuda_ids'],'new_science':False,'root_offserver_adopted':False})
    os.sched_setaffinity(0,prior_affinity)
    with conf.open('x',encoding='utf8') as f:f.write('[program:'+service+']\ncommand=/workspace/guardfed_envs/celeba-cu128-20261009/bin/python '+str(B/'resume_queue.py')+'\ndirectory=/workspace/GuardFed-celeba-expanded\nautostart=false\nautorestart=false\nstartsecs=1\nstartretries=0\nstopasgroup=true\nkillasgroup=true\nstdout_logfile='+str(B/'queue.log')+'\nstderr_logfile='+str(B/'queue.stderr.log')+'\nenvironment=PYTHONDONTWRITEBYTECODE="1",OMP_NUM_THREADS="1",MKL_NUM_THREADS="1",OPENBLAS_NUM_THREADS="1",NUMEXPR_NUM_THREADS="1"\n')
    for cmd in [['supervisorctl','reread'],['supervisorctl','update',service],['supervisorctl','start',service]]:
        p=subprocess.run(cmd,capture_output=True,text=True,timeout=30);need(p.returncode==0,(cmd,p.stdout,p.stderr))
    report={'status':'STARTED443_AFTER57_STRICT_REUSE_NOT_ROOT_ADOPTION','utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'cohort':pin(B/'cohort.json'),'auth':pin(B/'ROOT_RESUME_AUTH.json'),'source_seal':pin(B/'DEPLOYMENT_SEAL.json'),'reuse_gate':pin(B/'ROOT_PRIOR57_IDENTITY_GATE.json'),'reused':57,'remaining':443,'root_offserver_adopted':False}
    put(B/'ROOT_START.json',report);print(json.dumps(report))
if __name__=='__main__':main()
