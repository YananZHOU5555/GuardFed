"""One explicit infrastructure recovery; preserve57 CUDA receipts and failed attempt.

No science implementation here. Calls the sealed original-science worker in a
new namespace and stops on its first failure. Never adopts diagnostic outputs.
"""
from pathlib import Path
import datetime,fcntl,hashlib,json,os,subprocess,time,traceback
import dispatch as d

def stat_token(path):
    stat = path.stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino, 'device': stat.st_dev,
            'resolved_path': str(path.resolve())}

def verify_group(auth):
    old=Path(auth['prior_base'])
    d.need(d.pin(old/'QUEUE_FAILURE.json')==auth['prior_queue_failure'],'Prior failure changed')
    d.need(d.pin(old/'DEPLOYMENT_SEAL.json')==auth['prior_deployment_seal'],'Prior deployment seal changed')
    seal=d.read(old/'DEPLOYMENT_SEAL.json')
    for name,row in seal['files'].items():d.need(d.sha(old/name)==row['sha256'],'Prior source changed: '+name)
    for path,h in seal['external_code_pins'].items():d.need(d.sha(path)==h,'Prior external source changed')
    cohort=d.read(old/'cohort.json')
    d.need(d.pin(old/'cohort.json')==auth['prior_cohort'],'Prior cohort changed')
    d.need(cohort['records']==d.read(d.BASE/'cohort.json')['records'],'Original500 objects changed')
    failure=d.read(old/'QUEUE_FAILURE.json')
    d.need(failure[auth['completed_field']]==auth['reused_cuda_ids'] and failure['id']==auth['failed_id'],'Prior accepted/failure order changed')
    d.need(len(auth['reused_cuda_ids'])==len(set(auth['reused_cuda_ids']))==auth['expected_count'],'Exact producer accepted count required')
    for row in auth['reused_cuda_receipts']:
        identity=row['id'];task=old/'tasks'/identity
        d.need(identity in auth['reused_cuda_ids'] and d.pin(task/'ACCEPTED.json')==row['accepted'],'Prior ACCEPTED changed')
        a=d.read(task/'ACCEPTED.json');approval=d.read(a['approval']['path'])
        receipt=d.read(a['receipt']['path'])
        d.need(a['id']==identity and a['cohort']==auth['prior_cohort'],'Prior accepted cohort mixed')
        for key in ['approval','receipt','arrays']:
            d.need(d.pin(a[key]['path'])==a[key],'Prior '+key+' changed')
        comparison=a['acceptance']['native_comparison']
        d.need(comparison['accepted'] and comparison['max_abs_difference']<=1e-12,'Prior native strict check failed')
        d.need(receipt['runtime']['device']=='cuda:0' and receipt['runtime']['cuda_device_count']==1,'CPU record cannot be reused')
        d.need(not receipt['optimizer_created'] and not receipt['gradients_created'] and not receipt['test_labels_accessed'],'Prior wrong operations')
        d.need(receipt['prediction_arrays_sha256']==a['arrays']['sha256'],'Prior saved array identity differs')
        record=next(r for r in cohort['records'] if r['id']==identity)
        d.need(receipt['model_inventory_record_sha256']==d.canonical(record),'Prior original inventory identity differs')
        for field,key in [('checkpoint_sha256','checkpoint'),('original_result_sha256','result'),('original_job_sha256','raw_job')]:
            d.need(receipt[field]==record[key]['sha256'],'Prior checkpoint/job/result mixed')
        d.need(approval['native_tolerance']==1e-12 and approval['target_split']=='valid' and not approval['training'] and not approval['test'],'Prior protocol changed')
        d.need(d.pin(row['binding']['path'])==row['binding'],'Prior binding changed')
        binding=d.read(row['binding']['path'])
        d.need(binding['id']==identity and binding['approval_sha256']==a['approval']['sha256'],'Prior binding approval differs')
        if identity.startswith('GuardFed-AD2+_'):
            d.need(a['acceptance']['binding_sha256']==row['binding']['sha256'] and binding['receipt_sha256']==a['receipt']['sha256'],'Full Linux saved binding disconnected')
        else:
            d.need(a['acceptance']['bridge_receipt_sha256']==row['binding']['sha256'] and binding['scientific_body_receipt_sha256']==a['receipt']['sha256'],'Mechanism Linux saved binding disconnected')
        for role in ['source','artifact']:
            before=binding[role+'_before'];after=binding[role+'_after']
            d.need(before==after,'Prior '+role+' changed during science')
            for path,h in before.items():
                current={'sha256':d.sha(path),'resolved_path':str(Path(path).resolve()),'bytes':Path(path).stat().st_size,'stat':stat_token(Path(path))}
                d.need(current==h,'Prior frozen '+role+' changed: '+path)
        d.need(receipt['valid_n']==19867 and receipt['weights_before']==receipt['weights_after'],'Prior sample count or weights changed')
        d.need(a['acceptance']['views']==receipt['views'] and comparison==receipt['native_comparison'],'Prior receipt/saved gate differs')
        d.need((record['terminal_round'],record['original_split'],record['original_n_eval'])==(70,'valid',19867),'Prior terminal/split differs')
    d.need([r['id'] for r in auth['reused_cuda_receipts']]==auth['reused_cuda_ids'],'Wrong prior receipt order')
    return auth['reused_cuda_ids']

def prior_cuda_receipts(auth):
    fixed=d.read(d.BASE/'PINNED_PRIOR.json')
    d.need(len(auth['producer_groups'])==len(fixed['groups'])==2,'Exactly two original producers required')
    reused=[]
    for group,expected in zip(auth['producer_groups'],fixed['groups']):
        for key,value in expected.items():
            if key!='reused_cuda_receipts':d.need(group[key]==value,'Producer metadata differs: '+key)
        d.need(len(group['reused_cuda_receipts'])==len(expected['reused_cuda_receipts']),'Producer receipt count differs')
        for row,ref in zip(group['reused_cuda_receipts'],expected['reused_cuda_receipts']):
            d.need(row['id']==ref['id'] and row['accepted']==ref['accepted'],'Unpinned prior ACCEPTED')
            d.need(row['binding']['path']==ref['binding']['path'],'Prior binding path changed')
            if ref['binding']['sha256'] is not None:d.need(row['binding']==ref['binding'],'Prior binding SHA changed')
            old=Path(group['prior_base']);out=Path(row['binding']['path']).parent
            d.need(not (old/'tasks'/row['id']/'FAILURE.json').exists()
                   and not out.with_name(out.name+'.formal_failure.json').exists()
                   and not out.with_name(out.name+'.bridge_failure.json').exists(),'Prior accepted record now has failure')
        if 'prior_queue_auth' in group:d.need(d.pin(group['prior_queue_auth']['path'])==group['prior_queue_auth'],'Prior queue auth changed')
        reused.extend(verify_group(group))
    d.need(reused==fixed['reused_cuda_ids'] and len(reused)==len(set(reused))==57,'Exactly57 distinct prior CUDA acceptances required')
    return reused

def queue():
    d.sources();auth=d.read(d.BASE/'ROOT_RESUME_AUTH.json');qauth=d.read(d.BASE/'ROOT_QUEUE_AUTH.json')
    d.need(auth['root_explicit_recovery_authorized'] and not auth['automatic_retry'],'No explicit bounded recovery')
    d.need(d.pin(__file__)==auth['resume_queue_source'] and d.pin(d.BASE/'ROOT_QUEUE_AUTH.json')==auth['queue_auth'],'Recovery source/auth changed')
    d.need(auth['recover_once_id']==d.read(d.BASE/'PINNED_PRIOR.json')['recover_once_id'] and auth['one_failed_record_replay_budget']==1 and auth['scientific_implementation_changed'] is False,'Wrong recovery budget')
    d.need(not (d.BASE/'STARTED.json').exists() and not (d.BASE/'QUEUE_FAILURE.json').exists(),'No automatic recovery retry')
    lock=(d.BASE/'dispatch.lock').open('x');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    reused=prior_cuda_receipts(auth);fixed=d.read(d.BASE/'PINNED_PRIOR.json');d.need(d.pin(fixed['resume1_started']['path'])==fixed['resume1_started'],'Prior STARTED changed');old_start=d.read(fixed['resume1_started']['path'])
    order=[i for i in old_start['execution_order'] if i not in reused]
    d.need(order==fixed['remaining_order']==auth['remaining_order'] and len(order)==len(set(order))==443 and order[0]==auth['recover_once_id'],'Remaining frozen order differs')
    d.need(set(order)|set(reused)==set(d.read(d.BASE/'cohort.json')['ordered_ids']),'Incomplete comparison scope')
    d.save(d.BASE/'STARTED.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),pid=os.getpid(),stage='EXPLICIT_INFRASTRUCTURE_RESUME443_PRESERVE_CUDA57',auth=d.pin(d.BASE/'ROOT_RESUME_AUTH.json'),execution_order=order,reused_cuda_ids=reused,automatic_retry=False))
    new=[]
    for identity in order:
        try:
            d.need(time.time()<qauth['deadline_unix'],'Root queue deadline expired')
            prior_cuda_receipts(auth)
            progress=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_saved_accepted_ids=reused+new,accepted_prior_cuda=len(reused),accepted_new_cuda=len(new),active=identity,pending=442-len(new),failed=0,root_offserver_accepted=0)
            temp=d.BASE/'progress.tmp';temp.write_text(json.dumps(progress,indent=2)+'\n');temp.replace(d.BASE/'progress.json')
            argv=['taskset','-c','107','nice','-n','10','ionice','-c','3',d.PY,str(d.BASE/'dispatch.py'),'worker','--id',identity]
            with (d.BASE/'logs'/(identity+'.log')).open('x') as log:p=subprocess.run(argv,cwd=d.REPO,stdout=log,stderr=subprocess.STDOUT,timeout=2400)
            d.need(p.returncode==0,'Worker failed rc='+str(p.returncode))
            a=d.read(d.BASE/'tasks'/identity/'ACCEPTED.json');d.need(a['id']==identity and a['cohort']==d.pin(d.BASE/'cohort.json'),'Wrong saved acceptance')
            new.append(identity)
            progress.update(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),remote_saved_accepted_ids=reused+new,accepted_new_cuda=len(new),active=None,pending=443-len(new))
            temp=d.BASE/'progress.tmp';temp.write_text(json.dumps(progress,indent=2)+'\n');temp.replace(d.BASE/'progress.json')
        except BaseException as exc:
            d.save(d.BASE/'QUEUE_FAILURE.json',dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),id=identity,error=repr(exc),traceback=traceback.format_exc(),completed_prior_cuda=reused,completed_new_cuda=new,automatic_retry=False));raise
    prior_cuda_receipts(auth)
    d.save(d.BASE/'REMOTE_COMPLETE.json',dict(remote_saved_accepted=500,accepted_prior_cuda=57,accepted_new_cuda=443,root_offserver_accepted=0,cohort=d.pin(d.BASE/'cohort.json'),training=False,test=False))

if __name__=='__main__':queue()
