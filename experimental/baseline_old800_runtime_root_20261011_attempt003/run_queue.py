"""Finite original800 CUDA evaluation, canary then exact remaining cohort."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys,time
import guard
from reuse_fedavg import verify_reused
BASE=guard.BASE
PY='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
read=guard.read;sha=guard.sha;pin=guard.pin;need=guard.need
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['canary','remaining']);parser.add_argument('--canary-gate-sha256');args=parser.parse_args()
    guard.sources();auth=read(BASE/'ROOT_QUEUE_AUTH.json');scope=read(BASE/'EXACT800_INPUTS.json')
    need(auth['execution_authorized'] and auth['exact_cohort']==pin(BASE/'EXACT800_INPUTS.json'),'Unbound cohort')
    need(auth['ordered_ids']==scope['ordered_ids'] and len(scope['records'])==800,'Wrong original800')
    phase=BASE/args.phase
    need(not phase.exists(),'Preserve prior phase attempts; no automatic retry')
    canaries=auth['canary_ids'];reused=verify_reused(auth,scope)
    selected=[i for i in canaries if i not in reused] if args.phase=='canary' else [i for i in auth['ordered_ids'] if i not in canaries]
    need(len(canaries)==3 and len(set(canaries))==3,'Exactly three fixed source-schema canaries')
    if args.phase=='remaining':
        need(isinstance(args.canary_gate_sha256,str) and len(args.canary_gate_sha256)==64
             and sha(BASE/'ROOT_CANARY_ACCEPTANCE.json')==args.canary_gate_sha256,'Externally root-pinned canary gate required')
        gate=read(BASE/'ROOT_CANARY_ACCEPTANCE.json')
        need(gate['root_accepted'] and gate['ids']==canaries and gate['cohort']==auth['exact_cohort'],'Canary gate not root accepted')
        for i in canaries:
            item=gate['records'][i]
            expected=reused[i]['acceptance'] if i in reused else pin(BASE/'runs'/i/'formal_saved_acceptance.json')
            need(item['acceptance']==expected,'Canary acceptance changed')
    phase.mkdir();(BASE/'tasks').mkdir(exist_ok=True);(BASE/'runs').mkdir(exist_ok=True)
    completed=list(reused) if args.phase=='canary' else []
    newly_completed=[]
    for identity in selected:
        need(time.time()<auth['deadline_unix'],'Root deadline expired')
        task=BASE/'tasks'/identity;task.mkdir(exist_ok=False)
        record=next(r for r in scope['records'] if r['id']==identity)
        out=BASE/'runs'/identity
        try:
            need(not out.exists() and not out.with_name(out.name+'.formal_failure.json').exists(),'No output overwrite or retry')
            pre=subprocess.run(['nice','-n','10','ionice','-c','3','taskset','-c','109',PY,str(BASE/'guard.py'),'--id',identity],capture_output=True,timeout=60)
            save(task/'preflight_command.json',{'exit_code':pre.returncode,'stderr':pre.stderr.decode(errors='replace')})
            need(pre.returncode==0,'Fresh resource preflight rejected')
            fresh=json.loads(pre.stdout);save(task/'preflight.json',fresh)
            a=dict(auth['approval_base'],id=identity,execution_authorized=True,
                   record_canonical_sha256=hashlib.sha256(json.dumps(record,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest(),
                   ordered_input_ids=auth['ordered_ids'],Linux_preflight=pin(task/'preflight.json'),
                   output=str(out),deadline_unix=min(auth['deadline_unix'],time.time()+3600))
            save(task/'approval.json',a)
            command=['nice','-n','10','ionice','-c','3','taskset','-c','109',PY,str(BASE/'worker.py'),
                     '--approval',str(task/'approval.json'),'--approval-sha256',sha(task/'approval.json'),'--wall-seconds','1800']
            with (task/'worker.stdout').open('xb') as stdout,(task/'worker.stderr').open('xb') as stderr:
                result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=3700)
            save(task/'worker_command.json',{'command':command,'exit_code':result.returncode})
            need(result.returncode==0,'Original800 worker rejected; preserve evidence and stop')
            acceptance=read(out/'formal_saved_acceptance.json')
            need(acceptance['id']==identity and acceptance['status']=='PER_RECORD_ORIGINAL800_CUDA_SAVED_WHOLE_ACCEPTED'
                 and acceptance['native_comparison']['accepted'],'Wrong saved original800 acceptance')
            save(task/'ACCEPTED.json',{'id':identity,'receipt':pin(out/'receipt.json'),
                 'binding':pin(out/'formal_binding.json'),'arrays':pin(out/'validation_predictions.npz'),
                 'acceptance':pin(out/'formal_saved_acceptance.json'),'approval':pin(task/'approval.json'),
                 'native_max_abs_difference':acceptance['native_comparison']['max_abs_difference'],'root_offserver_adopted':False})
            completed.append(identity);newly_completed.append(identity)
            progress={'utc':datetime.now(timezone.utc).isoformat(),'phase':args.phase,'accepted_ids':completed,
                      'planned':len(canaries) if args.phase=='canary' else len(selected),
                      'new_planned':len(selected),'new_accepted_ids':newly_completed,
                      'reused_ids':list(reused) if args.phase=='canary' else [],
                      'failed':0,'remaining':len(selected)-len(newly_completed),'root_offserver_adopted':False}
            tmp=phase/'progress.new';tmp.write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8');os.replace(tmp,phase/'progress.json')
        except BaseException as exc:
            save(phase/'QUEUE_FAILURE.json',{'id':identity,'error_type':type(exc).__name__,'error':str(exc),
                  'accepted_ids':completed,'automatic_retry':False,'utc':datetime.now(timezone.utc).isoformat()})
            raise
    save(phase/'COMPLETE.json',{'ids':completed,'new_ids':newly_completed,
         'reused_ids':list(reused) if args.phase=='canary' else [],
         'cohort':auth['exact_cohort'],'server_saved_accepted':True,'root_offserver_adopted':False})
if __name__=='__main__':main()
