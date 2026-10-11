"""Finite FL100 CUDA evaluation: fixed3 source-interface canaries, externally gated remaining97."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json,os,subprocess,sys,time
import guard
BASE=guard.BASE
PY='/workspace/guardfed_envs/celeba-cu128-20261009/bin/python'
read=guard.read;sha=guard.sha;pin=guard.pin;need=guard.need
def save(p,v):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    with p.open('x',encoding='utf-8') as f:json.dump(v,f,indent=2,allow_nan=False);f.write('\n')
def main():
    parser=argparse.ArgumentParser();parser.add_argument('phase',choices=['canary','remaining']);parser.add_argument('--canary-gate-sha256');args=parser.parse_args()
    guard.sources();auth=read(BASE/'ROOT_QUEUE_AUTH.json');scope=read(BASE/'adapter/EXACT100.json')
    need(auth['execution_authorized'] and auth['exact_cohort']==pin(BASE/'adapter/EXACT100.json'),'Unbound cohort')
    need(auth['ordered_ids']==[r['id'] for r in scope['records']] and len(scope['records'])==100,'Wrong exact FL100')
    phase=BASE/args.phase
    need(not phase.exists(),'Preserve prior phase attempts; no automatic retry')
    canaries=auth['canary_ids'];selected=canaries if args.phase=='canary' else [i for i in auth['ordered_ids'] if i not in canaries]
    need(canaries==['FLGMM_Tg20_L2.0_lr0.001_IID_Benign_seed91001_screen', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91001_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_Sp-DFA_seed91001_fullcoverage'],'Only fixed outcome-independent source-schema canaries')
    need(len(selected)==(3 if args.phase=='canary' else 97),'Only fixed3 or ordered remaining97')
    need(len(canaries)==3 and len(set(canaries))==3,'Exactly three fixed FL source-schema canaries')
    if args.phase=='remaining':
        need(isinstance(args.canary_gate_sha256,str) and len(args.canary_gate_sha256)==64
             and sha(BASE/'ROOT_CANARY_ACCEPTANCE.json')==args.canary_gate_sha256,'Externally root-pinned canary gate required')
        gate=read(BASE/'ROOT_CANARY_ACCEPTANCE.json')
        need(gate['root_accepted'] and gate['ids']==canaries and gate['cohort']==auth['exact_cohort'],'Canary gate not root accepted')
        for i in canaries:
            item=gate['records'][i]
            need(item['acceptance']==pin(BASE/'runs'/i/'formal_saved_acceptance.json'),'Canary acceptance changed')
    phase.mkdir();(BASE/'tasks').mkdir(exist_ok=True);(BASE/'runs').mkdir(exist_ok=True)
    completed=[]
    for identity in selected:
        need(time.time()<auth['deadline_unix'],'Root deadline expired')
        task=BASE/'tasks'/identity;task.mkdir(exist_ok=False)
        record=next(r for r in scope['records'] if r['id']==identity)
        out=BASE/'runs'/identity
        try:
            need(not out.exists() and not out.with_name(out.name+'.formal_failure.json').exists(),'No output overwrite or retry')
            pre=subprocess.run(['nice','-n','10','ionice','-c','3','taskset','-c','111',PY,str(BASE/'guard.py'),'--id',identity],capture_output=True,timeout=60)
            save(task/'preflight_command.json',{'exit_code':pre.returncode,'stderr':pre.stderr.decode(errors='replace')})
            need(pre.returncode==0,'Fresh resource preflight rejected')
            fresh=json.loads(pre.stdout);save(task/'preflight.json',fresh)
            a=dict(auth['approval_base'],id=identity,execution_authorized=True,
                   native_index=record['native_index'],identity_canonical_sha256=record['identity_canonical_sha256'],
                   runtime_dependencies=read(BASE/'RUNTIME_DEPENDENCIES.json')[record['identity_source']],
                   ordered_input_ids=auth['ordered_ids'],Linux_preflight=pin(task/'preflight.json'),
                   output=str(out),deadline_unix=min(auth['deadline_unix'],time.time()+3600))
            save(task/'approval.json',a)
            command=['nice','-n','10','ionice','-c','3','taskset','-c','111',PY,str(BASE/'worker.py'),
                     '--approval',str(task/'approval.json'),'--approval-sha256',sha(task/'approval.json'),'--wall-seconds','1800']
            with (task/'worker.stdout').open('xb') as stdout,(task/'worker.stderr').open('xb') as stderr:
                result=subprocess.run(command,stdout=stdout,stderr=stderr,timeout=3700)
            save(task/'worker_command.json',{'command':command,'exit_code':result.returncode})
            need(result.returncode==0,'FL100 worker rejected; preserve evidence and stop')
            acceptance=read(out/'formal_saved_acceptance.json')
            need(acceptance['id']==identity and acceptance['status']=='PER_RECORD_FL100_CUDA_SAVED_WHOLE_ACCEPTED'
                 and acceptance['native_comparison']['accepted'],'Wrong original Linux whole FL acceptance')
            save(task/'ACCEPTED.json',{'id':identity,'receipt':pin(out/'receipt.json'),
                 'binding':pin(out/'formal_binding.json'),'arrays':pin(out/'validation_predictions.npz'),
                 'acceptance':pin(out/'formal_saved_acceptance.json'),'approval':pin(task/'approval.json'),
                 'native_max_abs_difference':acceptance['native_comparison']['max_abs_difference'],'root_offserver_adopted':False})
            completed.append(identity)
            progress={'utc':datetime.now(timezone.utc).isoformat(),'phase':args.phase,'accepted_ids':completed,
                      'planned':len(selected),'failed':0,'remaining':len(selected)-len(completed),'root_offserver_adopted':False}
            tmp=phase/'progress.new';tmp.write_text(json.dumps(progress,indent=2)+'\n',encoding='utf-8');os.replace(tmp,phase/'progress.json')
        except BaseException as exc:
            save(phase/'QUEUE_FAILURE.json',{'id':identity,'error_type':type(exc).__name__,'error':str(exc),
                  'accepted_ids':completed,'automatic_retry':False,'utc':datetime.now(timezone.utc).isoformat()})
            raise
    save(phase/'COMPLETE.json',{'ids':completed,'cohort':auth['exact_cohort'],'server_saved_accepted':True,'root_offserver_adopted':False})
if __name__=='__main__':main()
