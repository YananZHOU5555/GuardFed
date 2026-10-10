"""Build one exact compact Git45 input list. Reads only; writes a new JSON, never Git."""
from pathlib import Path
import argparse,json,sys
sys.dont_write_bytecode=True
from publish_increment45 import ROOT,H,PARENT,BRANCH,base,configure
TRAIN='docs/server_deployment_20260923/training_20260923/'
NATIVE=TRAIN+'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T060527Z'
GRAD='tmp/celeba_gradient64_delta_after1_20261010'
LOGO='tmp/celeba_logofair100_root_operations_20261010'
STATE=TRAIN+'TRAINING_STATE.json'
MUTABLE=[STATE,TRAIN+'RUNNING.md',TRAIN+'REBUTTAL_COMPLETION_20261009.md',
 TRAIN+'server_reactivation_20261009/MONITOR_HANDOFF.md','docs/返修实验总览.md',
 'tmp/update_reactivation_state_20261009.py','tmp/update_completion_current_20261009.py','tmp/update_overview_closure100_root_20261009.py']
PINNED={
 NATIVE+'/ROOT_DELTA_VERIFICATION.json':'796fa3fc3a7b673c9267696e6b3e122e85fef2a2ec8db0dfbd6fcddaeb18c415',
 GRAD+'/ROOT_ADOPTION_REVIEW.json':'e040d742c082dbc65dbb8b7cc36055cf869b5e75950dd927d24a362471b0ff95',
 LOGO+'/ROOT_STARTUP_REVIEW.json':'1ee8a6b002320e6dae552f816772f298d1ecf15e085a90fbecec775dbece8694'}
DIRECTORIES=[NATIVE,GRAD,LOGO,'tmp/celeba_gradient64_delta_after1_review_20261010',
 'tmp/celeba_logofair100_reader_v2_20261010','tmp/celeba_logofair100_reader_v2_review_20261010',
 'tmp/celeba_logofair100_summary_20261010','tmp/celeba_logofair100_summary_independent_review_20261010',
 'tmp/publication_increment45_20261010']
EXPLICIT=['tmp/adopt_gradient64_after1_root_20261010.py','tmp/review_logofair100_startup_root_20261010.py',
 'tmp/publication44_root_20261010/REMOTE_VERIFICATION.json','tmp/publication44_root_20261010/COMMIT_RECEIPT.json',
 TRAIN+'server_reactivation_20261009/root_live_20261010T061149Z.json',
 'tmp/celeba_gradient_screen64_v2_root_operations_20261010/ATTEMPT2_OBSERVATION_20261010T061148336387Z.json',
 'tmp/celeba_mechanism_remaining620_root_operations_20261010/OBSERVATION_20261010T061147994166Z.json']

def build():
    configure({'native':208,'three_view':200})
    for n,h in PINNED.items(): assert H((ROOT/n).read_bytes())==h,n
    seal_checks={}
    for n,h in [(GRAD+'/DELIVERY_FILES_SHA256.json','833affd3beb8264c139683fc2ab80397b7ec5c7b31a7f70ff1cd4f15fed8f7ba'),
      ('tmp/celeba_logofair100_summary_20261010/FILES_SHA256.json','af28607a600ce4ccb03d2ef88bc76e5b83bf3c0415040344950f2f684716a1a1')]:
        p=ROOT/n;assert H(p.read_bytes())==h
        entries=json.loads(p.read_bytes())['files']
        for rel,pin in entries.items():
            raw=(p.parent/rel).read_bytes();assert H(raw)==pin['sha256'] and len(raw)==pin['bytes']
        seal_checks[n]={'sha256':h,'all_members_checked':len(entries)}
    receipt=Path('F:/YananResearchStorage/GuardFed/git_publication/increment44/stage001/STAGE_RECEIPT.json')
    assert H(receipt.read_bytes())=='b2e01577c479553bc8f6cb03882b17cc8b146737ea42060f267deccadde0a5b1'
    parent=json.loads((ROOT/'tmp/publication44_root_20261010/REMOTE_VERIFICATION.json').read_bytes())
    assert parent['commit']==PARENT and parent['receipt_sha256']==H(receipt.read_bytes())
    blobs={e['path']:e for e in json.loads(receipt.read_bytes())['blobs']}
    names=set(MUTABLE+EXPLICIT);omitted=[]
    for folder in DIRECTORIES:
        for p in (ROOT/folder).rglob('*'):
            if not p.is_file():continue
            n=p.relative_to(ROOT).as_posix()
            if p.suffix.lower() not in {'.json','.py','.md','.patch','.csv','.sha256'} or '__pycache__' in p.parts:
                omitted.append({'source':n,'reason':'bulk/log/derived bytes excluded; original seal/path retained'});continue
            names.add(n)
    # The imported original transport seal is a dependency; unchanged members are parent references.
    old=ROOT/'tmp/publication_increment44_20261010/FILES_SHA256.json'
    names.add(old.relative_to(ROOT).as_posix())
    names.update((old.parent/n).relative_to(ROOT).as_posix() for n in json.loads(old.read_bytes())['files'])
    files=[];refs=[]
    for n in sorted(names):
        _,raw=base.source(n,[n]);e={'source':n,'sha256':H(raw),'bytes':len(raw)};dest=base.destination(n)
        prior=blobs.get(dest)
        if prior and prior['sha256']==e['sha256'] and prior['bytes']==e['bytes'] and n not in MUTABLE:
            refs.append(dict(e,destination=dest,parent_commit=PARENT));continue
        files.append(e)
    selected={e['source']:e for e in files}
    def binding(n,expect):
        e=selected[n];d=json.loads((ROOT/n).read_bytes())
        for k,v in expect.items():assert base.pointer(d,k)==v,(n,k)
        return dict(path=n,sha256=e['sha256'],expect=expect)
    bindings={
      'current_state':binding(STATE,base.REQUIRED_FACTS['current_state']),
      'gradient5':binding(GRAD+'/ROOT_ADOPTION_REVIEW.json',base.REQUIRED_FACTS['gradient5']),
      'LoGo100startup':binding(LOGO+'/ROOT_STARTUP_REVIEW.json',base.REQUIRED_FACTS['LoGo100startup'])}
    return dict(status='SOURCE_PREPARED_ACTUAL_INPUT_LIST_NOT_FROZEN_OR_PUBLISHED',parent=PARENT,branch=BRANCH,
      scope='CLOSED_DELTA_AND_ACTUAL_LOGO100_STARTUP_NO_BULK',accepted={'native':208,'three_view':200},
      test_started=False,goal_complete=False,bindings=bindings,
      optional_bindings={'native_delta':binding(NATIVE+'/ROOT_DELTA_VERIFICATION.json',{'/total_new_strict_and_offserver':208})},
      allowed_paths=sorted(selected),files=files,parent_recovery_references=refs,
      original_seal_member_checks=seal_checks,omitted_local_files=omitted,
      parent_verified_receipt=parent,
      limitations=['LoGo100 started, not fully accepted; summary source only',
       'Git contains compact evidence/source; raw restore requires the F archives named in storage indices',
       'Parent byte references are rechecked by original plan before freeze; no Git operation in this builder'])

if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--output',type=Path,required=True);x=a.parse_args()
    assert x.output.resolve().is_relative_to((ROOT/'tmp/publication_increment45_root_inputs_20261010').resolve())
    d=build();x.output.parent.mkdir(parents=True,exist_ok=True)
    with x.output.open('x',encoding='utf8',newline='\n') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps({'path':str(x.output),'sha256':H(x.output.read_bytes()),'files':len(d['files']),
      'bytes':sum(e['bytes'] for e in d['files']),'parent_references':len(d['parent_recovery_references'])}))
