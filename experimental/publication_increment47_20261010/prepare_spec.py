"""Exact compact Git47 list after root supplies the final snapshot and helper paths."""
from pathlib import Path
import argparse,json,sys
sys.dont_write_bytecode=True
from publish_increment47 import ROOT,H,PARENT,BRANCH,base,configure
TRAIN='docs/server_deployment_20260923/training_20260923/'
NATIVE=TRAIN+'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T070642Z'
FL='tmp/celeba_flgmm_fullcoverage_delta_after28_20261010'
GATES='tmp/celeba_gradient_fullcoverage_gates_prepare_20261010'
REVIEW='tmp/celeba_gradient_fullcoverage_gates_review_20261010'
SCOPE='tmp/celeba_added_baseline_three_view_scope_20261010'
DIRECTORIES=[NATIVE,FL,GATES,REVIEW,SCOPE,'tmp/publication_increment47_20261010']

MUTABLE=[TRAIN+'TRAINING_STATE.json',TRAIN+'RUNNING.md',TRAIN+'REBUTTAL_COMPLETION_20261009.md',
 TRAIN+'server_reactivation_20261009/MONITOR_HANDOFF.md','docs/返修实验总览.md',
 'tmp/update_reactivation_state_20261009.py','tmp/update_completion_current_20261009.py','tmp/update_overview_closure100_root_20261009.py']
EXPLICIT=['tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json',
 'tmp/publication46_root_20261010/REMOTE_VERIFICATION.json','tmp/publication46_root_20261010/COMMIT_RECEIPT.json']
PINS={NATIVE+'/ROOT_DELTA_VERIFICATION.json':'d6dd77d74ed1a29e69e51cad761145a8bb14840fda44d40764647f1328f60bed',
 FL+'/ROOT_ADOPTION_REVIEW.json':'ae1e65bf764f3ed3e6657e95fa3798617788c8659ac30eb542035a0031663cf3',
 GATES+'/FILES_SHA256.json':'f068ad7f51fd6981b2211725d39009a5cb5ebbecfa1de680cf0d46aa52ef7760',
 REVIEW+'/FILES_SHA256.json':'76c3d10fdb67d45a2d06d96dfbdf6f241faf6f8c271aa5e4222012da0f2f11a4',
 REVIEW+'/REVIEW.json':'47291c77f49777d64a1949ce09fef0e57def822ddecf741be883860f4a8abb68',
 SCOPE+'/FILES_SHA256.json':'1cc32cee3f41abcc644510251a79f5751a60fdf8185533c9f0352ba89c150564'}


def build(extras):
    configure({'native':218,'three_view':212})
    receipt=Path('F:/YananResearchStorage/GuardFed/git_publication/increment46/stage001/STAGE_RECEIPT.json')
    assert H(receipt.read_bytes())=='775aa77d1249e4cd739f528244f2aaf826a17ff42475940214a44619f3a5a85c'
    parent=json.loads((ROOT/EXPLICIT[-2]).read_bytes());assert parent['commit']==PARENT and parent['receipt_sha256']==H(receipt.read_bytes())
    blobs={r['path']:r for r in json.loads(receipt.read_bytes())['blobs']}
    for name,sha in PINS.items():assert H((ROOT/name).read_bytes())==sha,name
    for directory in (GATES,REVIEW,SCOPE):
        seal=ROOT/directory/'FILES_SHA256.json'
        for name,pin in json.loads(seal.read_bytes())['files'].items():
            b=(seal.parent/name).read_bytes();assert H(b)==pin['sha256'] and len(b)==pin['bytes']
    names=set(MUTABLE+EXPLICIT+extras);omitted=[]
    for directory in [ROOT/n for n in DIRECTORIES]:
        for p in directory.rglob('*'):
            if not p.is_file():continue
            n=p.relative_to(ROOT).as_posix()
            if p.suffix.lower() not in {'.json','.py','.md','.csv','.patch','.sha256'} or '__pycache__' in p.parts or p.name=='SERVER_GUIDE.md':
                omitted.append(n);continue
            names.add(n)
    files=[];refs=[]
    for n in sorted(names):
        _,b=base.source(n,[n]);e=dict(source=n,sha256=H(b),bytes=len(b));destination=base.destination(n)
        if destination in blobs and all(blobs[destination][k]==e[k] for k in ('sha256','bytes')) and n not in MUTABLE:
            refs.append(dict(e,destination=destination,parent_commit=PARENT))
        else:files.append(e)
    selected={r['source']:r for r in files}
    paths={'current_state':MUTABLE[0],'native218':NATIVE+'/ROOT_DELTA_VERIFICATION.json',
      'FL32':FL+'/ROOT_ADOPTION_REVIEW.json','gatesreview':REVIEW+'/REVIEW.json'}
    bindings={}
    for role,n in paths.items():
        d=json.loads((ROOT/n).read_bytes());expect=base.REQUIRED_FACTS[role]
        for k,v in expect.items():assert base.pointer(d,k)==v,(role,k)
        bindings[role]=dict(path=n,sha256=selected[n]['sha256'],expect=expect)
    return dict(status='SOURCE_PREPARED_ACTUAL_LIST_NOT_PUBLISHED',parent=PARENT,branch=BRANCH,
      scope='CLOSED218_REPLAY212_FL32_AND_SOURCE_REVIEWS_NO_BULK',accepted={'native':218,'three_view':212},
      bindings=bindings,optional_bindings={},test_started=False,goal_complete=False,
      allowed_paths=sorted(selected),files=files,parent_recovery_references=refs,
      omitted_bulk_or_logs=omitted,large_artifacts_git_restore=False,
      limitation='Only new native6 and FL4 closed evidence; views212 unchanged; gate14 and added-method review source only; LoGo100 root accepted0; bulk excluded')

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--extras',type=Path,required=True,help='Root-reviewed JSON list of exact small source/snapshot/review paths')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert a.out.resolve().is_relative_to(ROOT/'tmp/publication_increment47_root_inputs_20261010')
    extras=json.loads(a.extras.read_bytes());assert isinstance(extras,list) and all(isinstance(n,str) for n in extras)
    d=build(extras);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(a.out),sha256=H(a.out.read_bytes()),files=len(d['files']),bytes=sum(e['bytes'] for e in d['files']))))
