"""Exact compact Git46 list after root supplies the adopted table and final snapshot paths."""
from pathlib import Path
import argparse,json,sys
sys.dont_write_bytecode=True
from publish_increment46 import ROOT,H,PARENT,BRANCH,base,configure
TRAIN='docs/server_deployment_20260923/training_20260923/'
NATIVE=TRAIN+'server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T062622Z'
REPLAY='tmp/celeba_mechanism_remaining620_A12_root_adoption_20261010'
DIRECTORIES=[NATIVE,REPLAY,'tmp/celeba_remaining620_A12_transport_20261010',
 'tmp/celeba_gradient_fullcoverage_prepare_20261010','tmp/publication_increment46_20261010']
MUTABLE=[TRAIN+'TRAINING_STATE.json',TRAIN+'RUNNING.md',TRAIN+'REBUTTAL_COMPLETION_20261009.md',
 TRAIN+'server_reactivation_20261009/MONITOR_HANDOFF.md','docs/返修实验总览.md',
 'tmp/update_reactivation_state_20261009.py','tmp/update_completion_current_20261009.py','tmp/update_overview_closure100_root_20261009.py']
EXPLICIT=['tmp/prepare_mechanism_A12_transport_root_20261010.py','tmp/adopt_mechanism_A12_root_20261010.py',
 'tmp/publication45_root_20261010/REMOTE_VERIFICATION.json','tmp/publication45_root_20261010/COMMIT_RECEIPT.json']

def build(table_root,table_sha,extras):
    configure({'native':212,'three_view':212})
    table_root=Path(table_root).resolve();assert table_root.is_relative_to(ROOT/'docs')
    assert H(table_root.read_bytes())==table_sha
    receipt=Path('F:/YananResearchStorage/GuardFed/git_publication/increment45/stage001/STAGE_RECEIPT.json')
    assert H(receipt.read_bytes())=='7e09337e152a0f16a7b0242a24b26f2e501aa56ed7a949255733ba35de0b5f08'
    parent=json.loads((ROOT/EXPLICIT[-2]).read_bytes());assert parent['commit']==PARENT and parent['receipt_sha256']==H(receipt.read_bytes())
    blobs={r['path']:r for r in json.loads(receipt.read_bytes())['blobs']}
    seal=ROOT/'tmp/celeba_gradient_fullcoverage_prepare_20261010/FILES_SHA256.json'
    assert H(seal.read_bytes())=='ea5a565d35c4f8df7d108679d1010baec92993da8f4b181a14ccde359334ed39'
    for n,pin in json.loads(seal.read_bytes())['files'].items():
        b=(seal.parent/n).read_bytes();assert H(b)==pin['sha256'] and len(b)==pin['bytes']
    names=set(MUTABLE+EXPLICIT+extras);omitted=[]
    for directory in [ROOT/n for n in DIRECTORIES]+[table_root.parent]:
        for p in directory.rglob('*'):
            if not p.is_file():continue
            n=p.relative_to(ROOT).as_posix()
            if p.suffix.lower() not in {'.json','.py','.md','.csv','.patch','.sha256'} or '__pycache__' in p.parts:
                omitted.append(n);continue
            names.add(n)
    files=[];refs=[]
    for n in sorted(names):
        _,b=base.source(n,[n]);e=dict(source=n,sha256=H(b),bytes=len(b));destination=base.destination(n)
        if destination in blobs and all(blobs[destination][k]==e[k] for k in ('sha256','bytes')) and n not in MUTABLE:
            refs.append(dict(e,destination=destination,parent_commit=PARENT))
        else:files.append(e)
    selected={r['source']:r for r in files}
    paths={'current_state':MUTABLE[0],'native212':NATIVE+'/ROOT_DELTA_VERIFICATION.json',
      'replay212':REPLAY+'/ROOT_ADOPTION.json','tableA10':table_root.relative_to(ROOT).as_posix()}
    bindings={}
    for role,n in paths.items():
        d=json.loads((ROOT/n).read_bytes());expect=base.REQUIRED_FACTS[role]
        for k,v in expect.items():assert base.pointer(d,k)==v,(role,k)
        bindings[role]=dict(path=n,sha256=selected[n]['sha256'],expect=expect)
    return dict(status='SOURCE_PREPARED_ACTUAL_LIST_NOT_PUBLISHED',parent=PARENT,branch=BRANCH,
      scope='CLOSED212_A12_AND_ADOPTED_A_TABLE_NO_BULK',accepted={'native':212,'three_view':212},
      bindings=bindings,optional_bindings={},test_started=False,goal_complete=False,
      allowed_paths=sorted(selected),files=files,parent_recovery_references=refs,
      omitted_bulk_or_logs=omitted,large_artifacts_git_restore=False,
      limitation='A one complete scene; F Flip2 partial excluded; gradient fullcoverage source only; LoGo runtime observations are not accepted results')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--table-root',required=True);p.add_argument('--table-sha256',required=True)
    p.add_argument('--extras',type=Path,required=True,help='Root-reviewed JSON list of exact small source/snapshot/review paths')
    p.add_argument('--out',type=Path,required=True);a=p.parse_args()
    assert a.out.resolve().is_relative_to(ROOT/'tmp/publication_increment46_root_inputs_20261010')
    extras=json.loads(a.extras.read_bytes());assert isinstance(extras,list) and all(isinstance(n,str) for n in extras)
    d=build(a.table_root,a.table_sha256,extras);a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('x',encoding='utf8') as f:json.dump(d,f,ensure_ascii=False,indent=2);f.write('\n')
    print(json.dumps(dict(path=str(a.out),sha256=H(a.out.read_bytes()),files=len(d['files']),bytes=sum(e['bytes'] for e in d['files']))))
