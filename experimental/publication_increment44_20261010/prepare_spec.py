"""Read actual closed evidence, write a draft only; current-state/startup pins remain unset."""
from pathlib import Path
import hashlib,json
import publish_increment44 as p
H=lambda b:hashlib.sha256(b).hexdigest()
T='docs/server_deployment_20260923/training_20260923'
R='docs/server_deployment_20260923/revision_20260923'
N=T+'/server_reactivation_20261009/mechanism_science_backups_20261009/root_delta_20261010T045743Z'
C=T+'/celeba_mechanism_v1/three_view_C_full100_20261010'
selected=set(); omitted=[]; seals=[]

def add(name):
    f=p.ROOT/name
    assert f.is_file(),name
    p.relative(name)
    selected.add(name)

def tree(name):
    for f in sorted((p.ROOT/name).rglob('*')):
        if f.is_file() and '__pycache__' not in f.parts:
            add(f.relative_to(p.ROOT).as_posix())

def sealed(name,seal='FILES_SHA256.json',excluded=()):
    q=p.ROOT/name/seal;d=json.loads(q.read_bytes())
    members=d['files'];assert isinstance(members,dict)
    for rel,info in members.items():
        f=q.parent/rel;b=f.read_bytes();expected=info if isinstance(info,str) else info['sha256']
        assert H(b)==expected,(name,rel)
        if isinstance(info,dict):assert len(b)==info.get('bytes',info.get('size'))
        if rel in excluded:
            omitted.append(dict(source=name+'/'+rel,sha256=expected,reason='Runtime console/guide bytes remain in original delivery; not needed for compact scientific recovery'))
        else:add(name+'/'+rel)
    add(name+'/'+seal)
    seals.append(dict(path=name+'/'+seal,sha256=H(q.read_bytes()),all_local_members_verified=len(members)))

def binding(name,expect):
    add(name);b=(p.ROOT/name).read_bytes();d=json.loads(b)
    for k,v in expect.items():assert p.pointer(d,k)==v,(name,k)
    return dict(path=name,sha256=H(b),expect=expect)

if __name__=='__main__':
    for name in ['ROOT_DELTA_VERIFICATION.json','ARCHIVE_LOCATION.json','OFFSERVER_VERIFICATION.json',
        'root_delta_20261010T045743Z.tar.gz.receipt.json','verified_ledger.json','inspection/inspection.json']:
        add(N+'/'+name)
    for name in ['MECHANISM1_ROOT_ADOPTION.json','MECHANISM181_INDEX.json']:
        add('tmp/root_adopt_first_closed_20261010/'+name)
    tree('tmp/celeba_mechanism_remaining620_C100_root_adoption_20261010')
    x='tmp/celeba_remaining620_C19_transport_20261010'
    seal=json.loads((p.ROOT/x/'FILES_SHA256.json').read_bytes())
    excluded=[n for n in seal['files'] if n.endswith(('.stderr.txt','.stdout.txt')) or n=='GUIDE.stdout.md']
    sealed(x,excluded=excluded)
    tree('tmp/celeba_logofair32_root_adoption_20261010')
    for name in ['tmp/celeba_logofair_fullcoverage_20261010','tmp/celeba_logofair100_independent_review_20261010',
        'tmp/celeba_logofair100_root_operations_20261010','tmp/celeba_logofair100_input_staging_20261010']:
        sealed(name)
    add('tmp/celeba_logofair100_input_staging_20261010/STORAGE_WORDING_CLARIFICATION.json')
    sealed(C,'ACTUAL_FILES_SHA256.json');add(C+'/ROOT_VERIFICATION.json')
    sealed(R+'/rebuttal_integrated_C100_20261010');add(R+'/rebuttal_integrated_C100_20261010/ROOT_REVIEW.json')
    for name in ['tmp/adopt_C100_table_root_20261010.py','tmp/adopt_rebuttal_C100_root_20261010.py',
        'tmp/publication43_root_20261010/REMOTE_VERIFICATION.json',T+'/LOCAL_STORAGE_20261010.json',
        T+'/EXTERNAL_GIT_STORAGE_20261010.json']:
        add(name)
    bindings={
        'native200':binding(N+'/ROOT_DELTA_VERIFICATION.json',p.REQUIRED_FACTS['native200']),
        'replay200':binding('tmp/celeba_mechanism_remaining620_C100_root_adoption_20261010/ROOT_ADOPTION.json',p.REQUIRED_FACTS['replay200']),
        'tableC100':binding(C+'/ROOT_VERIFICATION.json',p.REQUIRED_FACTS['tableC100']),
        'rebuttalC100':binding(R+'/rebuttal_integrated_C100_20261010/ROOT_REVIEW.json',p.REQUIRED_FACTS['rebuttalC100']),
        'logofair32':binding('tmp/celeba_logofair32_root_adoption_20261010/ROOT_ADOPTION.json',p.REQUIRED_FACTS['logofair32']),
        'current_state':None}
    for name in ['publish_increment44.py','verify_increment44.py','prepare_spec.py','README.md',
        'LOGOFAIR_SUMMARY32.json','LOGOFAIR_STRICT32_INDEX.json']:
        add('tmp/publication_increment44_20261010/'+name)
    receipt=Path('F:/YananResearchStorage/GuardFed/git_publication/increment43/stage001/STAGE_RECEIPT.json')
    b=receipt.read_bytes();assert H(b)=='26747554a50e87bb6d2327c50aa9427b814154d527a7264dd0d2018aa2748631'
    prior={e['path']:e for e in json.loads(b)['blobs']}
    refs=[];files=[]
    for name in sorted(selected):
        _,b=p.source(name,[name]);sha=H(b);dest=p.destination(name)
        if dest in prior and prior[dest]['sha256']==sha and prior[dest]['bytes']==len(b):
            refs.append(dict(source=name,destination=dest,sha256=sha,bytes=len(b),parent_commit=p.PARENT))
        else:files.append(dict(source=name,sha256=sha,bytes=len(b)))
    # These exact entries must be freshly pinned by root; do not freeze current mutable bytes here.
    mutable=[T+'/TRAINING_STATE.json',T+'/RUNNING.md',T+'/REBUTTAL_COMPLETION_20261009.md',
        T+'/server_reactivation_20261009/MONITOR_HANDOFF.md','docs/返修实验总览.md',
        'tmp/update_reactivation_state_20261009.py','tmp/update_completion_current_20261009.py','tmp/update_overview_closure100_root_20261009.py']
    for name in mutable:files.append(dict(source=name,sha256=None,bytes=None))
    spec=dict(status='DRAFT_REQUIRES_ROOT_FRESH_STATE_PINS',parent=p.PARENT,branch=p.BRANCH,
        scope='CLOSED200_COMPACT_EVIDENCE_AND_AUTHOR_REVIEW_NO_BULK',accepted=p.ACCEPTED,test_started=False,goal_complete=False,
        bindings=bindings,optional_bindings={'hybrid27':None,'logofair100_startup':None},
        allowed_paths=sorted(e['source'] for e in files),files=files,parent_recovery_references=refs,
        original_seals_verified=seals,excluded_non_scientific_console=omitted,
        bulk_policy='Archives/models/arrays/cache remain at original SHA-bound external paths; this is not a raw-artifact Git backup',
        pending='Parent refreshes exact mutable inputs/live and optional actual receipts, then reviews plan before freeze/stage')
    p.ns['save'](p.HERE/'DRAFT_SPEC.json',spec)
    print(json.dumps(dict(selected=len(files),frozen=sum(e['sha256'] is not None for e in files),mutable=len(mutable),
        parent_unchanged=len(refs),known_bytes=sum(e['bytes'] or 0 for e in files),verified_seals=len(seals))))
