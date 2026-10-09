"""Derive only the approved15 outer lifecycle from the accepted remaining7."""
from pathlib import Path
import ast, difflib, hashlib, json, shutil

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
OLD=ROOT/'tmp/celeba_mechanism_valid_replay_20261009/remaining_seven_prepared_20261009'
PREP=ROOT/'tmp/celeba_mechanism_valid_incremental_v2_20261009'
REMOTE='/workspace/guardfed_checks/celeba_mechanism_valid_incremental_v2_execution_20261009'
SHA='70d0d920c4c5351c42efc9968fe3c38eed431d208b94bc8af486ba49d869a42d'
ROOTSHA='701f95342f1da3a7e0ba6247b85320a7f5043ccf92475c1c713105866636f8de'
def digest(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def put(p,b):
    p=HERE/p; p.parent.mkdir(parents=True,exist_ok=True)
    if isinstance(b,str): b=b.encode('utf-8')
    with p.open('xb') as f:f.write(b)
def save(p,x):put(p,json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
assert digest(PREP/'FILES_SHA256.json')==SHA
seal=json.loads((PREP/'FILES_SHA256.json').read_text())
for row in seal['members']:
    p=PREP/row['path'];assert digest(p)==row['sha256'] and p.stat().st_size==row['size']
    put(Path('sealed_source')/row['path'],p.read_bytes())
put(Path('sealed_source/FILES_SHA256.json'),(PREP/'FILES_SHA256.json').read_bytes())
approval=ROOT/'tmp/celeba_mechanism_valid_incremental_v2_root_approval_20261009/APPROVED.json'
assert digest(approval)==ROOTSHA
put(Path('ROOT_APPROVED.json'),approval.read_bytes())
scope=json.loads((PREP/'SCOPE.json').read_text());ids=scope['selected_ids']
outputs={i:REMOTE+'/runs/'+i for i in ids}
old=(OLD/'batch.py').read_text();new=old
new=new.replace('Prepared seven-terminal','Authorized fifteen-terminal').replace('PREPARED = HERE.parent',"PREPARED = HERE / 'sealed_source'")
new=new.replace("/workspace/guardfed_checks/celeba_mechanism_valid_replay_20261009/remaining_seven_prepared_20261009",REMOTE)
new=new.replace('ae28a69acbb35413fbe000c8be2a52ff2ed1977100c6e62580543e1c84dc0044',scope['bridge_sha256'])
new=new.replace('be4d1d34b21443572a25e6a189710e8d75b108a8fcbdb58896a71f2ba1d88cc8',scope['inventory_sha256'])
new=new.replace('4da76ee750afd7d8824cadc1db7cbc1466c6eeb599c287ca6aa14b17525be0d1',SHA)
new=new.replace("SELECTED = ['minus_U_IID_Benign_seed' + str(seed) for seed in (91001, 91003, 91004, 91005, 91006, 91007, 91008)]",'SELECTED = '+repr(ids))
new=new.replace('inventory_actual8_Full100refs.json','inventory_actual23_Full100refs.json')
a=new.index('def identities(');z=new.index('\ndef approved(',a)
replacement='''def identities(check_new_seal=True):
    require(digest(PREPARED / 'FILES_SHA256.json') == OLD_SEAL_SHA, 'Prepared19 seal changed')
    for row in read(PREPARED / 'FILES_SHA256.json')['members']:
        p=PREPARED / row['path']
        require(digest(p)==row['sha256'] and p.stat().st_size==row['size'], 'Prepared source/member changed')
    if check_new_seal:
        for row in read(HERE / 'EXECUTION_SOURCE_SHA256.json')['members']:
            p=HERE / row['path']
            require(digest(p)==row['sha256'] and p.stat().st_size==row['size'], 'Execution source changed')
    scope=read(PREPARED / 'SCOPE.json')
    require(scope['selected_ids']==SELECTED and scope['inventory_sha256']==INVENTORY_SHA
        and scope['bridge_sha256']==BRIDGE_SHA, 'Only approved15 frozen identities')
    return scope


def check_approval(approval, scope, scope_sha, seal_sha):
    require(digest(HERE / 'ROOT_APPROVED.json') == ROOT_APPROVAL_SHA, 'Root approval changed')
    root=read(HERE / 'ROOT_APPROVED.json')
    require(root['status']=='ROOT_REVIEW_PASS_BOUNDED_FIFTEEN_VALID_REPLAY' and root['execution_authorized_within_existing_user_request'] is True
        and root['source_seal_sha256']==OLD_SEAL_SHA and root['scope_sha256']==digest(PREPARED/'SCOPE.json')
        and root['inventory_sha256']==INVENTORY_SHA and root['bridge_sha256']==BRIDGE_SHA and root['selected_ids']==SELECTED,
        'Exact reviewed15 root authority required')
    require(approval.get('status')=='APPROVED_FIFTEEN_MECHANISM_VALID_REPLAY_ONLY'
        and approval.get('root_approval_sha256')==ROOT_APPROVAL_SHA and approval.get('execution_seal_sha256')==seal_sha,
        'Execution authority/seal changed')
    require(approval.get('scope_sha256')==scope_sha and approval.get('selected_ids')==SELECTED
        and approval.get('allowed_cpus')==CPUS and approval.get('compute_threads')==8 and approval.get('max_processes')==1,
        'Only15 IDs, one8-thread worker')
    require(approval.get('outputs')=={i:(REMOTE/'runs'/i).as_posix() for i in SELECTED}
        and approval.get('original_scope_outputs')==scope['outputs'] and approval.get('dependency_paths')==scope['dependency_paths'],
        'Exact new owned output mapping/dependencies required')
    require(approval.get('inventory_sha256')==INVENTORY_SHA and approval.get('bridge_sha256')==BRIDGE_SHA
        and approval.get('target_split')=='valid' and approval.get('native_tolerance')==1e-12
        and approval.get('final_test_dispatch') is False and approval.get('new_full_inference')==0
        and approval.get('automatic_retry_authorized') is False, 'Scientific scope changed')

'''
new=new[:a]+replacement+new[z:]
new=new.replace("digest(HERE / 'SCOPE.json'), digest(HERE / 'FILES_SHA256.json')","digest(PREPARED / 'SCOPE.json'), digest(HERE / 'EXECUTION_SOURCE_SHA256.json')")
new=new.replace("child['selected_ids'] == [identity]","child['selected_ids'] == SELECTED and child['selected_id'] == identity")
new=new.replace("scope='MECHANISM_TERMINAL_VALID_REPLAY_BRIDGE_V1'","scope='MECHANISM_TERMINAL_VALID_REPLAY_INCREMENTAL_V2'")
new=new.replace('selected_ids=[identity], device=', 'selected_ids=SELECTED, selected_id=identity, device=')
new=new.replace("SEVEN_VALID_REPLAYS_STRICT_ACCEPTED_BACKUP_PENDING","FIFTEEN_VALID_REPLAYS_STRICT_ACCEPTED_BACKUP_PENDING")
new=new.replace("excluded_already_accepted='minus_U_IID_Benign_seed91002'","excluded_already_accepted=read(PREPARED/'SCOPE.json')['already_closed_replay_ids']")
new=new.replace('Foreign/pending/Full or already accepted seed91002 refused','Foreign/pending/Full or any closed8 ID refused')
new=new.replace('PREPARED_NOT_APPROVED seven accepted-terminal','AUTHORIZED_FIFTEEN accepted-terminal')
new=new.replace("CPUS = list(range(112, 120))","CPUS = list(range(112, 120))\nROOT_APPROVAL_SHA = "+repr(ROOTSHA))
ast.parse(new);put(Path('batch.py'),new)
put(Path('resource_extra.py'),(OLD/'resource_extra.py').read_bytes())
service=(OLD/'service.sh').read_text().replace('/workspace/guardfed_checks/celeba_mechanism_valid_replay_20261009/remaining_seven_prepared_20261009',REMOTE)
conf=(OLD/'supervisor.conf').read_text().replace('guardfed_celeba_mechanism_valid_remaining7','guardfed_celeba_mechanism_valid_incremental15')
put(Path('service.sh'),service);put(Path('supervisor.conf'),conf)
put(Path('COORDINATOR_DIFF.patch'),''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='accepted_remaining7/batch.py',tofile='authorized15/batch.py')))
save(Path('PATH_MAPPING.json'),dict(reason='New execution-only owned outputs; prepared19 scope bytes unchanged',old=scope['outputs'],new=outputs))
save(Path('EXECUTION_DRAFT.json'),dict(status='APPROVED_FIFTEEN_MECHANISM_VALID_REPLAY_ONLY',root_approval_sha256=ROOTSHA,
    scope_sha256=digest(PREP/'SCOPE.json'),selected_ids=ids,allowed_cpus=list(range(112,120)),compute_threads=8,max_processes=1,
    outputs=outputs,original_scope_outputs=scope['outputs'],dependency_paths=scope['dependency_paths'],inventory_sha256=scope['inventory_sha256'],
    bridge_sha256=scope['bridge_sha256'],target_split='valid',native_tolerance=1e-12,final_test_dispatch=False,new_full_inference=0,automatic_retry_authorized=False))
print(json.dumps({'derived':True,'selected':len(ids),'new_training':False}))
