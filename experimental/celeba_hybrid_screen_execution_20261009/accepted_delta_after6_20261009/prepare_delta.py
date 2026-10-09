"""Rebind the accepted after4 collector to one frozen after6 ID delta only."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
B=Path(__file__).resolve().parent;H=B.parent;OLD=H/'accepted_delta_after4_20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(name,value):
 with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
latest=H/'LATEST_BACKUP.json';chain=H/'BACKUP_CHAIN_accepted_delta_after4_20261009.json'
assert sha(chain)=='231dda94d276174aaf488bb03aa71a8072b2bd0b4a0f63c241fa412e72d3fcd7'
previous=read(chain);assert previous['accepted_total']==6 and len(set(previous['accepted_job_ids']))==6
assert read(latest)['chain_sha256']==sha(chain) and read(latest)['accepted']==6
snapshot=read(B/'AUTHORIZED_SNAPSHOT.json');scope=read(H/'screen_scope.json')
assert snapshot['source_members_verified']==69 and not snapshot['screen_failure'] and not any(r['failures'] for r in snapshot['rows'])
closed=[r['id'] for r in snapshot['rows'] if r['terminal'] and r['result_exists'] and r['round']==r['result_rounds']==70]
wanted=[e['id'] for e in scope['jobs'] if e['id'] in set(closed)-set(previous['accepted_job_ids'])]
assert set(previous['accepted_job_ids'])<=set(closed) and len(wanted)==len(set(wanted))==4
shutil.copyfile(latest,B/'PREVIOUS_LATEST.json');shutil.copyfile(chain,B/'PREVIOUS_CHAIN.json')
save('EXACT_DELTA.json',dict(status='FROZEN_READONLY_TERMINAL_SET_DIFFERENCE',selected_ids=wanted,
 old_accepted=6,new_closed=4,total_if_root_adopts=10,planned=32,
 snapshot_sha256=sha(B/'AUTHORIZED_SNAPSHOT.json'),snapshot_utc=snapshot['utc'],previous_chain_sha256=sha(chain),
 no_later_arrivals=True,no_recipe_selection=True))
original=(OLD/'collect_once.py').read_text(encoding='utf8');assert sha(OLD/'collect_once.py')=='55ed636c2232568ad8b75d0856e31cbf1227c3dbeae38018529c6536f6f0356e'
source=original.replace('5357d4ba81bbdf50964cb655fd19bb01eb6618a920dc0b2e683d9e4f25c014c2',sha(chain)).replace('5bfe255142c9a6798f82a1331d83aaa45917b5e27781c4dac965028552257b84',sha(B/'AUTHORIZED_SNAPSHOT.json'))
old_ids=read(OLD/'EXACT_DELTA.json')['selected_ids'];assert source.count('wanted='+repr(old_ids))==1
source=source.replace('wanted='+repr(old_ids),'wanted='+repr(wanted)).replace("previous['accepted_total']==4","previous['accepted_total']==6").replace('old_accepted=4,accepted_total=4+len(wanted)','old_accepted=6,accepted_total=6+len(wanted)').replace('accepted_delta_after4_20261009','accepted_delta_after6_20261009').replace('hybrid_after4_delta.tar.gz','hybrid_after6_delta.tar.gz')
compile(ast.parse(source),'collect_once.py','exec')
with (B/'collect_once.py').open('x',encoding='utf8',newline='\n') as f:f.write(source)
with (B/'COLLECTOR_DIFF.patch').open('x',encoding='utf8',newline='\n') as f:f.writelines(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='accepted_after4/collect_once.py',tofile='accepted_after6/collect_once.py'))
save('SOURCE_RECEIPT.json',dict(scope='Frozen exact four terminal IDs only; no later arrivals, dispatch, inference or selection',
 original_collector_sha256=sha(OLD/'collect_once.py'),collector_sha256=sha(B/'collect_once.py'),
 previous_chain_sha256=sha(chain),previous_latest_sha256=sha(latest),authorized_snapshot_sha256=sha(B/'AUTHORIZED_SNAPSHOT.json'),
 inherited_source_receipt_sha256=sha(OLD/'SOURCE_RECEIPT.json'),inherited_root_adoption_sha256=sha(OLD/'ROOT_ADOPTION_REVIEW.json'),
 original_checked_body_unchanged=True,original_driver_writer_policy_unchanged=True,
 source_seal_sha256=sha(H/'FILES_SHA256.json'),source_members_not_repackaged=69,
 CPU=106,threads=1,nice=10,IO='idle',canonical_ledgers_not_modified=True,selected_ids=wanted))
print(json.dumps(dict(selected_ids=wanted,previous_chain_sha256=sha(chain),previous_latest_sha256=sha(latest),snapshot_sha256=sha(B/'AUTHORIZED_SNAPSHOT.json'),collector_sha256=sha(B/'collect_once.py'))))
