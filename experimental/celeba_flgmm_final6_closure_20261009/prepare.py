"""Prepare templates from actual accepted26 bytes; never collect or select a recipe."""
from pathlib import Path
import ast,difflib,hashlib,json,shutil
from closure_guard import EXPECTED_CHAIN,EXPECTED_PACKAGE,SOURCE_PINS,validate_previous
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
BASE=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_dispatch';RELEASE=ROOT/'tmp/celeba_flgmm_screen_20261009_v2_frozen_release';OLD=BASE/'accepted_delta_after19_v2_20261009'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):return json.loads(p.read_bytes())
def save(name,value):
 with (HERE/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def put(name,data):
 with (HERE/name).open('xb') as f:f.write(data)
chain_path=BASE/'BACKUP_CHAIN_accepted_delta_after19_v2_20261009.json';previous=read(chain_path);manifest=read(RELEASE/'jobs/manifest.json')
assert sha(chain_path)==EXPECTED_CHAIN and sha(OLD/'ROOT_ADOPTION_REVIEW.json')=='93a5ab127fb357be93cf7b07bda6d09fab8839f73bca87406222b3f270fd1134'
for name,digest in SOURCE_PINS.items():assert sha(RELEASE/name)==digest
for name,digest in read(RELEASE/'PACKAGE_SHA256.json')['files'].items():assert sha(RELEASE/name)==digest
wanted=validate_previous(previous,sha(chain_path),sha(RELEASE/'PACKAGE_SHA256.json'),manifest)
assert read(BASE/'LATEST_BACKUP.json')['chain_sha256']==EXPECTED_CHAIN and read(BASE/'LATEST_BACKUP.json')['accepted']==26
for source,name in [(chain_path,'PREVIOUS_CHAIN.json'),(BASE/'LATEST_BACKUP.json','PREVIOUS_LATEST.json'),(RELEASE/'jobs/manifest.json','manifest.json'),(RELEASE/'source/protocol.json','protocol.json'),(RELEASE/'frozen_score.py','frozen_score.py')]:put(name,source.read_bytes())
save('EXACT_DELTA.json',dict(status='PREPARED_MANIFEST_MINUS_ACCEPTED26_ONLY',selected_ids=wanted,previous_accepted=26,expected_new=6,total_after_future_acceptance=32,planned=32,actual_new_accepted=0,actual_terminal_snapshot_available=False))
collector_raw=(OLD/'collect_delta.py').read_bytes();assert sha(OLD/'collect_delta.py')=='57a3cea55a409cc3e9fa386d2f01d38d288a9fc7a8cfd98730b85dc72c8c39df'
text=collector_raw.decode();nl='\r\n' if '\r\n' in text else '\n';original=text
text=text.replace('from screen_common import accepted,digest,local_identity,read,repo_identity,write_json','from screen_common import accepted,digest,local_identity,read,repo_identity,write_json'+nl+'from closure_guard import validate_previous,validate_snapshot,validate_live')
text=text.replace("    assert not (BATCH/'PARTIAL_ACCEPTANCE.json').exists(), 'Existing batch must not be overwritten'","    assert not any((BATCH/name).exists() for name in ['PARTIAL_ACCEPTANCE.json','MEMBERS.json','BACKUP_SHA256.json','accepted_final6_delta.tar.gz']), 'Existing batch must not be overwritten'"+nl+"    assert not list(BATCH.glob('BACKUP_FAILURE_*.json')), 'Preserve failure; no automatic retry'")
text=text.replace('84be50b9c7dc44ceb3bf34e1bedd0c3ce36e5186b9020e1d6935b9cdc0c95dcd',EXPECTED_CHAIN).replace("previous['accepted_total']==19 and len(previous['accepted_job_ids'])==19","previous['accepted_total']==26 and len(previous['accepted_job_ids'])==26")
old="    assert digest(BATCH/'AUTHORIZED_SNAPSHOT.json')=='f87eb797aef52016d1e9a77de871b5cd4942474545c2e628184f993a2ebfc404'"
new=nl.join(["    bindings=read(BATCH/'EXECUTION_BINDINGS.json')","    assert bindings['status']=='ROOT_AUTHORIZED_FIXED_FINAL6_CLOSURE'","    assert isinstance(bindings['authorized_snapshot_sha256'],str) and len(bindings['authorized_snapshot_sha256'])==64","    assert digest(BATCH/'AUTHORIZED_SNAPSHOT.json')==bindings['authorized_snapshot_sha256']","    validate_snapshot(read(BATCH/'AUTHORIZED_SNAPSHOT.json'),manifest)","    validate_live(snapshot,manifest,resource)"])
assert text.count(old)==1;text=text.replace(old,new)
old_wanted=read(OLD/'EXACT_DELTA.json')['selected_ids'];assert text.count('wanted='+repr(old_wanted))==1
text=text.replace('wanted='+repr(old_wanted),'wanted='+repr(wanted)).replace("    assert read(BATCH/'EXACT_DELTA.json')['selected_ids']==wanted","    assert read(BATCH/'EXACT_DELTA.json')['selected_ids']==wanted"+nl+"    assert validate_previous(previous,digest(BATCH/'PREVIOUS_CHAIN.json'),digest(RELEASE/'PACKAGE_SHA256.json'),manifest)==wanted")
text=text.replace('accepted_total_including_previous=19+len(wanted),planned=32,complete=False','accepted_total_including_previous=26+len(wanted),planned=32,complete=True').replace('accepted_total=19+len(wanted)','accepted_total=26+len(wanted)').replace('accepted_delta_after19_v2.tar.gz','accepted_final6_delta.tar.gz').replace("tarfile.open(archive,'w:gz')","tarfile.open(archive,'x:gz')")
text=text.replace("'AUTHORIZED_SNAPSHOT.json','PREVIOUS_LATEST.json')","'AUTHORIZED_SNAPSHOT.json','PREVIOUS_LATEST.json','closure_guard.py','EXECUTION_BINDINGS.json')")
strict=lambda value:value[value.index('    for identity in wanted:'):value.index('    after=repo_identity')]
assert strict(text).encode()==strict(original).encode()
compile(ast.parse(text),'collect_delta.py','exec');put('collect_delta.py',text.encode())
diff=''.join(difflib.unified_diff(original.splitlines(True),text.splitlines(True),fromfile='accepted26_parent/collect_delta.py',tofile='prepared_final6/collect_delta.py'));put('COLLECTOR_DIFF.patch',diff.encode())
old_verify=(OLD/'verify_delta_offserver.py').read_bytes().decode();verify=old_verify.replace('84be50b9c7dc44ceb3bf34e1bedd0c3ce36e5186b9020e1d6935b9cdc0c95dcd',EXPECTED_CHAIN).replace("receipt['accepted_total']==19+receipt['accepted_new']","receipt['accepted_total']==26+receipt['accepted_new']==32"+nl+"    assert receipt['accepted_new']==6 and len(set(receipt['accepted_new_ids']))==6"+nl+"    assert sha(BATCH/'PREVIOUS_CHAIN.json')==receipt['previous_chain_sha256']"+nl+"    assert receipt['accepted_new_ids']==read(BATCH/'EXACT_DELTA.json')['selected_ids']").replace('accepted_delta_after19_v2.tar.gz','accepted_final6_delta.tar.gz')
compile(ast.parse(verify),'verify_delta_offserver.py','exec');put('verify_delta_offserver.py',verify.encode())
put('SOURCE_DIFF.patch',(diff+''.join(difflib.unified_diff(old_verify.splitlines(True),verify.splitlines(True),fromfile='accepted26_parent/verify_delta_offserver.py',tofile='prepared_final6/verify_delta_offserver.py'))).encode())
source_rows=[];all_records=[]
groups=[('BACKUP_CHAIN_first_two_20261009.json','backups/first_two_20261009'),('BACKUP_CHAIN_increment_20261009T1133Z.json','backups/increment_20261009T1133Z'),('BACKUP_CHAIN_increment_after6_20261009.json','accepted_delta_after6_20261009'),('BACKUP_CHAIN_accepted_delta_after13_20261009.json','accepted_delta_after13_20261009'),('BACKUP_CHAIN_accepted_delta_after19_v2_20261009.json','accepted_delta_after19_v2_20261009')]
for chain_name,delta_name in groups:
 c=read(BASE/chain_name);delta=BASE/delta_name;server=read(delta/'PARTIAL_ACCEPTANCE.json');offserver=read(delta/'OFFSERVER_ACCEPTANCE.json');receipt=read(delta/'BACKUP_SHA256.json')
 detail=c.get('new_batch',c);assert sha(delta/detail['archive'])==detail['archive_sha256']==receipt['archive_sha256']
 expected_strict=c.get('server_strict_sha256',c.get('server_acceptance_sha256',receipt['acceptance_sha256']));assert sha(delta/'PARTIAL_ACCEPTANCE.json')==expected_strict==receipt['acceptance_sha256']
 expected_off=c.get('offserver_proof_sha256',detail.get('offserver_acceptance_sha256'));assert sha(delta/'OFFSERVER_ACCEPTANCE.json')==expected_off
 verify_rows={r['id']:r for r in offserver['records']};assert len(verify_rows)==len(server['records'])
 for row in server['records']:assert all(row[key]==verify_rows[row['id']][key] for key in ('rounds','seed','distribution','attack','metrics','checkpoint_sha256'))
 all_records+=server['records']
 paths=[BASE/chain_name,delta/'PARTIAL_ACCEPTANCE.json',delta/'OFFSERVER_ACCEPTANCE.json',delta/'BACKUP_SHA256.json',delta/detail['archive']]
 source_rows.append(dict(server_path=(delta/'PARTIAL_ACCEPTANCE.json').relative_to(ROOT).as_posix(),offserver_path=(delta/'OFFSERVER_ACCEPTANCE.json').relative_to(ROOT).as_posix(),pins={p.relative_to(ROOT).as_posix():sha(p) for p in paths}))
assert len(all_records)==len({r['id'] for r in all_records})==26 and {r['id'] for r in all_records}==set(previous['accepted_job_ids'])
save('PRIOR26_RECORD_SOURCES.json',dict(status='ACCEPTED26_SOURCE_INDEX_ONLY_NO_RECIPE_SELECTION',parent_chain_path=chain_path.relative_to(ROOT).as_posix(),parent_chain_sha256=EXPECTED_CHAIN,sources=source_rows))
save('EXECUTION_BINDINGS_TEMPLATE.json',dict(status='PREPARED_NOT_AUTHORIZED',authorized_snapshot_sha256=None,scope='exact six remaining manifest IDs; all32 terminal, no active producers',automatic_retry=False,actual_new_accepted=0))
save('SOURCE_RECEIPT.json',dict(status='PREPARED_ONLY_NOT_EXECUTED',original_collector_sha256=sha(OLD/'collect_delta.py'),original_verifier_sha256=sha(OLD/'verify_delta_offserver.py'),collector_sha256=sha(HERE/'collect_delta.py'),verifier_sha256=sha(HERE/'verify_delta_offserver.py'),original_per_job_loop_bytes_exact=True,strict_loop_raw_sha256=hashlib.sha256(strict(original).encode()).hexdigest(),parent_fields_used=['accepted_total','accepted_job_ids'],nonexistent_parent_package_field_not_used=True,parent_chain_sha256=EXPECTED_CHAIN,parent_root_sha256=sha(OLD/'ROOT_ADOPTION_REVIEW.json'),source_pins=SOURCE_PINS,selected_ids=wanted,scientific_acceptor_score_source_data_calibration_unchanged=True,no_automatic_retry=True,actual_new_accepted=0))
print(json.dumps(dict(status='PREPARED_ONLY',remaining_ids=wanted,strict_loop_raw_sha256=hashlib.sha256(strict(original).encode()).hexdigest(),old26_records_bound=len(all_records))))
