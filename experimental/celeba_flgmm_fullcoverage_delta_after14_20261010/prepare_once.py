from pathlib import Path
import datetime,difflib,hashlib,json
R=Path.cwd();H=Path(__file__).resolve().parent;B=R/'tmp/celeba_flgmm_fullcoverage_incremental_20261009';O=R/'tmp/celeba_flgmm_fullcoverage_delta_after12_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();read=lambda p:json.loads(p.read_bytes())
def save(n,v):
 with (H/n).open('x',encoding='utf8') as f:json.dump(v,f,indent=2);f.write('\n')
latest=read(B/'LATEST_BACKUP.json');assert latest['accepted_total']==14
assert sha(R/latest['root_adoption_path'])==latest['root_adoption_sha256']=='801ec992899529dc7b38e68acfde3b101d2b90a7966def64f0bbf901cf88793b'
assert sha(R/latest['next_collector_previous_path'])==latest['next_collector_previous_sha256']=='2686c5c4d1121611d888462f1e288720465237ae0b80b069fc9e03a38a50513c'
assert sha(B/'FILES_SHA256.json')=='f6de59a56de25a8d316cf7e05c44eafc9751041757e4dc61c88f7bccfd988472'
for n,p in read(B/'FILES_SHA256.json')['files'].items():assert sha(B/n)==p['sha256']
prior=read(R/latest['next_collector_previous_path']);assert len(prior['accepted_job_ids'])==14==len(set(prior['accepted_job_ids']))
snapshot=H/'SNAPSHOT.json';s=read(snapshot);assert s['source_members_match'] and not s['changed_source_members'] and not s['failure_paths'] and not s['queue']['failed']
active={r['id'] for r in s['queue']['active']}
terminal=[r['id'] for r in s['rows'] if r['kind']=='new' and r['result_present'] and r['acceptance_present'] and r['progress']['round']==70 and r['id'] not in active]
assert set(prior['accepted_job_ids'])<=set(terminal)
ids=[identity for identity in terminal if identity not in prior['accepted_job_ids']]
for n,p in [('PREVIOUS_LATEST.json',B/'LATEST_BACKUP.json'),('PREVIOUS_OFFSERVER_ACCEPTANCE.json',R/latest['next_collector_previous_path'])]:
 with (H/n).open('xb') as f:f.write(p.read_bytes())
authorization=dict(status='FIXED_ONE_SNAPSHOT_EXACT_DELTA_NOT_ACCEPTANCE',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),snapshot_path=str(snapshot.relative_to(R)),snapshot_sha256=sha(snapshot),snapshot_utc=s['utc'],prior_root_path=latest['root_adoption_path'],prior_root_sha256=latest['root_adoption_sha256'],prior_offserver_sha256=latest['next_collector_previous_sha256'],prior_count=14,authorized_ids=ids,source_hashes=s['source_hashes'],no_future_terminal_ids_allowed=True,root_adoption_required=True)
save('AUTHORIZED_SNAPSHOT.json',authorization)
if not ids:
 save('VERIFIED_WAIT.json',dict(status='NO_NEW_TERMINAL_FIXED_SNAPSHOT',snapshot_sha256=sha(snapshot),prior_root_sha256=latest['root_adoption_sha256'],accepted_new=0,collector_calls=0,automatic_retry=False));print('NO_NEW_TERMINAL');raise SystemExit(0)
assert ids==['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed'+str(seed)+'_fullcoverage' for seed in (91006,91007)],'Actual snapshot shape changed; inspect before collecting'
original=(B/'collect_delta.py').read_text(encoding='utf8');marker='    if not wanted:\n';assert original.count(marker)==1
addition='    # Parent-authorized fixed snapshot exact2 only; later terminal IDs remain unaccepted.\n    authorized_ids='+repr(ids)+'\n    wanted=[identity for identity in wanted if identity in authorized_ids]\n'
actual=original.replace(marker,addition+marker);assert actual.replace(addition,'')==original
with (H/'collect_delta.py').open('x',encoding='utf8',newline='\n') as f:f.write(actual)
with (H/'COLLECTOR_SCOPE_DIFF.patch').open('x',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(original.splitlines(True),actual.splitlines(True),fromfile='original/collect_delta.py',tofile='fixed_exact2/collect_delta.py')))
with (H/'verify_delta_offserver.py').open('xb') as f:f.write((B/'verify_delta_offserver.py').read_bytes())
save('SOURCE_REUSE.json',dict(original_collector_sha256=sha(B/'collect_delta.py'),collector_sha256=sha(H/'collect_delta.py'),verifier_sha256=sha(H/'verify_delta_offserver.py'),sole_change='Exact two ID filter before original terminal acceptance; scientific strict/archiver/verifier unchanged',filter_removed_source_exact=True,authorized_ids=ids,original_helper_seal_sha256=sha(B/'FILES_SHA256.json')))
for name in ('dispatch_once.py','close_observation.py','preflight.py','transfer_verify_once.py','check_saved_tensors.py'):
 old=(O/name).read_text(encoding='utf8');actual=old
 if name=='preflight.py':actual=old.replace('(91004,91005)','(91006,91007)')
 elif name=='close_observation.py':actual=old.replace('celeba_flgmm_fullcoverage_delta_after12_20261010/collect_delta.py','celeba_flgmm_fullcoverage_delta_after14_20261010/collect_delta.py')
 elif name=='transfer_verify_once.py':actual=old.replace('seed91004_fullcoverage','seed91006_fullcoverage').replace('seed91005_fullcoverage','seed91007_fullcoverage').replace("receipt['accepted_total']==14","receipt['accepted_total']==16")
 elif name=='check_saved_tensors.py':actual=old.replace("proof['accepted_total']==14","proof['accepted_total']==16")
 with (H/name).open('x',encoding='utf8',newline='\n') as f:f.write(actual)
 with (H/(name+'.rebind.patch')).open('x',encoding='utf8') as f:f.write(''.join(difflib.unified_diff(old.splitlines(True),actual.splitlines(True),fromfile='after12/'+name,tofile='after14/'+name)))
print(json.dumps(dict(authorized_ids=ids,prior=14,collector_sha256=sha(H/'collect_delta.py'),snapshot_sha256=sha(snapshot))))
