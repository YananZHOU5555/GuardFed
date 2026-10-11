"""Local source/compact parent metadata only; never invokes collector or adopter."""
from pathlib import Path
import difflib,hashlib,io,json,tokenize
H=Path(__file__).resolve().parent;R=H.parents[1];P=R/'tmp/celeba_native_after341_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
def put(n,v):
 with (H/n).open('x',encoding='utf8',newline='') as f:f.write(v if isinstance(v,str) else json.dumps(v,ensure_ascii=False,indent=2)+'\n')
C=R/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009'
prior=C/'root_delta_20261010T232753Z';root=prior/'ROOT_DELTA_VERIFICATION.json';rp=read(root)
assert sha(root)=='706b528831145c2a2402eb040f1e8a19279b09b29171e99082aa1c410db69770'
ledger=prior/'verified_ledger.json';inspection=prior/'inspection/inspection.json'
assert sha(ledger)==rp['ledger_sha256']=='cfd096af7e787f419774e140ede4ebaadefe5770e5768c5754bd8932894050d0'
assert sha(inspection)==rp['inspection_sha256']=='9fa33070c6a874bde25444dc9aae49fb7974670808eb3edc97ec3207c33ad303'
assert len(read(ledger)['entries'])==49 and len(read(inspection)['records'])==450 and rp['total_new_strict_and_offserver']==350 and rp['root_adopted']
ids=[f'minus_F_non-IID_Benign_seed{s}' for s in range(91001,91011)]
assert not set(ids)&set(read(inspection)['accepted_new_ids'])
with (H/'PARENT_LEDGER.json').open('xb') as f:f.write(ledger.read_bytes())
scope=dict(status='SOURCE_ONLY_EXACT10_F_NONIID_BENIGN_PENDING_CLOSURE',candidate_ids=ids,parent_native=350,target_native=360,Full_reused=100,target_total_records=460,parent_root_path=root.relative_to(R).as_posix(),parent_root_sha256=sha(root),parent_inspection_path=inspection.relative_to(R).as_posix(),parent_inspection_sha256=sha(inspection),parent_ledger_sha256=sha(ledger),parent_remote_inspection_path='/workspace/guardfed_checks/server_reactivation_20261009/native_delta_after341_20261011/root_delta_20261010T232753Z/inspection/inspection.json',future_snapshot_path=None,future_snapshot_sha256=None,execution_authorized=False,scientific_execution_performed=False)
put('FROZEN_SCOPE.json',scope)
assert sha(P/'adopt_root.py')=='01fd3f9f4fcb342ce3a6ea390ae9894aa5e3ba3bb9e5e135d4c731fb8eaf1d23'
s=(P/'adopt_root.py').read_bytes().decode('utf8');tokens=[]
for t in tokenize.generate_tokens(io.StringIO(s).readline):
 if t.type==tokenize.NUMBER and t.string in {'341','350','441','48'}:t=t._replace(string={'341':'350','350':'360','441':'450','48':'49'}[t.string])
 tokens.append(t)
s=tokenize.untokenize(tokens)
for a,b in [('AFTER341','AFTER350_NONIID_BENIGN10'),('native_delta_after341_20261011','native_delta_after350_nonIID_Benign10_20261011'),('after341.root.tmp','after350_nonIID_Benign10.root.tmp'),('root_delta_20261010T222658Z','root_delta_20261010T232753Z'),('755f996594675c858b696ec1e0ac7038de9e8ad2a1fd011212fc78a38b736ba4',sha(root)),('58f108b1cba698559ff96902c9ebe87412061e833421a7dfa0b5e95e99f85898',sha(ledger)),('2854747a0dfcf3623d6b624a9943dc5c01a4c6af631565d4f1e48a66c9bee65d',sha(H/'run_once.py')),('minus_F_IID_Sp-DFA_seed','minus_F_non-IID_Benign_seed'),('old441_records_exact','old450_records_exact')]:s=s.replace(a,b)
s=s.replace('TOTAL>=360','TOTAL==360')
s=s.replace('assert set(REQUIRED_SCENE[1:])<=set(NEW_IDS) and REQUIRED_SCENE[0] not in NEW_IDS','assert NEW_IDS==REQUIRED_SCENE and len(NEW_IDS)==10')
put('adopt_root.py',s)
actual=read(P/'ACTUAL_INPUTS.template.json');actual.update(required_status='ROOT_REVIEWED_ACTUAL_NATIVE_DELTA_AFTER350_NONIID_BENIGN10_READY_TO_ADOPT',note='Root binds actual exact10 delta and all original strict/archive/offserver receipts after successful one-shot collection. No forecast archive cardinality; actual total must be360. Current snapshot/output fields remain null.')
put('ACTUAL_INPUTS.template.json',actual)
put('READINESS.template.json',dict(status='UNBOUND_TEMPLATE_NOT_EXECUTION_PERMISSION',required_ids=ids,required_rounds=70,parent_root_sha256=sha(root),snapshot_path=None,snapshot_sha256=None,snapshot_utc=None,all_ten_complete=None,selected_producers_absent=None,source_data_unchanged=None,failures_empty=None,CPU107_free=None,execution_review_path=None,execution_review_sha256=None,execution_authorized=False,inspection_delta_must_equal_exact_ids=True,extra_terminal_action='REFUSE_BEFORE_BACKUP_PRESERVE_INSPECTION_NO_RETRY'))
put('SOURCE_PARENT_PINS.json',dict(parent_wrapper_path=(P/'run_once.py').relative_to(R).as_posix(),parent_wrapper_sha256=sha(P/'run_once.py'),parent_adopter_path=(P/'adopt_root.py').relative_to(R).as_posix(),parent_adopter_sha256=sha(P/'adopt_root.py'),original_collector_path='tmp/celeba_native_after288_20261011/run_once.py',original_collector_sha256='9745525453d797b72a36aee66bade4fb3d3db04345fecd93535b68f44d142e88',original_science_path='tmp/celeba_mechanism_evidence_20261009/evidence_v4.py',original_science_sha256='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef',parent_root_sha256=sha(root),parent_inspection_sha256=sha(inspection),parent_ledger_sha256=sha(ledger)))
diff=''.join(''.join(difflib.unified_diff((P/n).read_text('utf8').splitlines(True),(H/n).read_text('utf8').splitlines(True),fromfile='after341/'+n,tofile='after350_Benign10/'+n)) for n in ['run_once.py','adopt_root.py'])
put('SOURCE_DIFF.patch',diff)
print(json.dumps(dict(status='SOURCE_PREPARED_NOT_EXECUTED',parent_native=350,exact_new=10,target_native=360,Full_reused=100,saved_views_change=0)))
