"""Derive source only from original incremental transport / strict fullstate verifier."""
from pathlib import Path
import ast,difflib,hashlib,itertools,json
H=Path(__file__).resolve().parent;R=H.parents[1]
OLD=R/'tmp/gradient_native_after42_20261011'
V=R/'tmp/gradient64_final_acceptance_prepared_20261011/verify_delta.py'
F=R/'tmp/gradient_source_freeze_deployment_index_20261011/freeze_actual001'
PKG=F/'code/Fed-NGA-gradient'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
save=lambda n,v:(H/n).write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8',newline='\n')
assert not (H/'SCOPE.json').exists()
assert sha(F/'ROOT_SOURCE_APPROVAL.json')=='79edd0dde78c9dc23583a22953d3babebb357a4fcebb7197941cc5726b2ff961'
m=read(PKG/'jobs/manifest.json');new=m['new_jobs'];reused=m['reused_jobs']
assert len(new)==96 and len(reused)==4 and m['total']==100
cells=[]
for x in new:
 p=PKG/'jobs'/x['job'];assert sha(p)==x['sha256'];j=read(p)
 assert j['id']==x['id'] and j['method']=='Fed-NGA-gradient' and j['tuning_candidate']=='FedNGA_eta0.001'
 assert j['config']['rounds']==70 and j['config']['celeba_evaluation_split']=='valid'
 cells.append((j['distribution'],j['attack'],j['config']['seed']))
cells += [(r['distribution'],r['attack'],r['seed']) for r in reused]
assert len(set(cells))==100 and set(cells)==set(itertools.product(['IID','non-IID'],['Benign','F Flip','FedSA','S-DFA','Sp-DFA'],range(91001,91011)))
acceptor=PKG/'snapshot/gradient_bridge_fullcoverage/accept_result.py'
contract=dict(status='FROZEN_SOURCE_SCOPE_NOT_RESULTS',new_ids=[r['id'] for r in new],reused_screen4=reused,
 expected_manifest_sha256=sha(PKG/'jobs/manifest.json'),expected_source_approval_sha256=sha(F/'ROOT_SOURCE_APPROVAL.json'),
 expected_source_seal_sha256=sha(PKG/'FILES_SHA256.json'),expected_checker_sha256=sha(acceptor),
 expected_new96=96,expected_total_with_explicit_screen4=100,recipe='FedNGA_eta0.001',evaluation_split='valid',terminal_round=70,
 primary_statistics='Per distribution/attack, all10 shared seeds mean/sampleSD(ddof1); explicit oldscreen4 join only after complete strict+offserver+root acceptance.',
 selection_history='seed91001 supplied oldscreen4 and participated in recipe selection; separately report fixed9 (91002..91010) and fixed6 (91005..91010) panels.',
 negatives='Retain all outcomes, including constant predictions. Zero gaps with collapsed predictions are not fairness wins.',
 runtime_equivalence_claim=False,source_local=PKG.relative_to(R).as_posix(),new_results_verified=0,actual_terminal_ids=None)
save('SCOPE.json',contract)
adapt={};patch=[]
def emit(n,parent,s):
 raw=parent.read_bytes();data=s.encode();compile(data,str(H/n),'exec')
 spans=[dict(a=a,b=b,c=c,d=d,old=raw[a:b].hex(),new=data[c:d].hex()) for tag,a,b,c,d in difflib.SequenceMatcher(None,raw,data,autojunk=False).get_opcodes() if tag!='equal']
 undo=data
 for e in reversed(spans):assert undo[e['c']:e['d']].hex()==e['new'];undo=undo[:e['c']]+bytes.fromhex(e['old'])+undo[e['d']:]
 assert undo==raw
 (H/n).write_bytes(data);adapt[n]=dict(parent_path=parent.relative_to(R).as_posix(),parent_sha256=sha(parent),sha256=sha(H/n),raw_inverse_spans=spans)
 patch.extend(difflib.unified_diff(raw.decode().splitlines(True),s.splitlines(True),fromfile=str(parent.relative_to(R)),tofile=str((H/n).relative_to(R))))

# The exact original helpers retain CPU110 collection, one thread, nice10 and idle IO.
parent=OLD/'pinned_collect_one.py';s=parent.read_bytes().decode().replace('guardfed_celeba_gradient_screen64_v2a','guardfed_celeba_fednga_fullcoverage_v1');emit('pinned_collect_one.py',parent,s)
for n,parent in [('collect_delta.py',OLD/'collect_delta.py'),('verify_delta.py',V),('run_once.py',OLD/'run_once.py')]:
 s=parent.read_bytes().decode()
 changes={
  "'/workspace/guardfed_checks/celeba_gradient_screen64_v2_20261010'":"'/workspace/guardfed_checks/celeba_gradient_fullcoverage_selected_20261011/Fed-NGA-gradient'",
  "'/workspace/celeba_gradient_screen64_v2_results_20261010'":"'/workspace/celeba_fednga_fullcoverage_results_20261011/attempt001'",
  'guardfed_celeba_gradient_screen64_v2a':'guardfed_celeba_fednga_fullcoverage_v1',
  '11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced':contract['expected_source_seal_sha256'],
  '2c5d7699c6e9967d32c9080fb56b4672beb821cee0b20187c68195fb37d204e9':contract['expected_checker_sha256'],
  '985f04d65afd8867cd606ac54b31b081ebeb3626b03b615245abe31a1d5504a7':sha(H/'pinned_collect_one.py'),
  'gradient64_delta_after42_closed46.tar.gz':'nga96_incremental.tar.gz','gradient64_delta_after46_closed64.tar.gz':'nga96_incremental.tar.gz',
  'snapshot/gradient_bridge_20261010':'snapshot/gradient_bridge_fullcoverage'}
 for a,b in changes.items():s=s.replace(a,b)
 if n=='collect_delta.py':
  start=s.index(" auth=read(HERE/'AUTHORIZED_SNAPSHOT.json');");end=s.index(' before=live();',start)
  s=s[:start]+" auth=read(HERE/'AUTHORIZED_SNAPSHOT.json');ids=auth['authorized_ids']\n from scope import validate,manifest_jobs\n prior,previous=validate(auth,read(HERE/'SCOPE.json'),HERE)\n"+s[end:]
  s=s.replace("require(before['service']['returncode']==0 and 'RUNNING' in before['service']['stdout'],'Queue unhealthy')", "require((before['service']['returncode']==0 and before['service']['stdout'].split()[:2]==['guardfed_celeba_fednga_fullcoverage_v1','RUNNING']) or (before['service']['returncode']==3 and before['service']['stdout'].split()[:2]==['guardfed_celeba_fednga_fullcoverage_v1','EXITED'] and not before['workers'] and before['queue']['strict_server_completed_ids']==read(HERE/'SCOPE.json')['new_ids']),'Queue neither healthy running nor exact96 terminal')")
  old="manifest=read(PKG/'jobs/manifest.json');require(len(manifest['jobs'])==len({x['id'] for x in manifest['jobs']})==64 and sha(PKG/'jobs/manifest.json')==auth['manifest_sha256'],'Exact frozen64 grid changed')"
  new="manifest=manifest_jobs(read(PKG/'jobs/manifest.json'),read(HERE/'SCOPE.json'));require(sha(PKG/'jobs/manifest.json')==auth['manifest_sha256'],'Frozen96+4 manifest changed')"
  assert old in s;s=s.replace(old,new)
  s=s.replace("stage=run_queue.STAGE;sys.path.insert(0,str(stage));import worker", "from types import SimpleNamespace\n run_queue.coverage_contract.validate(PKG,SimpleNamespace(source_approval=Path(auth['source_approval_path']),source_approval_sha256=auth['source_approval_sha256'],gate14_adoption=Path(auth['gate14_adoption_path']),gate14_adoption_sha256=auth['gate14_adoption_sha256']))\n stage=run_queue.STAGE;sys.path.insert(0,str(stage));import worker")
  s=s.replace('accepted_before=42,strict_cumulative=42+len(ids)','accepted_before=len(prior[\'accepted_ids\']),strict_cumulative=len(prior[\'accepted_ids\'])+len(ids)')
  old=" for n in ('AUTHORIZED_SNAPSHOT.json','PREVIOUS_ROOT_ADOPTION.json','PREVIOUS_OFFSERVER_ACCEPTANCE.json','collect_delta.py','pinned_collect_one.py'):add('evidence/'+n,HERE/n)"
  new=" for n in ('AUTHORIZED_SNAPSHOT.json','SCOPE.json','scope.py','collect_delta.py','pinned_collect_one.py'):add('evidence/'+n,HERE/n)\n if prior['accepted_ids']:\n  for n in ('PREVIOUS_ROOT_ADOPTION.json','PREVIOUS_OFFSERVER_ACCEPTANCE.json'):add('evidence/'+n,HERE/n)\n for label,key in [('source_approval','source_approval_path'),('gate14_adoption','gate14_adoption_path')]:add('evidence/'+label+'.json',auth[key])"
  assert old in s;s=s.replace(old,new)
 if n=='verify_delta.py':
  start=s.index("  for n,pin in [('PREVIOUS_ROOT_ADOPTION.json'");end=s.index("  pkg=out/'source/package'",start)
  s=s[:start]+"  from scope import validate,manifest_jobs\n  require(sha(out/'evidence/AUTHORIZED_SNAPSHOT.json')==sha(H/'AUTHORIZED_SNAPSHOT.json'),'Actual selection changed')\n  require(sha(out/'evidence/SCOPE.json')==sha(H/'SCOPE.json'),'Source scope changed')\n  for label in ('source_approval','gate14_adoption'):require(sha(out/'evidence'/(label+'.json'))==auth[label+'_sha256'],'Frozen source/gate14 proof archive differs')\n  prior,previous=validate(auth,read(H/'SCOPE.json'),out/'evidence')\n"+s[end:]
  s=s.replace("manifest=read(pkg/'jobs/manifest.json');ids=auth['authorized_ids']", "manifest=manifest_jobs(read(pkg/'jobs/manifest.json'),read(H/'SCOPE.json'));ids=auth['authorized_ids']")
  s=s.replace("len(manifest['jobs'])==len({x['id'] for x in manifest['jobs']})==64", "len(manifest['jobs'])==len({x['id'] for x in manifest['jobs']})==96")
  line="  require(len(ids)==18 and prior['accepted_ids']+ids==[x['id'] for x in manifest['jobs']],'Exact ordered64 closure required')\n";assert line in s;s=s.replace(line,'')
  s=s.replace('accepted_before=46,accepted_total=46+len(ids)',"accepted_before=len(prior['accepted_ids']),accepted_total=len(prior['accepted_ids'])+len(ids)")
  s=s.replace('old46_ordered_prefix_exact=True','previous_accepted_ordered_prefix_exact=True').replace('screen64_completed_claim=False','coverage100_completed_claim=False,reused_screen4_count=4,reused_screen4_repacked=0')
  s=s.replace('Frozen64 grid/order mismatch','Frozen96+4 grid/order mismatch')
 if n=='run_once.py':
  s=s.replace('PARENT_AUTHORIZED_FIXED_GRADIENT64_DELTA_AFTER42_CLOSED46','PARENT_AUTHORIZED_FIXED_NGA96_INCREMENTAL_DELTA')
  s=s.replace("len(review['exact_new_ids'])==len(auth['authorized_ids'])==4", "len(review['exact_new_ids'])==len(auth['authorized_ids'])>0")
  s=s.replace(" assert sha(R/auth['prior_root_path'])==auth['prior_root_sha256'] and sha(R/auth['prior_offserver_path'])==auth['prior_offserver_sha256']", " from scope import validate\n validate(auth,read(H/'SCOPE.json'),H)\n assert sha(auth['snapshot_path'])==auth['snapshot_sha256']")
  s=s.replace("['collect_delta.py','pinned_collect_one.py','AUTHORIZED_SNAPSHOT.json','PREVIOUS_ROOT_ADOPTION.json','PREVIOUS_OFFSERVER_ACCEPTANCE.json']", "(['collect_delta.py','pinned_collect_one.py','scope.py','SCOPE.json','AUTHORIZED_SNAPSHOT.json']+(['PREVIOUS_ROOT_ADOPTION.json','PREVIOUS_OFFSERVER_ACCEPTANCE.json'] if auth['accepted_prior_ids'] else []))")
  s=s.replace("('celeba_gradient64_delta' in x or 'gradient_native_after' in x)","('celeba_gradient64_delta' in x or 'gradient_native_after' in x or 'NGA96_incremental' in x)")
  s=s.replace("assert service.returncode==0 and 'RUNNING' in service.stdout", "assert (service.returncode==0 and service.stdout.split()[:2]==['guardfed_celeba_fednga_fullcoverage_v1','RUNNING']) or (service.returncode==3 and service.stdout.split()[:2]==['guardfed_celeba_fednga_fullcoverage_v1','EXITED'] and len(json.loads(Path('/workspace/celeba_fednga_fullcoverage_results_20261011/attempt001/QUEUE_PROGRESS.json').read_bytes())['strict_server_completed_ids'])==96)")
 emit(n,parent,s)
save('SOURCE_ADAPTATIONS.json',adapt);(H/'SOURCE_DIFF.patch').write_bytes(''.join(patch).encode())
auth=dict(status='UNBOUND_TEMPLATE_NOT_EXECUTION_AUTHORITY',authorized_ids=None,observed_terminal_ids=None,observed_failure_paths=None,accepted_prior_ids=None,accepted_prior_count=None,
 snapshot_path=None,snapshot_sha256=None,manifest_sha256=None,source_seal_sha256=None,checker_sha256=None,source_approval_path=None,source_approval_sha256=None,gate14_adoption_path=None,gate14_adoption_sha256=None,
 prior_root_path=None,prior_root_sha256=None,prior_offserver_path=None,prior_offserver_sha256=None,test=False,automatic_retry=False)
save('AUTHORIZED_SNAPSHOT.template.json',auth)
save('ROOT_REVIEW.template.json',dict(status='UNBOUND_TEMPLATE_NOT_AUTHORITY',root_adoption_performed=False,source_files_sha256=None,authorization_sha256=None,exact_new_ids=None))
save('FUTURE_ARTIFACTS.template.json',dict(actual_terminal_ids=None,actual_checker_sha256=None,actual_source_seal_sha256=None,actual_F_volume=None,actual_archive=None,actual_archive_sha256=None,actual_receipt_sha256=None,actual_strict_proof_sha256=None,actual_offserver_proof_sha256=None,actual_root_adoption=None))
save('DEPENDENCIES.json',dict(status='LOCAL_FROZEN_SOURCE_PINS_ONLY',files={p.relative_to(R).as_posix():dict(sha256=sha(p),bytes=p.stat().st_size) for p in [F/'ROOT_SOURCE_APPROVAL.json',F/'code/BOUND_INPUTS.json',PKG/'FILES_SHA256.json',PKG/'jobs/manifest.json',PKG/'run_queue.py',PKG/'coverage_contract.py',PKG/'snapshot/gradient_bridge_fullcoverage/worker.py',acceptor,PKG/'snapshot/gradient_bridge_fullcoverage/protocol.json',R/'tmp/guardfed_local_storage.py',R/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py']},missing_server_sources=[]))
print(json.dumps(dict(status='SOURCE_ONLY_PREPARED',new96=96,reused_screen4=4,authorized_delta_ids=None,science_executed=False)))
