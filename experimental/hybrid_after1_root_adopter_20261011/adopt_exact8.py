"""Root-only adoption of exact8 saved Hybrid70 records,parent1->9;no strict/Torch rerun."""
from pathlib import Path,PurePosixPath
import ast,datetime,hashlib,json,sys,tarfile
sys.dont_write_bytecode=True
ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'tmp'))
from guardfed_local_storage import check_bulk_storage
SRC=ROOT/'tmp/hybrid_native_after1_20261011'
PRIOR=ROOT/'tmp/celeba_hybrid_first1_root_adoption_20261010/ROOT_ADOPTION.json'
DEST=ROOT/'tmp/celeba_hybrid_native9_root_adoption_20261011'
F=Path('F:/YananResearchStorage/GuardFed/hybrid_native_after1_20261011');V=F/'verified';S=V/'stage'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert not sys.flags.optimize and not DEST.exists()
storage=check_bulk_storage()
expected=[f'CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed{s}_fullcoverage' for s in range(91003,91011)]
prior_id='CosineFairness_lam20.0_tau0.1_lr0.001_IID_Benign_seed91002_fullcoverage'
assert sha(PRIOR)=='04d62c367609d1c6d079ff538f45a575bff368944d4833f47fd1cf28159c5fa4'
prior=read(PRIOR);assert prior['status']=='ROOT_HYBRID_FIRST1_ORIGINAL_STRICT_OFFSERVER_RESTORE_CHAIN_ADOPTED'
assert (prior['prior_accepted'],prior['new_accepted'],prior['cumulative_accepted'])==(0,1,1) and prior['accepted_new_ids']==[prior_id]
for name,digest,count in [('DELIVERY_FILES_SHA256.json','da25f1792d677a81f057e12a699d8b113f919282d442792dfeefd4f521e95703',58),('FILES_SHA256.json','32c45838444f673ccecc7424f5f541a3e23e308c2e0a917e2129fd319b2634c3',19)]:
 assert sha(SRC/name)==digest and len(read(SRC/name)['files'])==count
 for member,pin in read(SRC/name)['files'].items():
  p=(SRC/member).resolve();assert p.is_relative_to(SRC.resolve())
  assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
assert sha(SRC/'ROOT_READY_HANDOFF.json')=='63cfa0267811f803a567700b59239af2d931eabca943f2e74aa13d18432b1dc8'
h=read(SRC/'ROOT_READY_HANDOFF.json');scope=read(SRC/'DELTA_SCOPE.json');off=read(V/'OFFSERVER_VERIFICATION.json');remote=read(F/'REMOTE_STRICT.json');receipt=read(F/'BACKUP_RECEIPT.json');tensors=read(SRC/'SAVED_TENSOR_IDENTITY.json');records=read(SRC/'RECORDS8.json')['records']
assert h['status']=='ROOT_READY_EXACT8_HYBRID70_NATIVE_STRICT_OFFSERVER_NOT_ADOPTED' and h['root_adopted'] is False and h['root_adoption_required']
assert (h['accepted_before'],h['accepted_new'],h['accepted_total'],h['planned_new'],h['reused_separate'],h['planned_total'])==(1,8,9,96,4,100)
assert h['accepted_new_ids']==scope['ids']==remote['accepted_new_ids']==off['local']['accepted_new_ids']==expected
assert h['accepted_ids']==off['accepted_ids']==remote['accepted_ids']==[prior_id]+expected and len(set(h['accepted_ids']))==9
assert sha(SRC/'PRIOR_ROOT_ADOPTION.json')==sha(PRIOR)==h['prior_root_sha256']
assert sha(SRC/'PRIOR_OFFSERVER.json')==prior['offserver_sha256']==h['prior_offserver_sha256']=='16dc834a07c7b54e439a1c2b99d0dc528eec0d631b50a9e8fecd3c06735f2755'
assert read(SRC/'PRIOR_OFFSERVER.json')['accepted_ids']==[prior_id] and scope['accepted_before']==1 and scope['no_later_completions'] is True
assert Path(h['archive_path'])==F/'delta.tar.gz' and Path(h['receipt_path'])==F/'BACKUP_RECEIPT.json' and Path(h['offserver_path'])==V/'OFFSERVER_VERIFICATION.json'
assert sha(F/'delta.tar.gz')==h['archive_sha256']==receipt['archive_sha256']=='ce7b6b2d3368b32e68a16fcd21c99bccf9358b91f9fe6dfda9cdd9eb2dd61ccf'
for name,key,fp in [('BACKUP_RECEIPT.json','receipt_sha256',F/'BACKUP_RECEIPT.json'),('OFFSERVER_VERIFICATION.json','offserver_sha256',V/'OFFSERVER_VERIFICATION.json'),('REMOTE_STRICT.json','remote_strict_sha256',F/'REMOTE_STRICT.json')]:assert sha(SRC/name)==sha(fp)==h[key]
assert off['status']=='PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON' and remote['status']=='PASS_SAVED_ORIGINAL_FULLCOVERAGE_RECORD' and receipt['status']=='REMOTE_STRICT_BACKUP_READY_NOT_OFFSERVER'
assert (off['accepted_total'],off['local']['accepted_before'],off['local']['accepted_new'],off['member_count'])==(9,1,8,272)
assert receipt['content_members']==271 and receipt['archive_members']==272 and receipt['accepted_new_ids']==expected
assert sha(SRC/'DELTA_SCOPE.json')==receipt['delta_scope_sha256']==h['delta_scope_sha256']
assert receipt['source_seal_sha256']==h['source_seal_sha256']==sha(SRC/'FILES_SHA256.json')
assert h['package_sha256']==receipt['package_sha256']==remote['package_sha256']==off['local']['package_sha256']==sha(S/'PACKAGE_SHA256.json')=='a87a050b497a184efbe18b4649ad6bde40b9ea29a16f9e3efddd0ef1156e3b04'
for name,digest in read(S/'PACKAGE_SHA256.json')['files'].items():assert sha(S/name)==digest
# Root rehashes existing archive bytes only;no extraction,strict checker,model load or inference.
with tarfile.open(F/'delta.tar.gz','r:gz') as tar:
 members=tar.getmembers();names=[m.name for m in members]
 assert len(names)==len(set(names))==272 and all(m.isfile() for m in members)
 assert all(not PurePosixPath(n).is_absolute() and '..' not in PurePosixPath(n).parts and '\\' not in n and ':' not in n for n in names)
 raw=tar.extractfile('MEMBERS.json').read();assert hashlib.sha256(raw).hexdigest()==h['inventory_sha256']==receipt['inventory_sha256']
 inventory=json.loads(raw)['files'];assert set(names)==set(inventory)|{'MEMBERS.json'}
 for member in members:
  if member.name=='MEMBERS.json':continue
  digest=hashlib.sha256();stream=tar.extractfile(member)
  for block in iter(lambda:stream.read(8*1024**2),b''):digest.update(block)
  pin=inventory[member.name];assert digest.hexdigest()==pin['sha256'] and member.size==pin['bytes']
  saved=V.joinpath(*PurePosixPath(member.name).parts);assert sha(saved)==pin['sha256'] and saved.stat().st_size==pin['bytes']
original=ROOT/'tmp/celeba_hybrid_fullcoverage_first_delta_20261010'
assert (SRC/'verify_saved.py').read_bytes()==(original/'verify_saved.py').read_bytes() and (SRC/'storage.py').read_bytes()==(original/'storage.py').read_bytes()
for name in ['collect_once.py','restore_verify.py']:assert (SRC/name).read_bytes().replace(b'108',b'107')==(original/name).read_bytes()
body=(S/'body.py').read_text('utf8');node=next(n for n in ast.parse(body).body if isinstance(n,ast.FunctionDef) and n.name=='checked')
assert hashlib.sha256(ast.get_source_segment(body,node).encode()).hexdigest()==h['original_checked_source_sha256']==remote['original_checked_source_sha256']=='ca307187f096ae17b57ff5b8b0b665aa85cc13c66db323000225469b36262ae9'
assert remote['source_data_before']==remote['source_data_after'] and h['source_data_before_after_exact'] and h['same_terminal_checkpoint_all_metrics']
for key in ['package_sha256','original_checked_source_sha256','gate_sha256','accepted_new','accepted_new_ids','accepted_before','accepted_ids','rounds','formal_table_samples','records']:assert off['local'][key]==remote[key]
assert sha(SRC/'SAVED_TENSOR_IDENTITY.json')==h['tensor_proof_sha256'] and (tensors['models'],tensors['tensors'])==(8,64)
assert tensors['status']=='PASS_ORIGINAL_TENSOR_DIGEST_FINITE_NO_FORWARD' and tensors['CNN_calls']==0 and tensors['CUDA_initialized'] is False
assert [x['id'] for x in records]==[x['id'] for x in tensors['records']]==[x['id'] for x in remote['records']]==expected
manifest=read(S/'manifest.json');entries=[e for e in manifest['jobs'] if e['id'] in expected]
assert [e['id'] for e in entries]==expected and not set(expected)&{x['id'] for x in manifest['reused_jobs']}
for entry,row,tensor,record in zip(entries,remote['records'],tensors['records'],records):
 d=S/entry['output'];result=read(d/'result.json');accept=read(d/'acceptance.json');job=read(S/entry['job']);replay=read(d/'native_replay.json');prov=read(d/'provenance.json')
 assert row=={k:record[k] for k in row} and entry['id']==tensor['id']==record['id']
 assert sha(d/'model.pt')==row['checkpoint_sha256']==tensor['checkpoint_sha256'] and sha(d/'acceptance.json')==row['acceptance_sha256']
 assert sha(S/entry['job'])==entry['job_sha256']==record['job_sha256']==accept['job_sha256']
 assert result['metrics']==replay['metrics']==row['metrics'] and result['config']==job['config']
 assert result['distribution']==job['distribution']=='IID' and result['attack']==job['attack']=='Benign' and result['seed']==job['config']['seed']==record['seed']
 assert (result['rounds'],record['rounds'],result['evaluation_stats']['prediction_count'])==(70,70,19867)
 assert [x['round'] for x in result['trajectory_metrics']]==[x['round'] for x in result['round_summaries']]==[x['round'] for x in read(d/'diagnostics.json')]==list(range(1,71))
 assert result['metrics']==result['trajectory_metrics'][-1]['metrics']
 contract=result['data_contract']['image_data_contract'];assert (contract['evaluation_split'],contract['actual_train_rows'],contract['actual_evaluation_rows'])==('valid',162770,19867)
 assert contract==record['data_contract'] and contract['train_eval_disjoint'] and contract['root_client_disjoint']
 assert prov==record['provenance'] and prov['source_hashes']==remote['source_data_before']
 assert accept['checkpoint_tensor_sha256']==tensor['tensor_sha256']==replay['checkpoint_tensor_sha256'] and len(tensor['tensors'])==8 and all(x['finite'] for x in tensor['tensors'])
 assert not list(d.glob('failure*.json')) and all(sha(d/n)==pin for n,pin in accept['artifact_hashes'].items())
assert h['server_runtime']['CPU']==[108] and h['server_runtime']['torch']=='2.11.0+cu128' and h['local_runtime']['torch']=='2.8.0+cpu'
assert h['runtime_equivalence_claim'] is False and h['CPU108_released'] and sha(SRC/'CPU_RELEASE.json')==h['release_sha256']
assert (h['collector_runs'],h['restore_runs'],h['new_CNN'],h['new_fit'],h['new_training'])==(1,1,0,0,0) and all(x==0 for x in h['actual_exit_codes'].values())
assert not h['final_test'] and not h['recipe_reselection'] and not h['prediction_arrays_recomputed'] and h['negative_results_preserved']
proof=dict(status='ROOT_HYBRID_EXACT8_ORIGINAL_STRICT_OFFSERVER_CHAIN_ADOPTED',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_new_ids=expected,accepted_ids=h['accepted_ids'],new_accepted=8,prior_accepted=1,cumulative_accepted=9,planned_new=96,reused_separate=4,planned_total=100,canaries_formal_records=0,previous_root_path=PRIOR.relative_to(ROOT).as_posix(),previous_root_sha256=sha(PRIOR),previous_offserver_sha256=h['prior_offserver_sha256'],handoff_path=(SRC/'ROOT_READY_HANDOFF.json').relative_to(ROOT).as_posix(),handoff_sha256=sha(SRC/'ROOT_READY_HANDOFF.json'),delivery_seal_sha256=sha(SRC/'DELIVERY_FILES_SHA256.json'),delivery_members=58,source_seal_sha256=sha(SRC/'FILES_SHA256.json'),archive_path=h['archive_path'],archive_sha256=h['archive_sha256'],archive_members=272,archive_all_members_rehashed=True,receipt_sha256=h['receipt_sha256'],offserver_path=h['offserver_path'],offserver_sha256=h['offserver_sha256'],remote_strict_sha256=h['remote_strict_sha256'],package_sha256=h['package_sha256'],saved_tensor_count=64,tensor_proof_sha256=h['tensor_proof_sha256'],records=remote['records'],rounds=70,evaluation_split='valid',n_eval=19867,all_metrics_same_terminal_checkpoint=True,source_data_before_after_exact=True,fresh_F_volume=storage,source_functions_preserved=True,CPU108_release_observation_sha256=h['release_sha256'],new_CNN=0,new_fit=0,new_training=0,strict_restore_Torch_rerun=False,final_test=False,whole100_complete=False,table_adopted=False,negative_results_preserved=True,runtime_equivalence_claim=False,limitations=h['limitations'][:-1]+['Exact8 is adopted by this root execution;prior1/reuse4 remain separate and unrepacked;the auxiliary table is not adopted here.'])
DEST.mkdir()
with (DEST/'ROOT_ADOPTION.json').open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(new_accepted=8,cumulative_accepted=9,root_path=str(DEST/'ROOT_ADOPTION.json'),root_sha256=sha(DEST/'ROOT_ADOPTION.json'))))
