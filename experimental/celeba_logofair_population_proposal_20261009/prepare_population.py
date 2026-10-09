"""Local population proposal only: no model imports, inference, fitting or valid labels."""
from pathlib import Path
import hashlib,json,tarfile,io,csv
import numpy as np
HERE=Path(__file__).resolve().parent
WS=HERE.parents[1]
BRIDGE=WS/'tmp/celeba_baselines/logofair_bridge_20261009'
SHARED=WS/'docs/server_deployment_20260923/training_20260923/celeba_shared_calibration_v1'
IDS=WS/'tmp/celeba_final_valid_replay_20261009/v3/phase1_execution_20261009/remote_receipts/phase1_useful/runs/FedAvg_IID_Benign_seed91001'
DOMAIN=b'GuardFed/LoGoFair/virtual-cohort/v1\x00'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def arrsha(a):return hashlib.sha256(np.ascontiguousarray(a,dtype='<i8').tobytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(p,x):
 with Path(p).open('x',encoding='utf-8') as f:json.dump(x,f,indent=2,allow_nan=False);f.write('\n')
def cohorts(ids,domain=DOMAIN):
 assert domain==DOMAIN,'Unapproved domain'
 assert ids.dtype==np.int64 and ids.ndim==1 and (ids>0).all() and len(np.unique(ids))==len(ids),'Unique positive int64 IDs required'
 return np.array([int.from_bytes(hashlib.sha256(domain+str(int(i)).encode('ascii')).digest(),'big')%20 for i in ids],dtype=np.int64)
def check_mapping(mapping,rootsha,validsha):
 assert set(mapping)=={'root_image_id','root_client_id','valid_image_id','valid_client_id'}
 for split,expected in [('root',rootsha),('valid',validsha)]:
  ids=mapping[split+'_image_id'];cid=mapping[split+'_client_id']
  assert arrsha(ids)==expected,'Image ID identity/order mismatch'
  assert cid.dtype==np.int64 and np.array_equal(cid,cohorts(ids)),'Cohort rule mismatch'
  assert set(cid)==set(range(20)),'Missing cohort'
 assert not np.intersect1d(mapping['root_image_id'],mapping['valid_image_id']).size

def main():
 assert not any((HERE/name).exists() for name in ['mapping.npz','mapping_metadata.json','population_support.json','cohort_counts.csv','INPUT_BINDINGS.json','CHECKS.json']), 'Existing proposal must never be overwritten'
 protected=[BRIDGE/'protocol.json',BRIDGE/'reuse_manifest.json',BRIDGE/'bridge.py',BRIDGE/'FILES_SHA256.json',*sorted((BRIDGE/'screen_jobs_draft').glob('*.json'))]
 before={str(p.relative_to(WS)).replace('\\','/'):sha(p) for p in protected}
 reuse=read(BRIDGE/'reuse_manifest.json');assert sha(SHARED/'manifest.json')==reuse['source_manifest_sha256'] and sha(SHARED/'final/accepted.json')==reuse['source_accepted_sha256']
 entries=[e for e in reuse['entries'] if e['source_job']['config']['seed']==91001 and e['source_job']['attack'] in ['Benign','S-DFA']];assert len(entries)==4
 invpath=WS/'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json';inventory={r['id']:r for r in read(invpath)['records']}
 idreceipt=read(IDS/'receipt.json');assert sha(IDS/'validation_predictions.npz')==idreceipt['prediction_arrays_sha256']
 # NpzFile is lazy: only these two image-ID members are materialized, never prediction/margin/label members.
 with np.load(IDS/'validation_predictions.npz',allow_pickle=False) as z:rootids=z['root_image_ids'];validids=z['valid_image_ids']
 assert len(rootids)==16277 and len(validids)==19867
 mapping={'root_image_id':rootids,'root_client_id':cohorts(rootids),'valid_image_id':validids,'valid_client_id':cohorts(validids)}
 rootsha=arrsha(rootids);validsha=arrsha(validids)
 assert rootsha==idreceipt['root_reconstruction']['root_image_ids_sha256'] and validsha==idreceipt['valid_image_ids_sha256']
 check_mapping(mapping,rootsha,validsha)
 archive=SHARED/'sharedcal700_and_baseline_gates_20260928.tar.gz';backup=read(SHARED/'sharedcal_backup.json');assert sha(archive)==backup['sha256']
 arcinv=read(SHARED/'backup_inventory.json');bindings=[];supports=[];baseline_root=None
 with tarfile.open(archive) as t:
  for e in entries:
   rec=inventory[e['id']];acc=e['accepted_record'];contract=rec['data_contract'];assert rec['checkpoint']['sha256']==acc['checkpoint_sha256'] and rec['result']['sha256']==acc['original_result_sha256']
   assert contract['root_image_ids_sha256']==rootsha and contract['evaluation_image_ids_sha256']==validsha
   assert rec['config']==e['source_job']['config'] and rec['source_hashes']==e['source_job']['source_hashes']
   member='results/revision_20260928/celeba_shared_calibration_v1/runs/'+e['id']+'/margins.npz';data=t.extractfile(member).read();assert hashlib.sha256(data).hexdigest()==acc['cache_sha256']==arcinv[member]['sha256'] and len(data)==arcinv[member]['bytes']
   # Raw file hashing includes labels/scores bytes, but these four NPZ members are never decoded.
   with np.load(io.BytesIO(data),allow_pickle=False) as z:
    assert set(z.files)=={'root_y','root_sensitive','root_margins','valid_y','valid_sensitive','valid_margins'}
    ry=z['root_y'];rs=z['root_sensitive']
   assert ry.shape==rs.shape==rootids.shape and set(ry)==set(rs)=={0,1}
   if baseline_root is None:baseline_root=(ry.copy(),rs.copy())
   else:assert np.array_equal(ry,baseline_root[0]) and np.array_equal(rs,baseline_root[1])
   rows=[]
   for cid in range(20):
    mask=mapping['root_client_id']==cid
    cells={f'y{y}_Male{s}':int(np.sum(mask&(ry==y)&(rs==s))) for y in [0,1] for s in [0,1]}
    rows.append(dict(cohort=cid,root_n=int(mask.sum()),valid_n=int(np.sum(mapping['valid_client_id']==cid)),**cells,missing_sensitive_group=any(not np.any(mask&(rs==s)) for s in [0,1]),missing_two_label_support=any(v==0 for v in cells.values())))
   supports.append(dict(id=e['id'],cell_id=e['cell_id'],rows=rows,all_40_cohort_sensitive_cells_have_two_root_labels=all(min(r[k] for k in ['y0_Male0','y0_Male1','y1_Male0','y1_Male1'])>0 for r in rows)))
   bindings.append(dict(id=e['id'],cell_id=e['cell_id'],cache_archive=str(archive),cache_archive_sha256=backup['sha256'],cache_member=member,cache_sha256=acc['cache_sha256'],checkpoint_reference=rec['checkpoint'],result_reference=rec['result'],source_job_reference=rec['raw_job'],config_sha256=rec['config_canonical_sha256'],source_hashes=rec['source_hashes'],root_image_ids_sha256=rootsha,valid_image_ids_sha256=validsha,root_y_array_sha256=arrsha(ry),root_Male_array_sha256=arrsha(rs),new_checkpoint_load_or_rehash=False))
 # Identity/order/reproduction checks exercise the actual arrays without any seed search.
 tests=[]
 for ids in [rootids,validids]:
  assert np.array_equal(cohorts(ids),cohorts(ids));assert np.array_equal(cohorts(ids[::-1])[::-1],cohorts(ids));tests.extend(['deterministic_recompute','order_independent_image_assignment'])
 for operation in [lambda:cohorts(rootids,DOMAIN+b'x'),lambda:cohorts(rootids.astype(float)),lambda:cohorts(np.array([1,1],dtype=np.int64)),lambda:check_mapping(dict(mapping,root_image_id=rootids[::-1]),rootsha,validsha),lambda:check_mapping(dict(mapping,root_client_id=(mapping['root_client_id']+1)%20),rootsha,validsha)]:
  try:operation()
  except AssertionError:tests.append('invalid_input_rejected')
  else:raise AssertionError('Invalid mapping not rejected')
 # All four contracts share exactly the ID order: mappings therefore identical across conditions.
 assert len({(b['root_image_ids_sha256'],b['valid_image_ids_sha256']) for b in bindings})==1
 tests.append('all_four_shared_image_assignments_identical')
 np.savez(HERE/'mapping.npz',**mapping)
 meta=dict(status='PREPARED_NOT_APPROVED',approved=False,semantics='declared_virtual_partition',true_training_client_identity=False,domain_utf8_with_null='GuardFed/LoGoFair/virtual-cohort/v1\\0',domain_hex=DOMAIN.hex(),rule='int.from_bytes(SHA256(domain bytes + ASCII canonical positive decimal image_id), big) mod 20',cohorts=20,mapping_sha256=sha(HERE/'mapping.npz'),root_image_ids_sha256=rootsha,valid_image_ids_sha256=validsha,input_fields_for_mapping=['image_id'],not_mapping_inputs=['label','Male','score','attack','distribution','seed'],domain_selection='Single fixed proposal; no search or selection by support/performance',population_meaning='20 virtual image-ID hash cohorts within the same clean root and central validation; not original federated training clients',execution_authorized=False)
 save(HERE/'mapping_metadata.json',meta);save(HERE/'population_support.json',supports)
 with (HERE/'cohort_counts.csv').open('x',newline='',encoding='utf-8') as f:
  rows=[dict(condition=s['cell_id'],**r) for s in supports for r in s['rows']];w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 save(HERE/'INPUT_BINDINGS.json',dict(cache_count=4,missing_caches=[],bindings=bindings,model_inventory_path=str(invpath),model_inventory_sha256=sha(invpath),id_source_path=str(IDS/'validation_predictions.npz'),id_source_sha256=sha(IDS/'validation_predictions.npz'),id_receipt_sha256=sha(IDS/'receipt.json'),original_bridge_files=before,fields_decoded_from_margin_cache=['root_y','root_sensitive'],fields_decoded_from_id_cache=['root_image_ids','valid_image_ids'],file_contains_valid_labels_and_scores=True,valid_labels_sensitive_scores_materialized=False,models_loaded=False,beta_fitted=False,performance_evaluated=False))
 for row in read(BRIDGE/'screen_jobs_draft/manifest.json')['jobs']:
  j=read(BRIDGE/'screen_jobs_draft'/row['job']);assert j['mapping_sha256'] is None and j['mapping_metadata_sha256'] is None
 assert before=={str(p.relative_to(WS)).replace('\\','/'):sha(p) for p in protected}
 save(HERE/'CHECKS.json',dict(status='POPULATION_PROPOSAL_CHECKS_PASS_NOT_APPROVED',tests=tests,all_four_root_support_complete=all(s['all_40_cohort_sensitive_cells_have_two_root_labels'] for s in supports),tie_risk='NOT_ASSESSED: no scores/thresholds decoded or fitted. Complete label support does not ensure distinct scores, Beta MLE convergence, absence of exact threshold ties, or useful classifier.',original32_mapping_hashes_remain_null=True,original_bridge_unchanged=True,CNN_or_beta_or_performance=False))
 print(json.dumps({'conditions':4,'root':len(rootids),'valid':len(validids),'support_complete':all(s['all_40_cohort_sensitive_cells_have_two_root_labels'] for s in supports),'root_min_max':[min(r['root_n'] for r in supports[0]['rows']),max(r['root_n'] for r in supports[0]['rows'])],'valid_min_max':[min(r['valid_n'] for r in supports[0]['rows']),max(r['valid_n'] for r in supports[0]['rows'])],'root_fourcell_min':min(r[k] for r in supports[0]['rows'] for k in ['y0_Male0','y0_Male1','y1_Male0','y1_Male1'])}))
if __name__=='__main__':main()
