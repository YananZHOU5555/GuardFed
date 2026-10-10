"""Original frozen rank() on 27 root-adopted plus 5 strictly verified records; no model inference."""
from pathlib import Path
import datetime,hashlib,json,runpy,sys
sys.dont_write_bytecode=True
B=Path(__file__).resolve().parent;A=B/'attempt_v2';ROOT=B.parents[1]
H=ROOT/'tmp/celeba_hybrid_screen_execution_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
def save(name,value):
    with (B/name).open('x',encoding='utf8',newline='\n') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
F=Path(read(A/'RAW_STORAGE_LOCATION.json')['directory'])
off=read(F/'OFFSERVER_MEMBER_TENSOR_PROOF.json');record=read(F/'LOCAL_RECORD_CHECKS.json')
assert off['accepted_new']==5 and len(record['records'])==5
assert off['status']=='ORIGINAL_SERVER_STRICT_PLUS_OFFSERVER_ALL_MEMBERS_AND_CPU_TENSORS_VERIFIED'
assert record['status']=='RECORD_BOUND_ORIGINAL_SCIENTIFIC_AND_WRITER_CHECKS_PASS'
pins=[];old_records={};chain=H/read(B/'PREVIOUS_LATEST.json')['chain_file'];expected=sha(B/'PREVIOUS_CHAIN.json')
while True:
    assert sha(chain)==expected
    c=read(chain);delta=H/c['delta_dir'];proof=delta/'PARTIAL_ACCEPTANCE.json'
    assert sha(proof)==c['server_strict_sha256']
    original=read(proof);assert original['source_seal_sha256']=='2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f'
    if 'root_adoption_path' in c:
        root=ROOT/c['root_adoption_path'];assert sha(root)==c['root_adoption_sha256']
        local=delta/'OFFSERVER_MEMBER_TENSOR_PROOF.json';assert sha(local)==c['offserver_proof_sha256']
    else:
        root=delta/'ROOT_RECORD_REVIEW.json';assert sha(root)==c['root_record_review_sha256']
        local=delta/'OFFSERVER_MEMBER_TENSOR_PROOF.json';assert sha(local)==c['offserver_tensor_sha256']
    pins.append(dict(chain=str(chain),chain_sha256=sha(chain),strict_path=str(proof),strict_sha256=sha(proof),offserver_path=str(local),offserver_sha256=sha(local),root_path=str(root),root_sha256=sha(root)))
    assert set(original['accepted_new_ids'])==set(c['accepted_new_ids'])=={r['id'] for r in original['records']}
    for row in original['records']:assert row['id'] not in old_records;old_records[row['id']]=row
    if not c.get('previous_chain_file'):break
    chain=H/c['previous_chain_file'];expected=c['previous_chain_sha256']
previous=read(B/'PREVIOUS_CHAIN.json');assert len(old_records)==27 and set(old_records)==set(previous['accepted_job_ids'])
server=read(A/'PARTIAL_ACCEPTANCE.json');new={r['id']:r for r in server['records']}
assert len(new)==5 and not set(new)&set(old_records) and set(new)==set(read(B/'EXACT_DELTA.json')['selected_ids'])
all_records=dict(old_records,**new);scope=read(H/'screen_scope.json');protocol=read(H/'runtime_protocol.json')
assert sha(H/'summarize.py')=='47a83a21b35983b6fab1d499784df15e01f81792a795b2ed3eb06361eec4403f'
assert sha(H/'runtime_protocol.json')=='bcc66477d22096eaf647e31f065f59ed6727716dcf01db814d5edcbe4ad525f1'
assert protocol['score']=='accuracy - .35*(.45*aeod + .45*aspd + .10*max(aeod,aspd)) - .10*max(0,max(aeod,aspd)-.06)'
rows=[]
for entry in scope['jobs']:
    job=read(H/entry['job']);source=all_records[entry['id']]
    assert sha(H/entry['job'])==entry['job_sha256']==source['original_provenance']['job_sha256']
    assert source['rounds']==70 and source['seed']==91001
    assert (source['distribution'],source['attack'],source['alpha'])==(job['distribution'],job['attack'],job['config']['client_alpha'])
    assert source['evaluation_stats']['prediction_count']==19867
    rows.append(dict(id=entry['id'],candidate=job['tuning_candidate'],distribution=job['distribution'],attack=job['attack'],seed=91001,metrics=source['metrics'],checkpoint_sha256=source['model_sha256'],acceptance_sha256=source['acceptance_sha256']))
assert len(rows)==len({r['id'] for r in rows})==32
sys.path.insert(0,str(H));original=runpy.run_path(str(H/'summarize.py'))
result=original['rank'](rows,protocol['candidates'])
summary=dict(status='ALL32_ORIGINAL_STRICT_OFFSERVER_VERIFIED_ORIGINAL_FROZEN_RANK_ROOT_PENDING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**result,
    seed_n=1,no_sample_std_or_significance=True,four_conditions_not_four_seeds=True,all_negative_results_preserved=True,final_test_evaluated=False,formal_multi_seed_records=0,
    source_seal_sha256=sha(H/'FILES_SHA256.json'),original_summarize_sha256=sha(H/'summarize.py'),runtime_protocol_sha256=sha(H/'runtime_protocol.json'),
    previous_root_adopted=27,new_strict_offserver_verified=5,new_root_adopted=0,source_bindings=pins,new_strict_sha256=sha(A/'PARTIAL_ACCEPTANCE.json'),new_offserver_sha256=sha(F/'OFFSERVER_MEMBER_TENSOR_PROOF.json'),new_record_check_sha256=sha(F/'LOCAL_RECORD_CHECKS.json'),
    no_repeated_strict_for_old27=True,no_CNN_or_training=True,original_main_not_called=True,selection_algorithm='Unmodified original rank/score; four condition scores averaged per candidate; exact ties by candidate ID lexicographic order.',
    limits=['One seed91001 exploratory screen, no sample SD/significance or formal100 claim.','No saved prediction/probability arrays supplied; no array recomputation claimed.','Original CUDA server strict and CPU offserver member/tensor/record checks remain different runtime roles.','Selected recipe is the frozen screen outcome pending root adoption; no formal100 was launched.'])
save('SUMMARY32.json',summary)
lines=['# Hybrid32 frozen validation screen — root review pending','','Eight candidates; four conditions each; one seed91001. Original score is computed per condition and then averaged over four conditions. Exact ties use candidate ID lexical order. No SD, significance, test, or formal100 claim.','','| Candidate | Mean ACC | Mean AEOD | Mean ASPD | Mean frozen score |','|---|---:|---:|---:|---:|']
for row in result['all_candidates']:
    m=row['mean_four_condition_metrics'];lines.append(f"| {row['candidate']} | {m['accuracy']:.9f} | {m['aeod']:.9f} | {m['aspd']:.9f} | {row['mean_four_condition_score']:.9f} |")
lines+=['',f"Frozen score selection: `{result['selected_recipe']}`.",f"ACC champion: `{result['accuracy_champion']}`.",'Three-metric Pareto: '+', '.join('`'+x+'`' for x in result['three_metric_Pareto'])+'.','','All conditions and negative results remain in SUMMARY32.json. Shared state and recipe files were not edited.']
with (B/'SUMMARY32.md').open('x',encoding='utf8',newline='\n') as f:f.write('\n'.join(lines)+'\n')
print(json.dumps(dict(summary_sha256=sha(B/'SUMMARY32.json'),records=32,candidates=8,selected_recipe=result['selected_recipe'],accuracy_champion=result['accuracy_champion'],Pareto=result['three_metric_Pareto'])))
