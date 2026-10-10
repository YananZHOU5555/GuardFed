"""One root-adopted Hybrid IID/Benign native scene;fixed10/9/6,original mean/sampleSD."""
from pathlib import Path
import hashlib,json,statistics
ROOT=Path(__file__).resolve().parents[2];HERE=Path(__file__).resolve().parent
OUT=ROOT/'outputs/guardfed_tables/celeba_hybrid_IID_Benign10_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
PINS={
'tmp/celeba_hybrid_native9_root_adoption_20261011/ROOT_ADOPTION.json':'1434b40de5116bf3d53bc5a6ae2bd3b90f4e54222f23ad099ff72b1fd3ef1775',
'tmp/celeba_hybrid_first1_root_adoption_20261010/ROOT_ADOPTION.json':'04d62c367609d1c6d079ff538f45a575bff368944d4833f47fd1cf28159c5fa4',
'tmp/hybrid_native_after1_20261011/IID_BENIGN10_NATIVE.json':'3c471be49e352faa93a3e2c0f15d45cff79ef31d2a65fd97c4cd8ba40519f1b4',
'tmp/hybrid_native_after1_20261011/DELIVERY_FILES_SHA256.json':'da25f1792d677a81f057e12a699d8b113f919282d442792dfeefd4f521e95703',
'tmp/hybrid_native_after1_20261011/ROOT_BOUND_ADOPTION.json':'41e9d664e17105b84b1266824fffec0cfef7b7e66a4fa7cfca0cfc168c3f10cf',
'tmp/celeba_hybrid32_final_collection_20261010/SUMMARY32.json':'46b5f8fdca9536166ed868e50d4c7bc2578f8a1100ffea97878cf044c95748ae'}
for p,v in PINS.items():assert sha(ROOT/p)==v
assert not OUT.exists()
root=read(ROOT/next(iter(PINS)));prior=read(ROOT/'tmp/celeba_hybrid_first1_root_adoption_20261010/ROOT_ADOPTION.json')
old=read(ROOT/'tmp/hybrid_native_after1_20261011/IID_BENIGN10_NATIVE.json');bound=read(ROOT/'tmp/hybrid_native_after1_20261011/ROOT_BOUND_ADOPTION.json');summary=read(ROOT/'tmp/celeba_hybrid32_final_collection_20261010/SUMMARY32.json')
assert root['status']=='ROOT_HYBRID_EXACT8_ORIGINAL_STRICT_OFFSERVER_CHAIN_ADOPTED' and root['cumulative_accepted']==9 and root['archive_members']==272
assert root['previous_root_sha256']==PINS['tmp/celeba_hybrid_first1_root_adoption_20261010/ROOT_ADOPTION.json'] and root['delivery_seal_sha256']==PINS['tmp/hybrid_native_after1_20261011/DELIVERY_FILES_SHA256.json']
assert (root['rounds'],root['n_eval'],root['evaluation_split'])==(70,19867,'valid') and root['all_metrics_same_terminal_checkpoint'] and not root['final_test']
winner_index=next(i for i,x in enumerate(summary['all_candidates']) if x['candidate']==bound['selected_recipe']['id'])
winner=summary['all_candidates'][winner_index];reuse_index=next(i for i,x in enumerate(winner['records']) if (x['distribution'],x['attack'],x['seed'])==('IID','Benign',91001));reuse=winner['records'][reuse_index]
assert summary['selected_recipe']==bound['selected_recipe']['id'] and bound['old_four_explicit_reuse'] and old['records'][0]['root32_sha256']==bound['root32_sha256']
new={x['id']:x for x in root['records']};delivery=ROOT/'tmp/hybrid_native_after1_20261011';extra=read(delivery/'RECORDS8.json')['records'];extra_by={x['id']:x for x in extra}
records=[]
for source in old['records']:
 seed=source['seed'];ID=source['id']
 if seed==91001:
  actual=reuse;role='root_bound_original_screen_reuse';path='tmp/celeba_hybrid32_final_collection_20261010/SUMMARY32.json';pointer=f'/all_candidates/{winner_index}/records/{reuse_index}';runtime=None
 elif seed==91002:
  assert prior['accepted_new_ids']==[ID];actual=dict(id=ID,metrics=prior['metrics'],checkpoint_sha256=prior['checkpoint_sha256'],acceptance_sha256=read(delivery/'PRIOR_OFFSERVER.json')['local']['records'][0]['acceptance_sha256']);role='prior_root_adopted_formal1';path='tmp/celeba_hybrid_first1_root_adoption_20261010/ROOT_ADOPTION.json';pointer='/';runtime=None
 else:
  actual=new[ID];role='actual_root_adopted_exact8';path=next(iter(PINS));pointer=f"/records/{next(i for i,x in enumerate(root['records']) if x['id']==ID)}";p=extra_by[ID]['provenance'];runtime={k:p[k] for k in ['torch','cuda_build','device','cpu_threads','cpu_affinity','gpu_uuid','gpu_name']}
 assert actual['id']==ID and actual['metrics']==source['metrics'] and actual['checkpoint_sha256']==source['checkpoint_sha256']
 records.append(dict(id=ID,seed=seed,distribution='IID',attack='Benign',method='CosineFairnessHybrid',rounds=70,evaluation_split='valid',n_eval=19867,metrics=actual['metrics'],checkpoint_sha256=actual['checkpoint_sha256'],acceptance_sha256=actual['acceptance_sha256'],source_role=role,source_path=path,source_sha256=PINS[path],source_pointer=pointer,selection_seed=seed==91001,training_runtime=runtime))
assert [x['seed'] for x in records]==list(range(91001,91011)) and len({x['id'] for x in records})==len({x['checkpoint_sha256'] for x in records})==10
METRICS=['accuracy','aeod','aspd'];panels=[]
for name,seeds in [('all10',list(range(91001,91011))),('exclude_selection_seed9',list(range(91002,91011))),('fixed_last6',list(range(91005,91011)))]:
 chosen=[x for x in records if x['seed'] in seeds];assert [x['seed'] for x in chosen]==seeds
 values={k:dict(mean=statistics.mean(x['metrics'][k] for x in chosen),sample_SD=statistics.stdev(x['metrics'][k] for x in chosen),n=len(seeds),ddof=1) for k in METRICS}
 display={k:f"{values[k]['mean']*(100 if k=='accuracy' else 1):.{6 if k=='accuracy' else 8}f} ± {values[k]['sample_SD']*(100 if k=='accuracy' else 1):.{6 if k=='accuracy' else 8}f}" for k in METRICS}
 panels.append(dict(name=name,n=len(seeds),seeds=seeds,ids=[x['id'] for x in chosen],checkpoint_sha256=[x['checkpoint_sha256'] for x in chosen],statistics=values,display=display))
OUT.mkdir(parents=True)
def save(name,v):
 with (OUT/name).open('x',encoding='utf8',newline='\n') as f:json.dump(v,f,indent=2);f.write('\n')
save('INPUTS.json',dict(status='ACTUAL_ADOPTED_NATIVE_INPUTS',files=PINS,builder_path=Path(__file__).relative_to(ROOT).as_posix(),builder_sha256=sha(Path(__file__)),record8_path='tmp/hybrid_native_after1_20261011/RECORDS8.json',record8_sha256=sha(delivery/'RECORDS8.json'),prior_offserver_sha256=sha(delivery/'PRIOR_OFFSERVER.json')))
save('records.json',dict(records=records,original_pending_ten_record_identity_and_metrics_exact=True,all_metrics_same_terminal_checkpoint=True,root_native9_sha256=PINS[next(iter(PINS))],reused_original_screen1=True))
save('tables.json',dict(status='BOUNDED_IID_BENIGN_NATIVE_TEN_SEED_TABLE',view='original_native_terminal_argmax',metric_units=dict(accuracy='fraction;Markdown displays percent',aeod='fraction',aspd='fraction'),panels=panels,mean_SD_scalars=18,display_cells=9))
save('COVERAGE.json',dict(unique_records=10,unique_checkpoints=10,complete_scenes=1,distribution='IID',attack='Benign',formal_new_root_accepted=9,reused_screen_records=1,total_search_reuse_separate=4,panels=[dict(name=p['name'],n=p['n'],seeds=p['seeds'],ids=p['ids']) for p in panels],partial_scenes_included=False,whole100_complete=False,three_view_table=False))
save('ENVIRONMENT_SCOPE.json',dict(recipe=bound['selected_recipe'],selection_seed=91001,selection_conditions='Original four exposed-valid screen conditions;fixed winner reused without reselection',prior_official_test_exposure=True,new_final_test_evaluation=False,training_runtime_reported_exactly_for_records_with_provenance=8,unreported_runtime_for_reuse91001_and_prior91002='Not reread or inferred here;their accepted checkpoint/source/runtime checks remain in original root chains.',server_saved_check='Original cu128 CPU-tensor record checker,CUDA hidden',local_saved_check='Windows torch2.8.0+cpu;source/runtime metadata bridge,no numerical-runtime equivalence',prediction_arrays_supplied=False,prediction_arrays_recomputed=False,new_CNN=0,new_fit=0,new_training=0,negative_results_retained=True,paired_comparison=False,significance_claim=False,necessity_claim=False,whole_method100_claim=False,final_endpoint_adoption=False))
md=['# Hybrid native validation: IID / Benign','', 'Frozen project CNN Hybrid adaptation (CosineFairnessHybrid),recipe learning_rate=0.001,fairness_lambda=20.0,threshold=0.1;native valid-only terminal outputs. The recipe is unchanged from the original four-condition search.', '', 'Mean ± sample SD (ddof=1), using each seed’s same terminal70-round checkpoint for all three metrics. ACC is displayed in percent; AEOD and ASPD remain fractions. All source checkpoints and seed IDs are retained in [records.json](records.json); fixed panels are listed in [COVERAGE.json](COVERAGE.json).','', '| Fixed panel | n | ACC (%) ↑ | AEOD ↓ | ASPD ↓ |','|---|---:|---:|---:|---:|']
for p in panels:md.append(f"| {p['name']} | {p['n']} | {p['display']['accuracy']} | {p['display']['aeod']} | {p['display']['aspd']} |")
md+=['','Ten seeds = original selected-screen reuse91001 + nine root-adopted formal records91002–91010. The9seed panel excludes selection seed91001;the6seed panel is fixed91005–91010. The panels use identical seed/checkpoint sets across all metrics;all outcomes remain available.','', 'These are exposed-validation development results. Seed91001 participated in the original four-condition recipe search. Prior official-test results had been viewed;no new final-test evaluation is performed here. This is one complete native scene,not whole100 completion,a three-view comparison,statistical significance or a claim of superiority. Training provenance and saved-check platform limits are recorded in [ENVIRONMENT_SCOPE.json](ENVIRONMENT_SCOPE.json);no uniform unmeasured runtime is asserted.','', 'Sources and actual root acceptance pins: [INPUTS.json](INPUTS.json). The earlier pending auxiliary table remains unchanged;this table uses the subsequently adopted native9 chain.']
(OUT/'TABLES.md').write_text('\n'.join(md)+'\n',encoding='utf8',newline='\n')
print(json.dumps(dict(output=OUT.as_posix(),records=10,panels=3,scalars=18,cells=9)))
