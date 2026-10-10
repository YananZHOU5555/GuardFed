from pathlib import Path
import datetime,hashlib,json,shutil
R=Path(__file__).resolve().parents[1]
H=R/'tmp/fl_seven_scene_table_prepared_20261011'; C=H/'candidate'
O=R/'outputs/guardfed_tables/celeba_flgmm_seven_scenes70_20261011'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
read=lambda p:json.loads(Path(p).read_bytes())
assert not O.exists(), 'Fresh root adoption only'
assert sha(H/'ROOT_BINDING.json')=='6b7c9c58acb6a0bdc4353dd38c5c04ce064168404983ef6b38010c24031873b3'
b=read(H/'ROOT_BINDING.json'); root=read(R/b['root71'])
assert sha(R/b['root71'])==b['root71_sha256']=='fe400060961fb923cbac79443f735fab6824b06689d45fb3bbf8d56307c95421'
v=read(C/'VERIFICATION.json')
assert v['status']=='INDEPENDENT_RECEIPT_COUNTS_FSUM_SAMPLE_SD_DISPLAY_PASS'
assert (v['records'],v['complete_scene_records'],v['retained_partial_records'],v['scene_count'])==(71,70,1,7)
assert (v['metrics_recomputed_from_counts'],v['integer_base_counts_verified'],v['new_scene_statistical_scalars_recomputed'])==(90,240,54)
assert v['old324_statistics_preserved_not_recomputed'] and v['max_count_metric_absolute_difference']==0
assert v['max_fsum_statistics_absolute_difference']<=1e-14 and not v['test'] and not v['arrays_read']
assert all(v[k]==0 for k in ['new_inference','new_fit','new_training'])
for n,s in v['outputs_sha256'].items(): assert sha(C/n)==s,n
d=read(C/'records71.json'); t=read(C/'tables.json')
assert len(d['records'])==71 and d['table_records']==70 and len(t['scenes'])==7
assert root['records']+root['prior_interface_explicitly_reused']==[r['root_adoption_record'] for r in d['records'][:60]]+[d['records'][61+i]['root_adoption_record'] for i in range(10)]+[d['records'][60]['root_adoption_record']]
O.mkdir(parents=True)
for p in C.iterdir():
    assert p.is_file() and p.stat().st_size<4_000_000,p
    shutil.copyfile(p,O/p.name); assert sha(p)==sha(O/p.name)
report={'status':'ROOT_FLGMM_SEVEN_COMPLETE_SCENES70_TABLE_ADOPTED','verified_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'root_adoption':True,'source_root71_sha256':b['root71_sha256'],'source_binding_sha256':sha(H/'ROOT_BINDING.json'),'original_verification_sha256':sha(C/'VERIFICATION.json'),'records_preserved':71,'complete_scene_records':70,'retained_partial_records':1,'scenes':7,'views':t['views'],'fixed_seed_panels':t['fixed_seed_panels'],'new_metrics_from_counts':90,'new_integer_counts':240,'new_statistical_scalars':54,'old324_statistics_preserved_not_recomputed':True,'total_statistical_scalars':378,'Markdown_cells':189,'CSV_cells':189,'max_count_metric_difference':v['max_count_metric_absolute_difference'],'max_independent_fsum_statistic_difference':v['max_fsum_statistics_absolute_difference'],'prior_windows_failure_preserved':True,'whole_windows_refit_equality_claim':False,'arrays_read':False,'new_fit':0,'new_inference':0,'new_training':0,'final_test':False,'whole_rebuttal_complete':False,'adopted_outputs':{p.name:sha(p) for p in sorted(O.iterdir())},'root_adoption_source_sha256':sha(__file__)}
with (O/'ROOT_VERIFICATION.json').open('x',encoding='utf8',newline='\n') as f:json.dump(report,f,indent=2);f.write('\n')
print(json.dumps({'path':str(O/'ROOT_VERIFICATION.json'),'sha256':sha(O/'ROOT_VERIFICATION.json'),'status':report['status'],'records':71,'table_records':70}))
