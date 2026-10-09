"""Adopt the actual single C scene; original U100 and frozen source stay unchanged."""
from pathlib import Path
import argparse,datetime,hashlib,json,statistics

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_mechanism_C_three_view_table_prepare_20261009'
OUT=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_Benign10_20261009'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
parser=argparse.ArgumentParser();parser.add_argument('--independent-review',type=Path,required=True)
parser.add_argument('--independent-sha256',required=True);args=parser.parse_args()
assert not OUT.exists() and sha(args.independent_review)==args.independent_sha256
assert sha(BASE/'FINAL_FILES_SHA256.json')=='68ee93369bb5d45bcb2063fd4bb1b1995e2a7b05abee4e78b9cc8a174a801f31'
seal=read(BASE/'FINAL_FILES_SHA256.json');assert len(seal['members'])==29
for item in seal['members']:
    assert sha(BASE/item['path'])==item['sha256'] and (BASE/item['path']).stat().st_size==item['bytes']
for path,pin in read(BASE/'INPUTS.json')['files'].items():
    assert sha(ROOT/path)==pin['sha256'] and (ROOT/path).stat().st_size==pin['bytes']
binding=read(BASE/'snapshot/SOURCE_BINDINGS.json')
assert sha(Path(binding['actual_root_review']))==binding['actual_root_review_sha256']=='8d064687ad7841e1050120a12ada9e10458fea5bb6a4d9da5aeed77584437fc5'
records=read(BASE/'snapshot/records.json')['records']
index={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
assert len(records)==len(index)==24
assert set(index)=={(v,'IID','Benign',s) for v in ('Full','minus_C') for s in range(91001,91011)}|{(v,'IID','F Flip',s) for v in ('Full','minus_C') for s in (91001,91002)}
table=read(BASE/'snapshot/tables.json');assert table['complete_scenes']==1 and table['partial_pairs']==2
assert table['table_model_records']==20 and not table['primary_endpoint_selected'] and not table['final_test']
errors=[];cells=[]
for panel in table['panels']:
    view=panel['view'];seeds=panel['seeds']
    assert seeds in (list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011)))
    assert len(panel['rows'])==3
    for row in panel['rows']:
        assert (row['distribution'],row['attack'])==('IID','Benign')
        assert row['n']==row['expected_n']==len(seeds) and row['complete']
        for metric,precision in (('accuracy_pct',3),('aeod',5),('aspd',5)):
            def value(v,s):
                metrics=index[v,'IID','Benign',s]['views'][view]
                return metrics['accuracy']*100 if metric=='accuracy_pct' else metrics[metric]
            values=[value('minus_C',s)-value('Full',s) if row['variant']=='minus_C minus Full' else value(row['variant'],s) for s in seeds]
            mean,sd=statistics.mean(values),statistics.stdev(values)
            errors.extend((abs(mean-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])))
            cells.append(f'{mean:.{precision}f} ± {sd:.{precision}f}')
assert len(errors)==162 and max(errors)<=1e-12 and len(cells)==81
text=(BASE/'snapshot/TABLES.md').read_text(encoding='utf-8')
assert text.count(' ± ')==82 and all(cell in text for cell in cells)
native=read(ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/native_C_Benign10_20261009/tables.json')
native_panels=[p for p in table['panels'] if p['view']=='native']
assert len(native_panels)==len(native['panels'])==3
for actual,old in zip(native_panels,native['panels']):
    assert actual['seeds']==old['seeds'] and actual['rows']==old['rows']
assert all(r['views']['native']==r['views']['shared_calibration'] for r in records)
review=read(args.independent_review)
assert review['status']=='PASS_INDEPENDENT_C_SINGLE_SCENE_THREE_VIEWS_SOURCE_AND_ARITHMETIC'
assert (review['independent_mean_sd_scalars'],review['independent_display_cells'],review['metric_checks_from_confusion_counts'],review['base_confusion_count_fields'],review['native_table_scalar_checks'])==(162,81,216,576,54)
assert review['actual_output_members_verified']==29 and review['prepared_source_pins_verified']==25
assert review['actual_C11_ROOT_sha256']==binding['actual_root_review_sha256']
assert review['original_U10022_sealed_members_unchanged'] and review['maximum_statistic_abs_difference']<=1e-12
assert review.get('new_CNN_inference',review.get('new_CNN',0))==0
OUT.mkdir(parents=True)
for item in seal['members']+[dict(path='FINAL_FILES_SHA256.json',sha256=sha(BASE/'FINAL_FILES_SHA256.json'))]:
    target=OUT/item['path'];target.parent.mkdir(parents=True,exist_ok=True)
    target.write_bytes((BASE/item['path']).read_bytes());assert sha(target)==item['sha256']
(OUT/'INDEPENDENT_REVIEW.json').write_bytes(args.independent_review.read_bytes())
proof=dict(status='ROOT_C_SINGLE_SCENE_THREE_VIEW_SOURCE_PAIRS_STATISTICS_ADOPTED',
    utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),source=str(BASE.relative_to(ROOT)),
    source_seal_sha256=sha(BASE/'FINAL_FILES_SHA256.json'),source_members_verified=29,
    C11_root_adoption_sha256=binding['actual_root_review_sha256'],independent_review_sha256=args.independent_sha256,
    source_input_pins_verified=25,complete_scenes=1,scene='IID Benign',variant='minus_C',paired_seeds=10,
    preserved_records=24,displayed_records=20,partial_F_Flip_pairs_excluded=2,seed_panels=[10,9,6],
    mean_SD_scalars=162,display_cells=81,max_abs_statistic_difference=max(errors),native_scalars_exact=54,
    native_shared_records_exact=24,original_U100_unchanged=True,other_C_scenes_complete=False,
    new_CNN=0,new_Full_inference=0,new_training=0,test=False,primary_endpoint_selected=False,
    mixed_devices_environment_selection_and_test_history_retained=True,necessity_or_causal_claim=False,
    goal_complete=False,table_path=(OUT/'snapshot/TABLES.md').relative_to(ROOT).as_posix())
with (OUT/'ROOT_VERIFICATION.json').open('x',encoding='utf-8',newline='\n') as stream:
    json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=proof['status'],root_proof_sha256=sha(OUT/'ROOT_VERIFICATION.json'),directory=str(OUT))))
