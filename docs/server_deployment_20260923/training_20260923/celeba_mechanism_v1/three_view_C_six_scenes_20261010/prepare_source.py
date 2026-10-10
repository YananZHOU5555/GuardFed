from pathlib import Path
import hashlib,json,difflib
R=Path.cwd();H=R/'tmp/celeba_mechanism_C_six_scenes_prepared_20261010';O=R/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_C_five_scenes_20261010'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=(O/'panels.py').read_text('utf8');p=p.replace("('IID','Sp-DFA')]","('IID','Sp-DFA'),('non-IID','Benign')]").replace('Only exact C IID Benign10, F Flip10, FedSA10 S-DFA10 and Sp-DFA10 scenes are publishable','Only exact five IID scenes plus non-IID Benign10 are publishable')
(H/'panels.py').write_text(p,encoding='utf8',newline='\n')
v=(O/'verify_numeric.py').read_text('utf8');v=v.replace('len(records)==100','len(records)==120').replace("len({r['id'] for r in records})==100","len({r['id'] for r in records})==120").replace('len(bycell)==100','len(bycell)==120').replace("('IID','Sp-DFA')}","('IID','Sp-DFA'),('non-IID','Benign')}").replace('len(rows)==15','len(rows)==18').replace('len(errors)==810','len(errors)==972').replace('metric_checks==900 and count_checks==2400','metric_checks==1080 and count_checks==2880')
v=v[:v.index('\ndef main():')]+'''\ndef main():
    import argparse,json
    from pathlib import Path
    import build as b
    parser=argparse.ArgumentParser();parser.add_argument('--snapshot',type=Path,required=True);args=parser.parse_args()
    bindings=b.read(args.snapshot/'SOURCE_BINDINGS.json');b.verify_future_binding(bindings['actual_C4_binding'])
    b.verify_inputs()
    for name,pin in b.read(args.snapshot/'FILES_SHA256.json')['files'].items():b.need(b.sha(args.snapshot/name)==pin['sha256'],'Snapshot changed '+name)
    records=b.read(args.snapshot/'records.json')['records'];tables=b.read(args.snapshot/'tables.json')
    result=verify(records,tables['panels'])
    result['display_mean_sd_cells']=b.displayed_cells((args.snapshot/'TABLES.md').read_text('utf8'),tables['panels'])
    b.need(result['display_mean_sd_cells']==486,'Display scope changed')
    iid=[r for r in records if r['distribution']=='IID']
    aggregate=args.snapshot/'cross_scene_seed_first.json'
    b.need(aggregate.read_bytes()==(b.OLD/'snapshot/cross_scene_seed_first.json').read_bytes(),'Old IID aggregate bytes changed')
    result['preserved_IID_seed_first']=verify_aggregate(iid,b.read(aggregate)['panels'])
    result['status']='INDEPENDENT_STDLIB_NUMERIC_AND_DISPLAY_PASS_PENDING_ROOT_REVIEW'
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
'''
(H/'verify_numeric.py').write_text(v,encoding='utf8',newline='\n')
base=json.loads((O/'INPUTS.json').read_bytes());files=base['files'].copy()
for p in [O/'ROOT_VERIFICATION.json',O/'ACTUAL_FILES_SHA256.json']+list((O/'snapshot').glob('*'))+[O/n for n in ['build.py','panels.py','verify_numeric.py','INPUTS.json']]:
 if p.is_file():files[p.relative_to(R).as_posix()]=dict(sha256=sha(p),bytes=p.stat().st_size)
C=R/'tmp/celeba_mechanism_valid_C_after50_20261010';A=C/'execution_candidate/backups/incremental_20261010T002635Z/ROOT_ADOPTION_REVIEW.json'
for p in [C/'FILES_SHA256.json',C/'bridge.py',C/'inventory_actual156_Full100refs.json',C/'execution_candidate/EXECUTION_SOURCE_SHA256.json',A]:files[p.relative_to(R).as_posix()]=dict(sha256=sha(p),bytes=p.stat().st_size)
inputs=dict(status='PREPARED_ONLY_FUTURE_EXACT4_SOURCE_AND_ADOPTION_REQUIRED',full900_path=base['full900_path'],files=files,prior_C6_adoption=A.relative_to(R).as_posix(),prior_C6_adoption_sha256=sha(A),future_C4_stage=None,future_C4_source_sha256=None,future_C4_inventory_sha256=None,future_C4_execution_sha256=None,future_C4_adoption=None,future_C4_adoption_sha256=None,expected_added_ids=[f'minus_C_non-IID_Benign_seed{s}' for s in range(91007,91011)],expected_records=120,expected_scene_scalars=972,expected_cells=486,expected_receipt_metrics=1080,expected_confusion_counts=2880,old_IID_seed_first_scalars_preserved=162,new_statistics_generated=False)
(H/'INPUTS.json').write_text(json.dumps(inputs,indent=2)+'\n',encoding='utf8')
