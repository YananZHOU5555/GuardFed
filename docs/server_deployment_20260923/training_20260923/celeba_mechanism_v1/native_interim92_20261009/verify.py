"""Independent fsum/sample-SD verification; no model or scientific imports."""
import hashlib
import json
import math
from pathlib import Path

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
SOURCE=ROOT/'docs/server_deployment_20260923/training_20260923/server_reactivation_20261009/mechanism_science_backups_20261009/mechanism_inspection_v4_root_delta_20261009T171247Z/inspection.json'
OLD=ROOT/'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/interim_tables_20261009T144558Z'
METRICS=['accuracy_pct','aeod','aspd']

def load(p):return json.loads(p.read_text(encoding='utf-8-sig'))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(n,j):
    text=json.dumps(j,ensure_ascii=False,indent=2,allow_nan=False)+'\n';p=HERE/n
    if p.exists():assert p.read_text(encoding='utf-8')==text
    else:p.write_text(text,encoding='utf-8')

assert sha(SOURCE)=='cd357e05eca1ae6a7d0c9bca170484af4e31d208a96c493681d70206b34e163a'
renderer=ROOT/'tmp/render_mechanism_interim_tables_20261009.py'
evidence=ROOT/'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
assert sha(renderer)=='023553f7ed63a5c4f152dbdaa7e02a9f9073e9530d5a66553a0ec6bf22628247'
assert sha(evidence)=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
source=load(SOURCE);tables=load(HERE/'tables.json');old=load(OLD/'tables.json')
assert source['new_count']==92 and source['reused_count']==100 and not source['invalid']
assert tables['complete_paired_scenes']==9 and old['complete_paired_scenes']==7
assert tables['inspection_sha256']==sha(SOURCE) and tables['evidence_source_sha256']==sha(evidence)
records=source['records'];bycell={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in records}
assert len(bycell)==len(records)==192 and {r['variant'] for r in records}=={'Full','minus_U'}
partial=[r for r in records if r['variant']=='minus_U' and r['distribution']=='non-IID' and r['attack']=='Sp-DFA']
assert len(partial)==2 and {r['seed'] for r in partial}=={91001,91002}
assert [p['seeds'] for p in tables['panels']]==[list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]
differences=[];old_scalar_checks=0;display_checks=0
for panel,prior in zip(tables['panels'],old['panels']):
    assert panel['seeds']==prior['seeds'] and panel['label']==prior['label']
    seeds=panel['seeds'];assert len(panel['rows'])==27
    assert len({(r['distribution'],r['attack']) for r in panel['rows']})==9
    for row in panel['rows']:
        assert not (row['distribution']=='non-IID' and row['attack']=='Sp-DFA')
        assert row['n']==row['expected_n']==len(seeds) and row['complete'] and row['seeds']==seeds
        for metric in METRICS:
            if row['variant']=='minus_U minus Full':
                values=[bycell['minus_U',row['distribution'],row['attack'],s][metric]-bycell['Full',row['distribution'],row['attack'],s][metric] for s in seeds]
            else:
                values=[bycell[row['variant'],row['distribution'],row['attack'],s][metric] for s in seeds]
            mu=math.fsum(values)/len(values)
            sd=math.sqrt(math.fsum((x-mu)**2 for x in values)/(len(values)-1))
            differences.extend([abs(mu-row[metric]['mean']),abs(sd-row[metric]['sample_sd_ddof1'])])
            precision=3 if metric=='accuracy_pct' else 5
            assert f'{mu:.{precision}f}'==f"{row[metric]['mean']:.{precision}f}"
            assert f'{sd:.{precision}f}'==f"{row[metric]['sample_sd_ddof1']:.{precision}f}"
            display_checks+=2
    current={(r['distribution'],r['attack'],r['variant']):r for r in panel['rows']}
    for row in prior['rows']:
        actual=current[row['distribution'],row['attack'],row['variant']]
        for metric in METRICS:
            for stat in ['mean','sample_sd_ddof1']:
                assert f'{actual[metric][stat]:.10f}'==f'{row[metric][stat]:.10f}'
                old_scalar_checks+=1
oldlines=[l for l in (OLD/'TABLES.md').read_text(encoding='utf-8').splitlines() if l.startswith('| IID |') or l.startswith('| non-IID |')]
newlines=[l for l in (HERE/'TABLES.md').read_text(encoding='utf-8').splitlines() if l.startswith('| IID |') or l.startswith('| non-IID |')]
filtered=[l for l in newlines if not(l.startswith('| non-IID | FedSA |') or l.startswith('| non-IID | S-DFA |'))]
assert len(oldlines)==42 and filtered==oldlines and len(newlines)==54
assert len(differences)==486 and max(differences)<1e-12
paths=[SOURCE,SOURCE.with_suffix('.sha256'),SOURCE.with_name('statistics.json'),renderer,evidence,OLD/'tables.json',OLD/'TABLES.md']
save('INPUTS.json',{'files':{str(p.relative_to(ROOT)).replace('\\','/'):{'sha256':sha(p),'bytes':p.stat().st_size} for p in paths}})
save('verification.json',{'status':'NATIVE92_NINE_COMPLETE_SCENES_INDEPENDENT_FSUM_SAMPLESD_PASS','inspection_sha256':sha(SOURCE),'renderer_source_sha256':sha(renderer),'renderer_unchanged':True,'native_records':92,'Full_reference_records':100,'complete_paired_scenes':9,'panels':3,'scalar_mean_sd_checks':486,'max_abs_difference':max(differences),'formatted_scalar_checks_including_JSON_deltas':display_checks,'prior_seven_scene_scalars':old_scalar_checks,'prior_seven_scene_displayed_rows_exact':42,'displayed_new_rows':54,'partial_SpDFA_ids_retained_but_not_averaged':[r['id'] for r in partial],'new_three_view_inference':0,'three_view_nine_scenes_claimed':False,'new_training':0,'test':False})
print(json.dumps({'scalars':len(differences),'max_diff':max(differences),'old_display_rows':len(oldlines)}))
