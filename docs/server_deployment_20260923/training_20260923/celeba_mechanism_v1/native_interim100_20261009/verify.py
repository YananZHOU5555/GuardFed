"""Independent fsum/sample-SD checks of all540 scalars and exact old-nine displays."""
import copy
import hashlib
import json
import math
from pathlib import Path
import build

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[1]
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    basis=read(HERE/'INPUTS.json')
    for name,pin in basis['files'].items():assert sha(ROOT/name)==pin['sha256']
    inspection=read(ROOT/basis['inspection']);rows=build.validate_scope(inspection)
    tables=read(HERE/'rendered/tables.json');old=read(ROOT/basis['prior_tables'])
    indexed={(r['variant'],r['distribution'],r['attack'],r['seed']):r for r in rows}
    assert len(indexed)==204
    checks=0;maximum=0.;old_checks=0
    for panel,prior,seeds in zip(tables['panels'],old['panels'],[list(range(91001,91011)),list(range(91002,91011)),list(range(91005,91011))]):
        assert panel['seeds']==prior['seeds']==seeds and len(panel['rows'])==30
        assert {(r['distribution'],r['attack']) for r in panel['rows']}==build.SCENES
        assert len({(r['distribution'],r['attack'],r['variant']) for r in panel['rows']})==30
        for row in panel['rows']:
            assert row['n']==row['expected_n']==len(seeds) and row['complete'] and row['seeds']==seeds
            for metric in build.METRICS:
                values=[]
                for seed in seeds:
                    key=(row['distribution'],row['attack'],seed)
                    values.append(indexed[('minus_U',*key)][metric]-indexed[('Full',*key)][metric] if row['variant']=='minus_U minus Full' else indexed[(row['variant'],*key)][metric])
                mean=math.fsum(values)/len(values)
                sd=math.sqrt(math.fsum((x-mean)**2 for x in values)/(len(values)-1))
                for stat,value in [('mean',mean),('sample_sd_ddof1',sd)]:
                    saved=row[metric][stat];assert f'{value:.10f}'==f'{saved:.10f}'
                    precision=3 if metric=='accuracy_pct' else 5
                    assert f'{value:.{precision}f}'==f'{saved:.{precision}f}'
                    maximum=max(maximum,abs(value-saved));checks+=1
        current={(r['distribution'],r['attack'],r['variant']):r for r in panel['rows']}
        for row in prior['rows']:
            assert current[row['distribution'],row['attack'],row['variant']]==row
            old_checks+=6
    assert checks==540 and old_checks==486 and maximum<1e-12
    oldlines=[l for l in (ROOT/basis['prior_markdown']).read_text(encoding='utf-8').splitlines() if l.startswith('| IID |') or l.startswith('| non-IID |')]
    newlines=[l for l in (HERE/'rendered/TABLES.md').read_text(encoding='utf-8').splitlines() if l.startswith('| IID |') or l.startswith('| non-IID |')]
    assert len(oldlines)==54 and len(newlines)==60 and [l for l in newlines if not l.startswith('| non-IID | Sp-DFA |')]==oldlines
    paired=read(HERE/'paired_differences.json');assert len(paired['records'])==len(paired['checkpoint_pairs'])==100
    paired_checks=0
    for row in paired['records']:
        key=row['distribution'],row['attack'],row['seed']
        for metric in build.METRICS:
            assert row[metric]==indexed[('minus_U',*key)][metric]-indexed[('Full',*key)][metric];paired_checks+=1
    for pair in paired['checkpoint_pairs']:
        key=pair['distribution'],pair['attack'],pair['seed']
        assert pair['checkpoint_sha256']==indexed[('minus_U',*key)]['checkpoint_sha256'] and pair['Full_checkpoint_sha256']==indexed[('Full',*key)]['checkpoint_sha256']
    coverage=read(HERE/'coverage.json');assert len(coverage['accepted_new104_ids'])==104 and coverage['original_record_count']==204
    c=next(r for r in coverage['variant_coverage'] if r['variant']=='minus_C');assert c['n']==4 and not c['complete']
    refusals=[]
    for name,mutate in [('drop_seed',lambda j:j['records'].pop(next(i for i,r in enumerate(j['records']) if r['variant']=='minus_U'))),('duplicate_record',lambda j:j['records'].append(copy.deepcopy(j['records'][0]))),('erase_C_partial',lambda j:j.__setitem__('records',[r for r in j['records'] if r['variant']!='minus_C'])),('mix_native_source',lambda j:j.__setitem__('source_script_sha256','0'*64))]:
        bad=copy.deepcopy(inspection);mutate(bad)
        try:build.validate_scope(bad)
        except AssertionError:refusals.append(name)
        else:raise AssertionError('Must reject '+name)
    out=dict(status='NATIVE100_TEN_SCENE_INDEPENDENT_STATISTICS_AND_IDENTITIES_PASS',inspection_sha256=sha(ROOT/basis['inspection']),statistic_source_sha256=sha(ROOT/basis['evidence']),renderer_source_sha256=sha(ROOT/basis['renderer']),original_renderer_unchanged=True,
        Full_records=100,minus_U_records=100,minus_C_partial_records=4,accepted_new_total=104,original_records=204,complete_scenes=10,panels=3,table_rows=90,
        scalar_mean_sd_checks=checks,max_abs_difference=maximum,decimal_precision_checked=10,prior_nine_scalar_values_exact=486,prior_nine_display_rows_exact=54,new_displayed_rows=60,
        unique_checkpoint_pair_ids=100,panel_paired_counts=[100,90,60],per_seed_delta_scalar_checks=paired_checks,checkpoint_pair_bindings=100,refusals=refusals,
        new_CNN_inference=0,new_training=0,test=False,three_view100_claimed=False,other_variants_complete=False,primary_endpoint='PENDING_USER',whole_rebuttal_complete=False)
    with (HERE/'verification.json').open('x',encoding='utf-8',newline='\n') as f:json.dump(out,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(dict(status=out['status'],checks=checks,max_abs_difference=maximum,old_exact=old_checks,paired=100,refusals=len(refusals))))


if __name__=='__main__':main()
