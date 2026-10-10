"""Independent fsum/sample-variance checks of an existing table snapshot; no inference."""
import argparse, hashlib, json, math
from pathlib import Path

HERE=Path(__file__).resolve().parent
METRICS={'accuracy':(100,2),'aeod':(1,4),'aspd':(1,4)}
SEEDS={'ten':list(range(91001,91011)),'nonselection_nine':list(range(91002,91011)),'matching_six':list(range(91005,91011))}
read=lambda p:json.loads(Path(p).read_bytes())
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()


def calculate(values):
    n=len(values);assert n>1
    mu=math.fsum(values)/n
    return mu, math.sqrt(math.fsum((x-mu)**2 for x in values)/(n-1))


def verify(folder):
    assert folder.resolve().is_relative_to(HERE) and not (folder/'NUMERIC_VERIFICATION.json').exists()
    rows=read(folder/'records_native_1000.json')['records'];tables=read(folder/'summary_statistics.json');aggregates=read(folder/'seed_first_aggregates.json')
    assert len(rows)==1000 and len({r['id'] for r in rows})==1000
    scalars=cells=aggregate_scalars=seed_means=0;max_error=0.0
    for panel,seeds in SEEDS.items():
        assert len(tables[panel])==100 and len(aggregates[panel])==30
        for g in tables[panel]:
            selected=[r for r in rows if (r['method'],r['distribution'],r['attack'])==(g['method'],g['distribution'],g['attack']) and r['seed'] in seeds]
            assert len(selected)==len(seeds) and {r['seed'] for r in selected}==set(seeds)
            assert g['IDs']==[r['id'] for r in selected] and g['seeds']==seeds and g['n']==len(seeds)
            for k,(scale,precision) in METRICS.items():
                mu,sd=calculate([r['metrics'][k] for r in selected])
                errors=[abs(mu-g[k]['mean']),abs(sd-g[k]['sample_sd'])];assert max(errors)<=1e-12
                assert g[k]['display']==f'{mu*scale:.{precision}f} ± {sd*scale:.{precision}f}'
                scalars+=2;cells+=1;max_error=max(max_error,*errors)
        for g in aggregates[panel]:
            dist=['IID','non-IID'] if g['scope']=='balanced_all10' else [g['scope']]
            assert g['n']==len(seeds) and [r['seed'] for r in g['seed_first']]==seeds
            for k in METRICS:
                values=[]
                for seed,reported in zip(seeds,g['seed_first']):
                    chosen=[r for r in rows if r['method']==g['method'] and r['distribution'] in dist and r['seed']==seed]
                    assert len(chosen)==reported['n_scenes']==5*len(dist)
                    value=math.fsum(r['metrics'][k] for r in chosen)/len(chosen)
                    assert abs(value-reported[k])<=1e-12;values.append(value);seed_means+=1
                mu,sd=calculate(values);errors=[abs(mu-g[k]['mean']),abs(sd-g[k]['sample_sd'])]
                assert max(errors)<=1e-12;aggregate_scalars+=2;max_error=max(max_error,*errors)
    assert (scalars,cells,aggregate_scalars)==(1800,900,540)
    proof=dict(status='INDEPENDENT_FSUM_NATIVE1000_TABLE_AND_SEED_FIRST_CHECKS_PASS',records=1000,
        per_scene_scalars=scalars,display_cells=cells,seed_first_aggregate_scalars=aggregate_scalars,
        independent_seed_means=seed_means,max_absolute_difference=max_error,
        input_sha256={name:sha(folder/name) for name in ('records_native_1000.json','summary_statistics.json','seed_first_aggregates.json','SOURCE_BINDINGS.json','TABLES.md')},
        no_significance_or_test_claim=True,new_inference=0,new_fit=0,root_adopted=False)
    (folder/'NUMERIC_VERIFICATION.json').write_text(json.dumps(proof,indent=2)+'\n')
    return proof


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('snapshot',type=Path);print(json.dumps(verify(p.parse_args().snapshot)))
