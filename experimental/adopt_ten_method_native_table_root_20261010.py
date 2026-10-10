"""Check rendered table cells and adopt the actual descriptive native1000 table."""
from pathlib import Path
import datetime,hashlib,json

ROOT=Path(__file__).resolve().parents[1]
read=lambda p:json.loads(p.read_bytes())
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
source=ROOT/'tmp/celeba_ten_method_native_table_20261010'
assert sha(source/'FILES_SHA256.json')=='979753cabe1b489908aba724784be34a356a1416b61da902ab0867db6bf2b028'
for name,pin in read(source/'FILES_SHA256.json')['files'].items():
    p=source/name
    assert sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes']
snap=source/'snapshot';numeric=read(snap/'NUMERIC_VERIFICATION.json')
assert numeric['status']=='INDEPENDENT_FSUM_NATIVE1000_TABLE_AND_SEED_FIRST_CHECKS_PASS'
assert (numeric['records'],numeric['per_scene_scalars'],numeric['display_cells'],numeric['seed_first_aggregate_scalars'],numeric['independent_seed_means'])==(1000,1800,900,540,2250)
assert numeric['max_absolute_difference']<=1e-12
for name,digest in numeric['input_sha256'].items():assert sha(snap/name)==digest
tables=read(snap/'summary_statistics.json');aggregates=read(snap/'seed_first_aggregates.json')
bindings=read(snap/'SOURCE_BINDINGS.json')
assert bindings['old_native_summary_metric_objects_exact']==810 and bindings['models']==1000 and bindings['methods']==10
assert not bindings['final_test'] and bindings['new_inference']==bindings['new_fit']==0
assert sha(Path(bindings['actual_logofair_root']['path']))==bindings['actual_logofair_root']['sha256']=='1529a852b3bd02561d274fdea832186706bb09725c8d02594d09114f44f977c2'
metrics={'ACC (%) ↑':('accuracy',100,2),'AEOD ↓':('aeod',1,4),'ASPD ↓':('aspd',1,4)}
attacks=['Benign','F Flip','FedSA','S-DFA','Sp-DFA']
sections=[];seen=set();rendered=aggregate_cells=0;aggregate=False
for line in (snap/'TABLES.md').read_text('utf8').splitlines():
    if line.startswith('## '):
        dist,panel=line[3:].split(' / ');sections.append((dist,panel));aggregate=False
    if line.startswith('# Seed-first'):aggregate=True
    if not line.startswith('| ') or ' ± ' not in line:continue
    parts=line.strip('| ').split(' | ')
    if not aggregate:
        method,label,*values=parts
        assert len(values)==5 and label in metrics
        key,_,_=metrics[label]
        for attack,value in zip(attacks,values):
            rows=[r for r in tables[panel] if (r['method'],r['distribution'],r['attack'])==(method,dist,attack)]
            assert len(rows)==1 and rows[0][key]['display']==value
            identity=(panel,dist,method,key,attack);assert identity not in seen;seen.add(identity);rendered+=1
    else:
        panel,scope,method,n,*values=parts
        rows=[r for r in aggregates[panel] if (r['scope'],r['method'])==(scope,method)]
        assert len(rows)==1 and rows[0]['n']==int(n) and len(values)==3
        for value,(key,scale,precision) in zip(values,metrics.values()):
            stat=rows[0][key]
            assert value==f"{stat['mean']*scale:.{precision}f} ± {stat['sample_sd']*scale:.{precision}f}"
            aggregate_cells+=1
assert len(sections)==len(set(sections))==6 and rendered==900 and aggregate_cells==270
out=ROOT/'outputs/guardfed_tables/celeba_ten_method_native_20261010'
assert not out.exists();out.mkdir(parents=True)
for name in (*numeric['input_sha256'],'NUMERIC_VERIFICATION.json'):
    with (out/name).open('xb') as f:f.write((snap/name).read_bytes())
    assert sha(out/name)==sha(snap/name)
proof=dict(status='ROOT_TEN_METHOD_NATIVE1000_DESCRIPTIVE_TABLE_ADOPTED',adopted_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    records=1000,methods=10,distributions=2,scenes_per_distribution=5,seeds=[10,9,6],
    original_nine_method810_metric_objects_exact=True,independent_numeric_sha256=sha(snap/'NUMERIC_VERIFICATION.json'),
    numeric_source_path=(snap/'NUMERIC_VERIFICATION.json').relative_to(ROOT).as_posix(),
    per_scene_scalars=1800,rendered_scene_cells=900,seed_first_aggregate_scalars=540,rendered_aggregate_cells=270,independent_seed_means=2250,
    source_seal_sha256=sha(source/'FILES_SHA256.json'),source_bindings_sha256=sha(out/'SOURCE_BINDINGS.json'),
    files_sha256={p.name:sha(p) for p in out.iterdir() if p.is_file()},
    canonical_table=(out/'TABLES.md').relative_to(ROOT).as_posix(),new_inference=0,new_fit=0,final_test=False,
    full17_complete=False,primary_endpoint_selected=False,incorporated_into_full_rebuttal=False,
    limitation='Native descriptive validation table; original per-method calibration/postprocessing retained. LoGoFair virtual20-cohort adaptation and fixed fitseed1719; negative/constant outcomes, mixed runtime and validation/test history retained. No necessity, causal, significance or universal superiority claim.')
target=out/'ROOT_REVIEW.json'
with target.open('x',encoding='utf8') as f:json.dump(proof,f,indent=2);f.write('\n')
print(json.dumps(dict(path=str(target),sha256=sha(target),status=proof['status'],cells=rendered,aggregate_cells=aggregate_cells)))
