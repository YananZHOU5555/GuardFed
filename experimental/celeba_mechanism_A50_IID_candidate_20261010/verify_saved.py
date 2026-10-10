"""Recompute saved statistics/count identities only; no source records are inferred."""
import hashlib
import json
from pathlib import Path
import runpy
import sys
sys.dont_write_bytecode = True
H = Path(__file__).resolve().parent
R = H.parents[1]
read = lambda p: json.loads(Path(p).read_bytes())
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()

def main():
    if sys.flags.optimize: raise ValueError('Optimized Python forbidden')
    base = runpy.run_path(str(H/'build.py'), run_name='reader_only')
    old = runpy.run_path(str(base['OLD']/'build.py'), run_name='reader_only')
    bindings = read(H/'SOURCE_BINDINGS.json')
    pins = read(H/'INPUTS.json')['original_A_inputs']
    for name, pin in read(H/'AGGREGATE_SOURCE_PINS.json').items():
        assert sha(R/name)==pin['sha256'] and (R/name).stat().st_size==pin['bytes']
    changes = bindings['scope_only_reversible_rebindings'][pins['C_numeric']]
    numeric = base['scoped_module'](old,pins['C_numeric'],[(c['before'],c['after']) for c in changes],{})
    aggregate_numeric = old['variant_module'](R/'tmp/celeba_mechanism_C100_table_20261010/verify_numeric.py')
    records = read(H/'records.json')['records']; panels = read(H/'tables.json')['panels']
    aggregate = read(H/'IID_SEED_FIRST.json')['panels']
    result = dict(per_scene=numeric.verify(records,panels),seed_first=aggregate_numeric.verify_aggregate(records,aggregate))
    previous=read(base['PREV']/'records.json')['records']; previous_ids={r['id'] for r in previous}
    assert [r for r in records if r['id'] in previous_ids]==previous and len(previous)==80
    fragments=base['record_fragments']
    assert [r for r in fragments((H/'records.json').read_text(encoding='utf8')) if r[0] in previous_ids]==fragments((base['PREV']/'records.json').read_text(encoding='utf8'))
    old_panels=read(base['PREV']/'tables.json')['panels']
    for panel,prior in zip(panels,old_panels):
        assert [row for row in panel['rows'] if row['attack']!='Sp-DFA']==prior['rows']
    md=(H/'TABLES.md').read_text(encoding='utf8'); tex=(H/'TABLES.tex').read_text(encoding='utf8')
    cells=0
    for panel in panels+aggregate:
        for row in panel['rows']:
            for metric in ['accuracy_pct','aeod','aspd']:
                digits=3 if metric=='accuracy_pct' else 5
                mean=f"{row[metric]['mean']:.{digits}f}"; sd=f"{row[metric]['sample_sd_ddof1']:.{digits}f}"
                assert mean+' ± '+sd in md and '$'+mean+r' \pm '+sd+'$' in tex
                cells+=1
    assert cells==486 and tex.count(r'\begin{table*}')==18 and tex.count(r'\end{table*}')==18
    assert all(r['distribution']=='IID' for r in records)
    assert bindings['limits']['excluded_partial_id']=='minus_A_non-IID_Benign_seed91001'
    result.update(status='PASS',all_scalars=972,all_display_cells=486,
        old80_object_bytes_order_exact=True,old648_scalars_and324_cells_exact=True,
        tex_fragment_tables=18,tex_compiled=False,new_inference=0,new_fit=0)
    print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
