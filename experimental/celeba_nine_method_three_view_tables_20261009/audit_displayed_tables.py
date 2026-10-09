"""Independent serialized-output audit; never imports the producer."""
from pathlib import Path
from collections import Counter
import ast
import csv
import hashlib
import json
import re
import numpy as np
from PIL import Image, ImageChops

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]
OLD = ROOT / 'outputs/guardfed_tables/celeba_nine_method_final_20261004'
read = lambda p: json.loads(Path(p).read_text(encoding='utf-8-sig'))
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()
bundle = read(OUT / 'records_three_views_900.json')
records = {r['id']: r for r in bundle['records']}
stats = read(OUT / 'summary_statistics.json')
coverage = read(OUT / 'coverage_alias_environment.json')
original = (OLD / 'build_tables.py').read_text(encoding='utf-8-sig')
ns = {}
for n in ast.parse(original).body:
    if isinstance(n, ast.Assign) and any(isinstance(t,ast.Name) and t.id in ['methods','metrics','attacks','dists','adapted'] for t in n.targets):
        exec(compile(ast.Module(body=[n],type_ignores=[]),'original/presentation_constants','exec'),ns)
fixed = {'ten':list(range(91001,91011)), 'nonselection_nine':list(range(91002,91011)), 'matching_six':list(range(91005,91011))}
panels = [('celeba_iid_ten_seed',['IID'],'ten'), ('celeba_noniid_ten_seed',['non-IID'],'ten'), ('celeba_iid_noniid_ten_seed',ns['dists'],'ten'), ('celeba_iid_noniid_exclude_selection',ns['dists'],'nonselection_nine'), ('celeba_iid_noniid_matching_six',ns['dists'],'matching_six')]
assert len(records) == 900
assert len({tuple(r['scientific_cell']) for r in records.values()}) == 900
assert all(r['same_checkpoint_all_views'] for r in records.values())
checks = Counter()
with (OUT / 'records_three_views_2700.csv').open(encoding='utf-8-sig',newline='') as f:
    csvrows = list(csv.DictReader(f))
assert len(csvrows) == len({(r['id'],r['view']) for r in csvrows}) == 2700
for row in csvrows:
    r = records[row['id']]
    assert row['checkpoint_sha256'] == r['checkpoint_sha256']
    for k,_,_,_ in ns['metrics']:
        assert float(row[k]) == r['views'][row['view']][k]
        checks['CSV_metric_roundtrips'] += 1
tables, border_checks = [], []
for view in ['raw','native','shared_calibration']:
    for name,dists,subset in panels:
        table = OUT / view / (name+'.md')
        lines = [l for l in table.read_text(encoding='utf-8').splitlines() if l.startswith('|')]
        assert len(lines) == 29
        body = [[c.strip() for c in l.strip('|').split('|')] for l in lines[2:]]
        assert all(len(row) == 3 + 5*len(dists) for row in body)
        conditions = [(d,a) for d in dists for a in ns['attacks']]
        md_numbers = []
        for mi,m in enumerate(ns['methods']):
            for ki,(k,_,scale,precision) in enumerate(ns['metrics']):
                row = body[mi*3+ki]
                for ci,(d,a) in enumerate(conditions):
                    ids = [r['id'] for r in records.values() if (r['method'],r['distribution'],r['attack']) == (m,d,a) and r['seed'] in fixed[subset]]
                    assert len(ids) == len(fixed[subset])
                    vals = np.array([records[rid]['views'][view][k] for rid in ids])
                    expect = f'{np.mean(vals)*scale:.{precision}f} ± {np.std(vals,ddof=1)*scale:.{precision}f}'
                    assert row[3+ci] == expect, (view,name,m,d,a,k)
                    md_numbers.append(row[3+ci])
                    checks['Markdown_numeric_cells_independently_checked'] += 1
        tex = table.with_suffix('.tex').read_text(encoding='utf-8')
        tex_numbers = [a+' ± '+b for a,b in re.findall(r'\$([0-9.]+) \\pm ([0-9.]+)\$',tex)]
        assert tex_numbers == md_numbers
        checks['TeX_numeric_cells_equal_Markdown'] += len(tex_numbers)
        if view == 'native':
            old_lines = [l for l in (OLD/table.name).read_text(encoding='utf-8-sig').splitlines() if l.startswith('|')]
            assert lines == old_lines
            checks['native_original_displayed_numeric_cells_exact'] += len(md_numbers)
        png = table.with_suffix('.png')
        with Image.open(png) as im:
            assert im.size == ((4370,2147) if len(dists)==2 else (2983,2147))
            rgb = im.convert('RGB')
            bbox = ImageChops.difference(rgb,Image.new('RGB',rgb.size,'white')).getbbox()
            assert bbox and bbox[0] > 10 and bbox[1] > 10 and bbox[2] < im.width-10 and bbox[3] < im.height-10
            border_checks.append({'path':str(png.relative_to(OUT)).replace('\\','/'),'size':list(im.size),'nonwhite_content_bbox':list(bbox),'all_four_page_margins_visible':True})
        tables.append({'view':view,'name':name,'subset':subset,'seed_count':len(fixed[subset]),'numeric_cells':len(md_numbers),'Markdown_sha256':sha(table),'TeX_sha256':sha(table.with_suffix('.tex')),'PNG_sha256':sha(png)})
for rel,pin in read(OUT/'input_files_SHA256.json').items():
    p = ROOT/rel
    assert p.stat().st_size == pin['bytes'] and sha(p) == pin['sha256'], ('Original input changed',rel)
    checks['source_files_rehashed_after_generation'] += 1
native_equal_raw_nonGF = sum(r['views']['native'] == r['views']['raw'] for r in records.values() if r['method']!='GuardFed-AD2+')
GF_equal_shared = sum(r['views']['native'] == r['views']['shared_calibration'] for r in records.values() if r['method']=='GuardFed-AD2+')
assert native_equal_raw_nonGF == 800 and GF_equal_shared == 100
constant_summary = {v:dict(Counter(records[rid]['method'] for rid in ids)) for v,ids in coverage['constant_predictions_retained'].items()}
result = {'status':'INDEPENDENT_SERIALIZED_OUTPUT_NUMERIC_AND_PAGE_MARGIN_AUDIT_PASS','checks':checks,'tables':tables,'PNG_border_checks':border_checks,'native_equals_raw_for_eight_baselines':native_equal_raw_nonGF,'GuardFed_native_equals_shared':GF_equal_shared,'constant_prediction_counts_by_view_method':constant_summary,'visual_review_status':'PENDING_HUMAN_AGENT_IMAGE_INSPECTION','forbidden_operations':{'new_CNN_inference':0,'threshold_refit':0,'test':0,'server':0,'Git':0}}
(OUT/'independent_display_audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps({'status':result['status'],'checks':checks,'constant_predictions':constant_summary},ensure_ascii=False))
