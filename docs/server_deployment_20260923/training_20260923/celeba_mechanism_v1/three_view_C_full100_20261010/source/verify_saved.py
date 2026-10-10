"""Independent fsum/sampleSD and confusion-count checks of saved C100 tables."""
import argparse
import json
from common import H,OLD,need,sha,read,module,displayed_cells,record_spans,verify_inputs


def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--snapshot',type=__import__('pathlib').Path,required=True);a=ap.parse_args()
    need(a.snapshot.resolve().parent==H.resolve(),'Only owned snapshot')
    verify_inputs()
    for name,pin in read(a.snapshot/'FILES_SHA256.json')['files'].items():
        need(sha(a.snapshot/name)==pin['sha256'] and (a.snapshot/name).stat().st_size==pin['bytes'],'Snapshot member differs')
    n=module('C100_numeric_verify',H/'verify_numeric.py')
    records=read(a.snapshot/'records.json')['records'];panels=read(a.snapshot/'tables.json')['panels']
    result=n.verify(records,panels)
    agg=read(a.snapshot/'cross_scene_additional.json')
    result['IID_seed_first']=n.verify_aggregate([r for r in records if r['distribution']=='IID'],read(a.snapshot/'cross_scene_seed_first.json')['panels'])
    result['nonIID_seed_first']=n.verify_aggregate([dict(r,distribution='IID') for r in records if r['distribution']=='non-IID'],agg['nonIID_five_scene_panels'])
    result['balanced_seed_first']=n.verify_balanced_aggregate(records,agg['balanced_ten_scene_panels'])
    need((a.snapshot/'cross_scene_seed_first.json').read_bytes()==(OLD/'snapshot/cross_scene_seed_first.json').read_bytes(),'Old IID bytes differ')
    need(record_spans((a.snapshot/'records.json').read_text('utf8'))[:160]==record_spans((OLD/'snapshot/records.json').read_text('utf8')),'Old records changed')
    kept=[dict(p,rows=[r for r in p['rows'] if not (r['distribution']=='non-IID' and r['attack'] in ('S-DFA','Sp-DFA'))]) for p in panels]
    need(kept==read(OLD/'snapshot/tables.json')['panels'],'Old panels differ')
    result['display_cells']=displayed_cells((a.snapshot/'TABLES.md').read_text('utf8'),panels)
    result.update(status='SAVED_C100_NUMERIC_AND_OLD_OUTPUT_REGRESSION_PASS',no_inference=True)
    print(json.dumps(result))

if __name__=='__main__':main()
