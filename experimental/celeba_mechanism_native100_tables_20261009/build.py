"""Thin native100 extraction through the unchanged native92 renderer/evidence tool."""
import importlib.util
import json
import sys
from pathlib import Path
import hashlib

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
METRICS = ['accuracy_pct', 'aeod', 'aspd']
SCENES = {(d,a) for d in ['IID','non-IID'] for a in ['Benign','F Flip','FedSA','S-DFA','Sp-DFA']}
SEEDS = set(range(91001,91011))


def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p): return json.loads(Path(p).read_text(encoding='utf-8-sig'))
def save(name,value):
    with (HERE/name).open('x',encoding='utf-8',newline='\n') as f:
        json.dump(value,f,ensure_ascii=False,indent=2,allow_nan=False);f.write('\n')


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m


def validate_scope(inspection):
    assert inspection['source_script_sha256']=='3d78db7e67275dce2d52bc8927dd86fc1c517708224a994345b70cadc56427ef'
    assert inspection['manifest_sha256']=='498ed0e033ef6eb5532286820ec987af3b44ca6251d8a84c42ce9c3e5ff5ab2d'
    assert not inspection['invalid'] and inspection['new_count']==104 and inspection['reused_count']==100
    rows=inspection['records'];cells={(r['variant'],r['distribution'],r['attack'],r['seed']) for r in rows}
    assert len(rows)==len(cells)==204 and len({r['id'] for r in rows})==204
    assert set(inspection['accepted_new_ids'])=={r['id'] for r in rows if r['variant']!='Full'}
    assert set(inspection['accepted_reused_ids'])=={r['id'] for r in rows if r['variant']=='Full'}
    for variant in ['Full','minus_U']:
        selected=[r for r in rows if r['variant']==variant]
        assert len(selected)==100 and {(r['distribution'],r['attack']) for r in selected}==SCENES
        for dist,attack in SCENES:
            assert {r['seed'] for r in selected if (r['distribution'],r['attack'])==(dist,attack)}==SEEDS
    other=[r for r in rows if r['variant'] not in ['Full','minus_U']]
    assert len(other)==4 and {r['variant'] for r in other}=={'minus_C'}
    assert {(r['distribution'],r['attack'],r['seed']) for r in other}=={('IID','Benign',s) for s in range(91001,91005)}
    assert all(len(r['checkpoint_sha256'])==64 and r['files'] for r in rows)
    return rows


def main():
    basis=read(HERE/'INPUTS.json')
    for name,pin in basis['files'].items():
        assert sha(ROOT/name)==pin['sha256'] and (ROOT/name).stat().st_size==pin['bytes'],name
    inspection_path=ROOT/basis['inspection'];inspection=read(inspection_path);rows=validate_scope(inspection)
    assert inspection_path.with_suffix('.sha256').read_text().strip()==sha(inspection_path)
    rootproof=read(ROOT/basis['root_proof'])
    assert rootproof['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS'
    assert rootproof['inspection_sha256']==sha(inspection_path) and rootproof['total_new_strict_and_offserver']==104
    assert rootproof['ledger_sha256']==sha(ROOT/basis['frozen_ledger_copy'])
    assert rootproof['receipt_sha256']==sha(ROOT/basis['backup_receipt']) and rootproof['offserver_proof_sha256']==sha(ROOT/basis['offserver_proof'])
    original_rows=read(ROOT/basis['prior_native92_inspection'])['records']
    indexed={r['id']:r for r in rows}
    assert all(indexed[r['id']]==r for r in original_rows), 'Historical native records changed'

    renderer=module('unchanged_native92_renderer',ROOT/basis['renderer'])
    saved_argv=sys.argv
    try:
        sys.argv=['unchanged_native92_renderer',str(inspection_path),str(HERE/'rendered')]
        renderer.main()
    finally:sys.argv=saved_argv
    tables=read(HERE/'rendered/tables.json')
    assert tables['complete_paired_scenes']==10 and len(tables['accepted_new_ids'])==104
    assert all({r['variant'] for r in p['rows']}=={'Full','minus_U','minus_U minus Full'} and len(p['rows'])==30 for p in tables['panels'])
    evidence=module('unchanged_native_evidence',ROOT/basis['evidence'])
    summary=evidence.summarize(rows)
    paired=[r for r in summary['paired_per_seed'] if r['variant']=='minus_U']
    assert len(paired)==100 and {(r['distribution'],r['attack'],r['seed']) for r in paired}=={(d,a,s) for d,a in SCENES for s in SEEDS}
    save('paired_differences.json',dict(status='NATIVE_ONLY_ORIGINAL_SUMMARIZE_PAIRED_DIFFERENCES',direction='minus_U minus Full',records=paired,
        checkpoint_pairs=[dict(id=r['id'],checkpoint_sha256=r['checkpoint_sha256'],Full_id=f['id'],Full_checkpoint_sha256=f['checkpoint_sha256'],seed=r['seed'],distribution=r['distribution'],attack=r['attack']) for r in rows if r['variant']=='minus_U' for f in rows if f['variant']=='Full' and (r['distribution'],r['attack'],r['seed'])==(f['distribution'],f['attack'],f['seed'])],
        original_source_sha256=sha(ROOT/basis['evidence']),inspection_sha256=sha(inspection_path),new_inference=0,test=False))
    save('coverage.json',dict(status='NATIVE_FULL100_MINUS_U100_COMPLETE_OTHER_VARIANTS_PARTIAL',accepted_new104_ids=inspection['accepted_new_ids'],Full100_ids=inspection['accepted_reused_ids'],all_original_records_retained_at=basis['inspection'],original_record_count=204,
        variant_coverage=[dict(variant=v,n=len([r for r in rows if r['variant']==v]),expected_n=100,complete=len([r for r in rows if r['variant']==v])==100,ids=[r['id'] for r in rows if r['variant']==v]) for v in evidence.VARIANTS],
        complete_Full_U_scenes=10,unique_paired_seeds=100,panel_paired_counts=[100,90,60],torch_counts=inspection['torch_counts_by_role'],dispatch_environment=inspection['dispatch_environment'],
        native_only=True,three_view100_claimed=False,other_components_complete=False,formal_primary_endpoint='PENDING_USER',whole_rebuttal_complete=False))
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    print(json.dumps(dict(status='NATIVE100_TEN_SCENES_THROUGH_UNCHANGED_RENDERER',Full=100,minus_U=100,minus_C_partial=4,paired=100,panels=3,no_CNN=True)))


if __name__=='__main__':main()
