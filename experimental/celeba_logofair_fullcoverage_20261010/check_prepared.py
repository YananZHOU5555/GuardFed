"""Metadata and exact source reuse checks only; no arrays/models/fits are loaded."""
import ast, copy, difflib, hashlib, json, sys
from pathlib import Path
import metadata as m

def functions(source):
    return {n.name:ast.get_source_segment(source,n) for n in ast.parse(source).body if isinstance(n,ast.FunctionDef)}

def main():
    for p in m.HERE.glob('*.py'):compile(p.read_text(encoding='utf8'),str(p),'exec')
    old=(m.SCREEN/'snapshot/logofair_bridge_20261010/bridge.py').read_text(encoding='utf8');new=m.bridge_source()
    a,b=functions(old),functions(new)
    exact=[n for n in a if a[n]==b[n]]
    assert set(a)-set(exact)=={'validate_job'}
    assert b['validate_job'].replace('job["seed"] not in protocol["seeds"]','job["seed"] != 91001')==a['validate_job']
    namespace={'LOCAL_FILES':{'bridge.py','prepare_reuse.py','protocol.json','reuse_manifest.json'}}
    exec(compile(ast.Module(body=[n for n in ast.parse(new).body if isinstance(n,ast.FunctionDef) and n.name=='validate_job'],type_ignores=[]),'seed_identity','exec'),namespace)
    job=m.read(m.SCREEN/'jobs'/m.read(m.SCREEN/'jobs/manifest.json')['jobs'][0]['job'])
    protocol=m.read(m.SCREEN/'snapshot/logofair_bridge_20261010/protocol.json');protocol['seeds']=m.SEEDS
    for seed in m.SEEDS:namespace['validate_job'](dict(job,seed=seed),protocol)
    rejected=[]
    for label,edited in [('seed-outside',dict(job,seed=91011)),('test',dict(job,evaluation_split='test')),('fit-seed',dict(job,fit_seed=91001))]:
        try:namespace['validate_job'](edited,protocol)
        except ValueError:rejected.append(label)
        else:raise AssertionError(label)
    refs=m.read(m.HERE/'CACHE_IDENTITIES100.json')['references'];expected={(d,a,s) for d in ('IID','non-IID') for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA') for s in m.SEEDS}
    assert len(refs)==100 and {(r['distribution'],r['attack'],r['seed']) for r in refs}==expected
    reusable=[r for r in refs if r['seed']==91001 and r['attack'] in ('Benign','S-DFA')]
    assert len(reusable)==4 and len(refs)-len(reusable)==96
    assert len({r['root_image_ids_sha256'] for r in refs})==10
    assert len({r['valid_image_ids_sha256'] for r in refs})==1
    assert all(r['current_mapping_sha256'] is None for r in refs if r['seed']!=91001)
    try:m.selection({},dict(status='PREPARED_NOT_APPROVED'),'0'*64,'1'*64)
    except ValueError:rejected.append('no-adopted32')
    else:raise AssertionError('Unadopted screen accepted')
    fixture=[]
    re={r['id']:r for r in m.read(m.SCREEN/'snapshot/logofair_bridge_20261010/reuse_manifest.json')['entries']}
    for e in m.read(m.SCREEN/'jobs/manifest.json')['jobs']:
        j=m.read(m.SCREEN/'jobs'/e['job']);source=re[j['baseline_id']]['source_job']
        fixture.append(dict(id=j['id'],candidate=j['candidate'],distribution=source['distribution'],attack=source['attack'],seed=91001,fit_seed=1719,metrics=dict(accuracy=.5,aeod=.1,aspd=.1),checkpoint_sha256='0'*64))
    summary=m.summarize(fixture);summary['strict_index_sha256']='1'*64
    auth=dict(status='ROOT_LOGOFAIR_SCREEN32_ADOPTED',accepted_count=32,summary_sha256='0'*64,strict_index_sha256='1'*64,source_seal_sha256=m.SCREEN_SEAL,test_evaluated=False,independent_acceptance_sha256='2'*64)
    assert m.selection(summary,auth,'0'*64,'1'*64)['id']=='LoGoFair-DP_00'
    try:m.selection(dict(summary,records=fixture[:-1]),auth,'0'*64,'1'*64)
    except ValueError:rejected.append('partial31')
    else:raise AssertionError('Partial summary accepted')
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    m.write(m.HERE/'SELF_CHECK.json',dict(status='SOURCE_METADATA_ONLY_PASS',grid100=True,new96_reuse4=True,accepted_cache_identities=100,root_populations=10,seed_predicate_positives=10,refusals=rejected,science_functions_source_exact=exact,only_changed_function='validate_job:seed membership only',summary_fixture='Synthetic equal-valued metadata only; no scientific result/recipe selected',no_Torch_or_NumPy_import=True,new_arrays=0,new_fits=0,new_CNN=0))
    (m.HERE/'BRIDGE_IDENTITY_DIFF.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),new.splitlines(True),fromfile='original32/bridge.py',tofile='future_bound100/bridge.py')),encoding='utf8')
    print('PASS metadata grid100, seed10, original science functions',len(exact),'refusals',len(rejected))

if __name__=='__main__':main()
