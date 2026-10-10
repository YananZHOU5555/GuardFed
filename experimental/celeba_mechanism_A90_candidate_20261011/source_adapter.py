"""Reversible A90 scope bridge over sealed A80; no new scientific arithmetic."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys
H=Path(__file__).resolve().parent
R=H.parents[1]
A80=R/'tmp/celeba_mechanism_A80_candidate_20261011'
A80_SEAL='f3b2bcafff59db58c6070aa74fa8079879eb1ddb7ec5a8a0dcaf541e8e7975df'
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def need(ok,message):
    if not ok:raise ValueError(message)
def mapped(name):
    need(sha(A80/'FILES_SHA256.json')==A80_SEAL,'A80 source seal drift')
    sealed=json.loads((A80/'FILES_SHA256.json').read_bytes())['files']
    for member in ('source_adapter.py','SOURCE_ADAPTATIONS.json',name):
        p=A80/member;pin=sealed[member]
        need(sha(p)==pin['sha256'] and p.stat().st_size==pin['bytes'],'A80 source drift: '+member)
    source=runpy.run_path(str(A80/'source_adapter.py'))['mapped'](name)
    contract=json.loads((H/'SOURCE_ADAPTATIONS.json').read_bytes())[name]
    need(hashlib.sha256(source.encode()).hexdigest()==contract['A80_mapped_source_sha256'],'A80 mapped source drift')
    text=source
    for before,after in contract['replacements']:
        need(text.count(before)==1,'Non-unique scope replacement: '+before)
        text=text.replace(before,after,1)
    inverse=text
    for before,after in reversed(contract['replacements']):
        need(inverse.count(after)==1,'Non-unique scope inverse: '+after)
        inverse=inverse.replace(after,before,1)
    need(inverse==source,'Original A80 mapped source not byte exact')
    ast.parse(text)
    return text
def namespace(name,filename):
    scope={'__file__':filename,'__name__':'A90_scoped_source_no_main'}
    exec(compile(mapped(name),filename+' [exact A80 scope bridge]','exec'),scope)
    return {k:v for k,v in scope.items() if k not in ('__file__','__name__','__builtins__')}
def preflight(name):
    need(not sys.flags.optimize,'Optimized Python forbidden')
    p=H/'ROOT_BINDING.json'
    need(p.is_file(),'Actual root-adopted MECHANISM295 inputs not bound; no generation authorized')
    b=json.loads(p.read_bytes())
    need(b['status']=='ACTUAL_ROOT295_BOUND_FOR_A90_TABLE' and b['root_adopted'] is True,'Actual root295 adoption required')
    for key in ('adoption','index'):
        need(sha(R/b[key])==b[key+'_sha256'],'Actual root/index pin drift')
    root=json.loads((R/b['adoption']).read_bytes());index=json.loads((R/b['index']).read_bytes())
    ids=[f'minus_A_non-IID_S-DFA_seed{seed}' for seed in (91009,91010)]+[f'minus_A_non-IID_Sp-DFA_seed{seed}' for seed in range(91001,91006)]
    need(b['expected_new_ids']==root['accepted_new_ids']==index['new_ids']==ids,'Exact7 endpoint IDs required')
    need(root['status']==b['root_status'] and root['status'].startswith('ROOT_A') and root['status'].endswith('_ADOPTED'),'Actual root status mismatch')
    need(root['prior_accepted']==288 and root['new_accepted']==7 and root['cumulative_accepted']==len(index['all_ids'])==295 and root['original288_unchanged'],'Root288-to295 prefix required')
    need(root['records_index_sha256']==b['index_sha256'],'Root/index identity mismatch')
    if name=='build.py':
        parser=argparse.ArgumentParser()
        for key in ('adoption','adoption-sha256','index','index-sha256'):parser.add_argument('--'+key,required=True)
        args=vars(parser.parse_args())
        need(all(args[k]==b[k] for k in args),'CLI differs from ROOT_BINDING')
    else:
        proof=json.loads((H/'SOURCE_BINDINGS.json').read_bytes())
        need(proof['actual_A90_root_adoption_sha256']==b['adoption_sha256'] and proof['accepted_index_sha256']==b['index_sha256'],'Generated source/root binding mismatch')
