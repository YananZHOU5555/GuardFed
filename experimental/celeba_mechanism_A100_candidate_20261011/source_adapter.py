"""Reversible A100 scope binding over sealed A90; original arithmetic only."""
import argparse
import ast
import hashlib
import json
from pathlib import Path
import runpy
import sys

H = Path(__file__).resolve().parent
R = H.parents[1]
A90 = R/'tmp/celeba_mechanism_A90_candidate_20261011'
A90_SEAL = '4054a32a1c39b39ab0b1f8943b28c12f2b2615d4ec8a9154eecdd500ac06159a'
PRIOR_ROOT = '70689e63467d8866caa3beb06d2be5f911defd0a4d5f7f061ab005d20bd1c1bb'
PRIOR_INDEX = '7c46fcb20c15c377b1df17378f14b6282ee394bdcaacecd8d4f92be344777bef'
EXACT5 = [f'minus_A_non-IID_Sp-DFA_seed{s}' for s in range(91006,91011)]
sha = lambda p: hashlib.sha256(Path(p).read_bytes()).hexdigest()


def need(ok, message):
    if not ok:
        raise ValueError(message)


def mapped(name):
    need(sha(A90/'FILES_SHA256.json') == A90_SEAL, 'A90 source seal drift')
    sealed = json.loads((A90/'FILES_SHA256.json').read_bytes())['files']
    for member in ('source_adapter.py','SOURCE_ADAPTATIONS.json',name):
        p = A90/member
        need(sha(p) == sealed[member]['sha256'] and p.stat().st_size == sealed[member]['bytes'], 'A90 source drift: '+member)
    source = runpy.run_path(str(A90/'source_adapter.py'))['mapped'](name)
    contract = json.loads((H/'SOURCE_ADAPTATIONS.json').read_bytes())[name]
    need(hashlib.sha256(source.encode()).hexdigest() == contract['A90_mapped_source_sha256'], 'Mapped A90 drift')
    text = source
    for before, after in contract['replacements']:
        need(text.count(before) == 1, 'Non-unique scope replacement: '+before)
        text = text.replace(before, after, 1)
    inverse = text
    for before, after in reversed(contract['replacements']):
        need(inverse.count(after) == 1, 'Non-unique scope inverse: '+after)
        inverse = inverse.replace(after, before, 1)
    need(inverse == source, 'A90 source inverse not byte exact')
    ast.parse(text)
    return text


def namespace(name, filename):
    scope = {'__file__': filename, '__name__': 'A100_scoped_source_no_main'}
    exec(compile(mapped(name), filename+' [exact A90 scope binding]', 'exec'), scope)
    return {k:v for k,v in scope.items() if k not in ('__file__','__name__','__builtins__')}


def preflight(name):
    need(not sys.flags.optimize, 'Optimized Python forbidden')
    p = H/'ROOT_BINDING.json'
    need(p.is_file(), 'Actual native300 and root-adopted three-view300 not bound; generation forbidden')
    b = json.loads(p.read_bytes())
    need(b['status'] == 'ACTUAL_NATIVE300_AND_REPLAY300_BOUND_FOR_A100_TABLE' and b['root_adopted'] is True, 'Actual dual adoption required')
    for key in ('adoption','index','native_root'):
        digest = b[key+'_sha256']
        need(isinstance(digest,str) and len(digest)==64 and sha(R/b[key])==digest, 'Actual external pin drift: '+key)
    root = json.loads((R/b['adoption']).read_bytes())
    index = json.loads((R/b['index']).read_bytes())
    native = json.loads((R/b['native_root']).read_bytes())
    need(root['status']==b['root_status'] and root['status'].startswith('ROOT_A') and root['status'].endswith('_ADOPTED'), 'Actual replay-root status differs')
    need((root['prior_accepted'],root['new_accepted'],root['cumulative_accepted'])==(295,5,300) and root['original295_unchanged'], 'Exact295+5=300 prefix required')
    need(root['accepted_new_ids']==index['new_ids']==b['expected_new_ids']==EXACT5, 'Only remaining five Sp-DFA IDs permitted')
    need(root['records_index_sha256']==b['index_sha256'] and len(index['all_ids'])==len(set(index['all_ids']))==300, 'Actual300 root/index differs')
    need(index['prior_adoption_sha256']==PRIOR_ROOT and index['prior_index_sha256']==PRIOR_INDEX, 'Actual295 parent pins differ')
    prior = json.loads((R/index['prior_index_path']).read_bytes())
    need(sha(R/index['prior_index_path'])==PRIOR_INDEX and index['all_ids']==prior['all_ids']+EXACT5, 'Old295 ID prefix differs')
    need(root['native_root_sha256']==b['native_root_sha256'] and (R/root['native_root_path']).resolve()==(R/b['native_root']).resolve(), 'Replay/native root mismatch')
    need(native['root_adopted'] is True and native['status']=='ROOT_ORIGINAL_STRICT_DELTA_ARCHIVE_AND_OFFSERVER_PASS' and native['total_new_strict_and_offserver']==300 and native['test'] is False, 'Actual native300 adoption required')
    need(native['inspection_sha256']==root['native_inspection_sha256']==index['native_inspection_sha256'], 'Native inspection identity differs')
    need(root['native_max_abs_difference']==0 and root['Full_inference']==root['new_CNN']==root['new_training']==0 and root.get('new_fit',0)==0 and root['test'] is False, 'Forbidden scientific operation')
    if name == 'build.py':
        parser = argparse.ArgumentParser()
        for key in ('adoption','adoption-sha256','index','index-sha256'):
            parser.add_argument('--'+key, required=True)
        args = vars(parser.parse_args())
        need(all(args[k]==b[k] for k in args), 'CLI differs from ROOT_BINDING')
    else:
        proof = json.loads((H/'SOURCE_BINDINGS.json').read_bytes())
        need(proof['actual_A100_root_adoption_sha256']==b['adoption_sha256'] and proof['accepted_index_sha256']==b['index_sha256'], 'Generated source/root binding mismatch')
