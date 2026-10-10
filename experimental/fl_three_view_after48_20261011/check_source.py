"""Focused source/metadata checks only; no scientific imports or execution."""
from pathlib import Path
import ast
import copy
import hashlib
import importlib.util
import json
import sys

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'celeba_flgmm_three_view_closed_batch_preparation_20261011'


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    return json.loads(p.read_text(encoding='utf-8-sig'))


def main():
    source = (HERE / 'candidate.py').read_text(encoding='utf-8')
    original = (OLD / 'candidate.py').read_text(encoding='utf-8')
    assert sha(OLD / 'candidate.py') == 'd8412c4aa782e767afd174d92b7568f0153afb5215f8f65fbff178d84960d1be'
    compile(source, 'candidate.py', 'exec')
    spec = importlib.util.spec_from_file_location('finite_flgmm_candidate', HERE / 'candidate.py')
    c = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(c)
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    m = read(HERE / 'MANIFEST.json')
    c.validate_manifest(m)
    skip = read(HERE / 'SKIP_AND_REUSE.json')
    root = read(Path(m['FL57_root_pin']['path']))
    assert sha(Path(m['FL57_root_pin']['path'])) == m['FL57_root_pin']['sha256']
    expected = [rid for rid in root['accepted_job_ids'] if rid not in skip['prior48_ids']]
    assert len(expected) == 13 and m['exact_ids'] == expected
    refusals = []
    cases = {}
    x = copy.deepcopy(m); x['records'][0]['n_eval'] = 1; cases['partial_valid'] = x
    x = copy.deepcopy(m); x['records'][0]['split'] = 'test'; cases['test'] = x
    x = copy.deepcopy(m); x['native_tolerance'] = 1e-10; cases['tolerance'] = x
    x = copy.deepcopy(m); x['threads'] = 4; cases['threads'] = x
    x = copy.deepcopy(m); x['new_scientific_acceptances'] = 47; cases['prepared_acceptance'] = x
    x = copy.deepcopy(m); x['records'][0]['id'] = x['exact_ids'][0] = 'observed_unaccepted_job'; cases['unaccepted_ID'] = x
    x = copy.deepcopy(m); x['exact_ids'][0] = skip['prior48_ids'][0]; x['records'][0]['id'] = skip['prior48_ids'][0]; cases['repeat_accepted_interface'] = x
    for name, bad in cases.items():
        try:
            c.validate_manifest(bad)
        except (ValueError, KeyError):
            refusals.append(name)
        else:
            raise AssertionError('Accepted invalid metadata: ' + name)
    oldnodes = {n.name: n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef)}
    newnodes = {n.name: n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    changed = {name for name in oldnodes if ast.get_source_segment(original, oldnodes[name]) != ast.get_source_segment(source, newnodes[name])}
    assert changed == {'validate_manifest', 'runtime_nodes', 'authorize', 'run'}
    recovered = source
    for name in changed:
        a = ast.get_source_segment(source, newnodes[name]); b = ast.get_source_segment(original, oldnodes[name])
        assert recovered.count(a) == 1
        recovered = recovered.replace(a, b)
    recovered = recovered.replace('Exact13 already-accepted FLGMM valid three-view candidate.', 'Exact47 already-accepted FLGMM valid three-view candidate.')
    assert recovered == original
    # All science definitions and the original core are unchanged, with exact per-function pins.
    reuse = read(HERE / 'originals/SOURCE_REUSE.json')
    fn = {}
    for entry, file in zip(reuse['function_sources'], ('evaluator.py', 'replay.py')):
        p = HERE / 'originals' / file
        assert sha(p) == entry['file']['sha256']
        txt = p.read_text(encoding='utf-8'); nodes = {n.name:n for n in ast.parse(txt).body if isinstance(n, ast.FunctionDef)}
        for name, h in entry['functions'].items():
            assert hashlib.sha256(ast.get_source_segment(txt, nodes[name]).encode()).hexdigest() == h
            fn[name] = h
    assert len(fn) == 17
    assert sha(HERE / 'originals/core.py') == reuse['original_core']['sha256']
    # The original runtime helper/replay AST adapter is unchanged except report prose.
    oldruntime = ast.get_source_segment(original, oldnodes['runtime_nodes'])
    newruntime = ast.get_source_segment(source, newnodes['runtime_nodes']).replace(
        'Exactly13 accepted FLGMM checkpoints; finite partial coverage, not full100, method ranking, final primary endpoint, test, or CUDA equivalence',
        'Exactly47 accepted FLGMM checkpoints; finite partial coverage, not full100, method ranking, final primary endpoint, test, or CUDA equivalence')
    assert oldruntime == newruntime
    compile(c.runtime_nodes(), '<compile-only-original-runtime>', 'exec')
    return {'status': 'SOURCE_AND_FINITE_METADATA_CHECK_PASS_NOT_EXECUTED', 'root_training_new':57, 'screen_reused':4,
        'existing_three_view_skipped':48, 'pending_exact':13, 'positive_metadata_scope':True,
        'focused_metadata_refusals':refusals, 'original_scientific_functions_exact':17,
        'original_scientific_function_sha256':fn, 'original_candidate_inverse_source_exact':True,
        'changed_entry_functions':sorted(changed), 'original_runtime_helpers_and_replay_AST_adapter_exact_except_scope_prose':True,
        'new_CNN':0, 'new_fit':0, 'new_training':0, 'new_three_view_acceptances':0, 'dispatch_authorized':False,
        'Torch_or_NumPy_imported':False, 'manifest_sha256':sha(HERE/'MANIFEST.json'), 'candidate_sha256':sha(HERE/'candidate.py')}


if __name__ == '__main__':
    result = main()
    with (HERE / 'SOURCE_CHECK.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))
