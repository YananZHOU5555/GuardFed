"""Focused metadata/source check only. Does not import Torch or call science."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import sys

import candidate as c

HERE = Path(__file__).resolve().parent


def main():
    m = c.read(HERE / 'MANIFEST.json')
    c.validate_manifest(m)
    cases = []

    def refuse(name, fn):
        try:
            fn()
        except (ValueError, KeyError, FileNotFoundError):
            cases.append(name)
        else:
            raise AssertionError('Refusal missing: ' + name)

    for name, change in [
        ('extra_record', lambda x: x['records'].append(copy.deepcopy(x['records'][0]))),
        ('wrong_order', lambda x: x['records'].reverse()),
        ('Huber_without_70round_proof', lambda x: x['records'][0].update(method='Huber-BRFL-gradient')),
        ('LoGo_is_not_ordinary_CNN', lambda x: x['records'][0].update(method='LoGoFair')),
        ('partial_round', lambda x: x['records'][0].update(terminal_round=69)),
        ('test_scope', lambda x: x['records'][0].update(split='test')),
        ('wrong_alpha', lambda x: x['records'][0].update(actual_alpha=5.0)),
        ('unsafe_Windows_runtime_path', lambda x: x['records'][0]['runtime_artifacts']['model'].update(server_path='F:/model.pt')),
        ('checkpoint_SHA', lambda x: x['records'][0]['runtime_artifacts']['model'].update(sha256='0'*64)),
        ('CPU_as_original_CUDA_provenance', lambda x: x['records'][0].update(original_training_device='cpu')),
        ('wrong_threads', lambda x: x.update(threads=1)),
        ('prepared_marked_accepted', lambda x: x.update(new_scientific_acceptances=3)),
        ('unapproved_dispatch', lambda x: x.update(dispatch_authorized=True)),
        ('relaxed_tolerance', lambda x: x.update(native_tolerance=1e-8)),
        ('changed_valid_config', lambda x: x['records'][0]['identity']['config'].update(celeba_evaluation_split='test')),
    ]:
        altered = copy.deepcopy(m)
        change(altered)
        refuse(name, lambda altered=altered: c.validate_manifest(altered))
    refuse('unknown_origin_path', lambda: c.resolve_origin('/unknown', m, False))
    refuse('server_inputs_not_opened_by_source_only_resolver', lambda: c.resolve_origin(m['records'][0]['identity']['checkpoint']['path'], m, False))

    # Positive only: route the server-mapped origins to existing original F files
    # in this local metadata test. Candidate runtime still only resolves Linux paths.
    original_resolver = c.resolve_origin
    def local_resolver(origin, manifest, runtime):
        pin = manifest['path_map'].get(str(origin).replace('\\', '/'))
        if not runtime and pin and pin['kind'] == 'server':
            return Path(origin)
        return original_resolver(origin, manifest, runtime)
    c.resolve_origin = local_resolver
    try:
        bridge, ev = c.bound_bridge(m, runtime=False)
        for row in m['records']:
            actual = bridge.identity_record(row['method'], row['id'])
            assert actual == row['identity']
    finally:
        c.resolve_origin = original_resolver
    assert 'torch' not in sys.modules, 'Prepared check imported Torch'
    assert ev.TOLERANCE == 1e-12
    original = ast.parse((HERE / 'originals/replay.py').read_text(encoding='utf-8'))
    base = {n.name: n for n in original.body if isinstance(n, ast.FunctionDef)}
    private = c.runtime_nodes()
    compile(private, '<runtime-compile-only>', 'exec')
    unchanged = []
    for n in private.body:
        if n.name != 'replay_one':
            assert ast.dump(n, include_attributes=False) == ast.dump(base[n.name], include_attributes=False)
            unchanged.append(n.name)
    old_replay = copy.deepcopy(base['replay_one'])
    new_replay = next(n for n in private.body if n.name == 'replay_one')
    # Invert exactly the permitted identity/path/report substitutions.
    class Inverse(ast.NodeTransformer):
        def visit_Call(self, n):
            n = self.generic_visit(n)
            if isinstance(n.func, ast.Name) and n.func.id == 'validate_external':
                return ast.parse('validate_original(original, record, repo)', mode='eval').body
            if isinstance(n.func, ast.Name) and n.func.id == 'runtime_checkpoint':
                return ast.parse("inside(repo, record['checkpoint']['member'])", mode='eval').body
            return n
        def visit_Constant(self, n):
            if n.value == 'external_identity_record_sha256':
                n.value = 'model_inventory_record_sha256'
            if isinstance(n.value, str) and n.value.startswith('Exactly three added-CNN valid interface gates'):
                n.value = 'Two bounded valid-only canaries are implementation evidence, not all900 replay or a new final performance table'
            return n
    assert ast.dump(Inverse().visit(copy.deepcopy(new_replay)), include_attributes=False) == ast.dump(old_replay, include_attributes=False)
    reuse = c.read(HERE / 'originals/SOURCE_REUSE.json')
    scientific_functions = 0
    for entry in reuse['function_sources']:
        file = original_resolver(entry['file']['path'], m, False)
        text = file.read_text(encoding='utf-8')
        functions = {n.name: n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
        for name, h in entry['functions'].items():
            assert hashlib.sha256(ast.get_source_segment(text, functions[name]).encode()).hexdigest() == h
            scientific_functions += 1
    patch = ''.join(difflib.unified_diff(ast.unparse(old_replay).splitlines(True), ast.unparse(new_replay).splitlines(True),
                                       fromfile='original900/replay_one', tofile='private_exact3/replay_one'))
    (HERE / 'RUNTIME_BINDING_DIFF.patch').write_text(patch, encoding='utf-8', newline='\n')
    for p in HERE.glob('*.py'):
        compile(p.read_text(encoding='utf-8'), str(p), 'exec')
    result = {'status': 'SOURCE_AND_METADATA_CHECK_PASS_NOT_RUNTIME', 'positive_original_identity_records': 3,
              'focused_refusals': len(cases), 'refusal_cases': cases, 'original_scientific_functions_source_exact': scientific_functions,
              'unchanged_original_runtime_functions': unchanged, 'replay_one_inverse_AST_exact': True,
              'runtime_binding_edits': ['external adopted identity instead of legacy900 checked_result',
                'exact mapped server checkpoint instead of original repo-relative member', 'external identity receipt key', 'exact3 claim-limit text'],
              'Torch_imported': False, 'CNN_calls': 0, 'fit_calls': 0, 'SSH': 0, 'actual_dispatch_authorized': False,
              'actual_runtime_positive_test': False, 'Linux_paths_hashes_resources_still_require_root_actual_preflight': True}
    with (HERE / 'CHECK_RESULTS.json').open('x', encoding='utf-8', newline='\n') as f:
        f.write(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(result, ensure_ascii=False))


if __name__ == '__main__':
    main()
