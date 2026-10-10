"""Pure metadata/source check; no Torch, science calls, model/array loads or dispatch."""
from pathlib import Path
import ast, copy, hashlib, json, sys
import candidate as c
import check_saved as saved


def ast_sha(node):
    return hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()


def main():
    m = c.read(c.HERE / 'MANIFEST.json'); c.validate_manifest(m)
    compiled = []
    for p in sorted(c.HERE.rglob('*.py')):
        compile(p.read_bytes(), str(p), 'exec'); compiled.append(p.relative_to(c.HERE).as_posix())
    for rel, pin in m['sources'].items():
        p = c.HERE / rel
        c.require(c.sha(p) == pin['sha256'] and p.stat().st_size == pin['bytes'], 'Original source/proof copy changed')
    # Read only the existing small F metadata required by the original private identity.
    saved.f_volume()
    bridge, ev = c.bound_bridge(m, runtime=False)
    actual = [bridge.identity_record(row['id'], checkpoint_sha256=row['identity']['checkpoint']['sha256']) for row in m['records']]
    c.require(actual == [r['identity'] for r in m['records']], 'Actual native8 identity changed')
    reuse = c.read(c.HERE / 'originals/SOURCE_REUSE.json')
    functions = {}
    for item, filename in zip(reuse['function_sources'], ('evaluator.py', 'replay.py')):
        text = (c.HERE / 'originals' / filename).read_text(encoding='utf-8')
        for node in ast.parse(text).body:
            if isinstance(node, ast.FunctionDef) and node.name in item['functions']:
                segment = ast.get_source_segment(text, node).encode()
                c.require(hashlib.sha256(segment).hexdigest() == item['functions'][node.name], 'Scientific source body changed')
                functions[node.name] = item['functions'][node.name]
    c.require(len(functions) == 17 and ev.TOLERANCE == 1e-12, 'Original17/tolerance changed')
    before, after = c.original_runtime_nodes(), c.runtime_nodes()
    raw_replay = next(n for n in before.body if n.name == 'replay_one')
    new_replay = copy.deepcopy(next(n for n in after.body if n.name == 'replay_one'))
    for node in ast.walk(new_replay):
        if isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith('Exactly8 adopted Hybrid IID Benign checkpoints;'):
            node.value = 'Exactly47 accepted FLGMM checkpoints; finite partial coverage, not full100, method ranking, final primary endpoint, test, or CUDA equivalence'
    c.require(ast_sha(new_replay) == ast_sha(raw_replay), 'Original replay science changed beyond report metadata')
    before_gate = next(n for n in before.body if n.name == 'resource_gate')
    new_gate = copy.deepcopy(next(n for n in after.body if n.name == 'resource_gate'))
    count = 0
    for node in ast.walk(new_gate):
        if isinstance(node, ast.Compare) and ast.unparse(node) == 'len(os.sched_getaffinity(0)) == len(RESOURCE_CPUS)':
            node.comparators = [ast.Constant(value=8)]; count += 1
    c.require(count == 1 and ast_sha(new_gate) == ast_sha(before_gate), 'Resource change exceeded affinity-cardinality binding')
    node, helpers = saved.selected_originals()
    original_block = next(n for n in node.body if isinstance(n, ast.With))
    block = helpers['saved_output_block'](original_block)
    c.require([ast_sha(n) for n in block.body] == [ast_sha(original_block.body[i]) for i in (0, 1, 4, 5, 6, 7, 8)], 'Seven original saved-output statements changed')
    refusals = []
    for label, change in [('skip91002', lambda x: x['exact_ids'].__setitem__(0, x['previously_accepted_skip_ids'][0])),
                          ('wrong_checkpoint', lambda x: x['records'][0]['runtime_artifacts']['model'].__setitem__('sha256', '0' * 64)),
                          ('test', lambda x: x['records'][0]['identity']['config'].__setitem__('celeba_evaluation_split', 'test'))]:
        changed = copy.deepcopy(m); change(changed)
        try:
            c.validate_manifest(changed)
        except (ValueError, KeyError) as exc:
            refusals.append(dict(case=label, error=str(exc)))
        else:
            raise AssertionError('Invalid metadata accepted: ' + label)
    c.require('torch' not in sys.modules, 'Source check imported Torch')
    return dict(status='PASS_EXACT8_SOURCE_AND_ACTUAL_METADATA_ONLY_NO_DISPATCH', exact_ids=c.IDS,
        candidate_sha256=c.sha(c.HERE / 'candidate.py'), manifest_sha256=c.sha(c.HERE / 'MANIFEST.json'),
        saved_consumer_sha256=c.sha(c.HERE / 'check_saved.py'), compiled=compiled,
        actual_private_identity_records=8, original_scientific_functions=functions,
        original_replay_AST_exact_after_reversing_scope_string=True, original_resource_gate_AST_exact_after_reversing_affinity_binding=True,
        original_whole_check_saved_source_sha256=saved.SAVED_SHA, seven_saved_output_statements_AST_exact=True,
        shared_calibration=ev.SHARED_CALIBRATION, first_native_canary=c.IDS[0], remaining_sequential=7,
        CPU_affinity_pending=True, affinity_cardinality_options=[8, 32], compute_threads=8,
        reserved_CPUs=list(range(11, 19)) + list(range(32, 64)) + list(range(102, 120)),
        Linux_whole_CPU110_requires_FL_EXITED_and_free=True, refusal_checks=refusals,
        torch_imported=False, model_or_array_opened=False, forward_calls=0, fit_calls=0,
        new_three_view_accepted=0, dispatch_authorized=False)


if __name__ == '__main__':
    print(json.dumps(main(), indent=2, allow_nan=False))
