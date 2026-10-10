"""Independent stdlib-only source review; no bind, Torch, arrays or image jobs."""
from pathlib import Path
import ast, copy, hashlib, importlib.util, json, sys
from unittest.mock import patch

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'tmp/celeba_gradient_fullcoverage_gates_prepare_20261010'
SEAL = 'f068ad7f51fd6981b2211725d39009a5cb5ebbecfa1de680cf0d46aa52ef7760'
H = lambda b: hashlib.sha256(b).hexdigest()
read = lambda p: json.loads(Path(p).read_bytes())


def functions(text, nested=False):
    tree = ast.parse(text)
    nodes = ast.walk(tree) if nested else tree.body
    return {n.name: ast.get_source_segment(text, n) for n in nodes if isinstance(n, ast.FunctionDef)}


def load(name):
    spec = importlib.util.spec_from_file_location(name, SOURCE / (name + '.py'))
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def run():
    seal_path = SOURCE / 'FILES_SHA256.json'
    assert H(seal_path.read_bytes()) == SEAL
    seal = read(seal_path)
    assert len(seal['files']) == 8
    for name, pin in seal['files'].items():
        data = (SOURCE / name).read_bytes()
        assert H(data) == pin['sha256'] and len(data) == pin['bytes']
    assert sum(row['bytes'] for row in seal['files'].values()) == 40350
    metadata, adapter = load('metadata'), load('adapter')
    coverage = metadata.coverage_source()
    protocol, manifest, _, _ = coverage.inputs()
    rendered, _ = coverage.render()
    original_worker = coverage.OLD / coverage.BRIDGE / 'worker.py'
    old_worker = metadata.OLD / 'snapshot/gradient_bridge_20261009/worker.py'
    assert original_worker.read_bytes() == old_worker.read_bytes()
    original_functions = functions(original_worker.read_text(encoding='utf8'))
    derived_functions = functions(rendered['worker.py'])
    science = sorted(set(original_functions) - {'validate_job', 'run'})
    assert len(science) == 14
    assert all(original_functions[n] == derived_functions[n] for n in science)
    original_gate = functions((metadata.OLD / 'gate.py').read_text(encoding='utf8'))
    adapted_gate, diff = adapter.render_gate_functions()
    assert (SOURCE / 'SOURCE_DIFF.patch').read_text(encoding='utf8') == diff
    old_nested = functions(original_gate['run_one'], True)
    new_nested = functions(adapted_gate['run_one'], True)
    exact_nested = ['retained_bundle', 'forbidden', 'root_adam', 'real_gradient', 'real_attack', 'progress']
    assert all(old_nested[n] == new_nested[n] for n in exact_nested)
    assert old_nested['real_aggregate'].replace("int(job['attack'] == 'S-DFA')",
        "int(job['attack'] in {'FedSA', 'S-DFA', 'Sp-DFA'})") == new_nested['real_aggregate']
    assert '70' in functions(rendered['accept_result.py'])['checked_result']
    assert '3' in adapted_gate['checked'] and "job['config']['seed']" in adapted_gate['checked']
    # Shape check only: these are declared metadata examples, never selected recipes.
    candidates = [next(c for c in protocol['candidates'] if c['method'] == m) for m in coverage.METHODS]
    jobs = [j for candidate in candidates for j in metadata.jobs_for(candidate, protocol)]
    metadata.validate_jobs(jobs, candidates, protocol)
    assert len({j['id'] for j in jobs}) == 14
    assert all((j['distribution'], j['config']['seed'], j['config']['client_alpha'],
        j['config']['rounds'], j['config']['batch_size'], j['config']['device']) ==
        ('non-IID', 91002, 5.0, 3, 64, 'cpu') for j in jobs)
    pairs = [(m, a) for m in coverage.METHODS for a in ('F Flip', 'FedSA', 'Sp-DFA')]
    assert len(pairs) == 6
    assert all({j['implementation'] for j in jobs if (j['method'], j['attack']) == pair} ==
        {'screen', 'coverage'} for pair in pairs)
    assert sum(j['attack'] == 'Benign' and j['implementation'] == 'screen' for j in jobs) == 2
    # Extra independent checks target runtime metadata, which the author's source suite did not exercise.
    scope = {'jobs': [{'id': j['id'], 'job_sha256': H(j['id'].encode())} for j in jobs]}
    receipt = dict(status='ROOT_APPROVED_GRADIENT14_CPU3_CANARIES_ONLY', scope_sha256='fixture',
        jobs={e['id']: e['job_sha256'] for e in scope['jobs']}, test_authorized=False,
        coverage192_authorized=False, automatic_retry=False, measured_unix=10000.0,
        exclusive_cpu_ids=list(range(8)), no_restricted_cpu_overlap=True, old_gradient64_exited=True,
        protected_main_healthy=True, no_duplicate_gate_worker=True, protected_main_growth_or_completed=True,
        nominal_existing_compute_threads=96, actual_cpu_quota=122.88, ram_available_bytes=8*1024**3)
    mutations = {
        'scope_sha': lambda r: r.update(scope_sha256='wrong'),
        'job_sha': lambda r: r['jobs'].update({scope['jobs'][0]['id']: 'wrong'}),
        'stale': lambda r: r.update(measured_unix=9879.9),
        'future': lambda r: r.update(measured_unix=10000.1),
        'duplicate_cpu': lambda r: r.update(exclusive_cpu_ids=[0]*8),
        'overlap': lambda r: r.update(no_restricted_cpu_overlap=False),
        'old64_alive': lambda r: r.update(old_gradient64_exited=False),
        'unhealthy_main': lambda r: r.update(protected_main_healthy=False),
        'duplicate_worker': lambda r: r.update(no_duplicate_gate_worker=False),
        'quota': lambda r: r.update(actual_cpu_quota=103.99),
        'ram': lambda r: r.update(ram_available_bytes=8*1024**3-1),
        'test_permission': lambda r: r.update(test_authorized=True),
        'coverage_permission': lambda r: r.update(coverage192_authorized=True),
        'retry_permission': lambda r: r.update(automatic_retry=True),
    }
    current = [receipt]
    with patch.object(adapter, 'H', lambda _: 'fixture'), patch.object(adapter, 'read', lambda _: current[0]), \
            patch.object(Path, 'read_bytes', lambda _: b'fixture'), patch.object(adapter.time, 'time', lambda: 10000.0):
        assert adapter.runtime_receipt(HERE, scope, HERE/'fixture.json', 'fixture') == receipt
        for name, mutate in mutations.items():
            current[0] = copy.deepcopy(receipt); mutate(current[0])
            try:
                adapter.runtime_receipt(HERE, scope, HERE/'fixture.json', 'fixture')
            except AssertionError:
                pass
            else:
                raise AssertionError('Runtime metadata drift passed: ' + name)
    comparison_tree = ast.parse((SOURCE / 'compare_saved.py').read_text(encoding='utf8'))
    called = {n.func.attr for n in ast.walk(comparison_tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)}
    assert not called.intersection({'run_one', 'train_pipeline', 'make_model', 'fit_group_thresholds', 'evaluate_for_reporting'})
    assert 'torch' not in sys.modules and 'numpy' not in sys.modules
    assert H(seal_path.read_bytes()) == SEAL
    for name, pin in seal['files'].items():
        assert H((SOURCE / name).read_bytes()) == pin['sha256']
    return dict(status='INDEPENDENT_SOURCE_REVIEW_PASS_NOT_RUNTIME_OR_DISPATCH_APPROVAL', source_adoptable=True,
        actual_execution_authorized=False, source_seal_sha256=SEAL,
        handoff_sha256=seal['files']['HANDOFF.json']['sha256'], source_members_verified=8, source_bytes=40350,
        original_worker_sha256=H(original_worker.read_bytes()), original_gate_sha256=metadata.PINS['gate.py'],
        coverage_preparation_seal_sha256=H((metadata.COVER/'FILES_SHA256.json').read_bytes()),
        screen64_source_seal_sha256=coverage.SEAL, science_functions_source_exact=science,
        gate_nested_oracles_source_exact=exact_nested,
        aggregate_oracle_only_expected_root_call_predicate_changed=True,
        declared_source_diff_exact=True, formal_acceptor_horizon_70_unchanged=True,
        metadata_examples_only=dict(unique_jobs=14, screen_coverage_pairs=6, fflip_benign_null_pairs=2,
            methods=2, seed=91002, distribution='non-IID', alpha=5.0, rounds=3, batch_size=64),
        extra_runtime_metadata_positive=1, extra_runtime_metadata_refusals=list(mutations),
        saved_comparison_does_not_call_inference_training_or_fitting=True,
        torch_imported=False, numpy_imported=False, author_suite_rerun=False,
        actual64_records_revalidated=False, frozen192_release_bound=False, actual_jobs_bound=0,
        actual_image_jobs=0, scientific_table_records=0, findings=[],
        limitations=['Actual complete64 root/summary and explicit frozen192 external source approval remain binding prerequisites.',
            'Runtime receipt is a root-measured 120-second resource attestation; it does not independently scan live CPU owners or GPU/IO health.',
            'Saved equality covers declared Python/global NumPy/Torch CPU RNG fingerprints, not every possible independent generator.',
            'The short CPU gate supplies no GPU equivalence, 70-round performance result, offserver adoption or 192-job launch permission.',
            'Metadata-only F Flip null is specific to unweighted CE with unchanged Smiling labels and image inputs.'])


if __name__ == '__main__':
    report = run()
    with (HERE / 'REVIEW.json').open('x', encoding='utf8') as handle:
        json.dump(report, handle, indent=2); handle.write('\n')
    print(json.dumps({'status': report['status'], 'source_members': 8, 'science_functions': 14,
        'extra_metadata_refusals': len(report['extra_runtime_metadata_refusals']), 'actual_jobs': 0}))
