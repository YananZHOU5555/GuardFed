"""Only the Supervisor terminal-code and zero-retry v2 engineering regression."""
import ast
import copy
import difflib
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch

sys.dont_write_bytecode = True
import evaluate_remaining as queue
from bridge_adapter import digest, read, require

HERE = Path(__file__).resolve().parent
OLD = HERE.parent / 'celeba_mechanism_remaining_evaluation_20261010'
OLD_SEAL = '101b0c2798456990885bd1db8305f7f47b6af49dca33c63b99edcc58e13e65f0'


def main():
    require(digest(OLD / 'FILES_SHA256.json') == OLD_SEAL, 'Original29 seal changed')
    for name, pin in read(OLD / 'FILES_SHA256.json')['files'].items():
        require(digest(OLD / name) == pin['sha256'] and (OLD / name).stat().st_size == pin['bytes'], 'Original29 member changed: ' + name)
    old = (OLD / 'evaluate_remaining.py').read_text(encoding='utf-8')
    new = (HERE / 'evaluate_remaining.py').read_text(encoding='utf-8')
    def funcs(text): return {n.name: ast.get_source_segment(text, n) for n in ast.parse(text).body if isinstance(n, ast.FunctionDef)}
    a, b = funcs(old), funcs(new)
    require(set(a) == set(b) and all(a[name] == b[name] for name in a if name != 'training_snapshot'), 'Non-guard function changed')
    require((HERE / 'bridge_adapter.py').read_bytes() == (OLD / 'bridge_adapter.py').read_bytes(), 'Original science projection changed')
    require(read(HERE / 'PLAN.json') == read(OLD / 'PLAN.json'), '620 plan/scope changed')
    plan = read(HERE / 'PLAN.json'); ids = [r['id'] for r in plan['all_manifest_entries']]
    results = []
    cases = [('rc0_RUNNING', 0, 'guardfed_celeba_mechanism_formal RUNNING pid1', 180, True),
             ('rc3_EXITED800', 3, 'guardfed_celeba_mechanism_formal EXITED Oct10', 800, True),
             ('rc3_EXITED_partial', 3, 'guardfed_celeba_mechanism_formal EXITED Oct10', 799, False),
             ('rc3_unknown', 3, 'guardfed_celeba_mechanism_formal UNKNOWN', 800, False)]
    for label, rc, stdout, n, expected in cases:
        observed = SimpleNamespace(returncode=rc, stdout=stdout, stderr='TEST_ONLY_CAPTURED_STDERR')
        snapshot = dict(completed=ids[:n], active=[], failed=[])
        with patch.object(queue, 'read', return_value=snapshot), patch.object(queue, 'digest', return_value='TEST_ONLY_SHA'), patch.object(queue.subprocess, 'run', return_value=observed):
            try:
                _, proof = queue.training_snapshot(plan, None)
                accepted = True
                require(proof['service_returncode'] == rc and proof['service_stderr'] == observed.stderr, 'Actual Supervisor input not retained')
            except ValueError as exc:
                accepted = False; proof = {'error': str(exc)}
        require(accepted == expected, 'Supervisor compatibility regression: ' + label)
        results.append(dict(case=label, accepted=accepted, expected=expected, guard_proof=proof))
    conf = (HERE / 'supervisor.conf.template').read_text(encoding='utf-8')
    require('startretries=0' in conf and 'autostart=false' in conf and 'autorestart=false' in conf, 'No-retry supervisor contract missing')
    require('torch' not in sys.modules, 'No-CNN check imported Torch')
    patch_text = ''.join(difflib.unified_diff(old.splitlines(True), new.splitlines(True), fromfile='sealed_v1/evaluate_remaining.py', tofile='v2/evaluate_remaining.py'))
    patch_text += ''.join(difflib.unified_diff((OLD / 'supervisor.conf.template').read_text().splitlines(True), conf.splitlines(True), fromfile='sealed_v1/supervisor.conf.template', tofile='v2/supervisor.conf.template'))
    with (HERE / 'ENGINEERING_DIFF.patch').open('x', encoding='utf-8') as f: f.write(patch_text)
    queue.save_new(HERE / 'V2_CHECK.json', dict(status='NO_CNN_SUPERVISOR_COMPATIBILITY_AND_ZERO_RETRY_PASS',
        original29_seal_sha256=OLD_SEAL, original29_members_unchanged=True, original_guard_other_functions_source_exact=True,
        original_science_projection_bytes_exact=True, original_PLAN_bytes_exact=digest(HERE / 'PLAN.json') == digest(OLD / 'PLAN.json'),
        cases=results, startretries=0, stderr_retained=True, CNN=0, Torch_imported=False, SSH=0, accepted_offserver=0))
    print(json.dumps({'status': 'PASS', 'cases': len(results), 'CNN': 0}))


if __name__ == '__main__':
    main()
