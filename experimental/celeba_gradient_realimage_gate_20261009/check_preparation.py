"""Preparation/closed-dispatch identity checks only; never imports torch or trains."""
import ast
import copy
from pathlib import Path
import subprocess
import sys
import tempfile

from gate import HERE, digest, dispatch, read, require, validate_scope, write


def main():
    scope = read(HERE / 'scope.json'); validate_scope(scope)
    for path in HERE.rglob('*.py'):
        ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    checks = ['all_python_ast', 'four_exact_full_image_job_identities', 'all_static_source_snapshot_hashes',
              'five_formal_decisions_unresolved', 'original_protocol_prepared', 'full_train_root_valid_limits']
    for receipt in (None, HERE / 'dispatch_receipt.PENDING.json'):
        try:
            dispatch(scope, receipt)
        except (ValueError, KeyError):
            pass
        else:
            raise ValueError('Unapproved preparation became executable')
    checks += ['no_receipt_rejected', 'pending_receipt_rejected']
    # Reject identity/CPU overlap before any scientific or resource work, no synthetic gradients.
    prepared = read(HERE / 'dispatch_receipt.PENDING.json')
    with tempfile.TemporaryDirectory(prefix='refusal_', dir=HERE) as directory:
        directory = Path(directory)
        base = dict(prepared, status='APPROVED_BOUNDED_EXPLORATORY_GATE_ONLY', exclusive_cpu_ids=list(range(24, 32)),
                    verified_no_overlap_with_live_cpu_gates=True)
        for name, changes in (('job_hash', {'jobs': {}}), ('overlap', {'exclusive_cpu_ids': list(range(8, 16))}),
                              ('formal_approval', {'formal_decisions_approved': True}), ('test', {'test_authorized': True}),
                              ('screen64', {'screen64_authorized': True}), ('resource', {'exclusive_cpu_ids': None})):
            candidate = copy.deepcopy(base); candidate.update(changes); path = directory / (name + '.json'); write(path, candidate)
            try:
                dispatch(scope, path)
            except (ValueError, KeyError):
                checks.append('reject_' + name)
            else:
                raise ValueError('Unsafe receipt accepted: ' + name)
    process = subprocess.run([sys.executable, str(HERE / 'gate.py'), 'run'], capture_output=True, text=True)
    require(process.returncode != 0 and 'CPU allocation/dispatch receipt required' in process.stderr, 'Missing receipt did not close launch')
    require(not (HERE / 'runs').exists() and 'torch' not in sys.modules, 'Preparation imported torch or created training outputs')
    checks += ['run_cli_refuses_before_torch_or_output', 'no_training_output_created']
    report = dict(status='PASS_PREPARATION_ONLY', checks=checks, count=len(checks), actual_image_runs=0,
                  actual_training_rounds=0, torch_imported=False, scope_sha256=digest(HERE / 'scope.json'))
    write(HERE / 'preparation_checks.json', report)
    print(report)


if __name__ == '__main__':
    main()
