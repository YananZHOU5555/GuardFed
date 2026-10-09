"""Reuse the accepted worker, with isolated controls and candidate-level assertions."""
import argparse
import importlib.util
import json
import math
import os
from pathlib import Path
import sys
from types import SimpleNamespace
from adapter import mechanism, verify_ledger


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module; spec.loader.exec_module(module)
    return module


def checked(original, job, job_path):
    out = Path(job['output'])
    assert not list(out.glob('failure*.json')), 'Preserved failure requires review'
    result = original.checked_result(job)
    if result is None:
        return None
    assert result['metrics'] == result['trajectory_metrics'][-1]['metrics']
    assert all(math.isfinite(result['metrics'][key]) for key in ('accuracy', 'aeod', 'aspd'))
    assert result['config']['celeba_evaluation_split'] == 'valid'
    assert result['seed'] == job['config']['seed']
    assert result['distribution'] == job['distribution'] and result['attack'] == job['attack']
    assert float(result['alpha']) == float(job['config']['client_alpha'])
    contract = result['data_contract']['image_data_contract']
    assert (contract['evaluation_split'], contract['actual_train_rows'], contract['actual_evaluation_rows']) == ('valid', 162770, 19867)
    assert contract['train_eval_disjoint'] and contract['root_client_disjoint']
    import torch
    state = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    assert state and all(torch.isfinite(t).all().item() for t in state.values())
    audit = json.loads((out / 'mechanism_acceptance.json').read_text())
    ledger = json.loads((out / 'candidate_mask_audit.json').read_text())
    assert audit['pass'] and audit['job_sha256'] == original.digest(job_path)
    assert audit['result_sha256'] == original.digest(out / 'result.json')
    assert audit['checkpoint_sha256'] == result['revision_job']['checkpoint_sha256']
    assert audit['candidate_audit_sha256'] == original.digest(out / 'candidate_mask_audit.json')
    assert audit['adapter_hashes'] == job['adapter_hashes']
    assert result['revision_job']['variant'] == job['variant']
    expected = verify_ledger(ledger, job['variant'], job['config']['rounds'])
    assert all(audit[k] == v for k, v in expected.items())
    return result


def worker(repo, job_path):
    if sys.flags.optimize:
        raise RuntimeError('Assertions must remain enabled')
    job = json.loads(job_path.read_text())
    out = Path(job['output'])
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.path.insert(0, str(repo / 'scripts')); sys.path.insert(0, str(repo))
    original = load('mechanism_original_worker', repo / 'scripts/run_revision_ablation.py')
    for name, expected in job['adapter_hashes'].items():
        assert original.digest(name) == expected, ('adapter drift', name)
    protocol = repo / 'results/revision_20261009/celeba_mechanism_v1/PROTOCOL.md'
    assert original.digest(protocol) == job['protocol_sha256']
    for name, expected in job['source_hashes'].items():
        assert original.digest(repo / name) == expected, ('source/data drift', name)
    prior = checked(original, job, job_path)
    if prior is not None:
        return
    assert not out.exists(), 'Partial output requires preserved recovery review'
    core = load('reproduce_paper_tables', repo / 'scripts/reproduce_paper_tables.py')
    with mechanism(core, job['variant']) as ledger:
        original.worker(SimpleNamespace(job=str(job_path)))
    result = original.checked_result(job)
    assert result is not None and result['metrics'] == result['trajectory_metrics'][-1]['metrics']
    assert result['config']['celeba_evaluation_split'] == 'valid'
    contract = result['data_contract']['image_data_contract']
    assert (contract['evaluation_split'], contract['actual_train_rows'], contract['actual_evaluation_rows']) == ('valid', 162770, 19867)
    assert contract['train_eval_disjoint'] and contract['root_client_disjoint']
    audit = verify_ledger(ledger, job['variant'], job['config']['rounds'])
    original.write_json(out / 'candidate_mask_audit.json', ledger)
    audit.update(job_sha256=original.digest(job_path), result_sha256=original.digest(out / 'result.json'),
                 checkpoint_sha256=result['revision_job']['checkpoint_sha256'],
                 candidate_audit_sha256=original.digest(out / 'candidate_mask_audit.json'),
                 adapter_hashes=job['adapter_hashes'])
    original.write_json(out / 'mechanism_acceptance.json', audit)
    assert checked(original, job, job_path) is not None


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--repo', type=Path, required=True); p.add_argument('--job', type=Path, required=True)
    args = p.parse_args(); worker(args.repo.resolve(), args.job.resolve())
