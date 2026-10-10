"""Private Hybrid identity view for twelve adopted native checkpoints; metadata only."""
from __future__ import annotations

import argparse
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
METHOD = 'CosineFairnessHybrid'
PACKAGE = 'a87a050b497a184efbe18b4649ad6bde40b9ea29a16f9e3efddd0ef1156e3b04'
INPUTS_SHA256 = 'bc4454c87ed5ce9e8f413448125a7e9d2aed87bbaca5f98051876071df036de6'


def require(ok, message):
    if not ok:
        raise ValueError(message)


def read_pin(pin, *, decode=True):
    """Only small source/JSON bytes: never open models, archives or arrays."""
    path = Path(pin['path'])
    require(path.suffix in {'.json', '.py'} and path.stat().st_size <= 2_000_000,
            'Not a compact metadata/source input: ' + str(path))
    raw = path.read_bytes()
    require(hashlib.sha256(raw).hexdigest() == pin['sha256'], 'Pinned bytes changed: ' + str(path))
    require(len(raw) == pin['bytes'], 'Pinned size changed')
    return json.loads(raw) if decode else raw


def inputs():
    raw = (HERE / 'INPUTS.json').read_bytes()
    require(hashlib.sha256(raw).hexdigest() == INPUTS_SHA256, 'Private registry bytes changed')
    return json.loads(raw)


def original_bridge():
    pin = inputs()['original_bridge']
    read_pin(pin, decode=False)
    spec = importlib.util.spec_from_file_location('_hybrid_private_original_bridge', pin['path'])
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def science_bindings(*, torch_module=None, pandas_module=None):
    """Original loader/code objects and frozen recipe; constructs, never calls science."""
    return original_bridge().science_bindings(torch_module=torch_module, pandas_module=pandas_module)


def one(records, rid):
    matches = [r for r in records if r['id'] == rid]
    require(len(matches) == 1, 'Missing/duplicate proof ID')
    return matches[0]


def identity_record(rid, *, checkpoint_sha256=None):
    m = inputs()
    require(rid in m['exact_ids'], 'ID not in adopted12')
    selected = m['records'][rid]
    batch = m['batches'][selected['batch']]
    root, strict, off, receipt, members = [read_pin(batch[k])
        for k in ('root', 'strict', 'offserver', 'receipt', 'members')]
    bound = read_pin(m['bound_root'])
    require(bound['status'] == 'ROOT_HYBRID100_BOUND_METADATA_ADOPTED'
            and bound['package_sha256'] == root['package_sha256'] == PACKAGE, 'Wrong bound package')
    require(root['status'] == batch['root_status'] and rid in root['accepted_new_ids'], 'Root did not adopt ID')
    require(root['remote_strict_sha256'] == batch['strict']['sha256']
            and root['offserver_sha256'] == batch['offserver']['sha256']
            and root['receipt_sha256'] == batch['receipt']['sha256'], 'Broken root proof links')
    require(off['status'] == 'PASS_FULL_MEMBER_SHA_AND_ORIGINAL_SAVED_COMPARISON'
            and strict['status'] == off['local']['status'] == 'PASS_SAVED_ORIGINAL_FULLCOVERAGE_RECORD', 'Original strict failed')
    require(off['receipt_sha256'] == batch['receipt']['sha256']
            and off['inventory_sha256'] == receipt['inventory_sha256'] == batch['members']['sha256']
            and off['archive_sha256'] == receipt['archive_sha256'] == root['archive_sha256'], 'Broken archive/member links')
    require(strict['source_data_before'] == strict['source_data_after'], 'Source/data drift')
    require(strict['package_sha256'] == off['local']['package_sha256'] == PACKAGE, 'Wrong strict package')
    sr, ore = one(strict['records'], rid), one(off['local']['records'], rid)
    require(sr == ore and rid in strict['accepted_new_ids'], 'Strict/offserver record drift')
    for prior_name, prior_pin in batch.get('prior', {}).items():
        require(root[prior_name] == prior_pin['sha256'], 'Root prefix pin drift')
        read_pin(prior_pin)
    docs = {}
    for kind, artifact in selected['metadata'].items():
        require(members['files'][artifact['member']] ==
                {'sha256': artifact['sha256'], 'bytes': artifact['bytes']}, 'Member binding changed')
        docs[kind] = read_pin(artifact)
    job, result, prov, acceptance, native = [docs[k]
        for k in ('job', 'result', 'provenance', 'acceptance', 'native_replay')]
    scope = read_pin(batch['scope'])
    require(strict['source_data_before'] == scope['protected_source_hashes'], 'Strict/source scope drift')
    require(hashlib.sha256(read_pin(batch['body'], decode=False)).hexdigest()
            == scope['local_hashes']['body.py'], 'Original body source drift')
    body_source = read_pin(batch['body'], decode=False).decode('utf-8')
    checked = next(n for n in ast.parse(body_source).body if isinstance(n, ast.FunctionDef) and n.name == 'checked')
    require(hashlib.sha256(ast.get_source_segment(body_source, checked).encode()).hexdigest()
            == strict['original_checked_source_sha256'] == m['hybrid_function_hashes']['body.checked'], 'Wrong original strict function')
    original_bridge().validate_metadata(METHOD, job, result, prov, acceptance, sr, ore, scope)
    require(job['id'] == rid and job['phase'] == 'fullcoverage'
            and job['tuning_candidate'] == bound['selected_recipe']['id']
            and job['adapter'] == bound['selected_recipe']['adapter']
            and job['config']['learning_rate'] == bound['selected_recipe']['learning_rate'], 'Recipe/job drift')
    require(acceptance['job_sha256'] == prov['job_sha256'] == selected['metadata']['job']['sha256']
            and acceptance['scope_sha256'] == prov['scope_sha256'] == batch['scope']['sha256'], 'Job/scope drift')
    require(selected['metadata']['acceptance']['sha256'] == sr['acceptance_sha256'], 'Acceptance bytes drift')
    for kind in ('result', 'provenance', 'native_replay'):
        require(acceptance['artifact_hashes'][kind + '.json'] == selected['metadata'][kind]['sha256'], 'Accepted artifact drift')
    model = members['files'][selected['checkpoint']['member']]
    require(model['sha256'] == sr['checkpoint_sha256'] == acceptance['artifact_hashes']['model.pt']
            and model['bytes'] == selected['checkpoint']['bytes'], 'Wrong model member')
    require(checkpoint_sha256 is None or checkpoint_sha256 == model['sha256'], 'Caller checkpoint mismatch')
    require([r['round'] for r in result['trajectory_metrics']] == list(range(1, 71))
            and [r['round'] for r in result['round_summaries']] == list(range(1, 71))
            and result['trajectory_metrics'][-1]['metrics'] == result['metrics'], 'Not the same terminal70 checkpoint')
    image = result['data_contract']['image_data_contract']
    require(native['metrics'] == result['metrics'] and native['prediction_count'] == 19867
            and native['checkpoint_tensor_sha256'] == acceptance['checkpoint_tensor_sha256']
            and native['root_group_label_total'] == result['data_contract']['root_clean_rows'] == 16277
            and native['root_image_ids_sha256'] == image['root_image_ids_sha256']
            and native['evaluation_image_ids_sha256'] == image['evaluation_image_ids_sha256'], 'Native/root/valid drift')
    # GPU provenance remains historical fact; no local CUDA queries or substitution.
    require(prov['torch'] == '2.11.0+cu128' and prov['cuda_build'] == '12.8'
            and prov['device'] == 'cuda:0' and prov['gpu_uuid'] == strict['server_runtime']['gpu_uuid'], 'Training provenance drift')
    return dict(id=rid, method=METHOD, source_method=job['method'], config=copy.deepcopy(job['config']),
        distribution=job['distribution'], attack=job['attack'], seed=job['config']['seed'],
        actual_alpha=result['alpha'], terminal_round=70, original_split='valid',
        config_canonical_sha256=original_bridge().science_bindings().canonical_sha(job['config']),
        data_contract=copy.deepcopy(image), prior_validation_metrics=copy.deepcopy(sr['metrics']),
        checkpoint=copy.deepcopy(selected['checkpoint']), result=copy.deepcopy(selected['metadata']['result']),
        raw_job=copy.deepcopy(selected['metadata']['job']), source_hashes=copy.deepcopy(prov['source_hashes']),
        adapter_source_hashes=copy.deepcopy(prov['local_hashes']), training_torch=prov['torch'],
        original_training_provenance=copy.deepcopy(prov), original_remote_output=selected['server_output'],
        external_proof_sha256={k: batch[k]['sha256'] for k in ('root', 'strict', 'offserver', 'receipt', 'members')},
        original_artifact_pins=copy.deepcopy(selected),
        status='PRIVATE_IDENTITY_ONLY_NO_NEW_PREDICTION_OR_FIT', dispatch_authorized=False)


def check():
    m = inputs()
    require(len(m['exact_ids']) == len(set(m['exact_ids'])) == 12, 'Only accepted12')
    roots = [read_pin(b['root']) for b in m['batches'].values()]
    require([r['cumulative_accepted'] for r in roots] == [1, 9, 12]
            and sum([r['accepted_new_ids'] for r in roots], []) == m['exact_ids']
            and roots[-1]['accepted_ids'] == m['exact_ids'], 'Root chain/order drift')
    for pin in m['source_pins'].values():
        read_pin(pin, decode=False)
    ev = science_bindings()
    reuse = read_pin(m['original_reuse'])
    hashes = {k: v for source in reuse['function_sources'] for k, v in source['functions'].items()}
    require(len(hashes) == 17 and ev.TOLERANCE == 1e-12, 'Original science/tolerance drift')
    records = [identity_record(rid) for rid in m['exact_ids']]
    refusals = []
    for name, rid, checkpoint in [('unknown_ID', 'unknown', None),
            ('wrong_checkpoint', m['exact_ids'][0], '0' * 64)]:
        try:
            identity_record(rid, checkpoint_sha256=checkpoint)
        except ValueError as exc:
            refusals.append(dict(name=name, error=str(exc)))
        else:
            raise AssertionError('Invalid metadata accepted: ' + name)
    require('torch' not in sys.modules, 'Metadata check imported Torch')
    return dict(status='PASS_ADOPTED12_PRIVATE_IDENTITY_BRIDGE_METADATA_ONLY',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        inputs_sha256=hashlib.sha256((HERE / 'INPUTS.json').read_bytes()).hexdigest(),
        exact_ids=m['exact_ids'], records_checked=len(records), original_scientific_functions=hashes,
        shared_calibration=ev.SHARED_CALIBRATION, hybrid_function_hashes=m['hybrid_function_hashes'],
        source_pins=m['source_pins'], refusal_checks=refusals, torch_imported=False,
        model_or_array_opened=False, original_GPU_strict_reexecuted=False, forward_calls=0, fit_calls=0,
        new_three_view_accepted=0, dispatch_authorized=False, full100_complete=False,
        native_replay_tolerance=ev.TOLERANCE, runtime_equivalence_claim=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true', required=True)
    parser.parse_args()
    print(json.dumps(check(), indent=2, allow_nan=False))
