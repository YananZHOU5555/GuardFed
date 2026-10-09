"""Local manifest preparation/check only. Never imports Torch or dispatches work."""
from pathlib import Path
import argparse
import ast
import collections
import copy
import difflib
import hashlib
import json
import sys

sys.dont_write_bytecode = True

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
OLD = REPO / 'tmp/celeba_final_valid_replay_20261009'
FAIL = OLD / 'v4/remaining872_attempt1/chunk_036'
DIAG = REPO / 'tmp/celeba_native_mismatch_diagnostic_execution_20261009'
GPU = REPO / 'tmp/celeba_native_mismatch_diagnostic_prepared_20261009'
RESTORE = REPO / 'docs/server_deployment_20260923/training_20260923/validation900_restore_20261009'
INVENTORY = REPO / 'docs/server_deployment_20260923/training_20260923/final_evaluation_prepared_20261009/model_inventory.json'
FAILED_ID = 'FairGuard_IID_F-Flip_seed91009'
PINS = {
    INVENTORY: '3486a9f185f40294553de9e487e9d9b5d142098a819f6cfb3adbbd2d6a6420cd',
    OLD / 'v4/remaining872_execution_20261009/cumulative_424_accepted.json': '75ec99bc1eb9fb2c5e20ead68aad7af867d11cb51f2cab991d01712568985fdd',
    OLD / 'v4/remaining872_prepared_v2_20261009/manifest.json': 'ad6eebf517f534fb8489acb241c51a9ec5328bb285406e55275f7dd9c0c3ed43',
    FAIL / 'strict_acceptance.json': '251fe691fedab5574ec3118e1e51f491155a3d037fbf78d50e65ea04a0f7f39b',
    FAIL / 'failure_input_bindings.json': 'c1dfbb8183aadb668ef3643c2c1b46e8d2628ce149142e06d7ab0f4acae842f2',
    FAIL / 'failure_offserver_verification.json': '8e37410b4ec83f57d5550da962997f449cfd8a07934627327344a4d6eddc5bdc',
    FAIL / 'failure_remote_archive_inventory.json': 'b3d4fa28c4e77eaa6cbd46584c2847b183c1b6fb387d01a852defc1c25364efd',
    RESTORE / 'storage_map.json': 'e160416101ccb82b42224c9ff1bb337de45ef72fa251197063d1c15e2910f949',
    RESTORE / 'restore_bundle.json': '0cfa391b0d317622551d4c6526f2d14ec33c3dda0f3e9e523066bd3718cbcf3a',
    RESTORE / 'restore_acceptance.json': '5114c2cd96e5b8ffaf46e40a341619dd8b3547f89d417263c19c6e7f1f33bf77',
    OLD / 'replay.py': '8476d651bd281d97d6fed42feac5569b45ab0e881b047a7962201e6c0b300803',
    OLD / 'v3/replay_v3.py': 'abb4560dd1c6752b9017a739381d2c54e68f52e278075c8e225a1c67df35cc6e',
    OLD / 'v4/replay_v4.py': '43b16d20d2497b7762cd0f6039f7f4bfc8fddd4e4c32979a16291b26e394ae5e',
    OLD / 'inputs/evaluator.py': '805eedf1fb08137cd86a543a80c83b9527e5c8937be02f7d2dca83a33b86e04c',
    GPU / 'gpu_replay_body.py': '2aafb0d08b2a0c84bbcfc224638dc216329f72d450facf37fd2e50d1cb0f0ad5',
    DIAG / 'attempt2/run_once.py': 'c0c6b199a4db9abbdcf5d2b1f6489b50608128f3386fe648ba3f7ef0b9d6974b',
    DIAG / 'attempt2/ATTEMPT2_DIFF.patch': '9a9200714f3e6301687e2ef5b3d35bfefb7a4f413a7dff2c1b2ee2b215f610e5',
    DIAG / 'FINAL_DELIVERY.json': 'c34f9d45b91fc7f1a91de87dd247cb6a76490f22651febe53d40474aaf9f4e83',
    DIAG / 'attempt2_backup/OFFSERVER_VERIFICATION.json': '26779b047b792b2b352153a5e16ba0a4acec413212fa9d3799c750cc9ce0ff92',
    DIAG / 'attempt2_backup/MEMBERS.json': 'd4ce8cdcaf86718df209071a1e44a5b256bf3c0364c0db67fb65c986e88146d8',
    DIAG / 'offline_comparison/OFFLINE_ACCEPTANCE.json': 'a50f7a7ce2a2fe81867c9d4475b91a85eaab24bdaa6ce010596249a71fc61f88',
    DIAG / 'offline_comparison/compare_saved_v2.py': '0c4eaf89f6c7d9f28f3cbb9253c795a1abf7fd5b5d5cedb1e91aac9f5b5fe849',
}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def write(name, value):
    with (HERE / name).open('x', encoding='utf-8', newline='\n') as out:
        json.dump(value, out, ensure_ascii=False, indent=2, allow_nan=False)
        out.write('\n')


def derive():
    for path, expected in PINS.items():
        need(digest(path) == expected, 'Frozen input changed: ' + str(path))
    inventory = read(INVENTORY)
    records = {row['id']: row for row in inventory['records']}
    prior = read(OLD / 'v4/remaining872_execution_20261009/cumulative_424_accepted.json')
    old = read(OLD / 'v4/remaining872_prepared_v2_20261009/manifest.json')
    strict = read(FAIL / 'strict_acceptance.json')
    diag = read(DIAG / 'FINAL_DELIVERY.json')
    accepted, partial = set(prior['accepted_ids']), set(strict['accepted_ids'])
    untouched = {identity for chunk in old['chunks'][37:] for identity in chunk['ids']}
    missing = set(records) - accepted
    cells = [(r['method'], r['distribution'], r['attack'], r['seed']) for r in records.values()]
    need(len(records) == len(cells) == len(set(cells)) == 900, '900 unique frozen scientific cells required')
    need(len(accepted) == prior['accepted_n'] == 424 and accepted <= set(records), 'Accepted424 identity drift')
    need(missing == set(prior['missing_ids']) and len(missing) == 476, 'Exact900 minus424 complement required')
    need(len(partial) == strict['accepted_n'] == 10 and strict['max_abs_native_metric_difference'] == 0.0, 'Historical CPU partial identity drift')
    need(strict['invalid'] == [{'id': FAILED_ID, 'error_type': 'ValueError', 'error': 'Worker receipt failed or incomplete'}], 'Original failure changed')
    need(set(old['chunks'][36]['ids']) == partial | {FAILED_ID}, 'Original chunk036 identity drift')
    need(len(untouched) == 465 and untouched | partial | {FAILED_ID} == missing and not untouched & partial, '465/10/1 partition drift')
    need(diag['id'] == FAILED_ID and diag['accepted_cohort_count_unchanged'] == 424, 'Diagnostic is not cohort inclusion')
    need(diag['GPU_native_comparison']['accepted'] and diag['GPU_native_comparison']['max_abs_difference'] == 0, 'Diagnostic scientific native check changed')
    mapping = {(r['id'], r['kind']): r for r in read(RESTORE / 'storage_map.json')['records']}
    need(len(mapping) == 2700, 'Unique exact900 three-kind storage binding required')
    rows = []
    for identity in old['remaining_ids']:
        if identity not in missing:
            continue
        r = records[identity]
        need(r['terminal_round'] == 70 and r['original_n_eval'] == 19867 and r['config']['device'] == 'cuda', 'Frozen checkpoint/split/config drift')
        artifacts = {}
        for kind in ('checkpoint', 'result', 'raw_job'):
            bound = mapping[(identity, kind)]
            need(all(bound[k] == r[kind][k] for k in ('sha256', 'bytes', 'archive', 'archive_sha256', 'member')), 'Inventory/storage artifact mismatch')
            artifacts[kind] = {k: bound[k] for k in ('target', 'sha256', 'bytes', 'archive', 'archive_sha256', 'member')}
        category = 'UNEXECUTED_465' if identity in untouched else ('CPU_STRICT_PARTIAL_10_NOT_REGISTERED' if identity in partial else 'GPU_DIAGNOSTIC_1_PENDING_COHORT_DECISION')
        evidence = None
        if identity in partial:
            inv = read(FAIL / 'failure_remote_archive_inventory.json')
            evidence = {'archive': str(FAIL / 'failure_chunk_evidence.tar.gz'), 'archive_sha256': inv['sha256'],
                        'proof_sha256': PINS[FAIL / 'failure_offserver_verification.json'],
                        'members': {name: inv['members']['batch/runs/' + identity + '/' + name] for name in ('receipt.json', 'validation_predictions.npz')}}
        elif identity == FAILED_ID:
            inv = read(DIAG / 'attempt2_backup/MEMBERS.json')
            evidence = {'archive': str(DIAG / 'attempt2_backup/evidence.tar.gz'), 'archive_sha256': diag['attempt2_archive_sha256'],
                        'proof_sha256': PINS[DIAG / 'attempt2_backup/OFFSERVER_VERIFICATION.json'],
                        'members': {name: inv['members']['runs/' + identity + '/' + name] for name in ('receipt.json', 'validation_predictions.npz')},
                        'separate_corrected_offline_acceptance_sha256': PINS[DIAG / 'offline_comparison/OFFLINE_ACCEPTANCE.json']}
        rows.append({'id': identity, 'canonical_cell_id': f"{r['method']}_{r['distribution']}_{r['attack']}_seed{r['seed']}",
                     'scientific_cell': [r['method'], r['distribution'], r['attack'], r['seed']],
                     'inventory_record_sha256': canonical(r), 'config_canonical_sha256': r['config_canonical_sha256'],
                     'data_contract_sha256': canonical(r['data_contract']), 'source_hashes_sha256': canonical(r['source_hashes']),
                     'adapter_source_hashes': r['adapter_source_hashes'], 'training_torch': r['training_torch'],
                     'original_remote_output': r['original_remote_output'], 'classification': category,
                     'artifacts': artifacts, 'preserved_evidence_not_cohort_acceptance': evidence,
                     'planned_action': 'PROPOSE_FRESH_GPU_VALID_REPLAY' if identity in untouched else 'WAIT_ROOT_EXPLICIT_REUSE_OR_RERUN_DECISION',
                     'accepted_by_this_package': False})
    need(len(rows) == 476, '476 ordered manifest rows required')
    return {'schema': 'existing_baseline900_valid_recovery_proposal_v1', 'status': 'PREPARED_NOT_DEPLOYED_NOT_DISPATCHED',
            'scope': 'Exact existing nine-method900 minus source-bound accepted424; excludes mechanism800',
            'inventory_sha256': digest(INVENTORY), 'accepted424_collector_sha256': digest(OLD / 'v4/remaining872_execution_20261009/cumulative_424_accepted.json'),
            'accepted424_ids': prior['accepted_ids'], 'accepted424_provenance': prior['accepted_provenance'],
            'remaining_n': 476, 'classification_counts': dict(collections.Counter(r['classification'] for r in rows)), 'records': rows,
            'root_review_decisions': {'reuse_cpu_partial10': None, 'include_gpu_diagnostic1': None, 'GPU_uuid_and_CPU_slot': None, 'execute_new465': None},
            'root_review_decisions_are_authorization': False, 'automatic_retry': False, 'old872_restart': False,
            'new_training': False, 'test_inference': False, 'seed_or_statistics_changed': False,
            'statistics_cohorts_unchanged': old['statistics_cohorts'], 'scientific_cli_bindings_unchanged': old['scientific_cli'],
            'scientific_source_pins_unchanged': old['scientific_source_sha256'], 'final_protocol_status': 'PREPARED_NOT_FROZEN',
            'CPU_failure_still_invalid': True, 'GPU_diagnostic_scientific_match_not_cohort_acceptance': True,
            'no_checkpoint_content_SHA_deduplication': True, 'no_new_evidence_accepted': True}


def validate_manifest(actual, expected):
    need(actual == expected, 'Prepared manifest differs from exact frozen derivation; no decision/ID/path/SHA changes allowed')


def selfcheck(manifest):
    original = (OLD / 'replay.py').read_text()
    fn = next(n for n in ast.parse(original).body if isinstance(n, ast.FunctionDef) and n.name == 'replay_one')
    expected_body = ast.get_source_segment(original, fn)
    restored = (GPU / 'gpu_replay_body.py').read_text().strip()
    changes = [
        (".to('cuda:0')", '.cpu()'),
        ("'SINGLE_MODEL_GPU_DIAGNOSTIC_NOT_ACCEPTED_COHORT'", "'VALID_ONLY_IMPLEMENTATION_PREFLIGHT'"),
        ("'DIAGNOSTIC_NATIVE_MATCH' if comparison['accepted'] else 'DIAGNOSTIC_NATIVE_MISMATCH'", "'NATIVE_VALID_REPLAY_PASS' if comparison['accepted'] else 'NATIVE_VALID_REPLAY_MISMATCH'"),
        ("'device': 'cuda:0'", "'device': 'cpu'"),
        ("'Single-model CUDA diagnostic only; never authorizes cohort inclusion, CPU failure invalidation, training, test or queue restart'", "'Two bounded valid-only canaries are implementation evidence, not all900 replay or a new final performance table'"),
    ]
    for before, after in changes:
        need(restored.count(before) == 1, 'Scientific diff grew beyond one device and four receipt fields')
        restored = restored.replace(before, after)
    need(restored == expected_body, 'Scientific body changed beyond declared device/receipt differences')
    bootstrap = ast.parse((DIAG / 'attempt2/run_once.py').read_text())
    need(not any(isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == 'set_num_interop_threads' for n in ast.walk(bootstrap)), 'Known duplicate interop bootstrap bug returned')
    cases = {}
    mutations = {
        'duplicate_id': lambda x: x['records'].append(copy.deepcopy(x['records'][0])),
        'omit_id': lambda x: x['records'].pop(),
        'accepted424_overlap': lambda x: x['records'][0].update(id=x['accepted424_ids'][0]),
        'wrong_storage_path': lambda x: x['records'][0]['artifacts']['checkpoint'].update(target='/workspace/other/model.pt'),
        'checkpoint_SHA_drift': lambda x: x['records'][0]['artifacts']['checkpoint'].update(sha256='0' * 64),
        'auto_accept_partial': lambda x: next(r for r in x['records'] if r['classification'] == 'CPU_STRICT_PARTIAL_10_NOT_REGISTERED').update(accepted_by_this_package=True),
        'reuse_without_review': lambda x: x['root_review_decisions'].update(reuse_cpu_partial10=True),
        'restart_old872': lambda x: x.update(old872_restart=True),
    }
    for name, mutate in mutations.items():
        bad = copy.deepcopy(manifest)
        mutate(bad)
        try:
            validate_manifest(bad, manifest)
        except ValueError:
            cases[name] = 'REJECTED'
        else:
            raise AssertionError('Mutation was accepted: ' + name)
    return {'status': 'LOCAL_PREPARATION_CHECK_PASS_NO_INFERENCE', 'frozen_input_pins_verified_n': len(PINS),
            'unique_scientific_cells': 900, 'prior_accepted_n_unchanged': 424, 'exact_complement_n': 476,
            'classification_counts': manifest['classification_counts'], 'refusal_checks': cases,
            'GPU_body_exact_v2_science_except_device_and_four_receipt_fields': True,
            'attempt2_has_no_duplicate_interop_setter': True,
            'GPU_or_CNN_execution': False, 'new_acceptance': 0}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    expected = derive()
    if args.check:
        validate_manifest(read(HERE / 'manifest.json'), expected)
        need((HERE / 'gpu_replay_body.py').read_bytes() == (GPU / 'gpu_replay_body.py').read_bytes(), 'GPU scientific body changed')
        print(json.dumps(selfcheck(expected)))
        return
    need(not (HERE / 'manifest.json').exists(), 'Never overwrite an existing preparation')
    write('manifest.json', expected)
    write('INPUT_PINS.json', {str(p.relative_to(REPO)): {'sha256': sha, 'bytes': p.stat().st_size} for p, sha in PINS.items()})
    (HERE / 'gpu_replay_body.py').write_bytes((GPU / 'gpu_replay_body.py').read_bytes())
    (HERE / 'BOOTSTRAP_ATTEMPT2_DIFF.patch').write_bytes((DIAG / 'attempt2/ATTEMPT2_DIFF.patch').read_bytes())
    old = (OLD / 'replay.py').read_text()
    fn = next(n for n in ast.parse(old).body if isinstance(n, ast.FunctionDef) and n.name == 'replay_one')
    source = ast.get_source_segment(old, fn) + '\n'
    gpu = (HERE / 'gpu_replay_body.py').read_text()
    diff = ''.join(difflib.unified_diff(source.splitlines(keepends=True), gpu.splitlines(keepends=True), fromfile='sealed_v2/replay_one', tofile='byte_exact_diagnostic_gpu/replay_one'))
    (HERE / 'SCIENCE_BODY_DIFF.patch').write_text(diff, encoding='utf-8', newline='\n')
    write('selfcheck.json', selfcheck(expected))
    print(json.dumps({'status': expected['status'], 'remaining_n': 476, 'counts': expected['classification_counts'], 'manifest_sha256': digest(HERE / 'manifest.json')}))


if __name__ == '__main__':
    main()
