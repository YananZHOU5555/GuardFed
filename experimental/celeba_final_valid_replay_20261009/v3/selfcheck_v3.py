"""Actual accepted archive-member identity tests; no real-image inference or batch execution."""
import copy
import json
from pathlib import Path
import sys
import tarfile
import tempfile
from types import SimpleNamespace

sys.dont_write_bytecode = True
import replay_v3 as v3
v2 = v3.v2
HERE, BASE = v3.HERE, v3.BASE
REPO = BASE.parents[1]


def reject(label, function, rows):
    try:
        function()
    except (ValueError, AssertionError, KeyError) as exc:
        rows.append({'name': label, 'error': str(exc)})
    else:
        raise AssertionError('Did not reject: ' + label)


def main():
    args = SimpleNamespace(inventory=BASE / 'inputs/model_inventory.json', storage_map=HERE / 'inputs/storage_map.json',
              storage_map_sha256='e160416101ccb82b42224c9ff1bb337de45ef72fa251197063d1c15e2910f949',
              restore_receipt=HERE / 'inputs/restore_bundle.json',
              restore_receipt_sha256='0cfa391b0d317622551d4c6526f2d14ec33c3dda0f3e9e523066bd3718cbcf3a',
              restore_acceptance=HERE / 'inputs/restore_acceptance.json',
              restore_acceptance_sha256='5114c2cd96e5b8ffaf46e40a341619dd8b3547f89d417263c19c6e7f1f33bf77')
    records, bindings = v3.read_inputs(args)
    inventory, mapping, bundle, acceptance = map(v2.read, (args.inventory, args.storage_map, args.restore_receipt, args.restore_acceptance))
    rejected = []
    def check(m=mapping, b=bundle, a=acceptance):
        return v3.storage_bindings(inventory, m, b, args.storage_map_sha256, args.restore_receipt_sha256, a)
    groups = [{'name': 'actual900_inventory_2700_map_2400_bundle_chain', 'passed': True,
               'models': len(records), 'mapped_files': sum(map(len, bindings.values()))}]
    for label, mutate in [
        ('duplicate id/kind', lambda m: m['records'].__setitem__(1, copy.deepcopy(m['records'][0]))),
        ('missing seed/kind', lambda m: m['records'].pop()),
        ('wrong source archive', lambda m: m['records'][0].update(archive_sha256='0' * 64)),
        ('wrong original member bytes', lambda m: m['records'][0].update(sha256='0' * 64)),
        ('storage escape', lambda m: m['records'][0].update(target='/workspace/elsewhere/model.pt')),
        ('cross-model checkpoint path', lambda m: m['records'][0].update(target=m['records'][3]['target'])),
    ]:
        bad = copy.deepcopy(mapping)
        mutate(bad)
        reject(label, lambda: check(m=bad), rejected)
    bad_acceptance = copy.deepcopy(acceptance)
    bad_acceptance['storage_map_sha256'] = '0' * 64
    reject('remote acceptance binds another map', lambda: check(a=bad_acceptance), rejected)
    bad_bundle = copy.deepcopy(bundle)
    bad_bundle['members'].pop(next(iter(bad_bundle['members'])))
    reject('missing restore bundle member', lambda: check(b=bad_bundle), rejected)
    tuning_full = next(r for r in records if r['method'] == 'GuardFed-AD2+' and 'GuardFed-celeba-tuning' in r['original_remote_output'])
    v2.require('GuardFed-celeba-tuning' in bindings[tuning_full['id']]['checkpoint']['target'], 'Tuning Full original path was rewritten')
    groups.append({'name': 'storage_tamper_rejections_and_original_tuning_Full_path', 'passed': True,
                   'tuning_full_id': tuning_full['id']})

    source_candidates = [REPO, REPO / 'tmp/revision-publish-20260928']
    frozen = next((p for p in source_candidates if (p / 'scripts/run_revision_ablation.py').exists()
                   and v2.digest(p / 'scripts/run_revision_ablation.py') == v2.RUNNER_SHA), None)
    v2.require(frozen is not None, 'Missing exact original checked_result source; do not use a wrong root source')
    original = v2.load('v3_selfcheck_original_checked', frozen / 'scripts/run_revision_ablation.py')
    record = next(r for r in records if r['id'] == 'FedAvg_IID_Benign_seed91001')
    other = next(r for r in records if r['id'] == 'FedAvg_IID_Benign_seed91002')
    archive = REPO / record['checkpoint']['archive']
    v2.require(v2.digest(archive) == record['checkpoint']['archive_sha256'], 'Original fixture archive SHA failed')
    with tempfile.TemporaryDirectory(dir=HERE) as folder:
        folder = Path(folder)
        paths = {'checkpoint': folder / 'actual_artifacts/model.pt', 'result': folder / 'actual_artifacts/result.json', 'raw_job': folder / 'separate_jobs/original.json'}
        source_receipt = {}
        with tarfile.open(archive, 'r:gz') as tar:
            for kind, path in paths.items():
                ref = record[kind]
                v2.require(ref['archive'] == record['checkpoint']['archive'], 'Fixture kinds use different archives')
                payload = tar.extractfile(ref['member']).read()
                v2.require(len(payload) == ref['bytes'] and v2.hashlib.sha256(payload).hexdigest() == ref['sha256'], 'Original fixture member failed')
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(payload)
                source_receipt[kind] = {'sha256': ref['sha256'], 'member': ref['member']}
            other_model = tar.extractfile(other['checkpoint']['member']).read()
            v2.require(v2.hashlib.sha256(other_model).hexdigest() == other['checkpoint']['sha256'], 'Second accepted fixture model SHA failed')
        raw_job_before = paths['raw_job'].read_bytes()
        validate, infer = v3.mapped_functions(record, paths, original)
        r = validate(original, record, frozen)
        v2.require(r['metrics'] == record['prior_validation_metrics'] and infer.__code__ is v2.replay_one.__code__, 'Mapped validator/scientific body differs')
        v2.require(paths['raw_job'].read_bytes() == raw_job_before and v2.read(paths['raw_job'])['output'] == record['original_remote_output'], 'Original job/output was rewritten')
        own_model = paths['checkpoint'].read_bytes()
        paths['checkpoint'].write_bytes(other_model)
        reject('real accepted but mixed checkpoint', lambda: validate(original, record, frozen), rejected)
        paths['checkpoint'].write_bytes(own_model)
        bad_record = copy.deepcopy(record)
        bad_record['data_contract']['root_image_ids_sha256'] = '0' * 64
        bad_validate, _ = v3.mapped_functions(bad_record, paths, original)
        reject('root ID tamper before inference', lambda: bad_validate(original, bad_record, frozen), rejected)
        changed_job = v2.read(paths['raw_job'])
        changed_job['output'] = str(folder / 'wrong_output')
        v2.save(paths['raw_job'], changed_job)
        reject('raw job historical output rewrite', lambda: validate(original, record, frozen), rejected)
        paths['raw_job'].write_bytes(raw_job_before)
        groups.append({'name': 'actual_nonFull_original_checked_result_via_mapped_storage', 'passed': True,
                       'archive_sha256': record['checkpoint']['archive_sha256'], 'members': source_receipt,
                       'historical_output_preserved': True, 'sealed_v2_inference_code_object_reused': True,
                       'actual_image_inference_performed': False})

        collection = folder / 'v2_reuse_partial.json'
        v3.collect(SimpleNamespace(inventory=args.inventory, acceptance=None, include_sealed_v2=True, output=collection))
        a = v2.read(collection)
        v2.require(a['accepted_n'] == 2 and not a['all900_native_valid_replayed'] and len(a['missing_ids']) == 898, 'Two canaries were misreported as900')
        # Synthetic descriptor tests joining only, not scientific acceptance/inference.
        duplicate = folder / 'synthetic_duplicate_acceptance.json'
        v2.save(duplicate, {'scope': v3.SCOPE, 'inventory_sha256': v2.INVENTORY_SHA,
                 'valid_image_ids_sha256': v2.VALID_IDS_SHA, 'calibration_core_sha256': v2.CORE_SHA,
                 'v2_source_sha256': v3.V2_SHA, 'test_labels_accessed': False,
                 'accepted_n': 1, 'accepted_ids': [v2.CANARY_IDS[0]], 'max_abs_native_metric_difference': 0})
        reject('duplicate ID across explicit v2 reuse and reviewed batch', lambda: v3.collect(SimpleNamespace(inventory=args.inventory,
                    acceptance=[(str(duplicate), v2.digest(duplicate))], include_sealed_v2=True, output=folder / 'must_not_exist.json')), rejected)
    groups.append({'name': 'cumulative_explicit_reuse_duplicate_rejection_partial_n', 'passed': True,
                   'actual_replayed_canaries': 2, 'new_v3_replayed_models': 0, 'overall_complete': False})
    report = {'status': 'PASS', 'groups': groups, 'rejected': rejected, 'v3_source_sha256': v2.digest(v3.__file__),
              'selfcheck_sha256': v2.digest(__file__), 'v2_source_sha256': v3.V2_SHA,
              'storage_map_sha256': args.storage_map_sha256, 'restore_receipt_sha256': args.restore_receipt_sha256,
              'restore_acceptance_sha256': args.restore_acceptance_sha256, 'test_labels_accessed': False,
              'actual_new_image_inference_performed': False, 'new_v3_replayed_models': 0}
    v2.save(HERE / 'selfcheck_v3.json', report)
    print(json.dumps({'status': 'PASS', 'groups': len(groups), 'rejection_cases': len(rejected), 'new_v3_replayed_models': 0}))


if __name__ == '__main__':
    main()
