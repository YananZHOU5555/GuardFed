"""No-network, no-CNN checks: actual saved first1 arrays plus adversarial metadata/tars."""
from pathlib import Path
import copy
import importlib.util
import io
import json
import shutil
import sys
import tarfile
import tempfile
import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('chunk_evidence_checked', HERE / 'evidence.py')
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
c = q.context(); root = HERE.parent.parent
prior, previous = q.prior_chain(c['paths']['base436'], q.sha(c['paths']['base436']), c, 0)
q.need(prior['accepted_n'] == 436 and previous is None, '436 base incorrect')
first = 'FairGuard_IID_FedSA_seed91003'; original = root / 'tmp/celeba_valid_gpu_recovery_execution_20261009/first1_backup/verified_extract'
old = dict(c); old['review'] = q.read(root / 'tmp/celeba_valid_gpu_recovery_execution_20261009/ROOT_REVIEW_FIRST1.json'); old['review_sha'] = q.sha(root / 'tmp/celeba_valid_gpu_recovery_execution_20261009/ROOT_REVIEW_FIRST1.json')
strict, batch = q.read(original / 'execution/strict_acceptance.json'), q.read(original / 'batch/batch_inputs.json')
remote = old['review']['output_parent'] + '/chunk_000'
q.check_run(original / 'batch/runs' / first, c['records'][first], old, strict, batch, remote)
rejected = []
def refuses(name, action):
    try: action()
    except (ValueError, KeyError): rejected.append(name)
    else: raise AssertionError('Unexpected pass: ' + name)
refuses('prior_SHA_drift', lambda: q.prior_chain(c['paths']['base436'], '0' * 64, c, 0))
refuses('skip_chunk0', lambda: q.prior_chain(c['paths']['base436'], q.sha(c['paths']['base436']), c, 1))
with tempfile.TemporaryDirectory(prefix='checks_', dir=HERE) as temporary:
    folder = Path(temporary).resolve(); q.need(folder.is_relative_to(HERE), 'Cleanup outside owned directory')
    fixture = folder / 'saved/verified_extract'; run = fixture / 'batch/runs' / first
    shutil.copytree(original / 'batch/runs', fixture / 'batch/runs')
    receipt, worker = q.read(run / 'receipt.json'), q.read(run.with_name(first + '.worker.json'))
    def rewrite(path, value): path.write_text(json.dumps(value), encoding='utf-8')
    for name, mutate in [('native_tolerance_change', lambda r: r['native_comparison'].update(tolerance=1e-8)), ('mixed_seed', lambda r: r.update(seed=91010)), ('mixed_checkpoint', lambda r: r.update(checkpoint_sha256='0' * 64)), ('count0_GPU', lambda r: r['runtime'].update(cuda_device_count=0)), ('nice0_GPU', lambda r: r['runtime'].update(nice=0)), ('optimizer_claim', lambda r: r.update(optimizer_created=True)), ('changed_saved_confusion', lambda r: r['views']['native']['group_confusion_counts']['0'].update(tp=999)), ('threshold_drift', lambda r: r['fits']['shared_calibration']['thresholds'].update({'0': r['fits']['shared_calibration']['thresholds']['0'] + 1e-4}))]:
        bad = copy.deepcopy(receipt)
        if name == 'changed_saved_confusion':
            group = next(iter(bad['views']['native']['group_confusion_counts'].values())); group['tp'] += 1
        else: mutate(bad)
        rewrite(run / 'receipt.json', bad); p = copy.deepcopy(worker); p['receipt_sha256'] = q.sha(run / 'receipt.json'); rewrite(run.with_name(first + '.worker.json'), p)
        refuses(name, lambda: q.check_run(run, c['records'][first], old, strict, batch, remote))
    rewrite(run / 'receipt.json', receipt); rewrite(run.with_name(first + '.worker.json'), worker)
    p = copy.deepcopy(worker); p['gpu_body_sha256'] = '0' * 64; rewrite(run.with_name(first + '.worker.json'), p)
    refuses('scientific_body_source_drift', lambda: q.check_run(run, c['records'][first], old, strict, batch, remote))
    rewrite(run.with_name(first + '.worker.json'), worker)
    expected = c['records'][first]['prior_validation_metrics']; changed = dict(expected, accuracy=expected['accuracy'] + 1 / 19867)
    q.need(not c['original']['check_native'](changed, expected)['accepted'], 'One-example native mismatch accepted')
    def tar_case(name, names, payloads, expected_payloads):
        archive = folder / (name + '.tar.gz')
        with tarfile.open(archive, 'w:gz') as bundle:
            for member, data in zip(names, payloads):
                item = tarfile.TarInfo(member); item.size = len(data); bundle.addfile(item, io.BytesIO(data))
        specs = {member: {'sha256': q.hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for member, data in zip(names, expected_payloads)}
        inventory = {'sha256': q.sha(archive), 'bytes': archive.stat().st_size, 'members': specs, 'member_n': len(specs)}
        return lambda: q.unpack(archive, inventory, folder / (name + '_extract'))
    tar_case('valid_member', ['evidence.json'], [b'{}'], [b'{}'])()
    refuses('member_tamper_with_rehashed_archive', tar_case('tampered', ['evidence.json'], [b'[]'], [b'{}']))
    refuses('duplicate_archive_member', tar_case('duplicate', ['same', 'same'], [b'a', b'a'], [b'a', b'a']))
    refuses('path_escape', tar_case('unsafe', ['../outside'], [b'a'], [b'a']))
    refuses('old_model_repack', tar_case('model', ['model.pt'], [b'a'], [b'a']))
    saved_here = q.HERE; q.HERE = folder
    (folder / 'PACKAGE_SHA256.json').write_text('{}'); package = q.sha(folder / 'PACKAGE_SHA256.json'); ids = c['plan']['chunks'][0]['ids']
    proof = {'status': 'ROOT_GPU464_CHUNK_OFFSERVER_SAVED_ARRAY_PASS', 'evidence_package_sha256': package, 'chunk_index': 0, 'accepted_new_ids': ids, 'remote_receipt_sha256': '2' * 64}
    proof_path = folder / 'proof.json'; rewrite(proof_path, proof)
    collector = {'base436_sha256': q.sha(c['paths']['base436']), 'evidence_package_sha256': package, 'new_proof_path': str(proof_path), 'new_proof_sha256': q.sha(proof_path), 'previous_collector_path': str(c['paths']['base436']), 'previous_collector_sha256': q.sha(c['paths']['base436']), 'last_chunk_index': 0, 'added_ids': ids, 'last_remote_receipt_sha256': '2' * 64, 'accepted_ids': c['base']['accepted_ids'] + ids, 'accepted_n': 447}
    collector_path = folder / 'collector.json'; rewrite(collector_path, collector)
    q.need(q.prior_chain(collector_path, q.sha(collector_path), c, 1)[0]['accepted_n'] == 447, '436→447 structural chain rejected')
    for name, mutate in [('wrong436_origin', lambda x: x.update(base436_sha256='0' * 64)), ('duplicate_collector_ID', lambda x: x['accepted_ids'].append(x['accepted_ids'][0])), ('changed_prior_proof', lambda x: x.update(new_proof_sha256='0' * 64))]:
        bad = copy.deepcopy(collector); mutate(bad); rewrite(collector_path, bad)
        refuses(name, lambda: q.prior_chain(collector_path, q.sha(collector_path), c, 1))
    q.HERE = saved_here
q.need('torch' not in sys.modules, 'Torch imported')
print(json.dumps({'status': 'PASS_LOCAL_NO_NETWORK_NO_CNN_NO_REGISTRATION', 'actual_first1_saved_metric_checks': 9, 'actual_first1_saved_confusion_checks': 24, 'actual_first1_saved_prediction_rules': 3, 'valid436_to447_metadata_fixture': True, 'refusal_checks': rejected, 'native_one_example_mismatch_rejected': True, 'new_network_calls': 0, 'new_CNN_inference': 0, 'cohort_changed': False, 'actual_remaining464_chunk_accepted_by_this_check': False}, indent=2))
