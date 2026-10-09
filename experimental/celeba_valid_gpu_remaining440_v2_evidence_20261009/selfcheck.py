"""No network/CNN: unchanged saved science, actual V1 array regression, synthetic V2 chain."""
from pathlib import Path
import ast
import copy
import importlib.util
import io
import json
import sys
import tarfile
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path); module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module); return module
q = load('V2_chunk_evidence_checked', HERE / 'evidence.py'); c = q.context(); root = HERE.parent.parent
parent_path = root / c['pins']['parent_evidence']['path']
q.need(q.sha(parent_path) == c['pins']['parent_evidence']['sha256'], 'Parent evidence changed')
old = load('original_V1_evidence_checked', parent_path)
functions = lambda path: {n.name: n for n in ast.parse(path.read_text()).body if isinstance(n, ast.FunctionDef)}
old_ast, new_ast = functions(parent_path), functions(HERE / 'evidence.py')
class ScopeOnly(ast.NodeTransformer):
    def visit_Constant(self, node):
        return ast.Name(id='SCOPE', ctx=ast.Load()) if node.value == 'EXISTING_BASELINE900_GPU_VALID_RECOVERY_V1' else node
q.need(ast.dump(ScopeOnly().visit(copy.deepcopy(old_ast['check_run']))) == ast.dump(new_ast['check_run']), 'Scientific saved-array body changed beyond V2 source scope')
for name in ('need', 'save', 'extract_functions', 'unpack'):
    q.need(ast.dump(old_ast[name]) == ast.dump(new_ast[name]), 'Original archive/scientific helper changed: ' + name)
parent_recovery = root / 'tmp/celeba_valid_gpu_recovery_implementation_20261009/recovery.py'
q.need(ast.dump(functions(parent_recovery)['receipt_guard']) == ast.dump(functions(c['paths']['recovery'])['receipt_guard']), 'Original receipt_guard changed')
prior, previous = q.prior_chain(c['paths']['base460'], q.sha(c['paths']['base460']), c, 0)
q.need(prior['accepted_n'] == 460 and previous is None, '460 origin incorrect')
first = 'FairGuard_IID_FedSA_seed91003'; original = root / 'tmp/celeba_valid_gpu_recovery_execution_20261009/first1_backup/verified_extract'
old_c = old.context(); old_c['review'] = q.read(root / 'tmp/celeba_valid_gpu_recovery_execution_20261009/ROOT_REVIEW_FIRST1.json'); old_c['review_sha'] = q.sha(root / 'tmp/celeba_valid_gpu_recovery_execution_20261009/ROOT_REVIEW_FIRST1.json')
strict, batch = q.read(original / 'execution/strict_acceptance.json'), q.read(original / 'batch/batch_inputs.json')
old.check_run(original / 'batch/runs' / first, old_c['records'][first], old_c, strict, batch, old_c['review']['output_parent'] + '/chunk_000')
rejected = []
def refuses(name, action):
    try: action()
    except (ValueError, KeyError): rejected.append(name)
    else: raise AssertionError('Unexpected pass: ' + name)
refuses('V1_saved_receipt_not_relabelled_V2', lambda: q.check_run(original / 'batch/runs' / first, c['records'][first], c, strict, batch, c['plan']['output_parent'] + '/chunk_000'))
refuses('prior_SHA_drift', lambda: q.prior_chain(c['paths']['base460'], '0' * 64, c, 0))
refuses('skip_chunk0', lambda: q.prior_chain(c['paths']['base460'], q.sha(c['paths']['base460']), c, 1))
with tempfile.TemporaryDirectory(prefix='checks_', dir=HERE) as temporary:
    folder = Path(temporary).resolve(); q.need(folder.is_relative_to(HERE), 'Cleanup outside ownership')
    def rewrite(path, value): path.write_text(json.dumps(value), encoding='utf-8')
    def tar_case(name, names, payloads, expected_payloads):
        archive = folder / (name + '.tar.gz')
        with tarfile.open(archive, 'w:gz') as bundle:
            for member, data in zip(names, payloads):
                item = tarfile.TarInfo(member); item.size = len(data); bundle.addfile(item, io.BytesIO(data))
        specs = {member: {'sha256': q.hashlib.sha256(data).hexdigest(), 'bytes': len(data)} for member, data in zip(names, expected_payloads)}
        inventory = {'sha256': q.sha(archive), 'bytes': archive.stat().st_size, 'members': specs, 'member_n': len(specs)}
        return lambda: q.unpack(archive, inventory, folder / (name + '_extract'))
    tar_case('valid_member', ['evidence.json'], [b'{}'], [b'{}'])()
    refuses('member_tamper_rehashed_archive', tar_case('tampered', ['evidence.json'], [b'[]'], [b'{}']))
    refuses('duplicate_archive_member', tar_case('duplicate', ['same', 'same'], [b'a', b'a'], [b'a', b'a']))
    refuses('path_escape', tar_case('unsafe', ['../outside'], [b'a'], [b'a']))
    refuses('old_model_repack', tar_case('model', ['model.pt'], [b'a'], [b'a']))
    ids = c['plan']['chunks'][0]['ids']; saved_here = q.HERE; q.HERE = folder
    (folder / 'PACKAGE_SHA256.json').write_text('{}'); package = q.sha(folder / 'PACKAGE_SHA256.json')
    proof = {'status': 'ROOT_GPU440_RESOURCE_GUARD_V2_CHUNK_OFFSERVER_SAVED_ARRAY_PASS', 'evidence_package_sha256': package, 'chunk_index': 0, 'accepted_new_ids': ids, 'remote_receipt_sha256': '2' * 64, 'source_package_sha256': c['recovery_sha'], 'queue_package_sha256': c['queue_sha'], 'recovery_scope': q.SCOPE, 'review_sha256': c['review_sha']}
    proof_path = folder / 'proof.json'; rewrite(proof_path, proof)
    collector = {'base460_sha256': q.sha(c['paths']['base460']), 'evidence_package_sha256': package, 'new_proof_path': str(proof_path), 'new_proof_sha256': q.sha(proof_path), 'previous_collector_path': str(c['paths']['base460']), 'previous_collector_sha256': q.sha(c['paths']['base460']), 'last_chunk_index': 0, 'added_ids': ids, 'last_remote_receipt_sha256': '2' * 64, 'accepted_ids': c['base']['accepted_ids'] + ids, 'accepted_n': 471, 'implementation_package_sha256': c['recovery_sha'], 'queue_package_sha256': c['queue_sha'], 'recovery_scope': q.SCOPE, 'review_sha256': c['review_sha']}
    collector_path = folder / 'collector.json'; rewrite(collector_path, collector)
    q.need(q.prior_chain(collector_path, q.sha(collector_path), c, 1)[0]['accepted_n'] == 471, 'Synthetic460→471 chain rejected')
    for name, mutate in [('wrong460_origin', lambda x: x.update(base460_sha256='0' * 64)), ('duplicate_collector_ID', lambda x: x['accepted_ids'].append(x['accepted_ids'][0])), ('changed_prior_proof', lambda x: x.update(new_proof_sha256='0' * 64)), ('V1_collector_scope', lambda x: x.update(recovery_scope='EXISTING_BASELINE900_GPU_VALID_RECOVERY_V1')), ('V1_runtime_identity', lambda x: x.update(implementation_package_sha256='0' * 64)), ('prior_queue_identity_drift', lambda x: x.update(queue_package_sha256='0' * 64)), ('prior_review_identity_drift', lambda x: x.update(review_sha256='0' * 64))]:
        bad = copy.deepcopy(collector); mutate(bad); rewrite(collector_path, bad)
        refuses(name, lambda: q.prior_chain(collector_path, q.sha(collector_path), c, 1))
    q.HERE = saved_here
    # Remote schema refusals run before unpack/array checks; these are synthetic metadata only.
    remote = folder / 'remote'; remote.mkdir()
    inv = {'chunk_index': 0, 'accepted_ids': ids, 'status': 'REMOTE_STRICT_ACCEPTED_ARCHIVE_VERIFIED_PENDING_OFFSERVER', 'offserver_verified': False, 'old_model_files_archived': 0, 'sha256': '3' * 64}
    rewrite(remote / 'remote_archive_inventory.json', inv)
    chain = {'status': 'REMOTE_CLOSED_PENDING_OFFSERVER', 'chunk_index': 0, 'remote_closed_ids': ids, 'previous_receipt_sha256': None, 'queue_package_sha256': c['queue_sha'], 'review_sha256': c['review_sha'], 'accepted_new_n': 0, 'cohort_registered': False, 'offserver_verified': False, 'implementation_package_sha256': c['recovery_sha'], 'recovery_scope': q.SCOPE, 'remote_closed_cumulative_n': 11, 'archive_sha256': inv['sha256'], 'archive_inventory_sha256': q.sha(remote / 'remote_archive_inventory.json')}
    for name, key, value in [('remote_V1_scope', 'recovery_scope', 'EXISTING_BASELINE900_GPU_VALID_RECOVERY_V1'), ('remote_runtime_drift', 'implementation_package_sha256', '0' * 64), ('remote_premature_acceptance', 'accepted_new_n', 11), ('remote_cumulative_count_drift', 'remote_closed_cumulative_n', 22)]:
        bad = copy.deepcopy(chain); bad[key] = value; rewrite(remote / 'REMOTE_PENDING_OFFSERVER.json', bad)
        refuses(name, lambda: q.verify_chunk(remote, c, 0, None))
expected = c['records'][first]['prior_validation_metrics']; changed = dict(expected, accuracy=expected['accuracy'] + 1 / 19867)
q.need(not c['original']['check_native'](changed, expected)['accepted'], 'One-example native mismatch accepted')
q.need('torch' not in sys.modules, 'Torch imported')
print(json.dumps({'status': 'PASS_LOCAL_NO_NETWORK_NO_CNN_NO_REGISTRATION', 'exact440_chunks40x11': True, 'actual_root_review_GATE_sha256': c['review_sha'], 'check_run_AST_unchanged_except_V2_scope_parameter': True, 'receipt_guard_AST_unchanged': True, 'actual_old_V1_saved_metric_checks': 9, 'actual_old_V1_saved_confusion_checks': 24, 'actual_old_V1_saved_prediction_rules': 3, 'synthetic460_to471_metadata_fixture': True, 'refusal_checks': rejected, 'native_tolerance': q.TOL, 'native_one_example_mismatch_rejected': True, 'new_network_calls': 0, 'new_CNN_inference': 0, 'cohort_changed': False, 'actual_V2_chunk_accepted_by_this_check': False}, indent=2))
