"""Read-only reuse of exactly one already Linux-saved-accepted CUDA record; no science."""
from pathlib import Path
import hashlib
import json

FIXED = {'id': 'FedAvg_IID_Benign_seed91001', 'reuse': {'task_accepted': {'path': '/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002/tasks/FedAvg_IID_Benign_seed91001/ACCEPTED.json', 'sha256': 'a0507114f5c5de34fcfcff6527df6cf3032f5126755771ac8fe410d346d0f2b1'}, 'deployment_seal': {'path': '/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002/DEPLOYMENT_SEAL.json', 'sha256': 'c411816b1f73c47e400f3cbc1adbd6e52e89975f360f131429f5279cddf70a4f'}, 'queue_auth': {'path': '/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002/ROOT_QUEUE_AUTH.json', 'sha256': 'c5684f028a11f59d8d7eab3dd229c91e22a0aa329f70441e453d526720b46270'}}, 'accepted': {'id': 'FedAvg_IID_Benign_seed91001', 'receipt': {'path': '/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002/runs/FedAvg_IID_Benign_seed91001/receipt.json', 'sha256': 'b7afd7b41ec85ce8eb1bf45597734fe88c7d035493a32d5eb4649a490329221d'}, 'binding': {'path': '/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002/runs/FedAvg_IID_Benign_seed91001/formal_binding.json', 'sha256': 'b2c6fe4369e8ca1e279f60ac17b63e962b1c7aff68a5aa894f38406182184656'}, 'arrays': {'path': '/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002/runs/FedAvg_IID_Benign_seed91001/validation_predictions.npz', 'sha256': '83c778e4c168a0315e2e85fa918aeb962eaccff16f2ae35c000dc21c5e491c9c'}, 'acceptance': {'path': '/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002/runs/FedAvg_IID_Benign_seed91001/formal_saved_acceptance.json', 'sha256': 'a81c218d03a2127fa5b1f601279ec05544323e159f34420b6ea18f5cf8906c0b'}, 'approval': {'path': '/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002/tasks/FedAvg_IID_Benign_seed91001/approval.json', 'sha256': 'f06b13006dbcb9b75baccdaf6fcfbff4a7370287e2ab3f2d3ef13dd169e053e0'}, 'native_max_abs_difference': 0.0, 'root_offserver_adopted': False}, 'cohort': {'path': '/workspace/guardfed_checks/uniform_cuda_formal_original800_20261011_attempt002/EXACT800_INPUTS.json', 'sha256': 'd39e29333c40e8a5b63f1344487225abac4ba83216dc32e75f64badc87bc0b71'}, 'record_sha256': 'bd3218f3335cc1c0617fa5256b9bbc742c2cf266f889f2a4586cb86be1e32771'}


def need(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_bytes())


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def stat_token(path):
    stat = path.stat()
    return {'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns, 'inode': stat.st_ino, 'device': stat.st_dev,
            'resolved_path': str(path.resolve())}


def current_identities(expected):
    need(isinstance(expected, dict) and expected, 'Missing original structured identities')
    observed = {}
    for name, identity in expected.items():
        p = Path(name)
        need(p.is_absolute() and '..' not in p.parts and isinstance(identity, dict)
             and set(identity) == {'sha256', 'resolved_path', 'bytes', 'stat'}, 'Wrong full_hashes schema')
        before = stat_token(p)
        actual = {'sha256': sha(p), 'resolved_path': str(p.resolve()), 'bytes': p.stat().st_size,
                  'stat': stat_token(p)}
        need(before == actual['stat'] and actual == identity, 'Original source/artifact changed: ' + name)
        observed[name] = actual
    return observed


def checked_pin(pin):
    need(isinstance(pin, dict) and set(pin) == {'path', 'sha256'}, 'Exact path/SHA pin required')
    p = Path(pin['path'])
    need(p.is_absolute() and '..' not in p.parts and p.is_file() and not p.is_symlink()
         and p.resolve() == p and sha(p) == pin['sha256'], 'Original pinned file changed: ' + str(p))
    return p


def verify_reused(auth, scope):
    """Return {fixed FedAvg ID: original ACCEPTED}; never runs fit/CNN or root adoption."""
    rid = FIXED['id']
    need(auth.get('reused_cuda') == {rid: FIXED['reuse']}, 'Only the fixed original FedAvg CUDA acceptance may be skipped')
    reuse = auth['reused_cuda'][rid]
    oldauth = read(checked_pin(reuse['queue_auth']))
    seal = read(checked_pin(reuse['deployment_seal']))
    base = Path(reuse['queue_auth']['path']).parent
    need(oldauth['execution_authorized'] is True and oldauth['exact_cohort'] == FIXED['cohort'], 'Changed original queue authorization')
    oldscope = read(checked_pin(FIXED['cohort']))
    need(scope == oldscope and len(scope['records']) == 800
         and len(set(scope['ordered_ids'])) == 800
         and scope['ordered_ids'] == [r['id'] for r in scope['records']], 'Not the original exact ordered800')
    need(auth['ordered_ids'] == oldauth['ordered_ids'] == scope['ordered_ids'], 'Successor changed original input cohort')
    record = next(r for r in scope['records'] if r['id'] == rid)
    need(canonical(record) == FIXED['record_sha256'], 'Fixed original record object changed')
    need((record['method'], record['distribution'], record['attack'], record['seed'], record['actual_alpha'])
         == ('FedAvg', 'IID', 'Benign', 91001, 5000.0), 'Wrong reused scenario')
    cfg = record['config']
    need((record['terminal_round'], record['original_split'], record['original_n_eval']) == (70, 'valid', 19867)
         and cfg['rounds'] == 70 and cfg['seed'] == 91001 and cfg['client_alpha'] == 5000.0
         and cfg['celeba_evaluation_split'] == 'valid' and not cfg['celeba_train_limit'] and not cfg['celeba_eval_limit']
         and canonical(cfg) == record['config_canonical_sha256'], 'Incomplete or changed original valid record')
    for rel, item in seal['files'].items():
        p = base / rel
        need(not Path(rel).is_absolute() and '..' not in Path(rel).parts and p.is_relative_to(base)
             and not p.is_symlink() and sha(p) == item['sha256'] and p.stat().st_size == item['bytes'], 'Old sealed deployment changed: ' + rel)
    need(seal['files']['ROOT_QUEUE_AUTH.json']['sha256'] == reuse['queue_auth']['sha256']
         and seal['files']['EXACT800_INPUTS.json']['sha256'] == FIXED['cohort']['sha256'], 'Old auth/cohort not sealed')
    for name, expected in seal['external_code_pins'].items():
        need(sha(name) == expected, 'Old external dependency changed: ' + name)
    accepted = read(checked_pin(reuse['task_accepted']))
    need(accepted == FIXED['accepted'], 'Original task ACCEPTED links changed')
    paths = {k: checked_pin(accepted[k]) for k in ('receipt', 'binding', 'arrays', 'acceptance', 'approval')}
    output = paths['receipt'].parent
    need(output == base / 'runs' / rid and paths['approval'] == base / 'tasks' / rid / 'approval.json', 'Mixed original task paths')
    need(not output.with_name(output.name + '.formal_failure.json').exists()
         and not any('fail' in p.name.lower() for d in (output, paths['approval'].parent) for p in d.glob('*.json')), 'Reused record has failure evidence')
    r, binding, passed, approval = (read(paths[k]) for k in ('receipt', 'binding', 'acceptance', 'approval'))
    need(approval['execution_authorized'] is True and approval['id'] == rid
         and approval['record_canonical_sha256'] == canonical(record)
         and approval['ordered_input_ids'] == scope['ordered_ids']
         and approval['exact_cohort'] == FIXED['cohort'] and approval['output'] == str(output), 'Mixed original per-record approval')
    need(binding['scope'] == passed['scope'] == 'UNIFORM_CUDA_ORIGINAL800_FORMAL_V1'
         and binding['id'] == passed['id'] == r['id'] == rid
         and binding['approval_sha256'] == accepted['approval']['sha256']
         and binding['input_scope_sha256'] == FIXED['cohort']['sha256']
         and binding['receipt_sha256'] == accepted['receipt']['sha256']
         and passed['binding_sha256'] == accepted['binding']['sha256'], 'Broken original saved/binding/receipt links')
    need(r['scope'] == 'FORMAL_ORIGINAL800_CUDA_PER_RECORD_V1' and r['status'] == 'NATIVE_VALID_REPLAY_PASS'
         and passed['status'] == 'PER_RECORD_ORIGINAL800_CUDA_SAVED_WHOLE_ACCEPTED'
         and r['runtime']['device'] == 'cuda:0' and r['runtime']['cuda_device_count'] == 1
         and r['runtime']['torch'] == '2.11.0+cu128' and r['valid_n'] == 19867
         and r['valid_image_ids_sha256'] == '64a15cf28caf1d177ac3dcf96a4408bc21091796974b243923ca947a37b554bf', 'Not original complete CUDA valid saved acceptance')
    need(all(r[k] == record[k] for k in ('method', 'distribution', 'attack', 'seed'))
         and r['model_inventory_record_sha256'] == canonical(record)
         and r['config_canonical_sha256'] == record['config_canonical_sha256']
         and r['original_training_torch'] == record['training_torch']
         and all(r[k] == record[a]['sha256'] for k,a in [('checkpoint_sha256','checkpoint'),('original_result_sha256','result'),('original_job_sha256','raw_job')]), 'Checkpoint/result/job identity mismatch')
    need(r['prediction_arrays_sha256'] == accepted['arrays']['sha256']
         and r['weights_before'] == r['weights_after'] and r['weights_before']
         and passed['views'] == r['views'] and passed['native_comparison'] == r['native_comparison'], 'Saved outputs or model identity mismatch')
    comparison = passed['native_comparison']
    need(comparison['accepted'] is True and comparison['tolerance'] == 1e-12
         and comparison['max_abs_difference'] == accepted['native_max_abs_difference'] == 0.0
         and set(comparison['differences']) == {'accuracy','aeod','aspd'}
         and all(comparison['differences'][m] == 0.0 and comparison['observed'][m] == comparison['expected'][m]
                 == record['prior_validation_metrics'][m] == r['views']['native'][m] for m in comparison['differences']), 'Original native-zero gate not met')
    need(all(r[k] is False for k in ('optimizer_created','gradients_created','test_labels_accessed','test_inference_performed'))
         and passed['new_training'] is False and passed['test'] is False and passed['cohort_complete'] is False
         and accepted['root_offserver_adopted'] is False, 'Wrong original operation or adoption claim')
    need(binding['source_before'] == binding['source_after'] == current_identities(binding['source_before'])
         and binding['artifact_before'] == binding['artifact_after'] == current_identities(binding['artifact_before']), 'Structured source/artifact closure changed')
    source = binding['source_before']; artifacts = binding['artifact_before']
    for k in ('approval',):
        need(source[accepted[k]['path']]['sha256'] == accepted[k]['sha256'], 'Approval outside original source closure')
    need(source[FIXED['cohort']['path']]['sha256'] == FIXED['cohort']['sha256'], 'Cohort outside original source closure')
    for k in ('checkpoint','result','raw_job'):
        need(any(v['sha256'] == record[k]['sha256'] and v['bytes'] == record[k]['bytes'] for v in artifacts.values()), 'Incomplete original artifact closure')
    need(len(artifacts) == 3, 'Wrong original checkpoint/result/raw_job closure')
    # No historical preflight-age test: this verifies preserved success, not permission for a new inference.
    return {rid: accepted}
