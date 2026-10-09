"""Explicitly represent only the frozen all-unprivileged undefined diagnostic."""
import math
import struct


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sanitize_result(value, job, job_sha256, expected_job_sha256):
    require(job_sha256 == expected_job_sha256, 'Frozen repair job SHA mismatch')
    require(job['method'] in {'CosineFairnessHybrid', 'GuardFed'} and job['attack'] == 'S-DFA'
            and job['distribution'] == 'non-IID', 'Only frozen non-IID S-DFA controls')
    cfg = job['config']
    require(cfg['fflip_mode'] == 'all_unprivileged' and cfg['num_malicious'] == 4
            and cfg['num_clients'] == 20 and cfg['rounds'] == 3 and cfg['seed'] == 91001
            and cfg['celeba_evaluation_split'] == 'valid', 'Undefined-correlation cause/config differs')
    require(value['config'] == cfg and value['method'] == job['method'] and value['attack'] == job['attack'],
            'Result identity differs from frozen job')
    audits = value['attack_audit']
    require(isinstance(audits, list) and len(audits) == 20, 'Incomplete client audit')
    expected_paths = {('attack_audit', i, 'fflip_label_corr_after') for i in range(4)}
    for i in range(4):
        audit = audits[i]
        require(audit['client_id'] == i and audit['is_malicious'] is True and audit['samples'] > 1
                and audit['attack_types'] == ['fflip', 'foe'] and audit['fflip_mode'] == 'all_unprivileged'
                and audit['fflip_overwrite_ratio'] == 1.0 and audit['fflip_requires_full_flip'] is False
                and audit['label_changed_count'] == 0, 'Not the frozen zero-variance sensitive overwrite')
        number = audit['fflip_label_corr_after']
        require(type(number) is float and math.isnan(number), 'Expected original undefined diagnostic, never a preexisting null/value')
    changed = []

    def visit(item, path=()):
        if isinstance(item, dict):
            return {key: visit(child, path + (key,)) for key, child in item.items()}
        if isinstance(item, list):
            return [visit(child, path + (i,)) for i, child in enumerate(item)]
        if isinstance(item, tuple):
            return tuple(visit(child, path + (i,)) for i, child in enumerate(item))
        if isinstance(item, float) and not math.isfinite(item):
            require(path in expected_paths and type(item) is float and math.isnan(item),
                    'Unrecognized nonfinite value refused: ' + repr(path))
            changed.append({'path': list(path), 'original_type': 'python.float', 'original_kind': 'NaN',
                            'original_ieee754_binary64_big_endian_hex': struct.pack('>d', item).hex(),
                            'serialized_value': None,
                            'reason': 'Undefined Pearson correlation: frozen all_unprivileged sensitive overwrite has zero variance',
                            'producer': 'frozen core client_runtime_data, fflip_label_corr_after else branch'})
            return None
        return item

    sanitized = visit(value)
    require({tuple(row['path']) for row in changed} == expected_paths and len(changed) == 4,
            'Exactly four mandatory undefined diagnostics required')
    return sanitized, changed


def check_sidecar(result, sidecar, job, expected_job_sha256, original_gate_sha256, original_core_sha256):
    require(sidecar['status'] == 'EXPLICIT_UNDEFINED_DIAGNOSTIC_ONLY' and sidecar['job_sha256'] == expected_job_sha256,
            'Wrong undefined diagnostic receipt')
    require(sidecar['original_gate_sha256'] == original_gate_sha256 and sidecar['original_core_sha256'] == original_core_sha256,
            'Undefined producer source changed')
    rows = sidecar['undefined_values']
    require(len(rows) == 4 and {tuple(row['path']) for row in rows} == {('attack_audit', i, 'fflip_label_corr_after') for i in range(4)},
            'Unknown/missing/duplicate undefined path')
    # Reconstruct only the recorded NaN representation, then repeat every cause and
    # nonfinite boundary check. Finite values and the original input are preserved.
    copied = {**result, 'attack_audit': [dict(row) for row in result['attack_audit']]}
    for row in rows:
        i = row['path'][1]
        require(result['attack_audit'][i]['fflip_label_corr_after'] is None and row['serialized_value'] is None
                and row['original_type'] == 'python.float' and row['original_kind'] == 'NaN', 'Undefined value meaning changed')
        number = struct.unpack('>d', bytes.fromhex(row['original_ieee754_binary64_big_endian_hex']))[0]
        require(math.isnan(number), 'Sidecar records a finite/infinite value')
        copied['attack_audit'][i]['fflip_label_corr_after'] = number
    sanitized, expected = sanitize_result(copied, job, expected_job_sha256, expected_job_sha256)
    require(sanitized == result and expected == rows, 'Result/undefined sidecar differs')
