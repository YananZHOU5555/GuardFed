"""Check actual archived108 raw-job/result schemas, without checked_result/inference.

The sealed v2 validate_original body is executed with accepted archived JSON as
the original.checked_result return value. This checks its path/schema assumptions,
not a fresh scientific/model acceptance. No model bytes are unpacked or copied.
"""
from __future__ import annotations
import ast
import hashlib
from pathlib import Path, PurePosixPath
import sys
import tarfile
import types
sys.dont_write_bytecode = True
import bridge as b


def main():
    root = Path(__file__).resolve().parents[2]
    inventory = b.read(b.HERE / 'inventory_actual8_Full100refs.json')
    baseline = b.read(root / 'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json')
    b.require(b.digest(root / 'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json') == b.BASELINE_INVENTORY_SHA, 'Baseline inventory drift')
    b.validate_inventory(inventory, baseline)
    records = [*inventory['records'], *(r for r in baseline['records'] if r['method'] == 'GuardFed-AD2+')]
    needs = {}
    for record in records:
        for kind in ('raw_job', 'result'):
            a = record[kind]
            group = needs.setdefault(a['archive'], {'sha256': a['archive_sha256'], 'members': {}})
            b.require(group['sha256'] == a['archive_sha256'], 'Conflicting archive SHA')
            group['members'][a['member']] = (record['id'], kind, a['sha256'], a['bytes'])
    payloads, archive_checks = {}, []
    for relative, group in needs.items():
        path = root / relative
        b.require(b.digest(path) == group['sha256'], 'Archive identity changed')
        seen = set()
        with tarfile.open(path, 'r|gz') as tar:
            for member in tar:
                if member.name not in group['members']:
                    continue
                b.require(member.isfile() and member.name not in seen, 'Duplicate/nonregular selected member')
                seen.add(member.name)
                model_id, kind, expected_sha, expected_size = group['members'][member.name]
                data = tar.extractfile(member).read()
                b.require(len(data) == expected_size and hashlib.sha256(data).hexdigest() == expected_sha, 'Archived job/result identity changed')
                import json
                payloads[model_id, kind] = json.loads(data)
        b.require(seen == set(group['members']), 'Missing archived raw-job/result')
        archive_checks.append({'archive': relative, 'sha256': group['sha256'], 'selected_members_verified': len(seen), 'checkpoint_members_extracted': 0})
    source_path = root / 'tmp/celeba_final_valid_replay_20261009/replay.py'
    b.require(b.digest(source_path) == b.V2_SHA, 'Sealed v2 changed')
    source = source_path.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == 'validate_original')
    checked = []
    for record in records:
        job, result = payloads[record['id'], 'raw_job'], payloads[record['id'], 'result']
        def fake_read(member):
            b.require(member == record['raw_job']['member'], 'Validator attempted an unbound input')
            return job
        def accepted_json(runtime_job):
            b.require(runtime_job == job, 'Original job was silently rewritten')
            return result
        namespace = {'require': b.require, 'read': fake_read, 'inside': lambda _repo, relative: relative, 'PurePosixPath': PurePosixPath}
        exec(compile(ast.Module(body=[node], type_ignores=[]), str(source_path), 'exec'), namespace)
        namespace['validate_original'](types.SimpleNamespace(checked_result=accepted_json), record, root)
        b.require(job['method'] == record['source_method'] == result['method'] and job['config'] == record['config'], 'Method/config path differs')
        if 'variant' in record:
            b.require(job['variant'] == record['variant'] == result['revision_job']['variant'], 'Mechanism variant changed')
        else:
            b.require(job['config']['ablation_component'] == 'none', 'Full contains an intervention')
        checked.append({'id': record['id'], 'role': 'new_mechanism' if 'variant' in record else 'Full_reference_only',
                        'raw_job_sha256': record['raw_job']['sha256'], 'result_sha256': record['result']['sha256'],
                        'raw_job_has_output': 'output' in job, 'historical_output': job['output'],
                        'original_output_unchanged': job['output'] == record['original_remote_output'],
                        'method': job['method'], 'ablation_component': job['config']['ablation_component'],
                        'original_config_source_data_terminal_schema': 'PASS', 'v2_validate_original_body_with_archived_JSON': 'PASS',
                        'actual_original_checked_result_rerun': False, 'new_inference': False})
    report = {'status': 'ALL108_ACTUAL_RAWJOB_RESULT_PATH_SCHEMAS_PASS_NO_INFERENCE',
              'new_mechanism_records': 8, 'Full_references': 100, 'output_field_present': sum(x['raw_job_has_output'] for x in checked),
              'v2_source_sha256': b.V2_SHA, 'validate_original_ast_sha256': hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest(),
              'checked': checked, 'archives': archive_checks, 'raw_job_result_members_verified': len(payloads),
              'checkpoint_members_extracted': 0, 'new_training': False, 'new_inference': False,
              'limitation': 'Actual archived path/schema precheck with authoritative accepted JSON; does not rerun original.checked_result, tensors, model inference or reaccept108 scientific results.'}
    b.save_new(b.HERE / 'actual108_path_schema_precheck.json', report)
    print(report['status'], len(payloads), 'job/result members; zero checkpoint extraction')


if __name__ == '__main__':
    main()
