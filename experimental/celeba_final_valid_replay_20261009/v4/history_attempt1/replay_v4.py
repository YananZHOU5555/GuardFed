"""Strict adapter-schema bridge; sealed v2 science and v3 runner remain unchanged.

Only SHA-bound FedAA/LASA input dictionaries receive documented in-memory views.
Run requires an externally SHA-bound successful semantic inspection of real900.
"""
from __future__ import annotations
import argparse
import copy
import json
from pathlib import Path, PurePosixPath
import sys
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
BASE = HERE.parent
sys.path.insert(0, str(BASE / 'v3'))
import replay_v3 as v3
v2 = v3.v2
V3_SHA = 'abb4560dd1c6752b9017a739381d2c54e68f52e278075c8e225a1c67df35cc6e'
SCOPE = 'VALID_ONLY_IMPLEMENTATION_REPLAY_V4'
EXTRA = None
BRIDGES = {}


def expected_paths(record):
    """Exactly the sealed inventory/storage-map derivation, never a user fallback."""
    full = record['method'] == 'GuardFed-AD2+'
    output = PurePosixPath(record['original_remote_output'])
    v2.require(output.is_absolute() and '..' not in output.parts and output.is_relative_to(v3.REPO / 'results'), 'Historical output escapes original repository results')
    raw = v3.REPO / record['raw_job']['member'] if full else v3.ARTIFACT_ROOT / 'artifact_store/archived_jobs' / (record['id'] + '.json')
    parent = output if full else v3.ARTIFACT_ROOT / 'artifact_store/original_outputs' / output.relative_to('/workspace')
    return {'raw_job': Path(raw), 'result': Path(parent / 'result.json'), 'checkpoint': Path(parent / 'model.pt')}


def normalized_views(record, paths, raw_job, raw_result):
    """Translate verified historical schemas, preserving every original dictionary."""
    require = v2.require
    require(paths == expected_paths(record), 'Read paths differ from exact inventory/storage derivation')
    job, result = copy.deepcopy(raw_job), copy.deepcopy(raw_result)
    require(job['id'] == PurePosixPath(record['raw_job']['member']).stem == record['id'], 'Original adapter job ID drift')
    require(job['dataset'] == 'celeba' and job['method'] == record['source_method'], 'Original adapter method/data drift')
    require(job['config'] == record['config'] and job['source_hashes'] == record['source_hashes'], 'Original adapter config/source drift')
    require((job['distribution'], job['attack']) == (record['distribution'], record['attack']), 'Original adapter scenario drift')
    require(job.get('adapter_hashes', job.get('adapter_source_hashes', {})) == record['adapter_source_hashes'], 'Original adapter source identity drift')
    method = record['method']
    if method in ('FedAA', 'LASA'):
        require('output' not in job, 'Unexpected output-bearing adapter rawjob schema')
        job['output'] = record['original_remote_output']
    else:
        require(job['output'] == record['original_remote_output'], 'Original output path drift')
    descriptor = {'schema': 'original_revision_job', 'raw_job_sha256': record['raw_job']['sha256'], 'result_sha256': record['result']['sha256'], 'checkpoint_sha256': record['checkpoint']['sha256'], 'historical_output': record['original_remote_output'], 'runtime_read_output': str(paths['result'].parent), 'derived_job_fields': [], 'derived_result_fields': [], 'original_artifact_bytes_modified': False}
    if method == 'FedAA':
        require('revision_job' not in result and 'alpha' not in result, 'Unexpected FedAA result schema')
        require(result['schema'] == 'fedaa_validation_screen_v1' and result['status'] in ('pilot_complete', 'coverage_complete'), 'Incomplete/unrecognized FedAA result')
        identity = result['identity']
        require(identity['job_sha256'] == record['raw_job']['sha256'], 'FedAA identity belongs to another rawjob')
        require(identity['config'] == result['config'] == job['config'], 'FedAA identity/config drift')
        require(identity['source_hashes'] == result['source_hashes'] == job['source_hashes'], 'FedAA result/source drift')
        require(identity['adapter_hashes'] == result['adapter_hashes'] == job['adapter_hashes'], 'FedAA result/adapter drift')
        require(identity['data_contract'] == result['data_contract'] == record['data_contract'], 'FedAA result/root/split identity drift')
        require(identity['environment']['torch'] == record['training_torch'], 'FedAA historical torch identity drift')
        require(identity['policy_seed'] == job['policy_seed'] == record['seed'], 'FedAA policy seed drift')
        require(identity['policy_config'] == result['policy_config'] == job['policy_config'], 'FedAA policy recipe drift')
        require(identity['aggre_num'] == result['aggre_num'] == job['aggre_num'], 'FedAA selected-client recipe drift')
        require((identity['distribution'], identity['attack'], result['id'], result['planned_rounds']) == (job['distribution'], job['attack'], job['id'], 70), 'FedAA result identity/horizon drift')
        require(result['checkpoint_sha256'] == record['checkpoint']['sha256'], 'FedAA result checkpoint drift')
        require(result['tuning_candidate'] == job['tuning_candidate'] and result['evidence_stage'] == job['evidence_stage'], 'FedAA selected recipe/history drift')
        result['revision_job'] = dict(job, checkpoint_sha256=result['checkpoint_sha256'])
        result['alpha'] = job['config']['client_alpha']
        result['data_contract'] = {'image_data_contract': result['data_contract']}
        descriptor.update(schema='FedAA_identity_v1_to_private_revision_view', derived_job_fields=['output <- inventory.original_remote_output'], derived_result_fields=['revision_job <- rawjob + original identity/checkpoint facts', 'alpha <- SHA-bound original config.client_alpha', 'data_contract.image_data_contract <- original data_contract'])
    else:
        provenance = result['revision_job']
        require(all(provenance.get(k) == value for k, value in job.items()), 'Original revision_job differs from complete original rawjob/output')
        require(provenance['checkpoint_sha256'] == record['checkpoint']['sha256'], 'Original result checkpoint drift')
        require(provenance['torch_version'] == record['training_torch'], 'Original historical torch identity drift')
        if method == 'LASA':
            descriptor.update(schema='LASA_rawjob_to_private_output_view', derived_job_fields=['output <- inventory.original_remote_output (also original revision_job.output)'])
    return job, result, descriptor


def mapped_functions(record, paths, original):
    """Keep original checked_result AND v2 validation/inference code objects intact."""
    v3.full_hashes({paths[k]: record[k]['sha256'] for k in v3.KINDS})
    raw_job, raw_result = v2.read(paths['raw_job']), v2.read(paths['result'])
    job, result, descriptor = normalized_views(record, paths, raw_job, raw_result)
    BRIDGES[record['id']] = descriptor
    by_member = {record[k]['member']: paths[k] for k in v3.KINDS}
    def located(repo, relative):
        return by_member.get(str(relative), v2.inside(repo, relative))
    def read_view(path):
        v2.require(Path(path) == paths['raw_job'], 'Private job view requested for another input')
        return copy.deepcopy(job)
    original_result_text = paths['result'].read_text()
    def loads_view(text):
        v2.require(text == original_result_text, 'Original checked_result opened a different result')
        return copy.deepcopy(result)
    checked = v3.private(original.checked_result, json=types.SimpleNamespace(loads=loads_view))
    def original_checked(actual_job):
        v2.require(actual_job == job, 'Private canonical job drift')
        return checked(dict(actual_job, output=str(paths['result'].parent)))
    validated = v3.private(v2.validate_original, inside=located, read=read_view)
    proxy = types.SimpleNamespace(checked_result=original_checked)
    def validate(_original, actual_record, repo):
        v2.require(actual_record == record, 'Bound record changed before original validation')
        return validated(proxy, actual_record, repo)
    inference = v3.private(v2.replay_one, inside=located, validate_original=validate)
    return validate, inference


def read_inputs(args, require_live=True):
    v2.require(v2.digest(BASE / 'v3/replay_v3.py') == V3_SHA, 'Sealed v3 source changed')
    records, bindings = v3.read_inputs(args, require_live)
    if args.command != 'inspect':
        v2.require(EXTRA.semantic_inspection is not None and EXTRA.semantic_inspection_sha256 is not None, 'Externally SHA-bound real900 semantic inspection required')
        p = EXTRA.semantic_inspection
        v2.require(v2.digest(p) == EXTRA.semantic_inspection_sha256, 'Semantic inspection SHA differs from reviewed receipt')
        r = v2.read(p)
        v2.require(r['status'] == 'ALL900_ORIGINAL_SEMANTICS_ACCEPTED_NO_INFERENCE' and r['accepted_n'] == 900 and not r['invalid'], 'Full900 original semantic gate did not pass')
        v2.require(r['v4_source_sha256'] == v2.digest(__file__) and r['sealed_v3_source_sha256'] == V3_SHA and r['inventory_sha256'] == v2.INVENTORY_SHA, 'Semantic inspection source/cohort drift')
        v2.require(r['storage_map_sha256'] == args.storage_map_sha256 and r['restore_acceptance_sha256'] == args.restore_acceptance_sha256, 'Semantic inspection restore chain drift')
        v3.check_source_tokens(r['source_before'])
    return records, bindings


def source_pins(repo, records, args):
    pins = v3.source_pins(repo, records, args)
    pins[BASE / 'v3/replay_v3.py'] = V3_SHA
    pins[Path(__file__).resolve()] = v2.digest(__file__)
    if EXTRA is not None and EXTRA.semantic_inspection is not None:
        pins[EXTRA.semantic_inspection.resolve()] = EXTRA.semantic_inspection_sha256
    return pins


def inspect(args):
    v2.require(sys.platform == 'linux' and v2.torch.__version__ == '2.11.0+cu128' and not sys.flags.optimize, 'Use unchanged isolated Linux cu128 with assertions')
    v2.require(not args.output.exists(), 'Preserve prior semantic inspection attempts')
    v3.bind_cpu(0)
    records, bindings = read_inputs(args)
    pins = source_pins(args.repo.resolve(), records, args)
    before = v3.full_hashes(pins)
    original = v2.load('v4_semantic_original_runner', args.repo / 'scripts/run_revision_ablation.py')
    accepted, invalid, artifacts = [], [], {}
    for record in records:
        paths = v3.mapping_paths(record, bindings)
        try:
            artifacts[record['id']] = v3.full_hashes({paths[k]: record[k]['sha256'] for k in v3.KINDS})
            validate, _ = mapped_functions(record, paths, original)
            result = validate(original, record, args.repo)
            accepted.append({'id': record['id'], 'record_sha256': v2.canonical(record), 'schema_bridge': BRIDGES[record['id']], 'original_native_metrics': result['metrics']})
        except Exception as exc:
            invalid.append({'id': record['id'], 'error_type': type(exc).__name__, 'error': str(exc)})
    after = v3.full_hashes(pins)
    v2.require(before == after, 'Source/data changed during semantic inspection')
    for values in artifacts.values():
        v3.check_source_tokens(values)
    report = {'scope': SCOPE, 'status': 'ALL900_ORIGINAL_SEMANTICS_ACCEPTED_NO_INFERENCE' if len(accepted) == 900 and not invalid else 'ORIGINAL_SEMANTICS_REJECTED_NO_INFERENCE', 'accepted_n': len(accepted), 'accepted': accepted, 'invalid': invalid, 'source_before': before, 'source_after': after, 'artifacts': artifacts, 'v4_source_sha256': v2.digest(__file__), 'sealed_v3_source_sha256': V3_SHA, 'inventory_sha256': v2.INVENTORY_SHA, 'storage_map_sha256': args.storage_map_sha256, 'restore_acceptance_sha256': args.restore_acceptance_sha256, 'new_image_inference': 0, 'semantic_labels_loaded': False, 'all900_native_valid_replayed': False, 'final_protocol_frozen': False}
    v2.save(args.output, report)
    print(json.dumps({'status': report['status'], 'accepted_n': len(accepted), 'invalid_n': len(invalid), 'sources': len(before), 'artifact_files': len(artifacts) * 3, 'receipt_sha256': v2.digest(args.output)}))
    return 0 if not invalid and len(accepted) == 900 else 1


def common_args(args):
    return v3.common_args(args) + ['--semantic-inspection', str(EXTRA.semantic_inspection), '--semantic-inspection-sha256', EXTRA.semantic_inspection_sha256]


OVERRIDES = dict(__file__=__file__, SCOPE=SCOPE, read_inputs=read_inputs, source_pins=source_pins, mapped_functions=mapped_functions, common_args=common_args)
_worker = v3.private(v3.worker, **OVERRIDES)

def worker(args):
    code = _worker(args)
    p = args.output.parent / (args.id + '.worker.json')
    if p.exists():
        proof = v2.read(p)
        proof['sealed_v3_runner_source_sha256'] = V3_SHA
        proof['v4_schema_bridge'] = BRIDGES.get(args.id)
        proof['semantic_inspection_sha256'] = EXTRA.semantic_inspection_sha256
        v2.save(p, proof)
    return code


def disabled_collect(_args):
    raise ValueError('Mixed v2/v3/v4 collector must be explicitly source-version bound; no900 completion claim is available here')


def main():
    global EXTRA
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--semantic-inspection', type=Path)
    parser.add_argument('--semantic-inspection-sha256')
    EXTRA, remainder = parser.parse_known_args()
    sys.argv = [sys.argv[0], *remainder]
    # v3 main's inspect handler historically discards a nonzero return, so wrap it.
    def inspect_or_raise(args):
        v2.require(inspect(args) == 0, 'Original900 semantic inspection rejected records; preserve receipt')
    return v3.private(v3.main, __file__=__file__, SCOPE=SCOPE, inspect=inspect_or_raise, run=v3.private(v3.run, **OVERRIDES), worker=worker, accept=v3.private(v3.accept, **OVERRIDES), collect=disabled_collect)()


if __name__ == '__main__':
    raise SystemExit(main())
