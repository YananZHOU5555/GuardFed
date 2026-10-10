"""Metadata projection onto the sealed C-after70 replay/accept body; no Torch import."""
import ast
import copy
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import types

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
SCOPE = 'MECHANISM_REMAINING620_TERMINAL_VALID_ONLY'
CPUS = list(range(112, 120))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def load(name, path, expected):
    require(digest(path) == expected, 'Dependency drift: ' + str(path))
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def project(parent, plan, identity, native_sha):
    """Only the two former batch-metadata gates change. bind_runtime code is reused."""
    source = Path(parent.__file__).read_text(encoding='utf-8')
    functions = {n.name: ast.get_source_segment(source, n) for n in ast.parse(source).body if isinstance(n, ast.FunctionDef)}
    ns = dict(parent.__dict__, SCOPE=SCOPE, ACCEPTED_IDS=[identity],
              EXCLUDED_PRIOR_IDS=plan['excluded180_ids'], REPLAY_IDS=[identity], INSPECTION_SHA=native_sha)
    for name, replacements in plan['metadata_replacements'].items():
        text = functions[name]
        for before, after in replacements:
            require(text.count(before) == 1, 'Metadata source anchor changed: ' + before)
            text = text.replace(before, after, 1)
        exec(compile(text, '<remaining620 metadata projection>', 'exec'), ns)
    require(ns['bind_runtime'].__code__ is parent.bind_runtime.__code__, 'Scientific body code changed')
    ns['bind_runtime'] = types.FunctionType(parent.bind_runtime.__code__, ns, parent.bind_runtime.__name__)
    return types.SimpleNamespace(**ns)


def inventory(plan, record, native_sha):
    identity = record['id']
    require(identity in plan['remaining620_ids'] and identity not in plan['excluded180_ids'], 'Closed/Full/foreign record')
    return dict(scope=SCOPE, status='PREPARED_NOT_APPROVED', baseline_inventory_sha256=plan['baseline_inventory_sha256'],
        mechanism_manifest_sha256=plan['manifest_sha256'], mechanism_inspection_sha256=native_sha,
        mechanism_adapter_hashes=plan['adapter_hashes'], mechanism_source_hashes=plan['source_hashes'],
        mechanism_protocol_sha256=plan['protocol_sha256'], native_tolerance=1e-12,
        views=['native', 'raw', 'shared_calibration'], full_references=plan['full_references'], records=[record],
        excluded_prior_replay_ids=plan['excluded180_ids'], selected_replay_ids=[identity],
        pending_new_ids_no_checkpoint=[i for i in plan['remaining620_ids'] if i != identity],
        legacy_pending_field_semantics='Other IDs in the frozen620; this per-ID inventory does not claim they lack checkpoints or evaluations.',
        new_image_inference_performed=False, full_weights_repacked=0)


def construct_record(parent, job, entry, result, row, full):
    # The assignment source is extracted from the original accepted constructor.
    path = HERE / 'record_constructor.py'
    refs = {}
    for kind, member, actual in [('checkpoint', 'runs/' + entry['id'] + '/model.pt', Path(entry['output']) / 'model.pt'),
                                  ('result', 'runs/' + entry['id'] + '/result.json', Path(entry['output']) / 'result.json'),
                                  ('raw_job', 'jobs/' + entry['id'] + '.json', Path(entry['job']))]:
        refs[kind] = dict(archive=None, archive_sha256=None, origin='LIVE_PRODUCER_CLOSED_ORIGINAL_STRICT_ACCEPTED',
                          member=member, sha256=row['files'][str(actual)], bytes=actual.stat().st_size)
    ns = dict(model_id=entry['id'], job=job, entry=entry, result=result, row=row, control=full,
              refs=refs, b=parent, copy=copy)
    exec(compile(path.read_text(encoding='utf-8'), str(path), 'exec'), ns)
    return ns['record']
