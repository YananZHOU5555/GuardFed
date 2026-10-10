"""Prepared exact3 binding. Import never executes CNN, root reconstruction or fit.

Root may explicitly call verify_one only after F transport/volume/label acceptance.
The scientific check_saved body is extracted unchanged from original900 source.
"""
import ast
import copy
import hashlib
import json
from pathlib import Path
import types

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'tmp/celeba_valid_gpu_remaining440_v2_evidence_20261009/chunk_039/verified_extract/sourcefreeze/062_saved_science.py'
CANDIDATE = ROOT / 'tmp/celeba_added_cnn_three_view_gate_preparation_20261010'
PACKAGE_SHA = '49c42b90214eae57d456f8ded1d2a4ff2de2762a78d6bcf5c4ae1a5766de9434'
KINDS = ('checkpoint', 'result', 'raw_job')

def need(ok, message):
    if not ok:
        raise ValueError(message)

def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def read(path):
    return json.loads(Path(path).read_bytes())

def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()

def runtime_record(row):
    """Same normalized metadata fields as immutable candidate.run, no method changes."""
    record = copy.deepcopy(row['identity'])
    record.update(distribution=row['distribution'], attack=row['attack'], seed=row['seed'],
                  result=record['original_artifact_pins']['result'],
                  raw_job=record['original_artifact_pins']['job'],
                  config_canonical_sha256=canonical(record['config']),
                  training_torch=row['original_training_torch'])
    return record

def measure(paths, record):
    observed = {}
    for kind, path in paths.items():
        actual = sha(path)
        need(actual == record[kind]['sha256'], 'Original accepted artifact SHA changed: ' + kind)
        observed[str(path)] = {'sha256': actual, 'bytes': Path(path).stat().st_size}
    return observed

def verify_one(row, run, *, bridge, evaluator, core, ids, y, sensitive, root_authorized_cached_refit=False):
    """Explicit later root action: saved arrays + original cached-root refit, no CNN.

    Caller owns F-volume/transport membership verification and supplies original
    metadata prefix (train+valid only) and source-pinned bridge/evaluator/core.
    Does not create acceptance files or mark a model root-adopted.
    """
    need(root_authorized_cached_refit is True, 'Cached-root refit requires explicit root call')
    need(sha(CANDIDATE/'FILES_SHA256.json') == PACKAGE_SHA, 'Candidate source seal changed')
    manifest = read(CANDIDATE/'MANIFEST.json')
    need(row in manifest['records'], 'Only one of the frozen exact3 records is allowed')
    run = Path(run).resolve()
    need(run.is_relative_to(Path('F:/YananResearchStorage/GuardFed').resolve()), 'Offserver arrays must be under GuardFed F storage')
    need(not list(run.glob('failure*.json')) and not (run/'FAILURE.json').exists(), 'Failure evidence must remain rejected')
    record = runtime_record(row)
    receipt = read(run/'receipt.json')
    need(receipt['id'] == row['id'] and receipt['status'] == 'NATIVE_VALID_REPLAY_PASS', 'Unfinished/mismatched receipt')
    need(receipt['external_identity_record_sha256'] == canonical(record), 'Normalized runtime identity differs')
    for key, expected in [('checkpoint_sha256', record['checkpoint']['sha256']),
                          ('original_result_sha256', record['result']['sha256']),
                          ('original_job_sha256', record['raw_job']['sha256']),
                          ('config_canonical_sha256', record['config_canonical_sha256'])]:
        need(receipt[key] == expected, 'Receipt identity differs: '+key)
    need(record['config']['rounds'] == 70 and record['config']['celeba_evaluation_split'] == 'valid'
         and receipt['valid_n'] == 19867, 'Wrong horizon/split/support')
    need(not any(receipt[k] for k in ('optimizer_created','gradients_created','test_labels_accessed','test_inference_performed','final_dispatch_created')), 'Inference-only contract failed')
    need(receipt['runtime']['device'] == 'cpu' and receipt['runtime']['cuda_device_count'] == 0
         and receipt['runtime']['original_config_device'] == record['config']['device'], 'CPU/original CUDA identity confused')
    need(len(ids) == 202599 and len(y) == len(sensitive) == 182637, 'Only official train+valid label prefix allowed')
    paths = {k: Path(record[k]['path']) for k in KINDS}
    before = measure(paths, record)
    second = measure(paths, record)
    # These are two real LOCAL observations, not invented remote before/after proof.
    proof = {'artifact_before': before, 'artifact_after': second}
    def validate_external(_original, r, _repo):
        need(bridge.identity_record(row['method'], row['id']) == row['identity'], 'Original accepted bridge identity changed')
        return read(paths['result'])
    v2 = types.SimpleNamespace(np=evaluator.rebuild_root.__globals__['np'], require=need,
                              rebuild_root=evaluator.rebuild_root, digest=sha, canonical=canonical,
                              VIEWS=['native','raw','shared_calibration'], check_native=evaluator.check_native)
    namespace = {'v2':v2, 'KINDS':KINDS,
                 'mapping_paths':lambda _r, _bindings:paths,
                 'full_hashes':lambda _pins:measure(paths, record),
                 'mapped_functions':lambda _r, _paths, _original:(validate_external, None)}
    pin = next(p for p in read(HERE/'SOURCE_PINS.json')['files'] if p['path'] == SOURCE.relative_to(ROOT).as_posix())
    need(sha(SOURCE) == pin['sha256'], 'Original saved-science source changed')
    text = SOURCE.read_text(encoding='utf-8')
    node = next(n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == 'check_saved')
    need(hashlib.sha256(ast.get_source_segment(text,node).encode()).hexdigest() == pin['functions']['check_saved']['sha256'], 'Original check_saved body changed')
    exec(compile(ast.Module(body=[node],type_ignores=[]), str(SOURCE), 'exec'), namespace)
    comparison = namespace['check_saved'](record, run, receipt, proof, types.SimpleNamespace(repo=ROOT),
                                          None, core, None, evaluator, ids, y, sensitive)
    after = measure(paths, record)
    need(before == second == after, 'Local original artifacts changed across saved-array validation')
    return {'id':row['id'], 'status':'LOCAL_SAVED_ARRAY_AND_ORIGINAL_CACHED_ROOT_REFIT_PASS_NOT_ROOT_ADOPTED',
            'native_comparison':comparison, 'receipt_sha256':sha(run/'receipt.json'),
            'array_sha256':sha(run/'validation_predictions.npz'), 'artifact_observations':{'before':before,'second_before':second,'after':after},
            'original_check_saved_AST_unchanged':True, 'root_refit_executed':True, 'new_CNN':0,
            'root_adopted':False, 'transport_verified_by_this_function':False}
