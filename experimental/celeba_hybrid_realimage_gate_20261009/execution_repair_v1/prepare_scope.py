"""Prepare an explicit two-canary repair scope; never approve or execute it."""
import ast
import hashlib
import json
from pathlib import Path
import tarfile

HERE = Path(__file__).resolve().parent
ORIGINAL = HERE.parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def save(name, value):
    target = HERE / name
    assert not target.exists(), 'Never overwrite a prepared repair artifact'
    target.write_bytes((json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n').encode())


scope = read(ORIGINAL / 'scope.json')
backup = ORIGINAL / 'terminal_failure_backup_20261009'
receipt = read(backup / 'backup_receipt.json'); verification = read(backup / 'offserver_verification.json')
assert sha(backup / 'hybrid_terminal_failure_evidence.tar.gz') == receipt['archive_sha256']
assert verification['pass'] and verification['different_host_observed'] and verification['members_verified'] == 46
refs = []
with tarfile.open(backup / 'hybrid_terminal_failure_evidence.tar.gz', 'r:gz') as handle:
    names = handle.getnames()
    for entry in scope['jobs'][:2]:
        suffix = entry['output'] + '/acceptance.json'
        matches = [name for name in names if name.endswith(suffix)]
        assert len(matches) == 1
        data = handle.extractfile(matches[0]).read(); accepted = json.loads(data)
        assert accepted['status'] == 'PASS' and accepted['job_sha256'] == entry['job_sha256']
        assert accepted['scope_sha256'] == sha(ORIGINAL / 'scope.json')
        refs.append(dict(id=entry['id'], output=entry['output'], job_sha256=entry['job_sha256'],
            acceptance_sha256=hashlib.sha256(data).hexdigest(), artifact_hashes=accepted['artifact_hashes'],
            archive_member=matches[0], offserver_archive_sha256=receipt['archive_sha256'],
            evidence_kind='ACCEPTED_REAL_IMAGE_3ROUND_CANARY_NOT_SCIENTIFIC_RESULT'))
save('reused_IID_references.json', dict(status='HASH_BOUND_TWO_ACCEPTED_IID_CANARIES_READ_ONLY_REUSE',
    records=refs, repeated_training_authorized=False, repeated_inference_authorized=False))
tree = ast.parse((ORIGINAL / 'gate.py').read_text(encoding='utf-8-sig'))
reused = {node.name: hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
          for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {'run_one', 'checked', 'compare', 'verify_scope', 'modules'}}
save('reused_scientific_functions.json', dict(original_gate_sha256=sha(ORIGINAL / 'gate.py'),
    original_core_sha256=scope['protected_source_hashes']['scripts/reproduce_paper_tables.py'],
    function_ast_sha256=reused,
    private_globals_only=['HERE points to isolated output overlay', 'write binds exact diagnostic-only strict writer',
                          'checked calls unchanged original acceptance then requires the exact undefined sidecar'],
    unchanged=['CNN/localAdam/data/model/train/eval/attack/aggregation source bytes',
               'original run_one/compare codeobjects', 'all original checked conditions and native equality']))
save('REPAIR_SCOPE.json', dict(status='PREPARED_NOT_APPROVED', scientific_table_records=0,
    original_scope_sha256=sha(ORIGINAL / 'scope.json'), original_gate_sha256=sha(ORIGINAL / 'gate.py'),
    original_failed_attempt_status='TERMINAL_FAILURE', original_failure_reclassified=False,
    original_failed_attempt_archive_sha256=receipt['archive_sha256'], new_jobs=scope['jobs'][2:],
    reused_IID_ids=[row['id'] for row in refs], reused_IID_references_sha256=sha(HERE / 'reused_IID_references.json'),
    repair='Exact attack_audit[0..3].fflip_label_corr_after original Python float NaN -> explicit JSON null plus NaN-bit/cause sidecar only',
    original_scientific_function_reuse_sha256=sha(HERE / 'reused_scientific_functions.json'),
    max_concurrent_cpu_processes=1, cpu_threads=8, proposed_exclusive_cpu_ids=list(range(8, 16)),
    device='cpu', torch='2.11.0+cu128', cuda_build='12.8', nice=10, ionice='idle',
    expected_real_train_rows=162770, expected_clean_root_rows=16277, expected_valid_rows=19867,
    new_actual_runs_if_later_approved=2, new_actual_rounds_if_later_approved=6,
    formal32_authorized=False, test_evaluation_authorized=False, automatic_retry_authorized=False,
    required_approval_status='APPROVED_TWO_UNACCEPTED_HYBRID_CANARIES_DIAGNOSTIC_WRITER_ONLY',
    limitations=['Component checks are not real-image training results',
      'Original failed third result was not retained; finite saved checkpoint does not count as acceptance',
      'Unrecognized diagnostic NaN, metric/weight/control nonfinite and nonfinite model remain fatal',
      'Original loader may materialize full-split attribute metadata; no test image inference/fitting/selection',
      'No seventy-round or CUDA/CPU equivalence/performance claims']))
print('PREPARED_NOT_APPROVED two new canaries; two accepted IID references; no execution')
