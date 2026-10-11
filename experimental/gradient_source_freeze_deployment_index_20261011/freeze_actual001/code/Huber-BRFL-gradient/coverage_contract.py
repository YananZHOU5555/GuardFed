"""Pure metadata dependencies for the original sequential worker loop; no approvals created."""
from pathlib import Path
import hashlib, json


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def need(ok, why):
    if not ok: raise ValueError(why)


def validate(base, args):
    base = Path(base); bound = base.parent; binding = read(bound / 'BOUND_INPUTS.json')
    need(args.source_approval is not None and args.source_approval_sha256 is not None
         and args.gate14_adoption is not None and args.gate14_adoption_sha256 is not None,
         'Actual source adoption and completed14 root/offserver adoption paths/SHA required')
    need(sha(args.source_approval) == args.source_approval_sha256 and sha(args.gate14_adoption) == args.gate14_adoption_sha256,
         'Late-bound actual proof changed')
    approved, gate = read(args.source_approval), read(args.gate14_adoption)
    need(approved['status'] == 'ROOT_GRADIENT200_FROZEN_SOURCE_ADOPTED' and approved['test'] is False
         and approved['bound_inputs_sha256'] == sha(bound / 'BOUND_INPUTS.json')
         and approved['manifest_sha256'][base.name] == sha(base / 'jobs/manifest.json'), 'Actual selected source approval mismatch')
    need(sha(binding['root_path']) == binding['root_sha256'] and sha(binding['summary_path']) == binding['summary_sha256'], 'Selected64 inputs changed')
    root = read(binding['root_path']); summary = read(binding['summary_path']); manifest = read(base / 'jobs/manifest.json')
    need(root['status'] == 'ROOT_GRADIENT64_COMPLETE_STRICT_OFFSERVER_ADOPTED' and root['accepted_count'] == 64
         and root['all64_offserver_verified'] is True and root['test'] is False
         and root['summary_sha256'] == binding['summary_sha256']
         and root['screen_source_seal_sha256'] == '11e2ae63c87e0440669a047c930f5465b13babf559cca17bd8808bf693ce7ced'
         and root['selected_candidates'] == binding['winners'] and len(summary['records']) == 64, 'Partial screen/recipe forbidden')
    need(len(manifest['new_jobs']) == len({x['id'] for x in manifest['new_jobs']}) == 96
         and manifest['total'] == 100 and len(manifest['reused_jobs']) == 4
         and manifest['reused_jobs'] == [r for r in summary['records'] if r['candidate'] == binding['winners'][base.name]], 'Exact96+4 required')
    need(gate['status'] == 'ROOT_GRADIENT14_SAME_HORIZON_STRICT_OFFSERVER_ADOPTED'
         and gate['root_adoption'] is True and gate['accepted_gate_jobs'] == 14
         and gate['all14_offserver_verified'] is True and gate['scientific_table_records'] == 0
         and gate['frozen192_approval_sha256'] == args.source_approval_sha256
         and gate['test'] is False and gate['automatic_retry_authorized'] is False, 'Actual14 offserver/root gate required')
    scope_path, comparison_path = Path(gate['scope_path']), Path(gate['comparison_path'])
    need(sha(scope_path) == gate['scope_sha256'] and sha(comparison_path) == gate['comparison_sha256'], 'Actual gate source/comparison changed')
    scope, comparison = read(scope_path), read(comparison_path)
    ids = [j['id'] for j in scope['jobs']]
    need(scope['frozen192_approval_sha256'] == args.source_approval_sha256
         and scope['root64_sha256'] == binding['root_sha256'] and scope['rounds'] == 3 and scope['seed'] == 91002
         and len(ids) == len(set(ids)) == 14 and gate['accepted_gate_ids'] == ids
         and comparison['status'] == 'EXACT14_CPU3_GATE_SAVED_COMPARISON_PASS'
         and comparison['scope_sha256'] == gate['scope_sha256']
         and comparison['checked_jobs'] == 14 and comparison['rounds_per_job'] == 3 and comparison['total_rounds'] == 42
         and set(comparison['strict_receipts_sha256']) == set(ids), 'Same-horizon14 identity/comparison required')
    return binding


def resource_dependencies(base, proof, args):
    validate(base, args)
    need(proof['source_approval_sha256'] == args.source_approval_sha256
         and proof['gate14_adoption_sha256'] == args.gate14_adoption_sha256
         and proof['selected_manifest_sha256'] == sha(Path(base) / 'jobs/manifest.json')
         and proof['old_gradient64_service_status'] == 'EXITED' and proof['old_gradient64_supervisor_rc'] == 3
         and proof['old_gradient64_live_workers'] == 0 and proof['old_gradient64_failures'] == 0
         and proof['new_jobs_authorized'] == 96 and proof['reused_jobs_authorized_for_training'] == 0,
         'Fresh old64 EXITED0worker/no-failure and exact96 runtime authorization required')
