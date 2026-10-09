"""Freeze only after actual-runtime image gates and independent Full100 acceptance."""
from pathlib import Path
import sys
import time

REPO = Path('/workspace/GuardFed-celeba-expanded')
STAGE = REPO / 'results/revision_20261009/celeba_mechanism_v1'
CHECKS = Path('/workspace/guardfed_checks/server_reactivation_20261009')
sys.path.insert(0,str(REPO/'deployment/celeba_mechanism_20261009'))
import runner
import torch

assert not sys.flags.optimize and torch.__version__ == '2.11.0+cu128'
manifest = runner.read(STAGE/'manifest.json')
runner.live_identity(REPO,STAGE,manifest)
assert not (STAGE/'dispatch_receipt.json').exists()
gate_proof = runner.read(CHECKS/'preflight_acceptance_cu128.json')
assert gate_proof['pass'] and gate_proof['accepted_pipeline_jobs'] == 20
assert gate_proof['runtime'] == torch.__version__
assert gate_proof['manifest_sha256'] == runner.digest(STAGE/'manifest.json')
backup_proof = runner.read(CHECKS/'preflight_cu128_offserver_verification.json')
backup = runner.read(CHECKS/'preflight_backup_cu128.json')
assert backup_proof['off_server_verified']
assert backup_proof['members_verified'] == len(backup['members'])
assert backup_proof['archive_sha256'] == backup['sha256']
for entry in manifest['preflight_jobs']+manifest['reference_jobs']:
    result = runner.read(Path(entry['output'])/'result.json')
    assert result['revision_job']['torch_version'] == torch.__version__
full_path = CHECKS/'full_inspection_v1/full_inspection.json'
full = runner.read(full_path)
assert full['status'] == 'FULL_REUSE_VERIFIED_100' and full['accepted_reused_count'] == 100
assert not full['invalid'] and not full['pending']
assert full['manifest_sha256'] == runner.digest(STAGE/'manifest.json')
assert full['inventory_sha256'] == runner.digest(CHECKS/'model_inventory.json')
assert full['inspector_sha256'] == runner.digest(CHECKS/'evidence.py') == '1c0961ae991d75d32d3269e967ac6bfdcdfb783afafcfcc965445a99a179617b'
full_by_id = {row['id']:row for row in full['full_identity_receipts']}
for entry in manifest['reused_full']:
    proof = full_by_id[entry['id']]; out=Path(entry['output'])
    assert runner.digest(out/'model.pt') == proof['checkpoint_sha256']
    assert runner.digest(out/'result.json') == proof['result_sha256']
    assert not list(out.glob('failure*.json'))
receipt = runner.freeze(REPO,STAGE,manifest)
receipt.update(strict_prerequisites={'gate_actual_runtime':torch.__version__,
    'preflight_acceptance_sha256':runner.digest(CHECKS/'preflight_acceptance_cu128.json'),
    'preflight_offserver_proof_sha256':runner.digest(CHECKS/'preflight_cu128_offserver_verification.json'),
    'full100_inspection_sha256':runner.digest(full_path),
    'full_inventory_sha256':full['inventory_sha256'], 'full_inspector_sha256':full['inspector_sha256'],
    'freezer_sha256':runner.digest(__file__), 'full_runtime_counts':full['torch_counts'],
    'cross_runtime_short_gate_exact':sum(r['metrics_and_diagnostics_exact'] and r['all_tensors_exact'] for r in gate_proof['cross_runtime_comparisons']),
    'full70round_cross_environment_equivalence_claimed':False, 'checked_at_unix':time.time()})
runner.save(STAGE/'dispatch_receipt.json',receipt)
print({'pass':True,'image_gates':receipt['image_gates'],'references':len(receipt['full_regressions']),
       'Full':len(receipt['reused_full']),'torch':receipt['torch_version'],
       'dispatch_receipt_sha256':runner.digest(STAGE/'dispatch_receipt.json')})
