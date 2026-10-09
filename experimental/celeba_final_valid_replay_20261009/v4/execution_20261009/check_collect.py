"""Exercise the real cumulative CLI and refusal paths, without dataset access."""
import copy
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import collect_valid_replay as collector

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
spec = json.loads((HERE / 'collection_inputs_17.json').read_text())
proofs = [(version, str(BASE / path), sha) for version, path, sha in spec['proofs']]
failures = [(str(BASE / path), sha) for path, sha in spec['failure_proofs']]
inventory = BASE / spec['inventory']
command = [sys.executable, str(HERE / 'collect_valid_replay.py'), '--inventory', str(inventory)]
for values in proofs:
    command += ['--proof', *values]
for values in failures:
    command += ['--failure-proof', *values]
destination = HERE / 'cumulative_17_accepted.json'
command += ['--output', str(destination)]
subprocess.run(command, check=True)
r = json.loads(destination.read_text())
assert r['accepted_n'] == len(set(r['accepted_ids'])) == 17
assert len(r['missing_ids']) == 883 and not r['all900_native_valid_replayed']
assert len(r['preserved_nonaccepted_attempts']) == 1 and r['preserved_nonaccepted_attempts'][0]['accepted_n'] == 0
assert len(r['preserved_nonaccepted_attempts'][0]['later_accepted_ids']) == 8
stock = collector.read(inventory)['records']
alias = collector.aliases(stock)
first = stock[0]
assert alias[first['id']] == alias[first['raw_job']['member']] == alias[first['original_remote_output']]
rejected = []

def refuse(name, values):
    try:
        collector.collect(inventory, values, failures)
    except (ValueError, KeyError) as exc:
        rejected.append({'case': name, 'error_type': type(exc).__name__, 'error': str(exc)})
    else:
        raise AssertionError('Invalid cumulative evidence accepted: ' + name)

refuse('duplicate_canonical_ID_across_proofs', proofs + [proofs[0]])
refuse('wrong_declared_source_version', [*proofs[:-1], ('v3', proofs[-1][1], proofs[-1][2])])
semantic = BASE / 'v4/semantic900_inspection.json'
refuse('static900_semantic_gate_is_not_actual900_replay', [('v4', str(semantic), collector.sha(semantic))])
refuse('failure_attempt_is_not_accepted_replay', [('v3', failures[0][0], failures[0][1])])
corrupt = HERE / 'collector_refusal_inputs'
corrupt.mkdir(exist_ok=False)
wrong = corrupt / 'tampered_proof_bytes.json'
wrong.write_bytes(Path(proofs[-1][1]).read_bytes() + b'\n')
refuse('proof_bytes_tampered', [('v4', str(wrong), proofs[-1][2])])
# Keep the real manifest/archive alongside the controlled modified proof path.
# Its path is deliberately another directory, which must not be trusted implicitly.
wrong_archive = copy.deepcopy(collector.read(Path(proofs[-1][1])))
wrong_archive['archive_sha256'] = '0' * 64
wrong = corrupt / 'wrong_archive_proof.json'
wrong.write_text(json.dumps(wrong_archive) + '\n')
try:
    collector.verified_archive(Path(proofs[-1][1]), wrong_archive)
except ValueError as exc:
    rejected.append({'case': 'archive_sha_not_bound_to_proof', 'error_type': type(exc).__name__, 'error': str(exc)})
else:
    raise AssertionError('Wrong archive accepted')
assert len(rejected) == 6
report = {'status': 'ACTUAL17_SOURCE_VERSION_COLLECTOR_AND_REFUSALS_PASS', 'collector_source_sha256': collector.sha(HERE / 'collect_valid_replay.py'), 'actual_unique_accepted_n': 17, 'rejected_cases': rejected, 'original_job_and_inventory_alias_resolve_same_key': True, 'source_versions': r['source_versions'], 'cumulative_receipt_sha256': collector.sha(destination), 'new_image_inference': 0, 'all900_native_valid_replayed': False}
target = HERE / 'collector_selfcheck.json'
assert not target.exists()
target.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps({'status': report['status'], 'selfcheck_sha256': collector.sha(target), 'cumulative_sha256': collector.sha(destination)}, indent=2))
