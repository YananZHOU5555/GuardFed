"""Build the exact620 source plan from accepted180; never read future run directories."""
import ast
import datetime
import difflib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
from bridge_adapter import SCOPE, digest, load, read, require
from evaluate_remaining import save_new

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = ROOT / 'tmp/celeba_mechanism_valid_C_after70_20261010'
TRAIN = 'docs/server_deployment_20260923/training_20260923'
MANIFEST = TRAIN + '/celeba_mechanism_v1/manifest.json'
BASELINE = 'tmp/celeba_final_valid_replay_20261009/inputs/model_inventory.json'
C80 = TRAIN + '/celeba_mechanism_v1/three_view_C_eight_scenes_20261010/ROOT_VERIFICATION.json'
ADOPTION = 'tmp/celeba_mechanism_valid_C_after70_20261010/execution_candidate/backups/incremental_20261010T025719Z/ROOT_ADOPTION_REVIEW.json'
PARENT_SHA = 'f337fce71153318d0a5046082a0f7ceda8b78f790fbd657796233bcbef8c9717'


def main():
    volume = json.loads(subprocess.check_output(['powershell', '-NoProfile', '-Command',
        'Get-Volume -DriveLetter F | Select-Object DriveLetter,FileSystemLabel,SizeRemaining,Size | ConvertTo-Json -Compress'], text=True))
    require(volume['DriveLetter'] == 'F' and volume['FileSystemLabel'] == 'Yanan 2TB' and volume['SizeRemaining'] >= 10 * 1024 ** 3, 'F storage identity/headroom failed')
    save_new(HERE / 'LOCAL_STORAGE.json', dict(checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        actual_volume=volume, large_files_written=0, arrays_written=0, archives_written=0,
        local_large_artifact_boundary='F only; runtime output is Linux server-only; E contains code/config/small reports'))
    parent = load('remaining620_prepare_original_bridge', OLD / 'bridge.py', PARENT_SHA)
    require(digest(ROOT / MANIFEST) == parent.MANIFEST_SHA and digest(ROOT / BASELINE) == parent.BASELINE_INVENTORY_SHA, 'Original manifest/inventory changed')
    prior = read(OLD / 'inventory_actual180_Full100refs.json'); baseline = read(ROOT / BASELINE)
    require(digest(OLD / 'inventory_actual180_Full100refs.json') == 'b4ffdf5f3dc549e397f82db49290cdceb1ee024478cdc4e58365f5c537d144a9', 'Prior180 inventory changed')
    parent.validate_inventory(prior, baseline)
    require(digest(ROOT / ADOPTION) == '3fc1e49e927a971a577d648dd9a7ff44ec7ac81552ea250026349d4f2e06d615', 'Actual180 closure changed')
    require(read(ROOT / ADOPTION)['cumulative_three_view_models'] == 180, 'Prior180 not root adopted')
    require(digest(ROOT / C80) == '71105f39c6345efb3706fe538a51686f80ad9e8a89f5601a04b349ebcc76e487', 'C80 adopted snapshot changed')
    manifest = read(ROOT / MANIFEST)
    excluded = [r['id'] for r in prior['records']]
    remaining = [e['id'] for e in manifest['jobs'] if e['id'] not in set(excluded)]
    require(len(excluded) == 180 and len(remaining) == len(set(remaining)) == 620, 'Wrong exact complement')
    require(sum(i.startswith('minus_U_') for i in excluded) == 100 and sum(i.startswith('minus_C_') for i in excluded) == 80, 'Prior U100+C80 differs')
    require(set(remaining) == set(prior['pending_new_ids_no_checkpoint']), 'Original pending620 boundary differs')
    dependency_local = dict(v2='tmp/celeba_final_valid_replay_20261009/replay.py',
        v3='tmp/celeba_final_valid_replay_20261009/v3/replay_v3.py', evaluator='tmp/celeba_final_valid_replay_20261009/inputs/evaluator.py',
        evidence_v4='tmp/celeba_mechanism_evidence_20261009/evidence_v4.py', baseline_inventory=BASELINE,
        manifest=MANIFEST, parent_bridge='tmp/celeba_mechanism_valid_C_after70_20261010/bridge.py')
    paths = read(OLD / 'execution_candidate/RUNTIME_BINDINGS.json')['dependency_paths']
    paths['parent_bridge'] = '/workspace/guardfed_checks/celeba_mechanism_valid_C_after70_20261010/bridge.py'
    hashes = {key: digest(ROOT / p) for key, p in dependency_local.items()}
    replacements = dict(validate_inventory=[
        ["len(records) == 180 and len({r['id'] for r in records}) == 180", "len(records) == 1 and len({r['id'] for r in records}) == 1"],
        ["r['variant'] in ('minus_U', 'minus_C')", "r['variant'] in VARIANTS[1:]"],
        ["len(missing) == len(set(missing)) == 620", "len(missing) == len(set(missing)) == 619"],
        ["set(missing) | {r['id'] for r in records} == expected_ids", "set(missing) | {r['id'] for r in records} | set(EXCLUDED_PRIOR_IDS) == expected_ids"],
        ['This inventory contains exactly180 actual accepted terminals', 'This per-ID inventory contains one original-strict accepted terminal'],
        ['Actual accepted180 snapshot changed', 'Once-bound terminal identity changed'],
        ['Excluded-prior170/selected10 boundary changed', 'Excluded-prior180/selected1 boundary changed'],
        ['Only adopted U100+C70 plus exact ten C terminals; no other scope', 'Only frozen620 controls may be bound; Full/prior180 remain excluded']],
        require_approval=[['len(ids) == len(set(ids)) == 10', 'len(ids) == len(set(ids)) == 1']])
    plan = dict(status='PREPARED_NOT_APPROVED_NO_FUTURE_CHECKPOINT_HASHES', scope=SCOPE,
        manifest_sha256=parent.MANIFEST_SHA, baseline_inventory_sha256=parent.BASELINE_INVENTORY_SHA,
        protocol_sha256=manifest['protocol_sha256'], source_hashes=manifest['source_hashes'], adapter_hashes=manifest['adapter_hashes'],
        prior180_adoption_sha256=digest(ROOT / ADOPTION), C80_root_sha256=digest(ROOT / C80),
        excluded180_ids=excluded, remaining620_ids=remaining, all_manifest_entries=manifest['jobs'],
        entries=[dict(e, checkpoint_sha256=None) for e in manifest['jobs'] if e['id'] in set(remaining)],
        full_references=prior['full_references'], remote_dependencies=paths, dependency_sha256=hashes,
        remote_training_progress='/workspace/GuardFed-celeba-expanded/results/revision_20261009/celeba_mechanism_v1/formal_queue_progress.json',
        metadata_replacements=replacements, native_tolerance=1e-12, Full_inference=0,
        checkpoint_binding='Only after original producer exit and original evidence_v4.accept_new; immutable atomic per-ID file, no intermediate model',
        recovery='NO_AUTOMATIC_RESUME_OR_RETRY. Preserve all partial/failure evidence; root must separately review a fresh complement for recovery.',
        remote_closure='REMOTE_STRICT_CLOSED_PENDING_OFFSERVER; accepted_offserver remains0')
    save_new(HERE / 'PLAN.json', plan)
    constructor_path = ROOT / 'tmp/celeba_mechanism_valid_incremental_next11_20261009/prepare.py'
    constructor = constructor_path.read_text(encoding='utf-8')
    nodes = [n for n in ast.walk(ast.parse(constructor)) if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'record' for t in n.targets)]
    require(len(nodes) == 1, 'Original metadata constructor changed')
    with (HERE / 'record_constructor.py').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(ast.get_source_segment(constructor, nodes[0]) + '\n')
    functions = {n.name: ast.get_source_segment(Path(parent.__file__).read_text(encoding='utf-8'), n)
                 for n in ast.parse(Path(parent.__file__).read_text(encoding='utf-8')).body if isinstance(n, ast.FunctionDef)}
    patch = []
    for name, changes in replacements.items():
        before = functions[name]; after = before
        for a, b in changes:
            require(after.count(a) == 1, 'Exact metadata anchor absent: ' + a); after = after.replace(a, b, 1)
        patch.extend(difflib.unified_diff(before.splitlines(True), after.splitlines(True), fromfile='sealed_C_after70/' + name, tofile='remaining620_projection/' + name))
    with (HERE / 'SCIENTIFIC_METADATA_DIFF.patch').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(''.join(patch))
    exact = [name for name in functions if name not in replacements]
    pins = {p: {'sha256': digest(ROOT / p), 'bytes': (ROOT / p).stat().st_size} for p in
            list(dependency_local.values()) + [ADOPTION, C80, 'tmp/celeba_mechanism_valid_C_after70_20261010/inventory_actual180_Full100refs.json',
            'tmp/celeba_mechanism_valid_C_after70_20261010/FULL100_ACTUAL_SOURCE_BINDINGS.json',
            'docs/server_deployment_20260923/training_20260923/celeba_mechanism_v1/three_view_interim100_20261009/ROOT_REVIEW.json',
            'tmp/celeba_mechanism_valid_incremental_next11_20261009/prepare.py']}
    save_new(HERE / 'SOURCE_BINDINGS.json', dict(input_pins=pins, dependency_local_paths=dependency_local,
        parent_bridge_sha256=PARENT_SHA, unchanged_parent_functions=exact,
        unchanged_function_source_sha256={name: __import__('hashlib').sha256(functions[name].encode()).hexdigest() for name in exact},
        scientific_body='Original bind_runtime bytecode/source, nested replay_one and accept_saved_predictions; no science rewrite',
        only_projected_functions=list(replacements), original_constructor_assignment_source_exact=True,
        no_Full_inference=True, no_optimizer=True, no_test_inference=True, actual_execution=False))
    with (HERE / 'REMAINING_620_IDS.txt').open('x', encoding='utf-8', newline='\n') as stream:
        stream.write('\n'.join(remaining) + '\n')
    print(json.dumps(dict(status='PREPARED_SOURCE_PLAN', remaining620=620, excluded180=180, Full_reference_only=100, future_checkpoint_sha256=None)))


if __name__ == '__main__':
    main()
