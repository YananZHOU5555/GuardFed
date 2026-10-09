"""Read-only, no-image-inference precheck of the two declared canaries using replay's validators."""
import os
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
import replay


def main():
    here = Path(__file__).resolve().parent
    repo = Path('/workspace/GuardFed-celeba-expanded')
    plan_path = here / 'inputs/valid_replay_plan.json'
    output = here / 'pre_inference_inspection.json'
    replay.require(not output.exists(), 'Preserve existing inspection receipt')
    inherited = sorted(os.sched_getaffinity(0))
    replay.require(len(inherited) >= 24, 'Missing coordinated CPU affinity slice')
    for task in Path('/proc/self/task').glob('*'):
        try:
            os.sched_setaffinity(int(task.name), inherited[16:24])
        except ProcessLookupError:
            pass
    os.nice(10)
    subprocess.run(['ionice', '-c', '3', '-p', str(os.getpid())], capture_output=True, check=True)
    plan = replay.read(plan_path)
    inventory = replay.read(plan_path.parent / plan['inventory_file'])
    replay.validate_inventory(inventory)
    selected = [next(r for r in inventory['records'] if r['id'] == i) for i in replay.CANARY_IDS]
    replay.require(all(r['training_torch'] == '2.11.0+cu128' for r in selected), 'Mixed cu130 canary refused')
    pinned = replay.input_paths(repo, selected, plan_path, plan)
    pinned[Path(__file__).resolve()] = replay.digest(__file__)
    before = replay.hash_pins(pinned)
    core = replay.load('inspect_frozen_core', repo / 'scripts/reproduce_paper_tables.py')
    original = replay.load('inspect_original_checked_result', repo / 'scripts/run_revision_ablation.py')
    cnn = replay.load('inspect_original_cnn', repo / 'src/celeba_data.py')
    resources = replay.live_snapshot(repo)
    replay.resource_gate(resources)
    ids, y, s, meta = replay.metadata(repo)
    inspected = []
    for record in selected:
        replay.validate_original(original, record, repo)
        cfg = core.ExperimentConfig(**record['config'])
        _, _, _, root = replay.rebuild_root(core, cfg, record, ids, y, s)
        state = replay.torch.load(replay.inside(repo, record['checkpoint']['member']), map_location='cpu', weights_only=True)
        replay.require(state and all(replay.torch.isfinite(t).all().item() for t in state.values()), 'Invalid checkpoint tensors')
        model = cnn.CelebACNN(seed=cfg.seed).cpu()
        model.load_state_dict(state, strict=True)
        inspected.append({'id': record['id'], 'root': root, 'checkpoint_keys_shapes_finite': True})
    after = replay.hash_pins(pinned)
    replay.require(before == after, 'Original inputs changed during inspection')
    replay.save(output, {'status': 'VALID_INPUTS_INSPECTED_NOT_REPLAYED', 'records': inspected, 'metadata': meta,
                         'input_sha256_before': before, 'input_sha256_after': after, 'resources': resources,
                         'inherited_allowed_cpus': inherited, 'selected_cpus': inherited[16:24],
                         'actual_image_inference_performed': False, 'test_labels_accessed': False,
                         'all900_native_valid_replayed': False, 'final_protocol_frozen': False})
    print({'status': 'VALID_INPUTS_INSPECTED_NOT_REPLAYED', 'models': len(inspected), 'inputs': len(before), 'receipt_sha256': replay.digest(output)})


if __name__ == '__main__':
    main()
