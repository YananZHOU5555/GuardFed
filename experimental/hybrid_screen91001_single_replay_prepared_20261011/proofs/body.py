"""Prepared CUDA pipeline/screen instrumentation derived from sealed CPU gate; no CLI dispatch."""
import argparse
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

HERE = Path(__file__).resolve().parent
REPO = Path('/workspace/GuardFed-celeba-expanded')
SEALED = HERE / 'scientific_snapshot'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    tmp.replace(path)


def tensor_sha(state):
    h = hashlib.sha256()
    for name, value in state.items():
        value = value.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(value.dtype).encode())
        h.update(str(tuple(value.shape)).encode()); h.update(value.numpy().tobytes())
    return h.hexdigest()


def snapshot():
    import resource
    stage = REPO / 'results/revision_20261009/celeba_mechanism_v1'
    queue = read(stage / 'formal_queue_progress.json')
    active = []
    for row in queue['active']:
        p = stage / 'runs' / row['id'] / 'progress.json'
        item = read(p) if p.exists() else {}
        active.append(dict(id=row['id'], pid=row['pid'], round=item.get('round'), updated_unix=item.get('updated_unix')))
    memory = {}
    for line in Path('/proc/meminfo').read_text().splitlines():
        k, v = line.split(':', 1)
        if k in ('MemTotal', 'MemAvailable'):
            memory[k] = v.strip()
    stats = os.statvfs(REPO)
    return dict(at_unix=time.time(), completed=len(queue['completed']), failed=queue['failed'], active=active,
        pending=queue['pending'], memory=memory, disk_free_bytes=stats.f_bavail * stats.f_frsize,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,temperature.gpu',
                                     '--format=csv,noheader'], text=True).strip(),
        loadavg=Path('/proc/loadavg').read_text().strip(),
        cpu_times={k:getattr(os.times(), k) for k in ('user', 'system', 'children_user', 'children_system', 'elapsed')},
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        cpu_affinity=sorted(os.sched_getaffinity(0)))


def verify_scope(scope):
    assert scope['status'] in ('FROZEN_BOUNDED_HYBRID_CUDA_GATE_ONLY', 'FROZEN_HYBRID_VALID_SCREEN_ONLY')
    assert scope['test_evaluation_authorized'] is False and scope['max_processes'] == 1 and scope['cpu_threads'] == 1
    for rel, expected in scope['protected_source_hashes'].items():
        assert digest(REPO / rel) == expected, ('protected source/data drift', rel)
    for rel, expected in scope['local_hashes'].items():
        assert digest(HERE / rel) == expected, ('prepared source drift', rel)
    for entry in scope['jobs']:
        assert digest(HERE / entry['job']) == entry['job_sha256']
    assert digest('/etc/vast-agents-guide.md') == scope['guide_sha256']


def modules():
    sys.path.insert(0, str(SEALED))
    spec = importlib.util.spec_from_file_location('sealed_hybrid_gate_worker', SEALED / 'worker.py')
    worker = importlib.util.module_from_spec(spec); sys.modules[spec.name] = worker; spec.loader.exec_module(worker)
    return worker, worker.load_core(REPO)


def checked(entry, scope):
    import torch
    out = HERE / entry['output']; job = read(HERE / entry['job'])
    if not (out / 'result.json').exists():
        return None
    assert not list(out.glob('failure*.json'))
    receipt = read(out / 'acceptance.json')
    assert receipt['status'] == 'PASS' and receipt['scope_sha256'] == digest(HERE / scope['scope_file'])
    assert receipt['job_sha256'] == entry['job_sha256']
    for name, expected in receipt['artifact_hashes'].items():
        assert digest(out / name) == expected
    result = read(out / 'result.json'); diag = read(out / 'diagnostics.json'); prov = read(out / 'provenance.json')
    assert result['config'] == job['config'] and result['method'] == job['method']
    assert result['dataset'] == 'celeba' and result['distribution'] == job['distribution']
    assert result['attack'] == job['attack'] and result['seed'] == 91001 and result['alpha'] == job['config']['client_alpha']
    assert result['evidence_stage'] == scope['evidence_stage'] and result['scientific_table_records'] == 0
    horizon = job['config']['rounds']; assert horizon == scope['rounds'] and horizon in (3, 70)
    assert [r['round'] for r in result['trajectory_metrics']] == list(range(1, horizon + 1))
    assert [r['round'] for r in result['round_summaries']] == list(range(1, horizon + 1))
    assert [r['round'] for r in diag] == list(range(1, horizon + 1))
    assert result['metrics'] == result['trajectory_metrics'][-1]['metrics']
    assert all(math.isfinite(result['metrics'][k]) for k in ('accuracy', 'aeod', 'aspd'))
    data = result['data_contract']; contract = data['image_data_contract']
    assert (contract['evaluation_split'], contract['actual_train_rows'], contract['actual_evaluation_rows']) == ('valid', 162770, 19867)
    assert contract['train_eval_disjoint'] and contract['root_client_disjoint']
    assert data['root_clean_rows'] == 16277 and data['root_synthetic_rows'] == 0
    assert data['synthetic_method'] == 'none' and data['feature_includes_label'] is False
    assert contract['cache_manifest_sha256'] == scope['protected_source_hashes']['data/celeba/derived/rgb64_v1/manifest.json']
    assert contract['evidence_stage'] == 'full_official_split'
    assert contract['model'] == 'Conv32/64/128_3x3_ReLU_MaxPool_GAP_Linear2'
    numeric = contract['numerical_execution']
    assert numeric['deterministic_algorithms'] and not numeric['cudnn_allow_tf32'] and not numeric['matmul_allow_tf32']
    assert result['evaluation_stats']['prediction_count'] == 19867
    assert prov['source_hashes'] == scope['protected_source_hashes'] and prov['local_hashes'] == scope['local_hashes']
    assert prov['torch'] == torch.__version__ == '2.11.0+cu128'
    assert prov['cuda_build'] == torch.version.cuda == '12.8' and prov['device'] == 'cuda:0' and prov['cpu_threads'] == 1
    assert prov['cuda_device_count'] == 1 and prov['cuda_visible_devices'] == scope['runtime_cuda_visible_device']
    assert prov['gpu_uuid'] == scope['runtime_gpu_uuid'] and prov['gpu_name'] == torch.cuda.get_device_name(0)
    state = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    assert state and all(torch.isfinite(x).all() for x in state.values())
    assert tensor_sha(state) == receipt['checkpoint_tensor_sha256']
    replay = read(out / 'native_replay.json')
    assert replay['metrics'] == result['metrics'] and replay['checkpoint_tensor_sha256'] == tensor_sha(state)
    assert replay['prediction_count'] == 19867 and replay['root_group_label_total'] == 16277
    assert all(n > 0 for n in replay['evaluation_group_label_counts'].values())
    assert replay['root_image_ids_sha256'] == contract['root_image_ids_sha256']
    assert replay['evaluation_image_ids_sha256'] == contract['evaluation_image_ids_sha256']
    assert all(r['legacy_delta_exact'] and r['legacy_selection_exact'] and r['legacy_trust_exact'] and r['rng_unchanged'] for r in diag)
    assert all(len(r['root_aeod']) == 20 and len(r['client_counts']) == 20 for r in diag)
    assert all(len(r['client_weights']) == 20 and abs(sum(r['client_weights']) - 1) < 1e-12 for r in diag)
    return result


def run_one(entry, scope, worker, core):
    import torch
    import numpy as np
    job = read(HERE / entry['job']); out = HERE / entry['output']
    if checked(entry, scope) is not None:
        return
    assert not out.exists(), ('partial output needs review', str(out))
    out.mkdir()
    started = time.time(); before = snapshot(); write(out / 'resource_before.json', before)
    prov = dict(scope_sha256=digest(HERE / scope['scope_file']), job_sha256=entry['job_sha256'],
        source_hashes=scope['protected_source_hashes'], local_hashes=scope['local_hashes'],
        torch=torch.__version__, cuda_build=torch.version.cuda, python=sys.version,
        device='cuda:0', cpu_threads=torch.get_num_threads(), cpu_affinity=sorted(os.sched_getaffinity(0)), pid=os.getpid(), started_unix=started,
        cuda_device_count=torch.cuda.device_count(), cuda_visible_devices=os.environ['CUDA_VISIBLE_DEVICES'],
        gpu_name=torch.cuda.get_device_name(0), gpu_uuid=scope['runtime_gpu_uuid'])
    write(out / 'provenance.json', prov)
    diagnostics = []
    original = core.aggregate_round
    original_bundle = core.load_bundle
    loaded_bundle = []
    hybrid = worker.aggregation_wrapper(original, job['adapter'])

    def retained_bundle(*args, **kwargs):
        bundle = original_bundle(*args, **kwargs)
        loaded_bundle.append(bundle)
        return bundle

    def audited(method, updates, counts, fairness, server_update, config, **kwargs):
        if method not in ('CosineFairnessHybrid', 'GuardFed'):
            raise ValueError('Unexpected method in bounded gate')
        rng_before = torch.get_rng_state().clone()
        cuda_rng_before = [x.clone() for x in torch.cuda.get_rng_state_all()]
        delta, info = hybrid('CosineFairnessHybrid', updates, counts, fairness, server_update, config, **kwargs)
        reference, legacy = original('GuardFed', updates, counts, fairness, server_update, config, **kwargs)
        exact = all(torch.equal(delta[k], reference[k]) for k in delta)
        trust_exact = info['trust_scores'] == legacy['trust_scores']
        selected_exact = info['selected_clients'] == legacy['selected_clients']
        rng_exact = torch.equal(rng_before, torch.get_rng_state()) and all(torch.equal(x,y) for x,y in zip(cuda_rng_before,torch.cuda.get_rng_state_all()))
        assert exact and trust_exact and selected_exact and rng_exact, 'Actual legacy path differs; preserve evidence'
        weights = [1. / len(info['selected_clients']) if i in info['selected_clients'] else 0. for i in range(20)]
        row = dict(round=len(diagnostics) + 1, legacy_delta_exact=exact, legacy_trust_exact=trust_exact,
            legacy_selection_exact=selected_exact, rng_unchanged=rng_exact, aggregate_tensor_sha256=tensor_sha(delta),
            global_tensor_sha256=tensor_sha(kwargs['global_state']), server_update_tensor_sha256=tensor_sha(server_update),
            attacked_client_tensor_sha256=[tensor_sha(x) for x in updates],
            root_aeod=[float(x['aeod']) for x in kwargs['fairness_details']],
            root_metrics=kwargs['fairness_details'], client_counts=list(counts), client_weights=weights,
            selected_clients=info['selected_clients'], trust_scores=info['trust_scores'],
            cumulative_seconds=time.time() - started)
        diagnostics.append(row); write(out / 'diagnostics.json', diagnostics)
        return (delta, info) if method == 'CosineFairnessHybrid' else (reference, legacy)

    def progress(item):
        write(out / 'progress.json', dict(item, id=entry['id'], pid=os.getpid(),
            elapsed_seconds=time.time() - started, updated_unix=time.time(), resource=snapshot()))
        print(json.dumps(dict(id=entry['id'], round=item['round'], elapsed_seconds=time.time() - started)), flush=True)

    try:
        core.aggregate_round = audited
        core.load_bundle = retained_bundle
        config = core.ExperimentConfig(**job['config'])
        result = core.run_experiment('celeba', job['distribution'], job['method'], job['attack'], config,
            scope['evidence_stage'], torch.device('cuda:0'), progress_callback=progress,
            checkpoint_path=out / 'model.pt')
        assert len(diagnostics) == scope['rounds']
        assert not any(not math.isfinite(float(x['aeod'])) for r in diagnostics for x in r['root_metrics'])
        result.update(evidence_stage=scope['evidence_stage'], scientific_table_records=0,
            revision_job=dict(job, job_sha256=entry['job_sha256'], scope_sha256=digest(HERE / scope['scope_file'])),
            method_impl_note=worker.LABEL if job['method'] == 'CosineFairnessHybrid' else 'Unchanged frozen core legacy GuardFed reference; native argmax')
        write(out / 'result.json', result)
        write(out / 'resource_after.json', snapshot())
        state = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
        assert len(loaded_bundle) == 1
        bundle = loaded_bundle[0]
        replay_model = core.make_model(bundle, config, torch.device('cuda:0'))
        replay_model.load_state_dict(state)
        replay = core.evaluate_for_reporting(job['method'], replay_model, bundle, config)
        assert {k:replay[k] for k in core.METRICS} == result['metrics']
        sy = bundle['y_test'].numpy(); ss = bundle['test_sensitive']
        ry = bundle['server_y'].numpy(); rs = bundle['server_sensitive']
        contract = bundle['image_data_contract']
        write(out / 'native_replay.json', dict(metrics={k:replay[k] for k in core.METRICS},
            checkpoint_tensor_sha256=tensor_sha(state), prediction_count=replay['prediction_count'],
            evaluation_group_label_counts={f's{s}_y{y}':int(((ss == s) & (sy == y)).sum()) for s in (0, 1) for y in (0, 1)},
            root_group_label_total=len(ry), root_group_label_counts={f's{s}_y{y}':int(((rs == s) & (ry == y)).sum()) for s in (0, 1) for y in (0, 1)},
            root_image_ids_sha256=contract['root_image_ids_sha256'], evaluation_image_ids_sha256=contract['evaluation_image_ids_sha256']))
        write(out / 'rng_final.json', dict(torch_sha256=hashlib.sha256(torch.get_rng_state().numpy().tobytes()).hexdigest(),
            cuda_rng_sha256=[hashlib.sha256(x.cpu().numpy().tobytes()).hexdigest() for x in torch.cuda.get_rng_state_all()],
            numpy_state_sha256=hashlib.sha256(repr(np.random.get_state()).encode()).hexdigest()))
        artifacts = ('result.json', 'model.pt', 'diagnostics.json', 'provenance.json', 'resource_before.json', 'resource_after.json', 'rng_final.json', 'native_replay.json')
        write(out / 'acceptance.json', dict(status='PASS', scope_sha256=digest(HERE / scope['scope_file']),
            job_sha256=entry['job_sha256'], artifact_hashes={n:digest(out / n) for n in artifacts},
            checkpoint_tensor_sha256=tensor_sha(state), elapsed_seconds=time.time() - started,
            limitation=scope['claim_limit']))
        assert checked(entry, scope) is not None
    except BaseException as error:
        write(out / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), at_unix=time.time()))
        raise
    finally:
        core.aggregate_round = original
        core.load_bundle = original_bundle
        loaded_bundle.clear()


def compare(scope):
    import torch
    pairs = []
    assert scope['kind'] == 'cuda_pipeline_gate' and len(scope['jobs']) == 4
    for a, b in zip(scope['jobs'][::2], scope['jobs'][1::2]):
        left = checked(a, scope); right = checked(b, scope)
        assert left is not None and right is not None
        for name in ('metrics', 'trajectory_metrics', 'attack_audit', 'evaluation_stats', 'data_contract', 'last10_metrics'):
            assert left[name] == right[name], ('legacy full-pipeline mismatch', name)
        for x, y in zip(left['round_summaries'], right['round_summaries']):
            assert {k:v for k,v in x.items() if k != 'aggregate'} == {k:v for k,v in y.items() if k != 'aggregate'}
            for key in ('selected_clients', 'trust_scores'):
                assert x['aggregate'][key] == y['aggregate'][key]
        ls = torch.load(HERE / a['output'] / 'model.pt', map_location='cpu', weights_only=True)
        rs = torch.load(HERE / b['output'] / 'model.pt', map_location='cpu', weights_only=True)
        assert ls.keys() == rs.keys() and all(torch.equal(ls[k], rs[k]) for k in ls)
        assert read(HERE / a['output'] / 'rng_final.json') == read(HERE / b['output'] / 'rng_final.json')
        ld = read(HERE / a['output'] / 'diagnostics.json'); rd = read(HERE / b['output'] / 'diagnostics.json')
        assert [{k:v for k,v in x.items() if k != 'cumulative_seconds'} for x in ld] == [{k:v for k,v in x.items() if k != 'cumulative_seconds'} for x in rd]
        pairs.append(dict(hybrid=a['id'], legacy=b['id'], all_metrics_model_attacks_diagnostics_rng_exact=True))
    return pairs

