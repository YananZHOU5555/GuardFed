"""Bounded real-image gradient pilot. Preparation never authorizes dispatch."""
from __future__ import annotations

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
SNAPSHOT = HERE / 'snapshot/gradient_bridge_20261009'
STAGE = 'exploratory_full_image_cpu_gradient_gate_only'
ARTIFACTS = {'job.json', 'result.json', 'model.pt', 'diagnostics.json', 'gradient_audit.json',
             'provenance.json', 'native_replay.json', 'resource_before.json', 'resource_after.json',
             'identity_before.json', 'identity_after.json'}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def write(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf8')
    temporary.replace(path)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda: stream.read(8 * 1024 * 1024), b''):
            h.update(part)
    return h.hexdigest()


def hashes(root, expected):
    root = Path(root).resolve()
    actual = {}
    for name, sha in expected.items():
        path = (root / name).resolve()
        require(path.is_relative_to(root) and path.is_file(), 'Missing/escaped identity: ' + name)
        actual[name] = digest(path)
        require(actual[name] == sha, 'Changed source/data: ' + name)
    return actual


def tensor_sha(state):
    h = hashlib.sha256()
    for name, tensor in state.items():
        tensor = tensor.detach().cpu().contiguous()
        h.update(name.encode()); h.update(str(tensor.dtype).encode())
        h.update(str(tuple(tensor.shape)).encode()); h.update(tensor.numpy().tobytes())
    return h.hexdigest()


def validate_scope(scope):
    require(scope['status'] == 'PREPARED_NOT_FROZEN' and not scope['execution_started'], 'Pilot preparation changed')
    require(scope['evidence_stage'] == STAGE and scope['scientific_table_records'] == 0 and
            scope['test_evaluation_authorized'] is False, 'Only exploratory valid-only gate is supported')
    require(scope['max_concurrent_cpu_processes'] == 1 and scope['cpu_threads'] == 8, 'Bounded CPU scope changed')
    protocol = read(SNAPSHOT / 'protocol.json')
    require(protocol['status'] == 'PREPARED_NOT_FROZEN' and len(protocol['protocol_decisions']) == 5 and
            all(x['status'] == 'UNRESOLVED' for x in protocol['protocol_decisions'].values()), 'Formal decisions were changed')
    require(digest(SNAPSHOT / 'protocol.json') == scope['formal_protocol_sha256'], 'Original protocol drift')
    hashes(HERE, scope['local_hashes'])
    expected_conditions = {('IID', 'Benign'), ('non-IID', 'S-DFA')}
    seen = set()
    for entry in scope['jobs']:
        require(digest(HERE / entry['job']) == entry['job_sha256'], 'Pilot job drift')
        job = read(HERE / entry['job'])
        require(job['id'] == entry['id'] and entry['output'] == 'runs/' + job['id'], 'Output/job mismatch')
        require(job['method'] in {'Fed-NGA-gradient', 'Huber-BRFL-gradient'} and
                (job['distribution'], job['attack']) in expected_conditions, 'Unknown bounded case')
        candidate = next(c for c in protocol['candidates'] if c['id'] == job['pilot_candidate'])
        require(job['adapter'] == candidate['adapter'] and job['method'] == candidate['method'], 'Pilot recipe drift')
        expected_config = dict(protocol['base_config'], seed=91001, rounds=3, device='cpu',
            client_alpha=protocol['distributions'][job['distribution']], use_reweighting=False,
            experiment_suite='celeba_gradient_realimage_gate_20261009', experiment_tag=job['id'])
        require(job['config'] == expected_config and job['evidence_stage'] == STAGE and
                job['scientific_table_records'] == 0, 'Wrong full-image pilot configuration')
        require(job['source_hashes'] == protocol['source_hashes'], 'Source/data contract changed')
        identity = (job['method'], job['distribution'], job['attack'])
        require(identity not in seen, 'Duplicate bounded job'); seen.add(identity)
    require(len(seen) == 4, 'Exactly two cases per method required')


def dispatch(scope, receipt_path):
    """Reject before creating output or importing torch when resource receipt is absent."""
    require(receipt_path is not None and Path(receipt_path).is_file(), 'CPU allocation/dispatch receipt required; prepare only')
    receipt = read(receipt_path)
    require(receipt['status'] == 'APPROVED_BOUNDED_EXPLORATORY_GATE_ONLY' and
            receipt['scope_sha256'] == digest(HERE / 'scope.json') and
            receipt['jobs'] == {x['id']: x['job_sha256'] for x in scope['jobs']}, 'Unapproved/mismatched pilot dispatch')
    require(receipt['source_hashes'] == scope['protected_source_hashes'] and
            receipt['local_hashes'] == scope['local_hashes'], 'Dispatch source identity mismatch')
    require(receipt['formal_decisions_approved'] is False and receipt['screen64_authorized'] is False and
            receipt['test_authorized'] is False, 'Bounded dispatch cannot approve a formal protocol')
    cpus = receipt['exclusive_cpu_ids']
    require(isinstance(cpus, list) and len(cpus) == len(set(cpus)) == 8 and
            all(isinstance(x, int) and not isinstance(x, bool) and x >= 0 for x in cpus), 'Explicit eight-CPU allocation required')
    require(not set(cpus).intersection(scope['protected_hybrid_cpu_ids']) and
            receipt['verified_no_overlap_with_live_cpu_gates'] is True, 'Hybrid/other CPU gate allocation not protected')
    return receipt


def snapshot():
    import resource
    stage = REPO / 'results/revision_20261009/celeba_mechanism_v1'
    queue = read(stage / 'formal_queue_progress.json')
    active = []
    for row in queue['active']:
        path = stage / 'runs' / row['id'] / 'progress.json'
        progress = read(path) if path.exists() else {}
        active.append(dict(id=row['id'], pid=row['pid'], round=progress.get('round'), updated_unix=progress.get('updated_unix')))
    stat = os.statvfs(REPO)
    return dict(at_unix=time.time(), completed=len(queue['completed']), failed=queue['failed'], active=active,
        pending=queue['pending'], cpu_affinity=sorted(os.sched_getaffinity(0)),
        cpu_user_seconds=os.times().user, cpu_system_seconds=os.times().system,
        peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        memory={line.split(':')[0]: line.split(':')[1].strip() for line in Path('/proc/meminfo').read_text().splitlines()
                if line.startswith(('MemTotal:', 'MemAvailable:'))},
        disk_free_bytes=stat.f_bavail * stat.f_frsize,
        gpu=subprocess.check_output(['nvidia-smi', '--query-gpu=index,utilization.gpu,memory.used,temperature.gpu',
                                     '--format=csv,noheader'], text=True).strip())


def formal_growth(before, after):
    old = {x['id']: x['round'] for x in before['active']}
    return after['completed'] > before['completed'] or any(
        row['id'] in old and old[row['id']] is not None and row['round'] is not None and row['round'] > old[row['id']]
        for row in after['active'])


def load_worker():
    spec = importlib.util.spec_from_file_location('sealed_real_gradient_worker', SNAPSHOT / 'worker.py')
    worker = importlib.util.module_from_spec(spec); sys.modules[spec.name] = worker; spec.loader.exec_module(worker)
    return worker


def checked(entry, scope, receipt):
    import torch
    out = HERE / entry['output']; job = read(HERE / entry['job'])
    require(not list(out.glob('failure*.json')), 'Failure evidence blocks reuse')
    if not (out / 'result.json').exists():
        return None
    accepted = read(out / 'acceptance.json')
    require(accepted['status'] == 'PASS' and set(accepted['artifact_hashes']) == ARTIFACTS, 'Incomplete acceptance receipt')
    require(accepted['scope_sha256'] == digest(HERE / 'scope.json') and
            accepted['job_sha256'] == entry['job_sha256'] and
            accepted['dispatch_receipt'] == receipt, 'Accepted pilot identity mismatch')
    hashes(out, accepted['artifact_hashes'])
    result = read(out / 'result.json'); provenance = read(out / 'provenance.json')
    require(read(out / 'job.json') == job and result['revision_job'] == job, 'Raw job mismatch')
    require(result['status'] == 'exploratory_gate_complete' and result['evidence_stage'] == STAGE and
            result['scientific_table_records'] == 0 and result['config'] == job['config'] and
            result['gradient_recipe'] == job['adapter'], 'Wrong scientific stage/configuration')
    for field in ('dataset', 'method', 'distribution', 'attack'):
        require(result[field] == job[field], 'Condition mismatch: ' + field)
    require((result['seed'], result['rounds'], result['alpha'], result['num_clients'], result['num_malicious']) ==
            (91001, 3, job['config']['client_alpha'], 20, 4), 'Endpoint/client identity mismatch')
    for field in ('trajectory_metrics', 'round_summaries'):
        require([x['round'] for x in result[field]] == [1, 2, 3], 'Incomplete or duplicate rounds')
    require(result['metrics'] == result['trajectory_metrics'][-1]['metrics'] and
            all(math.isfinite(x['metrics'][k]) and 0 <= x['metrics'][k] <= 1 for x in result['trajectory_metrics']
                for k in ('accuracy', 'aeod', 'aspd')), 'Mixed endpoint or invalid metrics')
    contract = result['data_contract']; image = contract['image_data_contract']
    require((contract['train_rows'], contract['test_rows'], contract['root_clean_rows'], contract['root_synthetic_rows'],
             image['actual_train_rows'], image['actual_evaluation_rows'], image['evaluation_split']) ==
            (162770, 19867, 16277, 0, 162770, 19867, 'valid'), 'Full real-image sample contract failed')
    require(image['train_eval_disjoint'] and image['root_client_disjoint'] and image['evidence_stage'] == 'full_official_split'
            and not contract['feature_includes_label'] and not contract['feature_includes_sensitive'], 'Data leakage/subset')
    require(image['model'] == 'Conv32/64/128_3x3_ReLU_MaxPool_GAP_Linear2' and
            image['target'] == 'Smiling' and image['sensitive'] == 'Male' and image['augmentation'] == 'none' and
            image['cache_manifest_sha256'] == scope['protected_source_hashes']['data/celeba/derived/rgb64_v1/manifest.json'], 'Image/model identity mismatch')
    numeric = image['numerical_execution']
    require(numeric['deterministic_algorithms'] and numeric['cudnn_deterministic'] and not numeric['cudnn_benchmark'] and
            not numeric['cudnn_allow_tf32'] and not numeric['matmul_allow_tf32'], 'Deterministic FP32 contract failed')
    require(provenance['source_hashes'] == scope['protected_source_hashes'] and provenance['local_hashes'] == scope['local_hashes']
            and provenance['job_sha256'] == entry['job_sha256'] and provenance['scope_sha256'] == digest(HERE / 'scope.json') and
            provenance['dispatch_receipt'] == receipt and provenance['device'] == 'cpu' and provenance['cpu_threads'] == 8 and
            provenance['cpu_affinity'] == sorted(receipt['exclusive_cpu_ids']) and
            provenance['torch'] == '2.11.0+cu128' and provenance['cuda_build'] == '12.8', 'Runtime/source provenance mismatch')
    require(read(out / 'identity_before.json') == read(out / 'identity_after.json') == scope['protected_source_hashes'], 'Source/data changed during run')
    require(read(out / 'diagnostics.json') == result['round_summaries'], 'Diagnostics/result mismatch')
    audit = read(out / 'gradient_audit.json')
    require(len(audit) == 60 and [x['client_id'] for x in audit] == list(range(20)) * 3 and
            [x['round'] for x in audit] == [r for r in (1, 2, 3) for _ in range(20)], 'Incomplete actual client gradient audit')
    require(all(x['same_point_unchanged'] and x['attack_sign_oracle_exact'] and x['local_optimizer_steps'] == 0 and
                x['rows'] == image['client_sample_counts'][x['client_id']] and x['objective_denominator'] == x['rows'] and
                math.isfinite(x['raw_norm']) and math.isfinite(x['uploaded_norm']) for x in audit), 'Wrong gradient/attack/sample semantics')
    require(sum(image['client_sample_counts']) == 146493 and all(x > 0 for x in image['client_sample_counts']), 'Client partition/count mismatch')
    for row in result['round_summaries']:
        info = row['aggregate']; records = [x for x in audit if x['round'] == row['round']]
        require(info['actual_parameters'] == job['adapter'] and info['counts'] == image['client_sample_counts'] and
                info['client_order'] == row['client_ids'] == list(range(20)) and info['local_optimizer_steps'] == 0 and
                info['projection'] == 'identity_Rp' and info['aggregation_weighting'] == 'raw_client_sample_count' and
                not info['defense_root_fairness_used'] and not info['defense_root_reference_used'], 'Gradient mechanism altered')
        require(all(x['global_point_sha256'] == info['global_point_sha256'] for x in records), 'Gradients at different model points')
        require(info['pilot_aggregation_oracle_pass'] and math.isfinite(info['server_step_norm']), 'Aggregation oracle failed')
        weights = info['pilot_effective_weights']
        require(len(weights) == 20 and all(math.isfinite(x) and 0 <= x <= 1 for x in weights) and
                math.isclose(sum(weights), 1., abs_tol=1e-12), 'Invalid actual aggregation weights')
        root = info['root_threat_reference']; attacked = job['attack'] == 'S-DFA'
        require(root['used'] == attacked and info['pilot_root_adam_calls'] == int(attacked), 'Undeclared root access')
        if attacked:
            require(root['mode'] == 'frozen_root_localadam_delta' and root['root_rows'] == 16277 and
                    not root['evaluation_labels_used'] and not root['defense_input'], 'Wrong threat reference')
        if job['method'] == 'Huber-BRFL-gradient':
            wanted = [job['adapter']['t0'] + job['adapter']['m'] / math.sqrt(n) for n in info['counts']]
            require(info['converged'] and info['stationarity_l2'] <= job['adapter']['tolerance'] and
                    all(math.isclose(a, b, rel_tol=1e-14, abs_tol=1e-15) for a, b in zip(info['fixed_thresholds'], wanted)) and
                    len(info['fixed_thresholds']) == 20, 'Huber convergence/threshold identity failed')
            trace = info['objective_trace']
            require(trace and all(math.isfinite(x) and x >= 0 for x in trace) and info['objective'] == trace[-1] and
                    info['iterations'] == len(trace) - 1 <= job['adapter']['max_iter'] and
                    all(b <= a + 1e-10 * max(1., abs(a)) for a, b in zip(trace, trace[1:])), 'Huber objective trace failed')
    require(not result['warnings'] and [x['client_id'] for x in result['attack_audit']] == list(range(20)) and
            not any(x.get('label_changed_count', 0) for x in result['attack_audit']), 'Attack/label audit failed')
    if job['attack'] == 'S-DFA':
        require(all(x['attack_types'] == ['fflip', 'foe'] and x['foe_mode'] == 'fedsa' and
                    'gradient sign-conjugacy' in x['foe_impl'] for x in result['attack_audit'][:4]), 'S-DFA semantics changed')
    state = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
    require(state and all(torch.isfinite(x).all() for x in state.values()), 'Nonfinite terminal checkpoint')
    replay = read(out / 'native_replay.json')
    require(tensor_sha(state) == accepted['checkpoint_tensor_sha256'] == replay['checkpoint_tensor_sha256'] and
            replay['metrics'] == result['metrics'] and replay['prediction_count'] == result['evaluation_stats']['prediction_count'] == 19867,
            'Terminal model/repredict mismatch')
    for prefix, count in (('evaluation', 19867), ('root', 16277)):
        support = replay[prefix + '_group_label_counts']
        require(set(support) == {'s0_y0', 's0_y1', 's1_y0', 's1_y1'} and all(x > 0 for x in support.values()) and
                sum(support.values()) == count and replay[prefix + '_image_ids_sha256'] == image[prefix + '_image_ids_sha256'],
                'Wrong group/label support or image IDs')
    require(not read(out / 'resource_after.json')['failed'], 'Protected formal queue failed')
    return result


def run_one(entry, scope, receipt, worker, core, components):
    import copy
    import numpy as np
    import torch
    job = read(HERE / entry['job']); out = HERE / entry['output']
    if out.exists():
        require(checked(entry, scope, receipt) is not None, 'Partial output: preserve and review; no automatic retry')
        return
    out.mkdir(parents=True, exist_ok=False); started = time.time()
    original = {name: getattr(core, name) for name in ('load_bundle', 'train_server_update', 'train_local_model',
                'evaluate_state_on_server', 'evaluate_model_calibrated')}
    empirical, attack, aggregate = components.empirical_gradient, worker.attack_gradient, worker.aggregate_gradients
    bundle_box, records, pending, root_calls = [], [], [], [0]
    current = {'point': None, 'model': None}
    try:
        write(out / 'job.json', job); write(out / 'identity_before.json', hashes(REPO, scope['protected_source_hashes']))
        write(out / 'resource_before.json', snapshot())
        write(out / 'provenance.json', dict(scope_sha256=digest(HERE / 'scope.json'), job_sha256=entry['job_sha256'],
            source_hashes=scope['protected_source_hashes'], local_hashes=scope['local_hashes'], dispatch_receipt=receipt,
            torch=torch.__version__, cuda_build=torch.version.cuda, python=sys.version, device='cpu',
            cpu_threads=torch.get_num_threads(), cpu_affinity=sorted(os.sched_getaffinity(0)), pid=os.getpid(), started_unix=started))

        def retained_bundle(*args, **kwargs):
            bundle = original['load_bundle'](*args, **kwargs); bundle_box.append(bundle)
            require(bundle['train_rows'] == 162770 and bundle['test_rows'] == 19867 and bundle['root_clean_rows'] == 16277,
                    'Full-image load contract failed before gradients')
            return bundle

        def forbidden(*args, **kwargs):
            raise RuntimeError('Forbidden client-Adam/root-fairness/calibration path in gradient pilot')

        def root_adam(model, bundle, config):
            before = worker.parameters_vector(model).clone()
            delta = original['train_server_update'](model, bundle, config)
            require(torch.equal(before, worker.parameters_vector(model)), 'Root reference mutated global model')
            root_calls[0] += 1
            return delta

        def real_gradient(model, client, batch_size, *, use_reweighting=False):
            require(not use_reweighting and client['cid'] == len(pending), 'Unexpected CE/client order')
            point = worker.parameters_vector(model).clone()
            if not pending:
                current.update(point=point, model=model)
            require(torch.equal(point, current['point']), 'Client gradients at different points')
            gradient = empirical(model, client, batch_size, use_reweighting=False)
            require(torch.equal(point, worker.parameters_vector(model)), 'Empirical gradient mutated model')
            row = dict(round=len(records) // 20 + 1, client_id=client['cid'], rows=client['n'],
                objective_denominator=client['n'], local_optimizer_steps=0, same_point_unchanged=True,
                global_point_sha256=worker.vector_digest(point), raw_gradient_sha256=worker.vector_digest(gradient),
                raw_norm=float(torch.linalg.vector_norm(gradient.double())))
            pending.append(row)
            return gradient

        def real_attack(gradient, client, root_descent, config, audit):
            uploaded, result_audit = attack(gradient, client, root_descent, config, audit)
            # This is a zero-origin *message algebra oracle*, never an Adam/local model surrogate.
            message = {'gradient_message': -gradient}; zero = {'gradient_message': torch.zeros_like(gradient)}
            reference, _ = core.apply_foe_if_needed(message, zero, client, copy.deepcopy(audit),
                                                   {'gradient_message': root_descent}, config)
            require(torch.equal(uploaded, -reference['gradient_message']), 'Frozen attack sign/norm algebra differs')
            row = pending[-1]
            row.update(uploaded_gradient_sha256=worker.vector_digest(uploaded), uploaded_norm=float(torch.linalg.vector_norm(uploaded.double())),
                attack_sign_oracle_exact=True, actual_attack_types=client['attack_types'], actual_foe_mode=client['foe_mode'],
                oracle_semantics='g_attack=-A(-g,r); zero-origin gradient-message algebra only; no local model/Adam substitution')
            if 'foe' in client['attack_types'] and row['raw_norm'] > 0 and float(torch.linalg.vector_norm(root_descent)) > 0:
                require(row['uploaded_norm'] <= config.fedsa_norm_ratio * row['raw_norm'] * (1 + 2e-6), 'Attack norm cap failed')
            write(out / 'gradient_audit.json', records + pending)
            return uploaded, result_audit

        def real_aggregate(method, uploaded, counts, adapter, comp):
            step, info = aggregate(method, uploaded, counts, adapter, comp)
            require(len(pending) == 20 and torch.equal(current['point'], worker.parameters_vector(current['model'])),
                    'Incomplete/mutated same-point gradient cohort')
            n = torch.as_tensor(counts, dtype=torch.float64); weights = n / n.sum()
            if method == 'Fed-NGA-gradient':
                x = uploaded.double(); norms = torch.linalg.vector_norm(x, dim=1, keepdim=True)
                expected = (-adapter['server_eta'] * (weights[:, None] * x / torch.where(norms > 0, norms, torch.ones_like(norms))).sum(0)).to(step)
                require(torch.equal(step, expected), 'Actual Eq9 aggregation oracle differs')
            else:
                weights = torch.as_tensor(info['normalized_final_weights'], dtype=torch.float64)
                center = (-step / adapter['server_eta']).double()
                residual = center[None] - uploaded.double(); distances = torch.linalg.vector_norm(residual, dim=1)
                thresholds = torch.as_tensor(info['fixed_thresholds'], dtype=torch.float64)
                raw_weights = n / n.sum() * torch.minimum(torch.ones_like(thresholds), thresholds / torch.where(distances > 0, distances, thresholds))
                # Solver center is cast to FP32 before applying eta; allow only that declared rounding error.
                error = torch.linalg.vector_norm((raw_weights[:, None] * residual).sum(0))
                rounding_bound = 32 * torch.finfo(step.dtype).eps * max(1., float(torch.linalg.vector_norm(center)))
                require(float(error) <= adapter['tolerance'] + rounding_bound, 'Actual fixed-Ti stationarity oracle failed')
                info['pilot_fp32_stationarity_l2'] = float(error); info['pilot_fp32_rounding_bound'] = rounding_bound
            info.update(pilot_aggregation_oracle_pass=True, pilot_effective_weights=weights.tolist(), pilot_root_adam_calls=root_calls[0])
            require(root_calls[0] == int(job['attack'] == 'S-DFA'), 'Undeclared root optimizer call')
            records.extend(pending); pending.clear(); root_calls[0] = 0
            return step, info

        core.load_bundle = retained_bundle; core.train_server_update = root_adam
        for name in ('train_local_model', 'evaluate_state_on_server', 'evaluate_model_calibrated'):
            setattr(core, name, forbidden)
        components.empirical_gradient = real_gradient; worker.attack_gradient = real_attack; worker.aggregate_gradients = real_aggregate
        config = core.ExperimentConfig(**job['config']); core.set_seed(91001, deterministic_image=True)
        def progress(item):
            write(out / 'progress.json', dict(item, job_id=job['id'], pid=os.getpid(), elapsed_seconds=time.time() - started,
                updated_unix=time.time(), resource=snapshot()))
            print(json.dumps(dict(id=job['id'], round=item['round'], elapsed_seconds=time.time() - started)), flush=True)
        result = worker.train_pipeline(core, components, job['method'], job['distribution'], job['attack'], config,
            job['adapter'], torch.device('cpu'), progress_callback=progress, checkpoint_path=out / 'model.pt',
            diagnostics_callback=lambda rows: worker.write_json(out / 'diagnostics.json', rows))
        require(len(records) == 60 and not pending and len(bundle_box) == 1, 'Incomplete real gradient pipeline')
        result.update(status='exploratory_gate_complete', evidence_stage=STAGE, scientific_table_records=0, revision_job=job)
        worker.write_json(out / 'result.json', result)
        bundle = bundle_box[0]; state = torch.load(out / 'model.pt', map_location='cpu', weights_only=True)
        model = core.make_model(bundle, config, torch.device('cpu')); model.load_state_dict(state)
        replay = core.evaluate_for_reporting(job['method'], model, bundle, config)
        require({k: replay[k] for k in core.METRICS} == result['metrics'], 'Terminal checkpoint native repredict differs')
        payload = dict(metrics={k: replay[k] for k in core.METRICS}, checkpoint_tensor_sha256=tensor_sha(state),
                       prediction_count=replay['prediction_count'])
        for prefix, labels, sensitive in (('evaluation', bundle['y_test'].numpy(), np.asarray(bundle['test_sensitive'])),
                                          ('root', bundle['server_y'].numpy(), np.asarray(bundle['server_sensitive']))):
            payload[prefix + '_group_label_counts'] = {f's{s}_y{y}': int(((sensitive == s) & (labels == y)).sum())
                                                       for s in (0, 1) for y in (0, 1)}
            payload[prefix + '_image_ids_sha256'] = bundle['image_data_contract'][prefix + '_image_ids_sha256']
        write(out / 'native_replay.json', payload); write(out / 'resource_after.json', snapshot())
        write(out / 'identity_after.json', hashes(REPO, scope['protected_source_hashes']))
        hashes(HERE, scope['local_hashes'])
        write(out / 'acceptance.json', dict(status='PASS', scope_sha256=digest(HERE / 'scope.json'),
            job_sha256=entry['job_sha256'], dispatch_receipt=receipt, checkpoint_tensor_sha256=tensor_sha(state),
            artifact_hashes={name: digest(out / name) for name in sorted(ARTIFACTS)}, elapsed_seconds=time.time() - started,
            limitation='Full-image exploratory CPU three-round gate only; formal five decisions unresolved; no test/GPU/70-round result'))
        require(checked(entry, scope, receipt) is not None, 'Pilot acceptance failed')
    except BaseException as error:
        write(out / 'failure.json', dict(error=repr(error), traceback=traceback.format_exc(), at_unix=time.time(),
            solver_diagnostics=getattr(error, 'solver_diagnostics', None), retained_client_gradient_count=len(records) + len(pending)))
        raise
    finally:
        for name, value in original.items():
            setattr(core, name, value)
        components.empirical_gradient = empirical; worker.attack_gradient = attack; worker.aggregate_gradients = aggregate
        bundle_box.clear()


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('action', choices=['inspect', 'run', 'summarize'])
    parser.add_argument('--dispatch-receipt', type=Path); args = parser.parse_args()
    scope = read(HERE / 'scope.json'); validate_scope(scope)
    if args.action == 'inspect':
        print(json.dumps(dict(status=scope['status'], jobs=4, unresolved_formal_decisions=5, execution_started=False))); return
    receipt = dispatch(scope, args.dispatch_receipt)
    require(sys.platform == 'linux' and str(HERE) == '/workspace/guardfed_checks/celeba_gradient_realimage_gate_20261009', 'Wrong server gate directory')
    require(digest('/etc/vast-agents-guide.md') == scope['guide_sha256'], 'Server guide changed')
    hashes(REPO, scope['protected_source_hashes'])
    require(set(receipt['exclusive_cpu_ids']).issubset(os.sched_getaffinity(0)), 'CPU allocation unavailable')
    os.environ['CUDA_VISIBLE_DEVICES'] = ''; os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        os.environ[name] = '8'
    import fcntl
    lock = (HERE / 'cpu_gate.lock').open('a'); fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    os.sched_setaffinity(0, receipt['exclusive_cpu_ids'])
    import torch
    torch.set_num_threads(8); torch.set_num_interop_threads(1)
    require(torch.__version__ == '2.11.0+cu128' and torch.version.cuda == '12.8', 'Unexpected isolated environment')
    worker = load_worker(); core = worker.load_core(REPO); components = worker.load_components()
    try:
        worker.validate_job(read(HERE / 'sealed_formal_job.json'), read(SNAPSHOT / 'protocol.json'))
    except ValueError as error:
        require('not frozen' in str(error), 'Formal freeze gate failed for unrelated identity reason')
    else:
        raise ValueError('Original formal64 draft must still be non-executable')
    if args.action == 'summarize':
        accepted = [checked(entry, scope, receipt) for entry in scope['jobs']]
        require(all(x is not None for x in accepted), 'Not all four image pilots accepted')
        print(json.dumps(dict(status='PASS', real_image_runs=4, rounds=12, scientific_table_records=0))); return
    require(not (HERE / 'gate_failure.json').exists(), 'Preserved gate failure blocks blind retry')
    before = snapshot(); write(HERE / 'protected_before.json', before)
    try:
        for entry in scope['jobs']:
            run_one(entry, scope, receipt, worker, core, components)
        after = snapshot(); write(HERE / 'protected_after.json', after)
        require(not after['failed'] and formal_growth(before, after), 'Protected formal800 queue failed or did not advance')
        hashes(REPO, scope['protected_source_hashes']); validate_scope(scope)
        write(HERE / 'gate_summary.json', dict(status='PASS', real_image_runs=4, actual_rounds=12, scientific_table_records=0,
            evidence_stage=STAGE, formal_decisions_unresolved=5, formal_gpu_progress_grew=True, test_evaluated=False,
            scope_sha256=digest(HERE / 'scope.json'), completed_unix=time.time()))
    except BaseException as error:
        write(HERE / 'gate_failure.json', dict(error=repr(error), traceback=traceback.format_exc(), at_unix=time.time()))
        raise


if __name__ == '__main__':
    main()
