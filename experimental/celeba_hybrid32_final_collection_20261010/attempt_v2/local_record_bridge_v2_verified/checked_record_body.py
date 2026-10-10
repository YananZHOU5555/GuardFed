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
    assert prov['torch'] == server_runtime['torch'] == '2.11.0+cu128'
    assert prov['cuda_build'] == server_runtime['cuda_build'] == '12.8' and prov['device'] == 'cuda:0' and prov['cpu_threads'] == 1
    assert prov['cuda_device_count'] == 1 and prov['cuda_visible_devices'] == scope['runtime_cuda_visible_device']
    assert prov['gpu_uuid'] == scope['runtime_gpu_uuid'] and prov['gpu_name'] == server_runtime['gpu_name_for_original_checker']
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
