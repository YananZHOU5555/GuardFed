def replay_one(core, original, cnn, evaluator, record, repo, ids, y, s, output, wall_limit):
    started, usage_start = time.monotonic(), resource.getrusage(resource.RUSAGE_SELF)
    signal.alarm(wall_limit)
    result = validate_original(original, record, repo)
    cfg = core.ExperimentConfig(**record['config'])  # original config bytes remain unchanged, including device string
    core.set_seed(cfg.seed, deterministic_image=True)
    root_ids, root_y, root_sensitive, root_receipt = rebuild_root(core, cfg, record, ids, y, s)
    model = cnn.CelebACNN(seed=cfg.seed).to('cuda:0')
    state = torch.load(inside(repo, record['checkpoint']['member']), map_location='cpu', weights_only=True)
    require(state and all(isinstance(v, torch.Tensor) and torch.isfinite(v).all().item() for v in state.values()), 'Invalid/nonfinite checkpoint tensors')
    model.load_state_dict(state, strict=True)
    model.eval()
    weights_before = evaluator.weights_identity(model)
    image_map = np.load(repo / 'data/celeba/derived/rgb64_v1/images.npy', mmap_mode='r', allow_pickle=False)
    require(image_map.shape == (202599, 3, 64, 64) and image_map.dtype == np.uint8, 'RGB64 image cache shape/type drift')
    root_X = torch.from_numpy(np.array(image_map[root_ids - 1], copy=True))
    valid_X = torch.from_numpy(np.array(image_map[162770:182637], copy=True))
    before = live_snapshot(repo)
    resource_gate(before)
    root_margins, valid_margins, fits, predictions = evaluator.extract_and_predict(
        core, model, {'server_X': root_X, 'server_y': torch.from_numpy(root_y), 'server_sensitive': root_sensitive},
        valid_X, s[162770:], record['method'], cfg, VIEWS, evaluator.SHARED_CALIBRATION)
    # Target labels first enter the scoring interface after fits and predictions are fixed.
    metrics = evaluator.evaluate_frozen_predictions(predictions, y[162770:], s[162770:])
    comparison = check_native(metrics['native'], result['metrics'])
    require(all(p.grad is None for p in model.parameters()), 'Inference created parameter gradients')
    weights_after = evaluator.weights_identity(model)
    require(weights_after == weights_before, 'Inference changed model tensor bytes')
    after = live_snapshot(repo)
    usage_end = resource.getrusage(resource.RUSAGE_SELF)
    output.mkdir(exist_ok=False)
    np.savez_compressed(output / 'validation_predictions.npz', root_image_ids=root_ids,
                        valid_image_ids=ids[162770:182637], root_margins=root_margins, valid_margins=valid_margins,
                        **{'prediction_' + k: v for k, v in predictions.items()})
    receipt = {
        'scope': 'SINGLE_MODEL_GPU_DIAGNOSTIC_NOT_ACCEPTED_COHORT', 'status': 'DIAGNOSTIC_NATIVE_MATCH' if comparison['accepted'] else 'DIAGNOSTIC_NATIVE_MISMATCH',
        'id': record['id'], 'method': record['method'], 'distribution': record['distribution'], 'attack': record['attack'], 'seed': record['seed'],
        'model_inventory_record_sha256': canonical(record), 'checkpoint_sha256': record['checkpoint']['sha256'],
        'original_result_sha256': record['result']['sha256'], 'original_job_sha256': record['raw_job']['sha256'],
        'config_canonical_sha256': record['config_canonical_sha256'], 'original_training_torch': record['training_torch'],
        'runtime': {'python': sys.executable, 'torch': torch.__version__, 'device': 'cuda:0', 'original_config_device': cfg.device,
                    'torch_threads': torch.get_num_threads(), 'interop_threads': torch.get_num_interop_threads(), 'loader_workers': 0,
                    'nice': os.getpriority(os.PRIO_PROCESS, 0), 'cuda_device_count': torch.cuda.device_count(), 'cpu': platform.processor()},
        'root_reconstruction': root_receipt, 'valid_n': len(valid_X), 'valid_image_ids_sha256': array_sha(ids[162770:182637]),
        'fits': fits, 'views': metrics, 'native_comparison': comparison,
        'prediction_arrays_sha256': digest(output / 'validation_predictions.npz'),
        'weights_before': weights_before, 'weights_after': weights_after, 'optimizer_created': False, 'gradients_created': False,
        'zero_margin_count': int(np.sum(valid_margins == 0)),
        'elapsed_seconds': time.monotonic() - started,
        'cpu_user_seconds': usage_end.ru_utime - usage_start.ru_utime, 'cpu_system_seconds': usage_end.ru_stime - usage_start.ru_stime,
        'peak_rss_kib': usage_end.ru_maxrss, 'before_resources': before, 'after_resources': after,
        'test_labels_accessed': False, 'test_inference_performed': False, 'final_dispatch_created': False,
        'claim_limit': 'Single-model CUDA diagnostic only; never authorizes cohort inclusion, CPU failure invalidation, training, test or queue restart',
    }
    receipt['effective_cpu_cores'] = (receipt['cpu_user_seconds'] + receipt['cpu_system_seconds']) / receipt['elapsed_seconds']
    save(output / 'receipt.json', receipt)
    signal.alarm(0)
    resource_gate(after)
    require(comparison['accepted'], 'Native metrics exceed fixed tolerance; preserve evidence and stop without retry')
    return receipt
