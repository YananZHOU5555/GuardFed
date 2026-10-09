def accept(args):
    v2.require(sys.platform == 'linux', 'Strict valid acceptance uses the original Linux input paths')
    bind_cpu(0)
    records, bindings = read_inputs(args)
    indexed = {r['id']: r for r in records}
    batch = v2.read(args.batch / 'batch_inputs.json')
    execution = v2.read(args.batch / 'batch_execution.json')
    v2.require(batch['scope'] == SCOPE and batch['v3_source_sha256'] == v2.digest(__file__) and execution['source_unchanged'], 'Batch source identity failed')
    v2.require(batch['storage_map_sha256'] == args.storage_map_sha256 and batch['restore_acceptance_sha256'] == args.restore_acceptance_sha256, 'Batch restore binding differs')
    selected = batch['selected_ids']
    v2.require(len(selected) == len(set(selected)) and set(selected) <= set(indexed), 'Duplicate or invalid accepted cohort')
    v2.require(execution['requested_ids'] == selected and len(execution['finished_zero_exit_ids']) == len(set(execution['finished_zero_exit_ids'])), 'Execution model identity drift')
    check_source_tokens(batch['source_before'])
    v2.require(full_hashes(source_pins(args.repo.resolve(), records, args)) == batch['source_before'] == execution['source_after'], 'Batch shared source bytes changed')
    core = v2.load('v3_accept_core', args.repo / 'scripts/reproduce_paper_tables.py')
    original = v2.load('v3_accept_original', args.repo / 'scripts/run_revision_ablation.py')
    evaluator = v2.load('v3_accept_evaluator', BASE / 'inputs/evaluator.py')
    ids, y, s, _ = v2.metadata(args.repo)
    accepted, invalid = [], []
    for model_id in selected:
        try:
            record, path = indexed[model_id], args.batch / 'runs' / model_id
            proof = v2.read(path.parent / (model_id + '.worker.json'))
            r = v2.read(path / 'receipt.json')
            v2.require(proof['status'] == r['status'] == 'DIAGNOSTIC_NATIVE_MATCH' and proof['artifacts_unchanged'], 'Worker receipt failed or incomplete')
            v2.require(proof['id'] == r['id'] == model_id and r['runtime']['device'] == 'cuda:0' and r['runtime']['cuda_device_count'] == 1, 'Worker/model/device identity changed')
            gpu_receipt_guard(proof, r)
            v2.require(not r['optimizer_created'] and not r['gradients_created'] and not r['test_labels_accessed'], 'Inference-only label/optimizer contract failed')
            for resources in (r['before_resources'], r['after_resources']):
                v2.require(all(cpus == proof['allowed_cpus'] for cpus in resources['thread_cpu_affinities'].values()), 'Worker escaped its eight-CPU slot')
            v2.require(proof['batch_receipt_sha256'] == v2.digest(args.batch / 'batch_inputs.json') and proof['storage_map_sha256'] == args.storage_map_sha256, 'Worker belongs to another batch/map')
            v2.require(proof['receipt_sha256'] == v2.digest(path / 'receipt.json') and r['model_inventory_record_sha256'] == v2.canonical(record), 'Worker receipt or original record identity changed')
            paths = mapping_paths(record, bindings)
            v2.require(full_hashes({paths[k]: record[k]['sha256'] for k in KINDS}) == proof['artifact_before'] == proof['artifact_after'], 'Accepted mapped artifact changed')
            validate, _ = mapped_functions(record, paths, original)
            original_result = validate(original, record, args.repo)
            cfg = core.ExperimentConfig(**record['config'])
            root_ids, root_y, root_s, root_receipt = v2.rebuild_root(core, cfg, record, ids, y, s)
            v2.require(root_receipt == r['root_reconstruction'] and r['weights_before'] == r['weights_after'], 'Root identity or model weights drift')
            with v2.np.load(path / 'validation_predictions.npz', allow_pickle=False) as z:
                v2.require(v2.digest(path / 'validation_predictions.npz') == r['prediction_arrays_sha256'], 'Prediction array SHA changed')
                v2.require(v2.np.array_equal(z['root_image_ids'], root_ids) and v2.np.array_equal(z['valid_image_ids'], ids[162770:182637]), 'Root/valid sample order drift')
                fits = evaluator.fit_views(core, record['method'], z['root_margins'], root_y, root_s, cfg, v2.VIEWS, evaluator.SHARED_CALIBRATION)
                v2.require(v2.canonical(fits) == v2.canonical(r['fits']), 'Root-only threshold fit changed')
                predictions = evaluator.predict_views(z['valid_margins'], s[162770:], fits)
                v2.require(all(v2.np.array_equal(predictions[v], z['prediction_' + v]) for v in v2.VIEWS), 'Saved predictions contradict frozen margin/tie rules')
                scored = evaluator.evaluate_frozen_predictions(predictions, y[162770:], s[162770:])
                comparison = v2.check_native(scored['native'], original_result['metrics'])
                v2.require(scored == r['views'] and comparison == r['native_comparison'] and comparison['accepted'], 'Recomputed common-checkpoint metrics failed')
            accepted.append(model_id)
        except Exception as exc:
            invalid.append({'id': model_id, 'error_type': type(exc).__name__, 'error': str(exc)})
    complete = len(accepted) == len(selected) and not invalid and not execution['failures']
    report = {'scope': SCOPE, 'status': 'SELECTED_VALID_REPLAY_ACCEPTED' if complete else 'PARTIAL_OR_INVALID_VALID_REPLAY',
              'requested_n': len(selected), 'accepted_n': len(accepted), 'accepted_ids': accepted, 'invalid': invalid,
              'all900_native_valid_replayed': complete and len(accepted) == 900, 'max_abs_native_metric_difference': None if not accepted else max(v2.read(args.batch / 'runs' / i / 'receipt.json')['native_comparison']['max_abs_difference'] for i in accepted),
              'wall_seconds': execution['wall_seconds'], 'models_per_second': len(accepted) / execution['wall_seconds'],
              'workers': execution['workers'], 'test_labels_accessed': False, 'final_protocol_status': 'PREPARED_NOT_FROZEN',
              'inventory_sha256': v2.INVENTORY_SHA, 'valid_image_ids_sha256': v2.VALID_IDS_SHA,
              'calibration_core_sha256': v2.CORE_SHA, 'v2_source_sha256': V2_SHA,
              'storage_map_sha256': args.storage_map_sha256, 'batch_path': str(args.batch.resolve()),
              'batch_inputs_sha256': v2.digest(args.batch / 'batch_inputs.json')}
    v2.require(not args.output.exists(), 'Preserve existing acceptance, including invalid results')
    v2.save(args.output, report)
    print(json.dumps({k: report[k] for k in ('status', 'accepted_n', 'requested_n', 'all900_native_valid_replayed')}))
    return 0 if complete else 1
