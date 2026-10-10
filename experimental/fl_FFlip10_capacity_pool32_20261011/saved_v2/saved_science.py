def check_saved(record, path, r, proof, args, bindings, core, original, evaluator, ids, y, s):
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
    return comparison
