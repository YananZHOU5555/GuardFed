"""Read-only original strict100 review and fixed-recipe descriptive tables; never fit."""
import argparse, ast, hashlib, importlib.util, json, math, os, statistics, sys
from pathlib import Path

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SOURCE = ROOT / 'tmp/celeba_logofair_fullcoverage_20261010'
SCREEN = ROOT / 'tmp/celeba_logofair_screen32_20261010'
ADOPTION = ROOT / 'tmp/celeba_logofair32_root_adoption_20261010/ROOT_ADOPTION.json'
METRICS = ('accuracy', 'aeod', 'aspd')
SEEDS = list(range(91001, 91011))
SCENES = [(d, a) for d in ('IID', 'non-IID') for a in ('Benign', 'F Flip', 'FedSA', 'S-DFA', 'Sp-DFA')]
PANELS = {'ten_seed': SEEDS, 'exclude_selection': SEEDS[1:], 'matching_six': SEEDS[4:]}
TOLERANCE = 1e-12


def read(path): return json.loads(Path(path).read_bytes())
def need(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(1024 ** 2), b''): h.update(b)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def pinned(path, wanted):
    need(isinstance(wanted, str) and len(wanted) == 64 and sha(path) == wanted, 'External SHA drift: ' + str(path))
    return read(path)


def f_read_path(path):
    path = Path(path).resolve()
    need(path.drive.upper() == 'F:', 'Existing bulk inputs must stay on F')
    return path


def sources():
    need(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'), 'Optimized Python forbidden')
    for path, wanted in read(HERE / 'INPUT_PINS.json').items(): need(sha(ROOT / path) == wanted, 'Frozen dependency drift: ' + path)
    sys.path.insert(0, str(SOURCE)); sys.path.insert(0, str(SCREEN))
    metadata = load('logofair100_original_metadata', SOURCE / 'metadata.py')
    metadata.verify_sources()
    return metadata


def grid(manifest, index, inventory, candidate):
    expected = {candidate['id'] + '_' + r['cell_id']: r for r in inventory}
    need(len(inventory) == len(expected) == 100 and {(r['distribution'], r['attack'], r['seed']) for r in inventory}
         == {(d, a, s) for d, a in SCENES for s in SEEDS}, 'Exact100 scientific cells required')
    old = {i for i, r in expected.items() if r['seed'] == 91001 and r['attack'] in ('Benign', 'S-DFA')}
    jobs, reused = manifest['jobs'], manifest['reused_jobs']
    need(len(jobs) == 96 and len(reused) == 4 and len({r['id'] for r in jobs + reused}) == 100, 'Exact96+4 unique partition required')
    need({r['id'] for r in jobs} == set(expected) - old and {r['id'] for r in reused} == old, 'Wrong new/reused scientific cells')
    need(all(r['cell_id'] == expected[r['id']]['cell_id'] for r in jobs + reused + index['records']), 'Cell-ID alias drift')
    need(manifest['candidate'] == candidate and manifest['scientific_stage'] == 'fixed_recipe_validation_postprocessing100'
         and manifest['new_CNN'] == 0 and manifest['final_test'] is False, 'Fixed adopted recipe/stage required')
    need(index['status'] == 'LOCAL_STRICT96_PLUS4_ROOT_REVIEW_PENDING' and index['reused_jobs'] == reused
         and [r['id'] for r in index['records']] == [r['id'] for r in jobs], 'Complete ordered original strict96+4 index required')
    return expected


def record_identity(result, job, row, candidate):
    need(job['candidate'] == candidate['id'] == 'LoGoFair-DP_07' and job['settings'] == candidate['settings'] and job['baseline_id'] == row['id'] and job['seed'] == row['seed']
         and job['fit_seed'] == 1719 and job['evaluation_split'] == 'valid' and job['settings']['post_rounds'] == 30,
         'Fixed recipe07/seed/fitseed1719/valid/30 postrounds required')
    need(result['job'] == job and result['status'] == 'complete' and result['settings'] == job['settings']
         and result['fit_seed'] == 1719 and [x['round'] for x in result['history']] == list(range(1, 31)), 'Incomplete/mixed postprocessing state')
    for name, key in [('checkpoint_sha256', 'checkpoint'), ('original_result_sha256', 'result'), ('accepted_margin_cache_sha256', 'cache')]:
        need(result[name] == row[key]['sha256'], 'Original accepted checkpoint/result/cache drift')
    need(result['root_image_ids_sha256'] == row['root_image_ids_sha256'] and result['valid_image_ids_sha256'] == row['valid_image_ids_sha256']
         and result['mapping_sha256'] == job['mapping_sha256'] and result['mapping_metadata_sha256'] == job['mapping_metadata_sha256'], 'Root/valid/mapping identity drift')
    need(all(math.isfinite(result['metrics'][k]) and 0 <= result['metrics'][k] <= 1 for k in METRICS), 'Nonfinite/out-of-range result')


def original_statistic():
    path = ROOT / 'tmp/celeba_mechanism_evidence_20261009/evidence_v4.py'
    tree = ast.parse(path.read_text(encoding='utf8'))
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'statistic')
    ns = {'statistics': statistics, 'require': need, 'METRICS': ('accuracy_pct', 'aeod', 'aspd')}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), 'exec'), ns)
    return ns['statistic']


def saved_metric_error(computed, reported):
    need(all(math.isfinite(x[k]) for x in (computed, reported) for k in METRICS), 'Nonfinite saved/reported metric')
    error = max(abs(computed[k] - reported[k]) for k in METRICS)
    need(error <= TOLERANCE, 'Saved prediction metrics differ; never change tolerance')
    return error


def describe(records):
    need(len(records) == 100 and len({(r['distribution'], r['attack'], r['seed']) for r in records}) == 100
         and {(r['distribution'], r['attack'], r['seed']) for r in records} == {(d, a, s) for d, a in SCENES for s in SEEDS}, 'No partial/duplicate summary')
    need(all(math.isfinite(r['metrics'][k]) and 0 <= r['metrics'][k] <= 1 for r in records for k in METRICS), 'Nonfinite/out-of-range summary')
    statistic = original_statistic(); panels = {}
    values = [dict(r, accuracy_pct=100 * r['metrics']['accuracy'], aeod=r['metrics']['aeod'], aspd=r['metrics']['aspd']) for r in records]
    for name, seeds in PANELS.items():
        scenes = [dict(distribution=d, attack=a, **statistic([r for r in values if (r['distribution'], r['attack']) == (d, a) and r['seed'] in seeds], len(seeds))) for d, a in SCENES]
        # Same statistics.mean operation as original evidence_v4.summarize, inside each seed first.
        seed_first = [dict(seed=s, n_scenarios=10, **{k: statistics.mean(r[k] for r in values if r['seed'] == s) for k in ('accuracy_pct', 'aeod', 'aspd')}) for s in seeds]
        panels[name] = dict(per_scene=scenes, cross_scene_per_seed=seed_first, cross_scene=statistic(seed_first, len(seeds)))
    return dict(panels=panels, units={'accuracy_pct': 'percent', 'aeod': 'absolute TPR gap', 'aspd': 'absolute positive-rate gap'},
                sample_SD='sample SD, ddof1', cross_scene_rule='Mean over all10 scenes within each model seed, then mean/sampleSD across retained seeds',
                fit_seed=1719, model_seed_panels=PANELS, constant_prediction_ids=[r['id'] for r in records if r['constant_prediction'] is not None],
                all_negative_and_constant_records_retained=True, recipe_search_or_score_ranking_performed=False)


def run(args):
    pinned(HERE / 'FILES_SHA256.json', args.source_sha)
    for name, pin in read(HERE / 'FILES_SHA256.json')['files'].items(): need(sha(HERE / name) == pin['sha256'], 'Summary source member drift')
    metadata = sources(); stage, _ = metadata.bulk_path(args.stage, 0); index_path, _ = metadata.bulk_path(args.index, 0)
    seal = pinned(stage / 'SOURCE_SHA256.json', args.stage_sha); manifest = pinned(stage / 'manifest.json', args.manifest_sha)
    for name, wanted in seal.items(): need(sha(stage / name) == wanted, 'Actual stage source/job drift')
    need(sha(stage / 'snapshot/logofair_bridge_20261010/bridge.py') == seal['snapshot/logofair_bridge_20261010/bridge.py'], 'Bound bridge identity drift')
    need((stage / 'snapshot/logofair_bridge_20261010/bridge.py').read_text(encoding='utf8') == metadata.bridge_source(), 'Only approved original bridge seed predicate may change')
    adoption = read(ADOPTION); candidate = adoption['selected_recipe']
    need(adoption['status'] == 'ROOT_LOGOFAIR_SCREEN32_ADOPTED' and adoption['accepted_count'] == 32 and candidate['id'] == 'LoGoFair-DP_07'
         and manifest['adoption_sha256'] == sha(ADOPTION) and manifest['summary_sha256'] == adoption['summary_sha256'], 'Original root-adopted recipe07 lineage drift')
    inputs = pinned(args.inputs, args.inputs_sha)
    need(inputs['status'] == 'ROOT_ACCEPTED_EXISTING_FEDAVG100_AND_FIXED_COHORT_MAPPINGS' and manifest['inputs_sha256'] == args.inputs_sha
         and inputs['cache_identity_sha256'] == sha(SOURCE / 'CACHE_IDENTITIES100.json'), 'Actual root-bound inputs required')
    index = pinned(index_path, args.index_sha); need(index['manifest_sha256'] == args.manifest_sha, 'Original index/manifest drift')
    need(not (index_path.parent / 'QUEUE_FAILURE.json').exists(), 'Preserved queue failure blocks complete100 review')
    inventory = read(SOURCE / 'CACHE_IDENTITIES100.json')['references']; expected = grid(manifest, index, inventory, candidate)
    locations = {r['id']: r for r in inputs['references']}; need(len(locations) == 100 and set(locations) == {r['id'] for r in inventory}, 'All100 actual references required')
    out, storage = metadata.bulk_path(args.out, 16 * 1024 ** 2)
    need(not out.exists() and not out.is_relative_to(stage) and not stage.is_relative_to(out)
         and not out.is_relative_to(index_path.parent), 'Fresh separate F summary output only; no overwrite/retry')
    out.mkdir(parents=True)
    try:
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '1'
        old_bridge = load('logofair100_original_screen_checker', SCREEN / 'snapshot/logofair_bridge_20261010/bridge.py')
        new_bridge = load('logofair100_bound_original_checker', stage / 'snapshot/logofair_bridge_20261010/bridge.py')
        old_bridge.torch.set_num_threads(1)
        if old_bridge.torch.get_num_interop_threads() != 1: old_bridge.torch.set_num_interop_threads(1)
        import netcal
        need(netcal.__version__ == '1.3.6', 'Use original netcal1.3.6 runtime')
        core = new_bridge.load_core(args.repo)
        old_jobs = {r['id']: r for r in read(SCREEN / 'jobs/manifest.json')['jobs']}
        rows = {r['id']: r for r in index['records'] + index['reused_jobs']}; records = []; artifacts = {str(stage/name):wanted for name,wanted in seal.items()}; worst = 0.0
        for entry in manifest['jobs'] + manifest['reused_jobs']:
            identity = entry['id']; row = expected[identity]; pin = rows[identity]; reused = identity in old_jobs
            job_path = SCREEN / 'jobs' / old_jobs[identity]['job'] if reused else stage / entry['job']; job = read(job_path)
            need(sha(job_path) == (old_jobs[identity] if reused else entry)['job_sha256'], 'Original job SHA drift')
            artifacts[str(job_path)] = (old_jobs[identity] if reused else entry)['job_sha256']
            folder = f_read_path(Path(pin['result']).parent)
            if not reused: need(folder == index_path.parent / identity, 'Foreign new output namespace')
            need(Path(pin['acceptance']).resolve() == folder / 'acceptance.json' and Path(pin['result']).resolve() == folder / 'result.json', 'Mixed result/acceptance directories')
            pinned(pin['result'], pin['result_sha256']); acceptance = pinned(pin['acceptance'], pin['acceptance_sha256'])
            reference = f_read_path(locations[row['id']]['path'])
            need(sha(reference / 'source_job.json') == row['source_job']['sha256'], 'Frozen raw-job bytes drift')
            if not reused:
                need(Path(entry['reference']).resolve() == reference and entry['mapping'] == inputs['mappings'][str(row['seed'])], 'Bound reference/mapping path drift')
            mapping = inputs['mappings'][str(row['seed'])]
            need(job['mapping_sha256'] == mapping['sha256'] and job['mapping_metadata_sha256'] == mapping['metadata_sha256'], 'Actual approved mapping/job drift')
            artifacts[str(f_read_path(mapping['path']))] = mapping['sha256']; artifacts[str(f_read_path(mapping['metadata']))] = mapping['metadata_sha256']
            result = (old_bridge if reused else new_bridge).checked_output(job_path, folder, core, reference)
            need(result is not None, 'Original checker returned partial'); record_identity(result, job, row, candidate)
            native_job = read(reference / 'result.json')['revision_job']
            for name, wanted in acceptance['artifact_hashes'].items(): artifacts[str(folder / name)] = wanted
            artifacts[str(reference / 'source_job.json')] = row['source_job']['sha256']
            for name, key in [('model.pt', 'checkpoint'), ('result.json', 'result'), ('margins.npz', 'cache')]: artifacts[str(reference / name)] = row[key]['sha256']
            # An additional independent read of saved decisions; no fit, threshold change, or CNN.
            with old_bridge.np.load(folder / 'scores_predictions.npz', allow_pickle=False) as cache:
                prediction = cache['prediction']; computed = core.compute_metrics(cache['valid_y'], prediction, cache['valid_sensitive'])
                need(len(prediction) == 19867 and set(old_bridge.np.unique(prediction)) <= {0, 1}, 'Invalid saved decisions')
                constant = int(prediction[0]) if old_bridge.np.unique(prediction).size == 1 else None
            error = saved_metric_error(computed, result['metrics']); worst = max(worst, error)
            records.append(dict(id=identity, baseline_id=row['id'], distribution=row['distribution'], attack=row['attack'], seed=row['seed'], fit_seed=1719,
                                checkpoint_sha256=result['checkpoint_sha256'], cache_sha256=result['accepted_margin_cache_sha256'], mapping_sha256=result['mapping_sha256'],
                                metrics=result['metrics'], constant_prediction=constant, reused_screen_record=reused, environment=result['environment'],
                                pretrained_source_runtime={k:native_job.get(k) for k in ('python_version','torch_version','visible_gpu','cpu_threads')},
                                result_path=pin['result'], result_sha256=pin['result_sha256'], acceptance_path=pin['acceptance'], acceptance_sha256=pin['acceptance_sha256']))
        need(all(sha(p) == wanted for p, wanted in artifacts.items()), 'Artifact changed during read-only review')
        pinned(stage / 'SOURCE_SHA256.json', args.stage_sha); pinned(stage / 'manifest.json', args.manifest_sha)
        pinned(index_path, args.index_sha); pinned(args.inputs, args.inputs_sha); sources()
        summary = describe(records)
        metadata.write(out / 'records100.json', records); metadata.write(out / 'SUMMARY100.json', summary)
        lines = ['# LoGoFair fixed recipe07: validation100', '', 'ACC%; AEOD absolute TPR gap; ASPD absolute positive-rate gap. Mean ± sampleSD(ddof1). Model seeds vary; fitseed1719 is fixed. No new recipe ranking.', '']
        for panel, group in summary['panels'].items():
            lines += ['## ' + panel, '', '| Distribution | Scene | n | ACC% | AEOD | ASPD |', '|---|---|---:|---:|---:|---:|']
            for r in group['per_scene'] + [dict(distribution='Seed-first', attack='All10 scenes', **group['cross_scene'])]:
                cells = [f"{r[k]['mean']:.{3 if k=='accuracy_pct' else 5}f} ± {r[k]['sample_sd_ddof1']:.{3 if k=='accuracy_pct' else 5}f}" for k in ('accuracy_pct', 'aeod', 'aspd')]
                lines.append('| ' + ' | '.join([r['distribution'], r['attack'], str(r['n']), *cells]) + ' |')
            lines.append('')
        lines += ['All negative/constant predictions are retained. Twenty image-ID virtual cohorts are not true training clients. Calibrated DP adaptation does not establish EO or aggregation-only effects.', '', 'Seed91001 participated in validation selection; other validation seeds were historically observed. Accepted FedAvg sources have mixed runtime/device histories; sigmoid of accepted float32 margins may differ from fresh softmax. Historical test attribute/split metadata exposure remains; this is valid-only, not final test. Primary manuscript endpoint and final claims remain author decisions.']
        (out / 'TABLES.md').write_text('\n'.join(lines) + '\n', encoding='utf8')
        metadata.write(out / 'ACCEPTANCE100.json', dict(status='ORIGINAL_STRICT100_AND_SAVED_PREDICTION_SUMMARY_PASS_ROOT_REVIEW_PENDING', accepted_n=100,
            original_new96=96, original_reused4=4, root_adopted=0, final_test=False, new_fits=0, new_CNN=0, source_seal_sha256=args.source_sha,
            stage_source_sha256=args.stage_sha, manifest_sha256=args.manifest_sha, index_sha256=args.index_sha, root_inputs_sha256=args.inputs_sha,
            screen32_root_adoption_sha256=sha(ADOPTION), saved_prediction_items=100*19867, saved_metric_checks=300, saved_metric_max_difference=worst,
            tolerance=TOLERANCE, artifact_hashes=artifacts, storage_preflight=storage, constant_prediction_ids=summary['constant_prediction_ids']))
    except BaseException as exc:
        import traceback
        metadata.write(out / 'SUMMARY_FAILURE.json', dict(error=repr(exc), traceback=traceback.format_exc(), automatic_retry=False)); raise


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source-sha', 'stage', 'stage-sha', 'manifest-sha', 'index', 'index-sha', 'inputs', 'inputs-sha', 'repo', 'out'): p.add_argument('--'+name, required=True)
    run(p.parse_args())
