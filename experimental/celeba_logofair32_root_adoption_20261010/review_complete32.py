"""Read-only complete32 review after the original summary action; never fit or run CNN."""
import argparse
import ast
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
SOURCE = ROOT / 'tmp/celeba_logofair_screen32_20261010'
META = ROOT / 'tmp/celeba_logofair_screen32_root_execution_20261010'
BULK = Path('F:/YananResearchStorage/GuardFed/logofair_screen32_20261010')
OUT = BULK / 'attempt001'
SOURCE_SHA = 'accd5cb8582a344f870188f1e70661b6dc9dc948cc88c9e6f6651e451607bc49'
REVIEW_SHA = '418868c324b3931ae590b590a72badaee240829a8c09d21476944827f88b27fe'


def read(path): return json.loads(Path(path).read_bytes())
def require(ok, message):
    if not ok: raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 ** 2), b''): h.update(block)
    return h.hexdigest()


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec); sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def gate():
    require(not sys.flags.optimize and not os.environ.get('PYTHONOPTIMIZE'), 'Assertions must remain enabled')
    require(sha(SOURCE / 'FILES_SHA256.json') == SOURCE_SHA, 'Frozen source seal drift')
    for name, expected in read(SOURCE / 'FILES_SHA256.json')['files'].items():
        require(sha(SOURCE / name) == expected, 'Source member drift: ' + name)
    require(sha(ROOT / 'tmp/celeba_logofair_screen32_independent_review_20261010/REVIEW.json') == REVIEW_SHA, 'Original source review drift')
    require((META / 'queue_EXIT.json').is_file(), 'WAIT: original queue_EXIT is absent; do not summarize partial results')
    require(read(META / 'queue_EXIT.json')['exit_code'] == 0 and not (OUT / 'QUEUE_FAILURE.json').exists(), 'Original queue did not close normally')
    index = read(OUT / 'STRICT32_INDEX.json')
    manifest = read(SOURCE / 'jobs/manifest.json')
    require(index['status'] == 'LOCAL_ORIGINAL_STRICT32_COMPLETE_ROOT_REVIEW_PENDING' and index['source_seal_sha256'] == SOURCE_SHA, 'Wrong strict index')
    require(len(index['records']) == len({r['id'] for r in index['records']}) == 32
            and [r['id'] for r in index['records']] == [j['id'] for j in manifest['jobs']], 'Exact ordered32 required')
    return index, manifest


def review(expected_summary_sha):
    index, manifest = gate()
    require(sha(OUT / 'SUMMARY32.json') == expected_summary_sha, 'Actual summary external SHA differs')
    require(read(META / 'summary_EXIT.json')['exit_code'] == 0, 'Original summary action must close normally')
    for key in ('CUDA_VISIBLE_DEVICES',): os.environ[key] = ''
    for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'NUMEXPR_NUM_THREADS'): os.environ[key] = '1'
    sys.path.insert(0, str(SOURCE))
    summary_module = load('logofair_original_summary', SOURCE / 'summarize.py')
    summary_module.verify_delivery()
    bridge_path = SOURCE / 'snapshot/logofair_bridge_20261010/bridge.py'
    bridge = load('logofair_original_closed_checker', bridge_path)
    bridge.torch.set_num_threads(1); bridge.torch.set_num_interop_threads(1)
    core = bridge.load_core(ROOT / 'tmp/revision-publish-20260928')
    refs = {r['id']: r for r in read(SOURCE / 'snapshot/logofair_bridge_20261010/reuse_manifest.json')['entries']}
    staged = read(BULK / 'inputs/INPUT_RECEIPT.json')
    require(staged['status'] == 'EXACT4_ACCEPTED_REFERENCES_AND_APPROVED_VIRTUAL_MAPPING_STAGED' and staged['source_seal_sha256'] == SOURCE_SHA, 'Actual staged identities differ')
    for name, expected in staged['files'].items(): require(sha(BULK / 'inputs' / name) == expected, 'Staged input drift')
    metadata = read(SOURCE / 'mapping_metadata.json')
    require(metadata['cohorts'] == 20 and metadata['semantics'] == 'declared_virtual_partition'
            and metadata['true_training_client_identity'] is False and metadata['input_fields_for_mapping'] == ['image_id'], 'Virtual population meaning changed')
    tree = ast.parse(bridge_path.read_text(encoding='utf8'))
    fit = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'fit_predict')
    calls = [n for n in ast.walk(fit) if isinstance(n, ast.Call) and ast.unparse(n.func) == 'post.fit']
    require(len(calls) == 1 and [ast.unparse(a) for a in calls[0].args] == ["probability['root_probability']", "labels['root_y']", "sensitive['root_sensitive']", "mapping['root_client_id']"], 'Root-only original fitting call changed')
    observed, pins, prediction_n = [], {}, 0
    for row, entry in zip(index['records'], manifest['jobs']):
        job_path = SOURCE / 'jobs' / entry['job']; job = read(job_path)
        require(sha(job_path) == entry['job_sha256'] and job['seed'] == 91001 and job['fit_seed'] == 1719 and job['settings']['post_rounds'] == 30, 'Frozen job/fit horizon differs')
        folder = OUT / row['id']
        require(Path(row['result']).resolve() == (folder / 'result.json').resolve()
                and Path(row['acceptance']).resolve() == (folder / 'acceptance.json').resolve(), 'Foreign output paths')
        require(sha(folder / 'result.json') == row['result_sha256'] and sha(folder / 'acceptance.json') == row['acceptance_sha256'], 'Index output SHA drift')
        for name, expected in read(folder / 'acceptance.json')['artifact_hashes'].items():
            require(sha(folder / name) == expected, 'Strict saved artifact changed'); pins[str(folder / name)] = expected
        result = bridge.checked_output(job_path, folder, core, BULK / 'inputs' / job['baseline_id'])
        require(result is not None, 'Original strict checker returned partial')
        source_job = refs[job['baseline_id']]['source_job']
        observed.append(dict(id=row['id'], candidate=job['candidate'], distribution=source_job['distribution'], attack=source_job['attack'],
            seed=job['seed'], fit_seed=job['fit_seed'], metrics=result['metrics'], checkpoint_sha256=result['checkpoint_sha256']))
        prediction_n += 19867
    summary = read(OUT / 'SUMMARY32.json')
    require(summary == dict(summary_module.summarize(observed), strict_index_sha256=sha(OUT / 'STRICT32_INDEX.json')), 'Summary differs from complete original frozen rule')
    independent = []
    for candidate in sorted({r['candidate'] for r in observed}):
        rows = [r for r in observed if r['candidate'] == candidate]
        averages = {k: math.fsum(r['metrics'][k] for r in rows) / 4 for k in ('accuracy', 'aeod', 'aspd')}
        scores = [summary_module.score(r['metrics']) for r in rows]
        independent.append(dict(candidate=candidate, **averages, score=math.fsum(scores) / 4))
    worst = max(abs(a[k] - b[k]) for a, b in zip(independent, summary['candidates']) for k in ('accuracy', 'aeod', 'aspd', 'score'))
    require(worst <= 1e-12 and min(independent, key=lambda r: (-r['score'], r['candidate']))['candidate']
            == summary['selected_per_method']['LoGoFair-DP-official-adapted']['candidate'], 'Frozen score/lexical selection differs')
    require(all(sha(p) == expected for p, expected in pins.items()), 'Inputs changed during read-only review')
    gate()
    proof = dict(status='COMPLETE32_ORIGINAL_STRICT_SAVED_PREDICTION_AND_FROZEN_SUMMARY_PASS_ROOT_ADOPTION_PENDING',
        source_seal_sha256=SOURCE_SHA, summary_sha256=expected_summary_sha, strict_index_sha256=sha(OUT / 'STRICT32_INDEX.json'),
        original_strict_rechecked_n=32, saved_prediction_items_checked=prediction_n, virtual_cohorts=20, true_training_client_fairness=False,
        root_only_fit_source_checked=True, seed_n=1, fit_seed=1719, post_rounds=30, candidate_means_and_scores_independent_max_difference=worst,
        selected_candidate=summary['selected_per_method']['LoGoFair-DP-official-adapted']['candidate'], all_candidates_and_negative_results_retained=True,
        summary_recipe_adopted=False, root_adopted=0, new_fits=0, new_CNN_calls=0, test_inference=False,
        artifact_hashes=pins, limitations=['Single seed91001; four conditions are not four independent seeds; no SD/significance',
        'Declared virtual image-ID cohorts, not original training clients; DP calibrated official adaptation, no EO claim',
        'Scores come from accepted float32 margins via sigmoid; possible 1ULP difference from fresh softmax',
        'Historical test attribute/split metadata exposure retained; no untouched-test claim; no formal100 or final test'])
    with (HERE / 'COMPLETE32_REVIEW.json').open('x', encoding='utf8') as f: f.write(json.dumps(proof, indent=2, allow_nan=False) + '\n')
    print(json.dumps(dict(status=proof['status'], summary_sha256=expected_summary_sha, original_strict_rechecked_n=32)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--summary-sha256', required=True)
    review(parser.parse_args().summary_sha256)
