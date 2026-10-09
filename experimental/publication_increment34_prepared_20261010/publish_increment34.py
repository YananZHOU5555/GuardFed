"""Prepared increment34 staging entry. No commit, push, SSH, science or shared-state writes."""
from pathlib import Path
import argparse, datetime, hashlib, json, re, shutil, subprocess, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT / 'tmp/revision-publish-20260928'
TRAIN = Path('docs/server_deployment_20260923/training_20260923')
CHECKS = TRAIN / 'server_reactivation_20261009'
NATIVE = CHECKS / 'mechanism_science_backups_20261009'
TAG = 'root_delta_20261009T230901Z'
C7 = Path('tmp/celeba_mechanism_valid_C_after40_20261010')
BRANCH = 'codex/revision-evidence-baselines-20260928'
PARENT = '5526afddfce841b2a771795918f33c31478c83c4'
EXCLUDED = {'__pycache__', 'verified_extract', 'verified', 'restored', '.git'}
IDS = ['minus_C_IID_Sp-DFA_seed' + str(n) for n in range(91001, 91008)]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def git(*args, binary=False, input=None):
    result = subprocess.check_output(['git', '-c', 'core.longpaths=true', *args], cwd=REPO, input=input)
    return result if binary else result.decode('utf-8').strip()

def closure_guard(proof, scope, science, execution):
    assert proof['status'] == 'ROOT_C_AFTER40_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
    assert (proof['prior_three_view_models'], proof['accepted_new'], proof['cumulative_three_view_models']) == (140, 7, 147)
    assert proof['accepted_new_ids'] == scope['selected_ids'] == IDS
    assert len(set(scope['excluded_prior_ids'])) == len(scope['excluded_prior_ids']) == 140
    assert not set(IDS) & set(scope['excluded_prior_ids'])
    assert proof['original140_unchanged'] is True and proof['all_native_differences_zero'] is True
    assert proof['server_strict_bound_in_saved_receipts'] is True and proof['negative_results_preserved'] is True
    assert proof['new_training'] == proof['new_Full_inference'] == proof['new_CNN_inference_for_root_review'] == 0
    assert proof['test_inference'] is False and proof['source_scope_complete'] is True
    assert proof['science_seal_sha256'] == science and proof['execution_seal_sha256'] == execution
    assert proof['prior140_root_adoption_sha256'] == 'eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a'

def verify_blobs(prefix, mapping):
    data = git('cat-file', '--batch', binary=True, input=''.join(prefix + name + '\n' for name in mapping).encode())
    offset = 0
    for name, digest in mapping.items():
        stop = data.index(b'\n', offset); header = data[offset:stop].split()
        assert len(header) == 3 and header[1] == b'blob', name
        size = int(header[2]); start = stop + 1; payload = data[start:start + size]
        assert len(payload) == size and hashlib.sha256(payload).hexdigest() == digest, name
        assert data[start + size:start + size + 1] == b'\n'
        offset = start + size + 1
    assert offset == len(data)

def plan(args):
    assert not sys.flags.optimize, 'Assertions are mandatory; do not use python -O'
    prepared = read(HERE / 'PREPARED_INPUTS.json')
    for rel, digest in prepared['ready_sha256'].items(): assert sha(ROOT / rel) == digest, rel
    assert git('rev-parse', 'HEAD') == PARENT and git('branch', '--show-current') == BRANCH
    assert not git('status', '--porcelain', '--untracked-files=all'), 'Publish worktree must be clean'
    adoption = args.C7_adoption.resolve()
    assert adoption.name == 'ROOT_ADOPTION_REVIEW.json'
    assert adoption.parent.parent == (ROOT / C7 / 'execution_candidate/backups').resolve()
    assert re.fullmatch('[0-9a-f]{64}', args.C7_adoption_sha256) and sha(adoption) == args.C7_adoption_sha256
    proof = read(adoption); scope = read(ROOT / C7 / 'SCOPE.json')
    science = sha(ROOT / C7 / 'FILES_SHA256.json'); execution = sha(ROOT / C7 / 'execution_candidate/EXECUTION_SOURCE_SHA256.json')
    closure_guard(proof, scope, science, execution)
    for name, key in [('incremental_valid_three_views.tar.gz', 'archive_sha256'), ('backup_receipt.json', 'backup_receipt_sha256'), ('OFFSERVER_VERIFICATION.json', 'offserver_verification_sha256')]:
        assert sha(adoption.parent / name) == proof[key], name
    off = read(adoption.parent / 'OFFSERVER_VERIFICATION.json')
    assert off['status'] == 'INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS'
    assert off['accepted_new_ids'] == off['all_accepted_ids'] == IDS
    assert (off['independent_metric_checks'], off['independent_confusion_count_checks'], off['prediction_rule_checks']) == (63, 168, 21)
    state = read(ROOT / TRAIN / 'TRAINING_STATE.json'); main = state['celeba_mechanism_v1']
    assert main['scientific_results_offserver_verified'] == main['three_view_new_models_offserver_verified'] == 147
    assert set(main['three_view_accepted_ids']) == set(scope['excluded_prior_ids']) | set(IDS)
    assert main['test_started'] is False and state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted'] == 900
    assert state['flgmm_fullcoverage_v2_20261009']['new_accepted'] == 7
    assert state['flgmm_fullcoverage_v2_20261009']['final_test'] is False
    assert state['hybrid_screen32_20261009']['offserver_accepted70round_jobs'] == 18
    assert state['hybrid_screen32_20261009']['selected_recipe'] is None and state['hybrid_screen32_20261009']['final_test'] is False
    sources = {}; mapping = {}
    def add(path, destination=None, expected=None):
        path = path.resolve(); rel = path.relative_to(ROOT); assert not EXCLUDED & set(rel.parts)
        assert path.is_file() and path.suffix.lower() not in {'.pt', '.pth', '.pem', '.key'} and path.name != '.env'
        assert path.stat().st_size < 100_000_000
        digest = sha(path); assert expected is None or digest == expected, rel
        dest = Path(destination) if destination else (Path('experimental') / rel.relative_to('tmp') if rel.parts[0] == 'tmp' else rel)
        assert not dest.is_absolute() and '..' not in dest.parts and not any(c in dest.as_posix() for c in '\n\r"')
        if path.suffix.lower() in {'.json', '.py', '.md', '.txt', '.log', '.sh', '.patch', '.template'}:
            assert not re.search(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|sk-proj-[A-Za-z0-9_-]{30,}', path.read_bytes()), rel
        name = dest.as_posix(); assert name not in mapping or mapping[name] == digest, name
        sources[name] = path; mapping[name] = digest
    def tree(folder):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and not EXCLUDED & set(path.relative_to(folder).parts): add(path)
    def seal(folder, filename):
        data = read(folder / filename)
        rows = data.get('members') or [{'path': n, **(v if isinstance(v, dict) else {'sha256': v})} for n, v in data['files'].items()]
        for row in rows: add(folder / row['path'], expected=row['sha256'])
        add(folder / filename)
    native = ROOT / NATIVE; delta = native / TAG
    assert read(delta / 'ROOT_INDEPENDENT_REVIEW.json')['native_accepted'] == 147
    assert read(delta / 'ROOT_DELTA_VERIFICATION.json')['new_ids'] == IDS
    for path in delta.iterdir():
        if path.is_file() and path.name != 'verified_ledger.json': add(path)
    tree(native / ('mechanism_inspection_v4_' + TAG))
    add(delta / 'verified_ledger.json', NATIVE / 'verified_ledger.json')
    for name in (TAG + '.tar.gz', TAG + '.tar.gz.receipt.json', TAG + '_offserver_verification.json'): add(native / name)
    seal(ROOT / C7, 'PACKAGE_SHA256.json')
    for folder in (ROOT / C7 / 'execution_candidate', ROOT / 'tmp/celeba_mechanism_C_after40_root_operations_20261010', ROOT / 'tmp/celeba_mechanism_C_after40_source_review_20261010'): tree(folder)
    review = read(ROOT / 'tmp/celeba_mechanism_C_after40_source_review_20261010/ROOT_INDEPENDENT_REVIEW.json')
    assert review['status'] == 'PASS_SOURCE_READY_FOR_ROOT_LINUX_PREFLIGHT_AND_EXACT7_APPROVAL' and review['source_adoptable'] is True
    assert review['package_sha256'] == sha(ROOT / C7 / 'PACKAGE_SHA256.json')
    assert review['science_seal_sha256'] == science and review['execution_seal_sha256'] == execution
    fl = ROOT / 'tmp/celeba_flgmm_fullcoverage_delta_after5_20261010'; seal(fl, 'DELIVERY_FILES_SHA256.json'); add(fl / 'ROOT_ADOPTION_REVIEW.json')
    add(ROOT / 'tmp/celeba_flgmm_fullcoverage_incremental_20261009/LATEST_BACKUP.json')
    hybrid = ROOT / 'tmp/celeba_hybrid_screen_execution_20261009'; hd = hybrid / 'accepted_delta_after17_20261010'
    seal(hd, 'DELIVERY_FILES_SHA256.json')
    for path in (hd / 'ROOT_ADOPTION_REVIEW.json', hybrid / 'LATEST_BACKUP.json', hybrid / 'BACKUP_CHAIN_accepted_delta_after17_20261010.json'): add(path)
    for rel in prepared['root_scripts']: add(ROOT / rel)
    for name in ('RUNNING.md', 'TRAINING_STATE.json', 'REBUTTAL_COMPLETION_20261009.md', 'celeba_mechanism_v1/EXECUTION.md', 'publication_closed_increment33_verified_20261009.json'): add(ROOT / TRAIN / name)
    for rel in ('docs/返修实验总览.md', str(CHECKS / 'MONITOR_HANDOFF.md'), str(CHECKS / 'latest_formal_live.json')): add(ROOT / rel)
    live = args.formal_live.resolve(); assert live.parent == (ROOT / CHECKS).resolve() and re.fullmatch(r'root_live_\d{8}T\d{6}Z.json', live.name)
    assert sha(live) == args.formal_live_sha256 == sha(ROOT / CHECKS / 'latest_formal_live.json')
    add(live); health = read(live)
    assert not health['failed'] and not health['failure_files']
    for key in ('flgmm_screen32_20261009', 'hybrid_screen32_20261009'):
        observation = state[key]['latest_readonly_terminal_observation']; source = ROOT / observation['entry']
        add(source, expected=observation['sha256'])
        raw = source.with_name(source.stem + '.RAW.json')
        if raw.is_file(): add(raw)
    fl_observation = state['flgmm_fullcoverage_v2_20261009']; source = ROOT / fl_observation['latest_readonly_observation_path']
    assert sha(source) == fl_observation['latest_readonly_observation_sha256']
    for path in source.parent.iterdir():
        if path.is_file(): add(path)
    seal(HERE, 'FILES_SHA256.json')
    archives = [digest for name, digest in mapping.items() if name.endswith(('.tar.gz', '.tar'))]
    assert len(archives) == len(set(archives)), 'Duplicate archive bytes must have only one published copy'
    assert sum(path.stat().st_size for path in sources.values()) < 100_000_000, 'Bounded increment total must be below 100MB'
    return sources, mapping, proof, health

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--C7-adoption', type=Path, required=True); parser.add_argument('--C7-adoption-sha256', required=True)
    parser.add_argument('--formal-live', type=Path, required=True); parser.add_argument('--formal-live-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--execute-stage', action='store_true')
    args = parser.parse_args(); assert args.execute_stage, 'Source is prepared only; explicit --execute-stage is required'
    output = args.output.resolve(); assert output.parent == HERE and not output.exists()
    sources, mapping, proof, health = plan(args)  # No writes or index changes before every source/closure gate passes.
    output.mkdir()
    receipt = dict(status='INCREMENT34_SOURCE_BYTES_PREPARED_FOR_INDEX', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous_commit=PARENT,
        copied_sha256=mapping.copy(), C7_root_adoption_path=args.C7_adoption.resolve().relative_to(ROOT).as_posix(), C7_root_adoption_sha256=args.C7_adoption_sha256,
        baseline_valid_replays_accepted=900, mechanism_offserver_verified=147, mechanism_three_view_offserver_verified=147,
        FLGMM_offserver_verified=32, FLGMM_fullcoverage_new_offserver_verified=7, Hybrid_offserver_verified=18,
        formal_live_path=args.formal_live.resolve().relative_to(ROOT).as_posix(), formal_live_sha256=args.formal_live_sha256,
        observed_main_terminal=health['queue_completed'], observed_main_active=len(health['active']), observed_terminal_is_not_acceptance=True,
        new_GPU_chunks_root_reviewed=[], duplicated_old_models=0, test_started=False, scientific_goal_complete=False,
        scope='Closed validation evidence increment only; no new recipe choice, full-coverage completion, manuscript application, or final test.')
    receipt_path = output / 'publication_closed_increment34_20261010.json'
    receipt_path.write_text(json.dumps(receipt, ensure_ascii=False, indent=2) + '\n', encoding='utf-8', newline='\n')
    receipt_rel = (TRAIN / receipt_path.name).as_posix(); sources[receipt_rel] = receipt_path; mapping[receipt_rel] = sha(receipt_path)
    try:
        for name, source in sources.items():
            assert sha(source) == mapping[name], name
            destination = REPO / name; assert destination.resolve().is_relative_to(REPO.resolve())
            destination.parent.mkdir(parents=True, exist_ok=True); shutil.copyfile(source, destination)
            assert sha(destination) == mapping[name], name
        attributes = REPO / '.gitattributes'; original = attributes.read_bytes() if attributes.exists() else b''
        attributes.write_bytes(original + (b'' if not original or original.endswith(b'\n') else b'\n') + ''.join('"' + n + '" -text\n' for n in mapping).encode('utf-8'))
        mapping['.gitattributes'] = sha(attributes)
        names = list(mapping)
        for start in range(0, len(names), 25): git('add', '--', *names[start:start + 25])
        for start in range(0, len(names), 25): git('add', '--renormalize', '--', *names[start:start + 25])
        verify_blobs(':', mapping)
        changed = set(filter(None, git('diff', '--cached', '--name-only', '-z', binary=True).decode().split('\0')))
        assert changed <= set(mapping) and git('rev-parse', 'HEAD') == PARENT
        result = dict(status='INDEX_BLOB_SHA_PASS_NOT_COMMITTED_OR_PUSHED', staged_paths=len(changed), blob_paths_verified=len(mapping), receipt_sha256=sha(receipt_path), index_sha256=mapping)
        (output / 'INDEX_VERIFICATION.json').write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8', newline='\n'); print(json.dumps(result))
    except Exception as error:
        (output / 'FAILURE.json').write_text(json.dumps(dict(status='FAILED_PRESERVE_WORKTREE_AND_INDEX_NO_AUTOMATIC_RETRY', error=repr(error)), indent=2) + '\n', encoding='utf-8')
        raise

if __name__ == '__main__': main()
