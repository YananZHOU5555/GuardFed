"""Prepared increment36 staging entry. No commit, push, SSH, science or shared-state writes."""
from pathlib import Path
import argparse, datetime, hashlib, json, re, shutil, subprocess, sys

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
REPO = ROOT / 'tmp/revision-publish-20260928'
TRAIN = Path('docs/server_deployment_20260923/training_20260923')
CHECKS = TRAIN / 'server_reactivation_20261009'
NATIVE = CHECKS / 'mechanism_science_backups_20261009'
TAG = 'root_delta_20261010T000754Z'
C6 = Path('tmp/celeba_mechanism_valid_C_after50_20261010')
BRANCH = 'codex/revision-evidence-baselines-20260928'
PARENT = 'fbe6027794ce045bdf79254b995c8d2b1de2fb56'
EXCLUDED = {'__pycache__', 'verified_extract', 'verified', 'restored', '.git'}
IDS = ['minus_C_non-IID_Benign_seed' + str(n) for n in range(91001, 91007)]

def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_bytes())
def git(*args, binary=False, input=None):
    result = subprocess.check_output(['git', '-c', 'core.longpaths=true', *args], cwd=REPO, input=input)
    return result if binary else result.decode('utf-8').strip()

def closure_guard(proof, scope, science, execution):
    assert proof['status'] == 'ROOT_C_AFTER50_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS'
    assert (proof['prior_three_view_models'], proof['accepted_new'], proof['cumulative_three_view_models']) == (150, 6, 156)
    assert proof['accepted_new_ids'] == scope['selected_ids'] == IDS
    assert len(set(scope['excluded_prior_ids'])) == len(scope['excluded_prior_ids']) == 150
    assert not set(IDS) & set(scope['excluded_prior_ids'])
    assert proof['original150_unchanged'] is True and proof['all_native_differences_zero'] is True
    assert proof['server_strict_bound_in_saved_receipts'] is True and proof['negative_results_preserved'] is True
    assert proof['new_training'] == proof['new_Full_inference'] == proof['new_CNN_inference_for_root_review'] == 0
    assert proof['test_inference'] is False and proof['source_scope_complete'] is True
    assert proof['science_seal_sha256'] == science and proof['execution_seal_sha256'] == execution
    assert proof['prior150_root_adoption_sha256'] == '3859d49fb57c3ecc4b23244012431224d255b02590dab221b2d6464e28aa7dd8'

def hybrid_unchanged_guard(proof,digest,previous_receipt,published_path,expected):
    assert digest == expected and previous_receipt['copied_sha256'][published_path] == digest
    assert proof['accepted_total'] == 19 and proof['new_inference'] == 0
    assert proof['selection_performed'] is proof['scientific_changes'] is proof['final_test'] is False

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
    assert sha(args.closed_inputs) == args.closed_inputs_sha256
    closed = read(args.closed_inputs)
    assert closed['status'] == 'ROOT_CLOSED_INCREMENT36_INPUTS' and closed['parent_commit'] == PARENT
    assert closed['counts'] == {'native':156, 'three_view':156, 'FL_new':11, 'Hybrid':19, 'baseline_valid':900}
    assert set(prepared['required_extra_paths']) <= set(closed['extra_pins'])
    assert all(re.fullmatch('[0-9a-f]{64}',v) for v in closed['extra_pins'].values())
    def pin(key):
        row = closed['closure_pins'][key]; p = (ROOT / row['path']).resolve(); p.relative_to(ROOT.resolve())
        assert re.fullmatch('[0-9a-f]{64}', row['sha256']) and sha(p) == row['sha256'], key
        return p, read(p)
    adoption, proof = pin('C6_adoption'); source_review, review = pin('C6_source_review')
    fl_adoption, fl_proof = pin('FL_adoption'); hy_adoption, hy_proof = pin('Hybrid_adoption')
    state_path, state = pin('state'); live, health = pin('formal_live'); previous_path, previous = pin('previous_publication')
    assert state_path == (ROOT / TRAIN / 'TRAINING_STATE.json').resolve()
    assert live.parent == (ROOT / CHECKS).resolve() and re.fullmatch(r'root_live_\d{8}T\d{6}Z.json',live.name)
    assert closed['extra_pins'][(CHECKS / 'latest_formal_live.json').as_posix()] == sha(live)
    assert previous_path == (ROOT / TRAIN / 'publication_closed_increment35_verified_20261010.json').resolve()
    assert adoption.name == 'ROOT_ADOPTION_REVIEW.json' and adoption.parent.parent == (ROOT / C6 / 'execution_candidate/backups').resolve()
    scope = read(ROOT / C6 / 'SCOPE.json'); science = sha(ROOT / C6 / 'FILES_SHA256.json'); execution = sha(ROOT / C6 / 'execution_candidate/EXECUTION_SOURCE_SHA256.json')
    closure_guard(proof, scope, science, execution)
    assert review['status'] == 'PASS_SOURCE_READY_FOR_ROOT_LINUX_PREFLIGHT_AND_EXACT6_APPROVAL' and review['source_adoptable'] is True
    assert review['package_sha256'] == sha(ROOT / C6 / 'PACKAGE_SHA256.json') and review['science_seal_sha256'] == science and review['execution_seal_sha256'] == execution
    for name, key in [('incremental_valid_three_views.tar.gz','archive_sha256'),('backup_receipt.json','backup_receipt_sha256'),('OFFSERVER_VERIFICATION.json','offserver_verification_sha256')]: assert sha(adoption.parent / name) == proof[key], name
    off = read(adoption.parent / 'OFFSERVER_VERIFICATION.json')
    assert off['status'] == 'INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS' and off['accepted_new_ids'] == off['all_accepted_ids'] == IDS
    assert (off['independent_metric_checks'],off['independent_confusion_count_checks'],off['prediction_rule_checks']) == (54,144,18)
    prior_receipt_path = ROOT / TRAIN / 'publication_closed_increment35_20261010.json'
    assert sha(prior_receipt_path) == previous['publication_receipt_sha256']
    hybrid_unchanged_guard(hy_proof,sha(hy_adoption),read(prior_receipt_path),'experimental/' + hy_adoption.relative_to(ROOT / 'tmp').as_posix(),prepared['unchanged_Hybrid_root_sha256'])
    assert previous['status'] == 'COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit'] == PARENT and previous['branch'] == BRANCH
    assert [previous[k] for k in ('mechanism_offserver_verified','mechanism_three_view_offserver_verified','FLGMM_fullcoverage_new_offserver_verified','Hybrid_offserver_verified')] == [150,150,9,19]
    assert fl_adoption.parent == (ROOT / 'tmp/celeba_flgmm_fullcoverage_delta_after9_20261010').resolve()
    assert fl_proof['status'] == 'ROOT_FL96_LINKED_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS'
    assert (fl_proof['accepted_before'],fl_proof['accepted_new'],fl_proof['accepted_total']) == (9,2,11)
    assert fl_proof['accepted_new_ids'] == ['FLGMM_Tg20_L2.0_lr0.001_IID_F Flip_seed' + str(n) + '_fullcoverage' for n in (91001,91002)]
    assert len(fl_proof['accepted_new_ids']) == len(set(fl_proof['accepted_new_ids'])) == 2 and len(set(fl_proof['accepted_job_ids'])) == 11
    assert fl_proof['previous_root_adoption_sha256'] == 'ecaaa936289589c2b8b28ff42fa81e7e4eb09206fce49a187781be9ed4143577'
    assert fl_proof['old_models_repacked'] == fl_proof['root_new_CNN'] == 0 and fl_proof['final_test'] is False and fl_proof['negative_results_preserved'] is True
    assert fl_proof['original_checker_replayed_by_offserver_verifier'] is True and fl_proof['planned_new'] == 96 and fl_proof['reused_separately'] == 4
    assert hy_adoption.parent == (ROOT / 'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after18_20261010').resolve()
    assert hy_proof['status'] == 'ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS'
    assert (hy_proof['accepted_before'],hy_proof['accepted_new'],hy_proof['accepted_total']) == (18,1,19)
    assert hy_proof['previous_chain_sha256'] == 'e11d04c3df01d42f0613b7101a976e84a67812c55b0a271be270ecbbf41defef'
    assert hy_proof['new_inference'] == 0 and hy_proof['original_acceptor_replayed_by_offserver_tool'] is True
    assert all(hy_proof[k] is False for k in ('selection_performed','scientific_changes','final_test','formal100_started'))
    main = state['celeba_mechanism_v1']
    assert main['scientific_results_offserver_verified'] == main['three_view_new_models_offserver_verified'] == 156
    assert set(main['three_view_accepted_ids']) == set(scope['excluded_prior_ids']) | set(IDS) and main['test_started'] is False
    assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted'] == 900
    assert state['flgmm_fullcoverage_v2_20261009']['new_accepted'] == 11 and state['flgmm_fullcoverage_v2_20261009']['final_test'] is False
    assert state['hybrid_screen32_20261009']['offserver_accepted70round_jobs'] == 19 and state['hybrid_screen32_20261009']['selected_recipe'] is None and state['hybrid_screen32_20261009']['final_test'] is False
    assert not health['failed'] and not health['failure_files']
    assert git('rev-parse','HEAD') == PARENT and git('branch','--show-current') == BRANCH
    assert not git('status','--porcelain','--untracked-files=all'), 'Publish worktree must be clean'
    tracked = set(git('ls-tree','-r','--name-only',PARENT).splitlines()); sources = {}; mapping = {}
    def add(path, destination=None, expected=None):
        path = path.resolve(); rel = path.relative_to(ROOT); assert not EXCLUDED & set(rel.parts)
        if path.is_relative_to(ROOT / 'tmp/celeba_hybrid_screen_execution_20261009'): assert path == hy_adoption, 'Already published Hybrid scope allows unchanged ROOT proof only'
        assert path.is_file() and path.suffix.lower() not in {'.pt','.pth','.pem','.key'} and path.name != '.env' and path.stat().st_size < 100_000_000
        digest = sha(path); assert expected is None or digest == expected, rel
        dest = Path(destination) if destination else (Path('experimental') / rel.relative_to('tmp') if rel.parts[0] == 'tmp' else rel)
        assert not dest.is_absolute() and '..' not in dest.parts and not any(c in dest.as_posix() for c in '\n\r"')
        if path.suffix.lower() in {'.json','.py','.md','.txt','.log','.sh','.patch','.template'}: assert not re.search(rb'gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{30,}|-----BEGIN (?:[A-Z ]+ )?PRIVATE KEY-----|sk-proj-[A-Za-z0-9_-]{30,}',path.read_bytes()), rel
        name = dest.as_posix(); assert name not in mapping or mapping[name] == digest, name
        if name.endswith(('.tar.gz','.tar')): assert name not in tracked, 'Old archive must not be republished: '+name
        sources[name] = path; mapping[name] = digest
    def tree(folder):
        for path in sorted(folder.rglob('*')):
            if path.is_file() and not EXCLUDED & set(path.relative_to(folder).parts): add(path)
    def seal(folder, filename, expected=None):
        assert expected is None or sha(folder / filename) == expected
        data = read(folder / filename); rows = data.get('members') or [{'path':n,**(v if isinstance(v,dict) else {'sha256':v})} for n,v in data['files'].items()]
        for row in rows: add(folder / row['path'],expected=row['sha256'])
        add(folder / filename)
    native = ROOT / NATIVE; delta = native / TAG
    assert read(delta / 'ROOT_INDEPENDENT_REVIEW.json')['native_accepted'] == 156 and read(delta / 'ROOT_DELTA_VERIFICATION.json')['new_ids'] == IDS
    for path in delta.iterdir():
        if path.is_file() and path.name != 'verified_ledger.json': add(path)
    tree(native / ('mechanism_inspection_v4_' + TAG)); add(delta / 'verified_ledger.json',NATIVE / 'verified_ledger.json')
    for name in (TAG+'.tar.gz',TAG+'.tar.gz.receipt.json',TAG+'_offserver_verification.json'): add(native / name)
    seal(ROOT / C6,'PACKAGE_SHA256.json'); tree(ROOT / C6 / 'execution_candidate'); add(source_review)
    seal(fl_adoption.parent,'DELIVERY_FILES_SHA256.json',fl_proof['delivery_seal_sha256']); add(fl_adoption)
    add(hy_adoption)  # Already published35; no old Hybrid seal tree/archive.
    for path in (state_path,live,previous_path,args.closed_inputs.resolve()): add(path)
    for rel,digest in closed['extra_pins'].items(): add(ROOT / rel,expected=digest)
    assert closed.get('C50_table') is None, 'C50 table already published35; do not rebind old C3 to C6'
    args.table_included = False
    seal(HERE,'FILES_SHA256.json')
    archives = [digest for name,digest in mapping.items() if name.endswith(('.tar.gz','.tar'))]
    assert len(archives) == len(set(archives)) and sum(p.stat().st_size for p in sources.values()) < 100_000_000
    args.C6_adoption=adoption; args.C6_adoption_sha256=sha(adoption); args.formal_live=live; args.formal_live_sha256=sha(live)
    return sources,mapping,proof,health

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closed-inputs', type=Path, required=True); parser.add_argument('--closed-inputs-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True); parser.add_argument('--execute-stage', action='store_true')
    args = parser.parse_args(); assert args.execute_stage, 'Source is prepared only; explicit --execute-stage is required'
    output = args.output.resolve(); assert output.parent == HERE and not output.exists()
    sources, mapping, proof, health = plan(args)  # No writes or index changes before every source/closure gate passes.
    output.mkdir()
    receipt = dict(status='INCREMENT36_SOURCE_BYTES_PREPARED_FOR_INDEX', created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), previous_commit=PARENT,
        copied_sha256=mapping.copy(), C6_root_adoption_path=args.C6_adoption.resolve().relative_to(ROOT).as_posix(), C6_root_adoption_sha256=args.C6_adoption_sha256,
        baseline_valid_replays_accepted=900, mechanism_offserver_verified=156, mechanism_three_view_offserver_verified=156,
        FLGMM_offserver_verified=32, FLGMM_fullcoverage_new_offserver_verified=11, Hybrid_offserver_verified=19,
        formal_live_path=args.formal_live.resolve().relative_to(ROOT).as_posix(), formal_live_sha256=args.formal_live_sha256,
        observed_main_terminal=health['queue_completed'], observed_main_active=len(health['active']), observed_terminal_is_not_acceptance=True,
        closed_inputs_sha256=args.closed_inputs_sha256, C50_table_included=args.table_included, new_GPU_chunks_root_reviewed=[], duplicated_old_models=0, test_started=False, scientific_goal_complete=False,
        scope='Closed validation evidence increment only; no new recipe choice, full-coverage completion, manuscript application, or final test.')
    receipt_path = output / 'publication_closed_increment36_20261010.json'
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
        for start in range(0, len(names), 25): git('add', '-f', '--', *names[start:start + 25])
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
