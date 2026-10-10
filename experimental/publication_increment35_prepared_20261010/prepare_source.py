"""Prepare increment35 from the successful lean34-v2 sources; no Git calls."""
from pathlib import Path
import ast,difflib,hashlib,json
H=Path(__file__).resolve().parent;R=H.parents[1];O=H.with_name('publication_increment34_prepared_v2_20261010')
sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
assert sha(O/'publish_increment34.py')=='59309c9ae03971efdddee3027e0a045248bfb92f3fb5f9673127568e07af2953'
assert sha(O/'verify_increment34.py')=='347d8817a6342e6cef325d5d5a6e1a10ea73c5ce7d82f9466d373f54216c19d5'
original=(O/'publish_increment34.py').read_text(encoding='utf-8-sig');s=original
changes={
 'increment34':'increment35','INCREMENT34':'INCREMENT35',
 "TAG = 'root_delta_20261009T230901Z'":"TAG = 'root_delta_20261009T233607Z'",
 "C7 = Path('tmp/celeba_mechanism_valid_C_after40_20261010')":"C3 = Path('tmp/celeba_mechanism_valid_C_after47_20261010')",
 "PARENT = '5526afddfce841b2a771795918f33c31478c83c4'":"PARENT = 'ae94e17a7b1c3b3f6298eaa9d7f09bbf31cca17d'",
 'range(91001, 91008)':'range(91008, 91011)',
 'ROOT_C_AFTER40_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS':'ROOT_C_AFTER47_INCREMENT_ARCHIVE_MEMBER_AND_SAVED_ARRAY_CHECKS_PASS',
 '(140, 7, 147)':'(147, 3, 150)',
 "== 140":"== 147",'original140_unchanged':'original147_unchanged',
 'prior140_root_adoption_sha256':'prior147_root_adoption_sha256',
 'eb1f6c3120febb138e32af25484ac790cf962cd15d5ea9921140119bc5a3317a':'64732337f35f81bed49e40c60fa1d5229fb6a7557c45272505fee488c5ad20ea',
}
# Apply once against original tokens; no cascading scientific/count replacement.
import re
s=re.sub('|'.join(re.escape(k) for k in sorted(changes,key=len,reverse=True)),lambda m:changes[m.group(0)],s)
start=s.index('def plan(args):');end=s.index('\ndef main():',start)
plan='''def plan(args):
    assert not sys.flags.optimize, 'Assertions are mandatory; do not use python -O'
    prepared = read(HERE / 'PREPARED_INPUTS.json')
    for rel, digest in prepared['ready_sha256'].items(): assert sha(ROOT / rel) == digest, rel
    assert sha(args.closed_inputs) == args.closed_inputs_sha256
    closed = read(args.closed_inputs)
    assert closed['status'] == 'ROOT_CLOSED_INCREMENT35_INPUTS' and closed['parent_commit'] == PARENT
    assert closed['counts'] == {'native':150, 'three_view':150, 'FL_new':9, 'Hybrid':19, 'baseline_valid':900}
    assert set(prepared['required_extra_paths']) <= set(closed['extra_pins'])
    assert all(re.fullmatch('[0-9a-f]{64}',v) for v in closed['extra_pins'].values())
    def pin(key):
        row = closed['closure_pins'][key]; p = (ROOT / row['path']).resolve(); p.relative_to(ROOT.resolve())
        assert re.fullmatch('[0-9a-f]{64}', row['sha256']) and sha(p) == row['sha256'], key
        return p, read(p)
    adoption, proof = pin('C3_adoption'); source_review, review = pin('C3_source_review')
    fl_adoption, fl_proof = pin('FL_adoption'); hy_adoption, hy_proof = pin('Hybrid_adoption')
    state_path, state = pin('state'); live, health = pin('formal_live'); previous_path, previous = pin('previous_publication')
    assert state_path == (ROOT / TRAIN / 'TRAINING_STATE.json').resolve()
    assert live.parent == (ROOT / CHECKS).resolve() and re.fullmatch(r'root_live_\\d{8}T\\d{6}Z.json',live.name)
    assert closed['extra_pins'][(CHECKS / 'latest_formal_live.json').as_posix()] == sha(live)
    assert previous_path == (ROOT / TRAIN / 'publication_closed_increment34_verified_20261010.json').resolve()
    assert adoption.name == 'ROOT_ADOPTION_REVIEW.json' and adoption.parent.parent == (ROOT / C3 / 'execution_candidate/backups').resolve()
    scope = read(ROOT / C3 / 'SCOPE.json'); science = sha(ROOT / C3 / 'FILES_SHA256.json'); execution = sha(ROOT / C3 / 'execution_candidate/EXECUTION_SOURCE_SHA256.json')
    closure_guard(proof, scope, science, execution)
    assert review['status'] == 'PASS_SOURCE_READY_FOR_ROOT_LINUX_PREFLIGHT_AND_EXACT3_APPROVAL' and review['source_adoptable'] is True
    assert review['package_sha256'] == sha(ROOT / C3 / 'PACKAGE_SHA256.json') and review['science_seal_sha256'] == science and review['execution_seal_sha256'] == execution
    for name, key in [('incremental_valid_three_views.tar.gz','archive_sha256'),('backup_receipt.json','backup_receipt_sha256'),('OFFSERVER_VERIFICATION.json','offserver_verification_sha256')]: assert sha(adoption.parent / name) == proof[key], name
    off = read(adoption.parent / 'OFFSERVER_VERIFICATION.json')
    assert off['status'] == 'INCREMENTAL_INDEPENDENT_SAVED_ARRAYS_THREE_VIEWS_PASS' and off['accepted_new_ids'] == off['all_accepted_ids'] == IDS
    assert (off['independent_metric_checks'],off['independent_confusion_count_checks'],off['prediction_rule_checks']) == (27,72,9)
    assert previous['status'] == 'COMMITTED_BLOB_SHA_AND_REMOTE_BRANCH_PASS' and previous['commit'] == PARENT and previous['branch'] == BRANCH
    assert [previous[k] for k in ('mechanism_offserver_verified','mechanism_three_view_offserver_verified','FLGMM_fullcoverage_new_offserver_verified','Hybrid_offserver_verified')] == [147,147,7,18]
    assert fl_adoption.parent == (ROOT / 'tmp/celeba_flgmm_fullcoverage_delta_after7_20261010').resolve()
    assert fl_proof['status'] == 'ROOT_FL96_LINKED_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS'
    assert (fl_proof['accepted_before'],fl_proof['accepted_new'],fl_proof['accepted_total']) == (7,2,9)
    assert len(fl_proof['accepted_new_ids']) == len(set(fl_proof['accepted_new_ids'])) == 2 and len(set(fl_proof['accepted_job_ids'])) == 9
    assert fl_proof['previous_root_adoption_sha256'] == '4403439d39196206e169f14428d68b59b779e1cdd4a5a9fb7d0dd0a3b13dabcf'
    assert fl_proof['old_models_repacked'] == fl_proof['root_new_CNN'] == 0 and fl_proof['final_test'] is False and fl_proof['negative_results_preserved'] is True
    assert fl_proof['original_checker_replayed_by_offserver_verifier'] is True and fl_proof['planned_new'] == 96 and fl_proof['reused_separately'] == 4
    assert hy_adoption.parent == (ROOT / 'tmp/celeba_hybrid_screen_execution_20261009/accepted_delta_after18_20261010').resolve()
    assert hy_proof['status'] == 'ROOT_LOCKED_AUXILIARY_DELTA_ARCHIVE_MEMBER_AND_ORIGINAL_RECORD_ACCEPTANCE_PASS'
    assert (hy_proof['accepted_before'],hy_proof['accepted_new'],hy_proof['accepted_total']) == (18,1,19)
    assert hy_proof['previous_chain_sha256'] == 'e11d04c3df01d42f0613b7101a976e84a67812c55b0a271be270ecbbf41defef'
    assert hy_proof['new_inference'] == 0 and hy_proof['original_acceptor_replayed_by_offserver_tool'] is True
    assert all(hy_proof[k] is False for k in ('selection_performed','scientific_changes','final_test','formal100_started'))
    main = state['celeba_mechanism_v1']
    assert main['scientific_results_offserver_verified'] == main['three_view_new_models_offserver_verified'] == 150
    assert set(main['three_view_accepted_ids']) == set(scope['excluded_prior_ids']) | set(IDS) and main['test_started'] is False
    assert state['final_evaluator_runtime_20261009']['actual_native_valid_image_replays_accepted'] == 900
    assert state['flgmm_fullcoverage_v2_20261009']['new_accepted'] == 9 and state['flgmm_fullcoverage_v2_20261009']['final_test'] is False
    assert state['hybrid_screen32_20261009']['offserver_accepted70round_jobs'] == 19 and state['hybrid_screen32_20261009']['selected_recipe'] is None and state['hybrid_screen32_20261009']['final_test'] is False
    assert not health['failed'] and not health['failure_files']
    assert git('rev-parse','HEAD') == PARENT and git('branch','--show-current') == BRANCH
    assert not git('status','--porcelain','--untracked-files=all'), 'Publish worktree must be clean'
    tracked = set(git('ls-tree','-r','--name-only',PARENT).splitlines()); sources = {}; mapping = {}
    def add(path, destination=None, expected=None):
        path = path.resolve(); rel = path.relative_to(ROOT); assert not EXCLUDED & set(rel.parts)
        assert path.is_file() and path.suffix.lower() not in {'.pt','.pth','.pem','.key'} and path.name != '.env' and path.stat().st_size < 100_000_000
        digest = sha(path); assert expected is None or digest == expected, rel
        dest = Path(destination) if destination else (Path('experimental') / rel.relative_to('tmp') if rel.parts[0] == 'tmp' else rel)
        assert not dest.is_absolute() and '..' not in dest.parts and not any(c in dest.as_posix() for c in '\\n\\r"')
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
    assert read(delta / 'ROOT_INDEPENDENT_REVIEW.json')['native_accepted'] == 150 and read(delta / 'ROOT_DELTA_VERIFICATION.json')['new_ids'] == IDS
    for path in delta.iterdir():
        if path.is_file() and path.name != 'verified_ledger.json': add(path)
    tree(native / ('mechanism_inspection_v4_' + TAG)); add(delta / 'verified_ledger.json',NATIVE / 'verified_ledger.json')
    for name in (TAG+'.tar.gz',TAG+'.tar.gz.receipt.json',TAG+'_offserver_verification.json'): add(native / name)
    seal(ROOT / C3,'PACKAGE_SHA256.json'); tree(ROOT / C3 / 'execution_candidate'); add(source_review)
    seal(fl_adoption.parent,'DELIVERY_FILES_SHA256.json',fl_proof['delivery_seal_sha256']); add(fl_adoption)
    seal(hy_adoption.parent,'DELIVERY_FILES_SHA256.json',hy_proof['delivery_seal_sha256']); add(hy_adoption)
    for path in (state_path,live,previous_path,args.closed_inputs.resolve()): add(path)
    for rel,digest in closed['extra_pins'].items(): add(ROOT / rel,expected=digest)
    table = closed.get('C50_table'); args.table_included = table is not None
    if table is not None:
        assert re.fullmatch('[0-9a-f]{64}',table['root_proof']['sha256']) and re.fullmatch('[0-9a-f]{64}',table['seal']['sha256'])
        folder = (ROOT / table['directory']).resolve(); root_path = ROOT / table['root_proof']['path']; assert root_path.resolve().parent == folder
        assert sha(root_path) == table['root_proof']['sha256']; table_proof = read(root_path)
        assert table_proof['status'].startswith('ROOT_C50_') and table_proof['status'].endswith('_ADOPTED')
        assert [table_proof[k] for k in ('unique_records','paired_models','complete_scenes','mean_SD_scalars_recomputed','display_cells','count_metrics_recomputed')] == [100,50,5,810,405,900]
        assert table_proof[table['C3_adoption_field']] == sha(adoption) and table_proof['seed_panels'] == [10,9,6]
        assert table_proof['all_negative_results_retained'] is True and table_proof['new_CNN'] == table_proof['new_training'] == table_proof['new_Full_inference'] == 0 and table_proof['test'] is False
        assert table_proof['primary_endpoint'] == 'PENDING_AUTHOR' and table_proof['whole_rebuttal_complete'] is False
        seal(folder,table['seal']['filename'],table['seal']['sha256']); add(root_path)
    seal(HERE,'FILES_SHA256.json')
    archives = [digest for name,digest in mapping.items() if name.endswith(('.tar.gz','.tar'))]
    assert len(archives) == len(set(archives)) and sum(p.stat().st_size for p in sources.values()) < 100_000_000
    args.C3_adoption=adoption; args.C3_adoption_sha256=sha(adoption); args.formal_live=live; args.formal_live_sha256=sha(live)
    return sources,mapping,proof,health
'''
s=s[:start]+plan+s[end:]
s=s.replace("    parser.add_argument('--C7-adoption', type=Path, required=True); parser.add_argument('--C7-adoption-sha256', required=True)\n    parser.add_argument('--formal-live', type=Path, required=True); parser.add_argument('--formal-live-sha256', required=True)","    parser.add_argument('--closed-inputs', type=Path, required=True); parser.add_argument('--closed-inputs-sha256', required=True)")
s=s.replace('C7_root_adoption','C3_root_adoption').replace('args.C7_adoption','args.C3_adoption')
s=s.replace('mechanism_offserver_verified=147, mechanism_three_view_offserver_verified=147','mechanism_offserver_verified=150, mechanism_three_view_offserver_verified=150')
s=s.replace('FLGMM_fullcoverage_new_offserver_verified=7, Hybrid_offserver_verified=18','FLGMM_fullcoverage_new_offserver_verified=9, Hybrid_offserver_verified=19')
s=s.replace("        new_GPU_chunks_root_reviewed=[]", "        closed_inputs_sha256=args.closed_inputs_sha256, C50_table_included=args.table_included, new_GPU_chunks_root_reviewed=[]")
ast.parse(s)
with (H/'publish_increment35.py').open('x',encoding='utf-8',newline='\n') as f:f.write(s)
v=(O/'verify_increment34.py').read_text(encoding='utf-8-sig')
for a,b in [('increment34','increment35'),("5526afddfce841b2a771795918f33c31478c83c4","ae94e17a7b1c3b3f6298eaa9d7f09bbf31cca17d"),('[900, 147, 147, 7, 18]','[900, 150, 150, 9, 19]'),('mechanism_offserver_verified=147, mechanism_three_view_offserver_verified=147','mechanism_offserver_verified=150, mechanism_three_view_offserver_verified=150'),('FLGMM_fullcoverage_new_offserver_verified=7, Hybrid_offserver_verified=18','FLGMM_fullcoverage_new_offserver_verified=9, Hybrid_offserver_verified=19')]:v=v.replace(a,b)
ast.parse(v)
with (H/'verify_increment35.py').open('x',encoding='utf-8',newline='\n') as f:f.write(v)
with (H/'MINIMAL_SOURCE_DIFF.patch').open('x',encoding='utf-8',newline='\n') as f:
 f.writelines(difflib.unified_diff(original.splitlines(True),s.splitlines(True),fromfile='sealed34-v2/publish_increment34.py',tofile='increment35/publish_increment35.py'))
 f.writelines(difflib.unified_diff((O/'verify_increment34.py').read_text().splitlines(True),v.splitlines(True),fromfile='sealed34-v2/verify_increment34.py',tofile='increment35/verify_increment35.py'))
print(json.dumps({'publisher_lines':len(s.splitlines()),'verifier_lines':len(v.splitlines()),'source_only':True,'Git_calls':0}))
