"""PREPARED root adopter: exact6 after accepted38; execute only after root review."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, json, tarfile

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'tmp/celeba_flgmm_fullcoverage_incremental_20261009'
ATTEMPT = ROOT / 'tmp/celeba_flgmm_fullcoverage_delta_after38_20261010'
BATCH = Path('F:/YananResearchStorage/GuardFed/celeba_flgmm_fullcoverage_delta_after38_20261010/batch')
sha = lambda p: hashlib.sha256(p.read_bytes()).hexdigest()
read = lambda p: json.loads(p.read_bytes())

# Direct materialization of previously root-executed checks; no wrapper execution.
from guardfed_local_storage import check_bulk_storage
check_bulk_storage(0)
assert not (ATTEMPT/'ROOT_ADOPTION_REVIEW.json').exists()
assert sha(ATTEMPT/'RAW_STORAGE_INDEX.json')=='4f4faceae702b65013ddc378e8d5147a46e3748e9449c0b8649c6507b71f6baf'
index=read(ATTEMPT/'RAW_STORAGE_INDEX.json');external=Path(index['raw_storage_root'])
assert external.resolve().drive.upper()=='F:' and len(index['files'])==72
for name,pin in index['files'].items():
    path=external/name
    assert path.resolve().is_relative_to(external.resolve()) and sha(path)==pin['sha256'] and path.stat().st_size==pin['bytes']
assert sha(ATTEMPT/'SAVED_TENSOR_STATE_CHECK.json')=='04e0aeec7b93b46c321cad54b16c554b73b660938b7d5189eb068a2e1b815abd'
tensor=read(ATTEMPT/'SAVED_TENSOR_STATE_CHECK.json')
assert sum(x['tensor_count'] for x in tensor['records'])==48 and sum(x['elements'] for x in tensor['records'])==561036
assert tensor['CNN_forward_calls']==tensor['optimizer_calls']==tensor['data_loads']==0
assert read(ATTEMPT/'ROOT_READY_HANDOFF.json')['CPU110_released']

assert sha(ATTEMPT/'ROOT_READY_HANDOFF.json') == '9f2d5d48d20cd812cdfac505d5f9e8763c6a7347caeba1141f430d6a90945861'
handoff = read(ATTEMPT/'ROOT_READY_HANDOFF.json')
assert sha(ATTEMPT/'DELIVERY_FILES_SHA256.json') == '6c79c9761ad152a9a354dcc9fdbf00137a6221182cec423bf91b33a02cfd974e'
for name, row in read(ATTEMPT/'DELIVERY_FILES_SHA256.json')['files'].items():
    p = ATTEMPT/name
    assert p.resolve().is_relative_to(ATTEMPT.resolve())
    assert sha(p) == row['sha256'] and p.stat().st_size == row['bytes']
assert sha(BASE/'FILES_SHA256.json') == 'f6de59a56de25a8d316cf7e05c44eafc9751041757e4dc61c88f7bccfd988472'
for name, row in read(BASE/'FILES_SHA256.json')['files'].items():
    assert sha(BASE/name) == row['sha256']

latest_path = BASE/'LATEST_BACKUP.json'
assert sha(latest_path) == handoff['previous_latest_sha256'] == 'f60d4b12f4165e9701cb6ffbe5b773ac09cf46bc783fb44a5a7c8c6849640114'
latest = read(latest_path)
prior_root_path = ROOT/latest['root_adoption_path']
assert sha(prior_root_path) == latest['root_adoption_sha256'] == '5e95872f3216a9842e95c534ba87629b2211c298d72a95d810bdaf9428021051'
prior_path = ROOT/latest['next_collector_previous_path']
assert sha(prior_path) == latest['next_collector_previous_sha256'] == handoff['previous_actual_offserver_sha256'] == 'b508fd4d0f0d36c05e4ea145ff960551d95d415b79db6c19f4d2da95935da5cf'
prior, proof = read(prior_path), read(BATCH/'OFFSERVER_ACCEPTANCE.json')
assert sha(BATCH/'OFFSERVER_ACCEPTANCE.json') == handoff['offserver_acceptance_sha256'] == '31c466412278dd88086df0178f1fd29018f157650284bdfebc5baa14164a13d7'
expected = ['FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91001_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91002_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91003_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91004_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91005_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_IID_Sp-DFA_seed91006_fullcoverage']
assert proof['new_ids'] == handoff['accepted_new_ids'] == expected
assert prior['accepted_total'] == latest['accepted_total'] == 38
assert proof['accepted_new'] == handoff['accepted_new'] == 6
assert proof['accepted_total'] == handoff['accepted_new_cumulative'] == 44
assert proof['accepted_job_ids'] == prior['accepted_job_ids'] + expected
assert not set(expected) & set(prior['accepted_job_ids'])
assert proof['previous_chain_sha256'] == sha(prior_path)
assert (proof['planned_new'], proof['planned_total'], proof['reused_separately']) == (96, 100, 4)
assert proof['package_sha256'] == handoff['source_package_sha256'] == prior['package_sha256'] == '6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
assert proof['status'] == 'PARTIAL_ACCEPTED_OFFSERVER_VERIFIED'
assert proof['original_checked_result_replayed_locally'] and proof['training_runtime_unchanged']
assert proof['source_data_rehashed_on_server_only'] and proof['no_old_models_repackaged']
assert proof['CNN_calls'] == 0 and not proof['final_test']
receipt, inventory = read(BATCH/'BACKUP_SHA256.json'), read(BATCH/'MEMBERS.json')
assert sha(BATCH/'BACKUP_SHA256.json') == handoff['server_backup_receipt_sha256'] == proof['server_backup_receipt_sha256']
assert sha(BATCH/'accepted_delta.tar.gz') == receipt['archive_sha256'] == proof['archive_sha256'] == handoff['archive_sha256']
assert sha(BATCH/'MEMBERS.json') == receipt['inventory_sha256'] == proof['inventory_sha256'] == handoff['inventory_sha256']
with tarfile.open(BATCH/'accepted_delta.tar.gz') as bundle:
    assert len(bundle.getnames()) == len(set(bundle.getnames())) == handoff['archive_members'] == 67
    assert set(bundle.getnames()) == set(inventory['members']) | {'MEMBERS.json'}
    for item in bundle:
        rel = PurePosixPath(item.name)
        assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
        payload = bundle.extractfile(item).read()
        pin = inventory['members'].get(item.name)
        if pin:
            assert len(payload) == pin['size'] and hashlib.sha256(payload).hexdigest() == pin['sha256']
        else:
            assert payload == (BATCH/'MEMBERS.json').read_bytes()
    for identity in expected:
        def load(name): return json.load(bundle.extractfile('runs/'+identity+'/'+name))
        result, job, accept, controller = [load(n) for n in ('result.json', 'job.json', 'acceptance.json', 'state.json')]
        row = next(r for r in proof['records'] if r['id'] == identity)
        assert result['metrics'] == row['metrics'] and result['evaluation_stats'] == row['evaluation_stats']
        assert result['seed'] == job['config']['seed'] == row['seed'] == int(identity.split('_seed')[1].split('_')[0])
        assert result['config'] == job['config']
        assert result['rounds'] == len(result['round_summaries']) == result['round_summaries'][-1]['round'] == controller['round_index'] == 70
        assert (controller['warmup_rounds'], controller['control_width']) == (20, 2.0)
        assert result['config']['celeba_evaluation_split'] == accept['evaluation_split'] == 'valid'
        assert result['config']['client_alpha'] == 5000 and result['config']['learning_rate'] == 0.001
        assert (accept['train_rows'], accept['evaluation_rows']) == (162770, 19867)
        for field, name in [('checkpoint_sha256','model.pt'),('job_sha256','job.json'),('original_acceptance_sha256','acceptance.json')]:
            assert row[field] == inventory['members']['runs/'+identity+'/'+name]['sha256']
        assert row['source_hashes'] == prior['records'][0]['source_hashes']
        assert row['adapter_source_hashes'] == prior['records'][0]['adapter_source_hashes']

review = dict(status='ROOT_FL96_LINKED_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), accepted_before=38, accepted_new=6, accepted_total=44,
    accepted_new_ids=expected, accepted_job_ids=proof['accepted_job_ids'], planned_new=96, reused_separately=4,
    archive_members_verified=67, archive_sha256=sha(BATCH/'accepted_delta.tar.gz'),
    offserver_acceptance_sha256=sha(BATCH/'OFFSERVER_ACCEPTANCE.json'), server_receipt_sha256=sha(BATCH/'BACKUP_SHA256.json'),
    delivery_seal_sha256=sha(ATTEMPT/'DELIVERY_FILES_SHA256.json'), source_package_sha256=proof['package_sha256'],
    previous_root_adoption_path=latest['root_adoption_path'], previous_root_adoption_sha256=sha(prior_root_path),
    previous_offserver_path=latest['next_collector_previous_path'], previous_offserver_sha256=sha(prior_path),
    archive_local_path=(BATCH/"accepted_delta.tar.gz").as_posix(), raw_storage_index_sha256="4f4faceae702b65013ddc378e8d5147a46e3748e9449c0b8649c6507b71f6baf", helper_CPU=110, old_models_repacked=0, root_new_CNN=0, final_test=False, original_checker_replayed_by_offserver_verifier=True,
    original_training_runtime_not_recreated=True, negative_results_preserved=True, not_complete_scenario_or_final_summary=True)
target = ATTEMPT/'ROOT_ADOPTION_REVIEW.json'
with target.open('x', encoding='utf8') as f:
    json.dump(review, f, indent=2); f.write('\n')
next_latest = dict(accepted_total=44, root_adoption_path=target.relative_to(ROOT).as_posix(), root_adoption_sha256=sha(target),
    next_collector_previous_path=(BATCH/'OFFSERVER_ACCEPTANCE.json').as_posix(),
    next_collector_previous_sha256=sha(BATCH/'OFFSERVER_ACCEPTANCE.json'))
assert sha(latest_path) == handoff['previous_latest_sha256']
latest_path.write_text(json.dumps(next_latest, indent=2)+'\n', encoding='utf8')
print(json.dumps(review | {'root_proof_sha256': sha(target)}))
