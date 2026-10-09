"""Bind the first actual FL96 backup to its sealed delivery and original strict result."""
from pathlib import Path, PurePosixPath
import datetime, hashlib, json, tarfile

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'tmp/celeba_flgmm_fullcoverage_incremental_20261009'
ATTEMPT=BASE/'attempt_20261009T210943751719Z';BATCH=ATTEMPT/'batch'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert sha(BASE/'FILES_SHA256.json')=='f6de59a56de25a8d316cf7e05c44eafc9751041757e4dc61c88f7bccfd988472'
for name,row in read(BASE/'FILES_SHA256.json')['files'].items():assert sha(BASE/name)==row['sha256']
assert sha(ATTEMPT/'DELIVERY_FILES_SHA256.json')=='9ff37ebf329e303ad5bea7b8136100db4b12fe681769d6ccaa91449f53d8b90f'
for name,row in read(ATTEMPT/'DELIVERY_FILES_SHA256.json')['files'].items():
    p=ATTEMPT/name;assert sha(p)==row['sha256'] and p.stat().st_size==row['bytes']
assert sha(BATCH/'OFFSERVER_ACCEPTANCE.json')=='2d4fe6ec3178b3f5325797951e7617fe9e31d54fb258105658f1b241864c5bd7'
assert sha(BATCH/'BACKUP_SHA256.json')=='e52d4c956abb0b01820e2e8421e5dea3d23d1b809dcca31ce766ce74af264674'
proof=read(BATCH/'OFFSERVER_ACCEPTANCE.json');receipt=read(BATCH/'BACKUP_SHA256.json');inventory=read(BATCH/'MEMBERS.json')
expected=['FLGMM_Tg20_L2.0_lr0.001_IID_Benign_seed91003_fullcoverage']
assert proof['status']=='PARTIAL_ACCEPTED_OFFSERVER_VERIFIED'
assert proof['accepted_new']==proof['accepted_total']==receipt['accepted_total']==1
assert proof['accepted_job_ids']==proof['new_ids']==receipt['accepted_new_ids']==expected
assert proof['original_checked_result_replayed_locally'] and proof['training_runtime_unchanged']
assert proof['source_data_rehashed_on_server_only'] and proof['no_old_models_repackaged']
assert proof['CNN_calls']==0 and not proof['final_test']
assert (proof['planned_new'],proof['planned_total'],proof['reused_separately'])==(96,100,4)
assert proof['package_sha256']=='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
assert sha(BATCH/'accepted_delta.tar.gz')==proof['archive_sha256']==receipt['archive_sha256']
assert sha(BATCH/'MEMBERS.json')==proof['inventory_sha256']==receipt['inventory_sha256']
with tarfile.open(BATCH/'accepted_delta.tar.gz') as bundle:
    assert len(bundle.getnames())==len(set(bundle.getnames()))==17
    assert set(bundle.getnames())==set(inventory['members'])|{'MEMBERS.json'}
    for item in bundle:
        rel=PurePosixPath(item.name);assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
        payload=bundle.extractfile(item).read()
        if item.name in inventory['members']:
            row=inventory['members'][item.name];assert len(payload)==row['size'] and hashlib.sha256(payload).hexdigest()==row['sha256']
    def load(name):return json.load(bundle.extractfile('runs/'+expected[0]+'/'+name))
    result=load('result.json');job=load('job.json');accept=load('acceptance.json');controller=load('state.json')
    row=proof['records'][0]
    assert result['metrics']==row['metrics'] and result['evaluation_stats']==row['evaluation_stats']
    assert result['seed']==job['config']['seed']==row['seed']==91003 and result['config']==job['config']
    assert result['rounds']==len(result['round_summaries'])==result['round_summaries'][-1]['round']==controller['round_index']==70
    assert (controller['warmup_rounds'],controller['control_width'])==(20,2.0)
    assert result['config']['celeba_evaluation_split']==accept['evaluation_split']=='valid'
    assert result['config']['client_alpha']==5000 and result['config']['learning_rate']==0.001
    assert (accept['train_rows'],accept['evaluation_rows'])==(162770,19867)
    assert row['checkpoint_sha256']==inventory['members']['runs/'+expected[0]+'/model.pt']['sha256']
    assert row['job_sha256']==inventory['members']['runs/'+expected[0]+'/job.json']['sha256']
    assert row['original_acceptance_sha256']==inventory['members']['runs/'+expected[0]+'/acceptance.json']['sha256']
review=dict(status='ROOT_FL96_FIRST_DELTA_ARCHIVE_SOURCE_CHECKPOINT_AND_ORIGINAL_STRICT_BINDING_PASS',
    checked_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),accepted_new=1,accepted_total=1,
    accepted_new_ids=expected,planned_new=96,reused_separately=4,archive_members_verified=17,
    archive_sha256=sha(BATCH/'accepted_delta.tar.gz'),offserver_acceptance_sha256=sha(BATCH/'OFFSERVER_ACCEPTANCE.json'),
    server_receipt_sha256=sha(BATCH/'BACKUP_SHA256.json'),delivery_seal_sha256=sha(ATTEMPT/'DELIVERY_FILES_SHA256.json'),
    source_package_sha256=proof['package_sha256'],old_models_repacked=0,root_new_CNN=0,final_test=False,
    original_checker_replayed_by_offserver_verifier=True,original_training_runtime_not_recreated=True,
    negative_results_preserved=True,not_complete_scenario_or_final_summary=True)
target=ATTEMPT/'ROOT_ADOPTION_REVIEW.json'
with target.open('x',encoding='utf8') as f:json.dump(review,f,indent=2);f.write('\n')
latest=dict(accepted_total=1,root_adoption_path=target.relative_to(ROOT).as_posix(),root_adoption_sha256=sha(target),
    next_collector_previous_path=(BATCH/'OFFSERVER_ACCEPTANCE.json').relative_to(ROOT).as_posix(),
    next_collector_previous_sha256=sha(BATCH/'OFFSERVER_ACCEPTANCE.json'))
with (BASE/'LATEST_BACKUP.json').open('x',encoding='utf8') as f:json.dump(latest,f,indent=2);f.write('\n')
print(json.dumps(review|{'root_proof_sha256':sha(target)}))
