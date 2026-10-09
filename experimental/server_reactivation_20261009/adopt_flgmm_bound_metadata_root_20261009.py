"""Finish local metadata verification after a Python3.10 extraction API failure; no SSH."""
from pathlib import Path, PurePosixPath
import datetime
import hashlib
import json
import tarfile

ROOT=Path(__file__).resolve().parents[1]
ATTEMPT=ROOT/'tmp/celeba_flgmm_fullcoverage_root_operations_20261009/attempt_20261009T200518912319Z'
OUT=ATTEMPT/'verified_manual_v2'
PROOF=ROOT/'tmp/celeba_flgmm_fullcoverage_binding_20261009/ROOT_BOUND_ADOPTION.json'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
read=lambda p:json.loads(p.read_bytes())
assert not OUT.exists() and not PROOF.exists()
assert read(ATTEMPT/'OPERATION_FAILURE.json')['error']=="TypeError(\"TarFile.extractall() got an unexpected keyword argument 'filter'\")"
assert not any((ATTEMPT/'verified').iterdir())
command=json.loads(read(ATTEMPT/'REMOTE_BIND.json')['stdout'])
assert command['returncode']==0
receipt=read(ATTEMPT/'BOUND_BACKUP.json')
assert json.loads(command['stdout'])==receipt
assert receipt['package_sha256']=='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
archive=ATTEMPT/'bound_metadata.tar.gz'
assert sha(archive)==receipt['archive_sha256']=='9d6ea7126224343dc08bb08113181de6edbc19111d4a15b335b81fc72e5be24c'
assert sha(ATTEMPT/'BOUND_MEMBERS.json')==receipt['inventory_sha256']
members=read(ATTEMPT/'BOUND_MEMBERS.json')['members']
with tarfile.open(archive) as bundle:
    assert len(bundle.getnames())==len(set(bundle.getnames()))==receipt['members']==144
    assert set(bundle.getnames())==set(members)|{'BOUND_MEMBERS.json'}
    payloads={}
    for item in bundle:
        rel=PurePosixPath(item.name)
        assert item.isfile() and not rel.is_absolute() and '..' not in rel.parts
        assert (OUT/item.name).resolve().is_relative_to(OUT.resolve())
        data=bundle.extractfile(item).read()
        pin=members.get(item.name,dict(sha256=receipt['inventory_sha256'],bytes=(ATTEMPT/'BOUND_MEMBERS.json').stat().st_size))
        assert len(data)==pin['bytes'] and hashlib.sha256(data).hexdigest()==pin['sha256']
        payloads[item.name]=data
OUT.mkdir()
for name,data in payloads.items():
    path=OUT/name;path.parent.mkdir(exist_ok=True,parents=True)
    with path.open('xb') as stream:stream.write(data)
stage=OUT/'stage';package=read(stage/'PACKAGE_SHA256.json')
assert sha(stage/'PACKAGE_SHA256.json')==receipt['package_sha256']
for name,digest in package['files'].items():assert sha(stage/name)==digest
manifest=read(stage/'manifest.json');protocol=read(stage/'source/protocol.json')
assert (len(manifest['jobs']),len(manifest['reused_jobs']),len(manifest['preflight_jobs']))==(96,4,5)
adopt=read(ROOT/'tmp/celeba_flgmm_final6_closure_20261009/ROOT_SUMMARY_ADOPTION.json')
assert protocol['selected_recipe']==adopt['selected_recipe'] and protocol['status']=='FROZEN'
assert protocol['execution']['status']=='NOT_AUTHORIZED_TO_START' and not manifest['execution_authorized']
cells=set()
for item in manifest['jobs']+manifest['preflight_jobs']:
    job=read(stage/'jobs'/item['job']);cfg=job['config']
    assert sha(stage/'jobs'/item['job'])==item['job_sha256']
    assert job['adapter']==adopt['selected_recipe']['adapter'] and cfg['learning_rate']==adopt['selected_recipe']['learning_rate']
    assert job['source_hashes']==protocol['source_hashes']
    assert cfg['celeba_evaluation_split']=='valid' and cfg['celeba_train_limit']==cfg['celeba_eval_limit']==0
    assert cfg['client_alpha']==(5000.0 if job['distribution']=='IID' else 5.0)
    assert cfg['batch_size']==64 and cfg['local_epochs']==1 and cfg['optimizer']=='adam' and not cfg['ad2_calibration_enabled']
    assert cfg['rounds']==(70 if job['phase']=='fullcoverage' else 3)
    for name,digest in job['adapter_source_hashes'].items():assert sha(stage/'source'/name)==digest
    if job['phase']=='fullcoverage':
        key=(job['distribution'],job['attack'],cfg['seed']);assert key not in cells;cells.add(key)
for item in manifest['reused_jobs']:
    row=item['accepted_record'];key=(row['distribution'],row['attack'],row['seed'])
    assert key not in cells;cells.add(key)
    assert row in read(stage/'SELECTED_SUMMARY.json')['records'] and row['candidate']==adopt['selected_recipe']['id']
    assert row['rounds']==70 and row['seed']==91001
    assert item['legacy_release']=='/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2'
assert cells=={(d,a,s) for d in ('IID','non-IID') for a in ('Benign','F Flip','FedSA','S-DFA','Sp-DFA') for s in range(91001,91011)}
identity=read(OUT/'BOUND_IDENTITY.json')
assert identity['status']=='BOUND_METADATA_AND_ORIGINAL4_CHECKED_NO_EXECUTION'
assert identity['package_sha256']==receipt['package_sha256'] and identity['source_data_before']==identity['source_data_after']==protocol['source_hashes']
assert len(identity['source_data_before'])==21 and len(identity['exact_six_targets'])==6
assert len(identity['strict_reused_records'])==4 and not identity['run_canaries'] and not identity['new_training'] and not identity['test']
assert not identity['services_changed']
proof=dict(status='ROOT_ACTUAL_BOUND96_PLUS4_METADATA_MEMBER_AND_SCOPE_PASS',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    attempt_path=ATTEMPT.relative_to(ROOT).as_posix(),verified_path=OUT.relative_to(ROOT).as_posix(),
    package_sha256=receipt['package_sha256'],archive_sha256=sha(archive),archive_members_verified=144,
    receipt_sha256=sha(ATTEMPT/'BOUND_BACKUP.json'),bound_identity_sha256=sha(OUT/'BOUND_IDENTITY.json'),
    manifest_sha256=sha(stage/'manifest.json'),protocol_sha256=sha(stage/'source/protocol.json'),
    source_seal_sha256=identity['source_seal_sha256'],source_review_sha256=identity['source_review_sha256'],
    selected_recipe=adopt['selected_recipe'],planned_new=96,reused=4,planned_total=100,prepared_new_canaries=5,original_references=2,
    original_four_strict_checked_on_server=True,local_original_four_tensor_recheck=False,
    local_extract_failure_preserved=True,local_repair='Exact144 member SHA checked, then safe new-path per-file write; Python3.10-compatible',
    repeated_remote_binding=0,checkpoint_repacked=0,new_CNN_inference=0,canaries_started=0,formal100_started=False,final_test=False)
with PROOF.open('x',encoding='utf8',newline='\n') as stream:json.dump(proof,stream,indent=2);stream.write('\n')
print(json.dumps(dict(status=proof['status'],proof_sha256=sha(PROOF),package_sha256=proof['package_sha256'],new=96,reused=4,canaries_started=0)))
