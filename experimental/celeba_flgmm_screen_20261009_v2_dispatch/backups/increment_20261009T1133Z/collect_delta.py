"""Bounded read-only original-checker acceptance and incremental backup; no training."""
import datetime
import hashlib
import json
import os
from pathlib import Path
import sys
import tarfile
import time
import traceback

BATCH=Path(__file__).resolve().parent
RELEASE=Path('/workspace/guardfed_checks/celeba_flgmm_screen_20261009/release_v2')
HYBRID=Path('/workspace/guardfed_checks/celeba_hybrid_screen_execution_20261009')
REPO=Path('/workspace/GuardFed-celeba-expanded')
sys.path.insert(0,str(RELEASE))
from screen_common import accepted,digest,local_identity,read,repo_identity,write_json

def execute():
    assert digest('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
    assert not (BATCH/'PARTIAL_ACCEPTANCE.json').exists(), 'Existing batch must not be overwritten'
    assert set(os.sched_getaffinity(0))=={106} and os.getpriority(os.PRIO_PROCESS,0)>=10
    resource=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid(): continue
        try:
            argv=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
            if not argv or 'python' not in Path(argv[0]).name: continue
            cpus=os.sched_getaffinity(int(proc.name))
            assert not (len(cpus)<=16 and 106 in cpus), 'Another restricted Python process occupies CPU106'
            resource.append(dict(pid=int(proc.name),argv=argv,affinity=sorted(cpus) if len(cpus)<=16 else dict(count=len(cpus))))
        except (FileNotFoundError,ProcessLookupError,PermissionError): pass
    snapshot=read(BATCH/'live_snapshot.json')
    previous=read(BATCH/'PREVIOUS_CHAIN.json')
    assert digest(BATCH/'PREVIOUS_CHAIN.json')=='61d40a7e91ae4d05f4bafaa3bfcffaca0292a68f58e3fe9d491bd41aa3588b11'
    assert previous['accepted']==2 and len(previous['accepted_job_ids'])==2
    wanted=[row['id'] for row in snapshot['flgmm']['rows'] if row['terminal_acceptance'] and row['progress']['round']==70 and row['id'] not in previous['accepted_job_ids']]
    assert len(wanted)==len(set(wanted))==4, 'Only the frozen snapshot delta is authorized in this batch'
    queue=read(RELEASE/'queue_progress.json')
    assert not set(wanted)&{row['id'] for row in queue['active']}
    assert not list(RELEASE.glob('*FAILURE*.json'))
    protocol,manifest=local_identity()
    assert digest(RELEASE/'PACKAGE_SHA256.json')==previous['package_sha256']==snapshot['flgmm']['source_seal_sha256']
    assert digest(RELEASE/'source/protocol.json')==snapshot['flgmm']['protocol_sha256']
    before=repo_identity(REPO,protocol)
    # Identity-only check for the healthy unfinished Hybrid; no old gate audit or model load.
    hseal=read(HYBRID/'FILES_SHA256.json')
    assert digest(HYBRID/'FILES_SHA256.json')=='2c496ae11369465d27ed223f8552321ec8cd424e5fd87e9f2d34e5ea8531e06f'
    for name,expected in hseal['files'].items(): assert digest(HYBRID/name)==expected, 'Hybrid frozen source drift: '+name
    assert digest(HYBRID/'screen_scope.json')=='d76d5fdff375c58b3b42354fc4256e19b26b29f3fdb4387530f0ae8617920f94'
    assert not (HYBRID/'screen_failure.json').exists()
    records=[]; files={}
    for identity in wanted:
        item=next(row for row in manifest['jobs'] if row['id']==identity)
        out=RELEASE/'runs'/identity
        # Producers must be gone, even when the coordinator retains idle log descriptors.
        assert not any('--job-id' in row['argv'] and identity in row['argv'] for row in resource)
        result=accepted(item,out)
        assert result is not None and result['rounds']==70
        assert read(out/'progress.json')['round']==70
        assert result['data_contract']['root_clean_rows']==16277
        assert result['config']['client_alpha']==protocol['distributions'][result['distribution']]
        assert read(out/'job.json')==result['revision_job']
        assert result['data_contract']['image_data_contract']['numerical_execution']['deterministic_algorithms']
        for path in sorted(out.iterdir()):
            assert path.is_file() and not path.name.endswith('.tmp'), 'Unexpected complete output entry'
            files[str(path.relative_to(RELEASE))]=dict(sha256=digest(path),size=path.stat().st_size)
        log=RELEASE/'logs'/(identity+'.log')
        files[str(log.relative_to(RELEASE))]=dict(sha256=digest(log),size=log.stat().st_size)
        records.append(dict(id=identity,candidate=result['tuning_candidate'],seed=result['seed'],
            distribution=result['distribution'],attack=result['attack'],rounds=result['rounds'],metrics=result['metrics'],
            evaluation_stats=result['evaluation_stats'],checkpoint_sha256=digest(out/'model.pt'),
            original_acceptance_sha256=digest(out/'acceptance.json'),screen_identity_sha256=digest(out/'screen_identity.json'),
            job_sha256=item['job_sha256'],source_hashes=result['revision_job']['source_hashes'],
            adapter_source_hashes=result['revision_job']['adapter_source_hashes']))
    after=repo_identity(REPO,protocol); assert before==after
    local_identity()
    acceptance=dict(status='PARTIAL_ACCEPTED_OFFSERVER_PENDING',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        snapshot_utc=snapshot['utc'],snapshot_sha256=digest(BATCH/'live_snapshot.json'),
        accepted_new_ids=wanted,accepted_new=4,accepted_total_including_previous=6,planned=32,complete=False,
        previous_chain_sha256=digest(BATCH/'PREVIOUS_CHAIN.json'),previous_ids_reused_without_recheck=previous['accepted_job_ids'],
        package_sha256=digest(RELEASE/'PACKAGE_SHA256.json'),protocol_sha256=digest(RELEASE/'source/protocol.json'),
        original_checker_sha256=digest(RELEASE/'source/accept_result.py'),original_screen_common_sha256=digest(RELEASE/'screen_common.py'),
        collector_sha256=digest(__file__),records=records,before_source_data=before,after_source_data=after,
        helper_cpu=106,helper_threads=1,helper_nice=os.getpriority(os.PRIO_PROCESS,0),
        no_training_or_inference=True,no_old_models_repackaged=True,candidate_selection_performed=False,
        source_host=os.uname().nodename,final_test=False,formal100=False,
        logs='Original job producers absent; artifact and closed log bytes verified before/after archive',
        hybrid=dict(terminal_at_snapshot=0,active_round_at_snapshot=48,frozen_source_members_verified=len(hseal['files'])))
    write_json(BATCH/'PARTIAL_ACCEPTANCE.json',acceptance)
    for name in ('PARTIAL_ACCEPTANCE.json','live_snapshot.json','PREVIOUS_CHAIN.json','collect_delta.py'):
        files[name]=dict(sha256=digest(BATCH/name),size=(BATCH/name).stat().st_size)
    write_json(BATCH/'MEMBERS.json',dict(members=files))
    archive=BATCH/'accepted_delta_four.tar.gz'
    with tarfile.open(archive,'w:gz') as tar:
        for name in files: tar.add(RELEASE/name if name.startswith(('runs/','logs/')) else BATCH/name,arcname=name)
        tar.add(BATCH/'MEMBERS.json',arcname='MEMBERS.json')
    for name,expected in files.items():
        source=RELEASE/name if name.startswith(('runs/','logs/')) else BATCH/name
        assert digest(source)==expected['sha256'] and source.stat().st_size==expected['size'], 'Artifact changed while archiving'
    with tarfile.open(archive) as tar:
        assert len(tar.getnames())==len(set(tar.getnames())) and set(tar.getnames())==set(files)|{'MEMBERS.json'}
        for member in tar:
            value=tar.extractfile(member).read()
            expected=dict(sha256=digest(BATCH/'MEMBERS.json'),size=(BATCH/'MEMBERS.json').stat().st_size) if member.name=='MEMBERS.json' else files[member.name]
            assert hashlib.sha256(value).hexdigest()==expected['sha256'] and len(value)==expected['size']
    receipt=dict(status='PARTIAL_ACCEPTED_SERVER_ARCHIVE_VERIFIED_OFFSERVER_PENDING',archive_sha256=digest(archive),
        archive_size=archive.stat().st_size,inventory_sha256=digest(BATCH/'MEMBERS.json'),acceptance_sha256=digest(BATCH/'PARTIAL_ACCEPTANCE.json'),
        archived_member_count=len(files)+1,accepted_new_ids=wanted,accepted_new=4,accepted_total=6,planned=32,
        previous_chain_sha256=digest(BATCH/'PREVIOUS_CHAIN.json'),package_sha256=digest(RELEASE/'PACKAGE_SHA256.json'))
    write_json(BATCH/'BACKUP_SHA256.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':
    try: execute()
    except BaseException as error:
        write_json(BATCH/('BACKUP_FAILURE_'+str(time.time_ns())+'.json'),dict(error=repr(error),traceback=traceback.format_exc()))
        raise
