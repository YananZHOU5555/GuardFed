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
    assert 'idle' in __import__('subprocess').check_output(['ionice','-p',str(os.getpid())],text=True)
    protocol,manifest=local_identity()
    queue_snapshot=read(RELEASE/'queue_progress.json')
    active={row['id'] for row in queue_snapshot['active']}
    rows=[]
    for item in manifest['jobs']:
        out=RELEASE/'runs'/item['id']
        progress=read(out/'progress.json') if (out/'progress.json').exists() else None
        rows.append(dict(id=item['id'],active=item['id'] in active,progress=progress,terminal_acceptance=all((out/name).exists() for name in ['result.json','acceptance.json','screen_identity.json']) and item['id'] not in active))
    snapshot=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=queue_snapshot,flgmm=dict(rows=rows,source_seal_sha256=digest(RELEASE/'PACKAGE_SHA256.json'),protocol_sha256=digest(RELEASE/'source/protocol.json')))
    write_json(BATCH/'live_snapshot.json',snapshot)
    previous=read(BATCH/'PREVIOUS_CHAIN.json')
    assert digest(BATCH/'PREVIOUS_CHAIN.json')=='84be50b9c7dc44ceb3bf34e1bedd0c3ce36e5186b9020e1d6935b9cdc0c95dcd'
    assert previous['accepted_total']==19 and len(previous['accepted_job_ids'])==19
    assert digest(BATCH/'AUTHORIZED_SNAPSHOT.json')=='f87eb797aef52016d1e9a77de871b5cd4942474545c2e628184f993a2ebfc404'
    wanted=['FLGMM_Tg20_L2.0_lr0.0005_non-IID_S-DFA_seed91001_screen', 'FLGMM_Tg20_L2.0_lr0.001_IID_Benign_seed91001_screen', 'FLGMM_Tg20_L2.0_lr0.001_IID_S-DFA_seed91001_screen', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_Benign_seed91001_screen', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_S-DFA_seed91001_screen', 'FLGMM_Tg20_L3.0_lr0.0005_IID_Benign_seed91001_screen', 'FLGMM_Tg20_L3.0_lr0.0005_IID_S-DFA_seed91001_screen']
    assert read(BATCH/'EXACT_DELTA.json')['selected_ids']==wanted
    terminals={row['id'] for row in snapshot['flgmm']['rows'] if row['terminal_acceptance'] and row['progress']['round']==70}
    assert set(wanted)<=terminals and not set(wanted)&set(previous['accepted_job_ids'])
    queue=read(RELEASE/'queue_progress.json')
    assert not set(wanted)&{row['id'] for row in queue['active']}
    assert not list(RELEASE.glob('*FAILURE*.json'))
    protocol,manifest=local_identity()
    assert digest(RELEASE/'PACKAGE_SHA256.json')==snapshot['flgmm']['source_seal_sha256']==read(BATCH/'AUTHORIZED_SNAPSHOT.json')['source_sha256']['PACKAGE_SHA256.json']=='aec95ceb5e8c9b7aa9cec89e9d70d2700c2f242c29e918792088648d6269bad4'
    assert digest(RELEASE/'source/protocol.json')==snapshot['flgmm']['protocol_sha256']
    before=repo_identity(REPO,protocol)
    records=[]; files={}
    for identity in wanted:
        item=next(row for row in manifest['jobs'] if row['id']==identity)
        out=RELEASE/'runs'/identity
        # Producers must be gone, even when the coordinator retains idle log descriptors.
        assert not any('--job-id' in row['argv'] and identity in row['argv'] for row in resource)
        assert not Path('/proc',str(read(out/'progress.json')['pid'])).exists(), 'Original producer PID still exists'
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
        accepted_new_ids=wanted,accepted_new=len(wanted),accepted_total_including_previous=19+len(wanted),planned=32,complete=False,
        previous_chain_sha256=digest(BATCH/'PREVIOUS_CHAIN.json'),previous_ids_reused_without_recheck=previous['accepted_job_ids'],
        package_sha256=digest(RELEASE/'PACKAGE_SHA256.json'),protocol_sha256=digest(RELEASE/'source/protocol.json'),
        original_checker_sha256=digest(RELEASE/'source/accept_result.py'),original_screen_common_sha256=digest(RELEASE/'screen_common.py'),
        collector_sha256=digest(__file__),records=records,before_source_data=before,after_source_data=after,
        helper_cpu=106,helper_threads=1,helper_nice=os.getpriority(os.PRIO_PROCESS,0),
        no_training_or_inference=True,no_old_models_repackaged=True,candidate_selection_performed=False,
        source_host=os.uname().nodename,final_test=False,formal100=False,
        logs='Original job producers absent; artifact and closed log bytes verified before/after archive',
        acceptance_runtime=dict(python=sys.version,executable=sys.executable,torch=__import__('torch').__version__,cuda_initialized=__import__('torch').cuda.is_initialized()),canonical_ledger_not_modified=True)
    write_json(BATCH/'PARTIAL_ACCEPTANCE.json',acceptance)
    for name in ('PARTIAL_ACCEPTANCE.json','live_snapshot.json','PREVIOUS_CHAIN.json','collect_delta.py','COLLECTOR_DIFF.patch','SOURCE_RECEIPT.json','EXACT_DELTA.json','AUTHORIZED_SNAPSHOT.json','PREVIOUS_LATEST.json'):
        files[name]=dict(sha256=digest(BATCH/name),size=(BATCH/name).stat().st_size)
    write_json(BATCH/'MEMBERS.json',dict(members=files))
    archive=BATCH/'accepted_delta_after19_v2.tar.gz'
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
        archived_member_count=len(files)+1,accepted_new_ids=wanted,accepted_new=len(wanted),accepted_total=19+len(wanted),planned=32,
        previous_chain_sha256=digest(BATCH/'PREVIOUS_CHAIN.json'),package_sha256=digest(RELEASE/'PACKAGE_SHA256.json'))
    write_json(BATCH/'BACKUP_SHA256.json',receipt)
    print(json.dumps(receipt))

if __name__=='__main__':
    try: execute()
    except BaseException as error:
        write_json(BATCH/('BACKUP_FAILURE_'+str(time.time_ns())+'.json'),dict(error=repr(error),traceback=traceback.format_exc()))
        raise
