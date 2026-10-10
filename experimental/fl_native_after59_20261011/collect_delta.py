"""Bounded read-only original-checker acceptance and incremental backup; no training."""
import argparse
import shutil
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
RELEASE=Path('/workspace/guardfed_checks/celeba_flgmm_fullcoverage_v2_20261009/stage')
PACKAGE='6200c40f22a8ea670cfa02f0e74bbd81dc4ef9adecc258f979ddacc078a5d230'
HELPER=Path(__file__).resolve().parent
CPU=None
if sys.flags.optimize:raise RuntimeError('Optimized Python forbidden')
REPO=Path('/workspace/GuardFed-celeba-expanded')

def execute():
    assert digest('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa'
    assert not (BATCH/'PARTIAL_ACCEPTANCE.json').exists(), 'Existing batch must not be overwritten'
    assert set(os.sched_getaffinity(0))=={CPU} and os.getpriority(os.PRIO_PROCESS,0)>=10
    resource=[];nominal_threads=1
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid(): continue
        try:
            argv=[v.decode(errors='replace') for v in (proc/'cmdline').read_bytes().split(b'\0') if v]
            if not argv or 'python' not in Path(argv[0]).name: continue
            cpus=os.sched_getaffinity(int(proc.name))
            assert not (len(cpus)<=16 and CPU in cpus), 'Another restricted Python process occupies selected collector CPU'
            env=dict(x.split('=',1) for x in (proc/'environ').read_text().split('\0') if '=' in x)
            nominal_threads+=max([int(env[k]) for k in ('GUARDFED_CPU_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS') if env.get(k,'').isdigit()] or [1])
            resource.append(dict(pid=int(proc.name),argv=argv,affinity=sorted(cpus) if len(cpus)<=16 else dict(count=len(cpus))))
        except (FileNotFoundError,ProcessLookupError,PermissionError): pass
    assert 'idle' in __import__('subprocess').check_output(['ionice','-p',str(os.getpid())],text=True)
    quota,period=Path('/sys/fs/cgroup/cpu.max').read_text().split()
    assert quota!='max' and nominal_threads<=int(quota)/int(period)
    protocol,manifest=local_identity()
    assert digest(RELEASE/'PACKAGE_SHA256.json')==PACKAGE
    assert len(manifest['jobs'])==96 and len(manifest['reused_jobs'])==4
    queue_snapshot=read(RELEASE/'queue_progress.json')
    assert not queue_snapshot['failed'], 'Frozen queue failure forbids collection'
    active={row['id'] for row in queue_snapshot['active']}
    rows=[]
    for item in manifest['jobs']:
        out=RELEASE/'runs'/item['id']
        progress=read(out/'progress.json') if (out/'progress.json').exists() else None
        rows.append(dict(id=item['id'],active=item['id'] in active,progress=progress,terminal_acceptance=all((out/name).exists() for name in ['result.json','acceptance.json','screen_identity.json']) and item['id'] not in active))
    snapshot=dict(utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),queue=queue_snapshot,flgmm=dict(rows=rows,source_seal_sha256=digest(RELEASE/'PACKAGE_SHA256.json'),protocol_sha256=digest(RELEASE/'source/protocol.json')))
    write_json(BATCH/'live_snapshot.json',snapshot)
    previous=read(BATCH/'PREVIOUS_CHAIN.json')
    assert previous['package_sha256']==PACKAGE
    assert len(previous['accepted_job_ids'])==len(set(previous['accepted_job_ids']))
    assert set(previous['accepted_job_ids'])<={x['id'] for x in manifest['jobs']}
    prior_count=len(previous['accepted_job_ids'])
    wanted=[row['id'] for row in snapshot['flgmm']['rows'] if row['terminal_acceptance'] and row['progress']['round']==70 and row['id'] not in previous['accepted_job_ids']]
    # Parent-authorized fixed snapshot exact4 only; later terminal IDs remain unaccepted.
    authorized_ids=['FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91003_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91004_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91005_fullcoverage', 'FLGMM_Tg20_L2.0_lr0.001_non-IID_F Flip_seed91006_fullcoverage']
    wanted=[identity for identity in wanted if identity in authorized_ids]
    assert wanted==authorized_ids, 'All four authorized outputs must be terminal, unique and ordered'
    if not wanted:
        write_json(BATCH/'NO_NEW_TERMINAL.json',dict(status='NO_NEW_TERMINAL_IN_THIS_SNAPSHOT',snapshot_sha256=digest(BATCH/'live_snapshot.json'),prior_accepted_new=prior_count,backup_created=False));return
    assert len(wanted)==len(set(wanted)) and not set(wanted)&set(previous['accepted_job_ids']), 'Only this single fresh snapshot delta beyond the pinned offserver chain is authorized'
    queue=read(RELEASE/'queue_progress.json')
    assert not queue['failed'], 'Queue failure forbids collection'
    assert not set(wanted)&{row['id'] for row in queue['active']}
    assert not list(RELEASE.glob('*FAILURE*.json'))
    protocol,manifest=local_identity()
    assert digest(RELEASE/'PACKAGE_SHA256.json')==previous['package_sha256']==snapshot['flgmm']['source_seal_sha256']
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
        accepted_new_ids=wanted,accepted_new=len(wanted),accepted_total_including_previous=prior_count+len(wanted),accepted_job_ids=previous['accepted_job_ids']+wanted,planned_new=96,planned_total=100,reused_separately=4,complete=False,
        previous_chain_sha256=digest(BATCH/'PREVIOUS_CHAIN.json'),previous_ids_reused_without_recheck=previous['accepted_job_ids'],
        package_sha256=digest(RELEASE/'PACKAGE_SHA256.json'),protocol_sha256=digest(RELEASE/'source/protocol.json'),
        original_checker_sha256=digest(RELEASE/'source/accept_result.py'),original_screen_common_sha256=digest(RELEASE/'screen_common.py'),
        collector_sha256=digest(__file__),records=records,before_source_data=before,after_source_data=after,
        helper_cpu=CPU,helper_threads=1,helper_nice=os.getpriority(os.PRIO_PROCESS,0),
        no_training_or_inference=True,no_old_models_repackaged=True,metadata_archive_sha256='9d6ea7126224343dc08bb08113181de6edbc19111d4a15b335b81fc72e5be24c',candidate_selection_performed=False,
        source_host=os.uname().nodename,final_test=False,scope='96_new_70round_valid_only',
        logs='Original job producers absent; artifact and closed log bytes verified before/after archive',
        acceptance_runtime=dict(python=sys.version,executable=sys.executable,torch=__import__('torch').__version__,cuda_initialized=__import__('torch').cuda.is_initialized()),canonical_ledger_not_modified=True)
    write_json(BATCH/'PARTIAL_ACCEPTANCE.json',acceptance)
    for name in ('collect_delta.py','verify_delta_offserver.py'):
        shutil.copyfile(HELPER/name,BATCH/name)
    for name in ('PARTIAL_ACCEPTANCE.json','live_snapshot.json','PREVIOUS_CHAIN.json','INPUT_BINDING.json','collect_delta.py','verify_delta_offserver.py'):
        files[name]=dict(sha256=digest(BATCH/name),size=(BATCH/name).stat().st_size)
    write_json(BATCH/'MEMBERS.json',dict(members=files))
    archive=BATCH/'accepted_delta.tar.gz'
    with tarfile.open(archive,'x:gz') as tar:
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
        archived_member_count=len(files)+1,accepted_new_ids=wanted,accepted_new=len(wanted),accepted_total=prior_count+len(wanted),planned_new=96,planned_total=100,reused_separately=4,
        previous_chain_sha256=digest(BATCH/'PREVIOUS_CHAIN.json'),package_sha256=digest(RELEASE/'PACKAGE_SHA256.json'))
    write_json(BATCH/'BACKUP_SHA256.json',receipt)
    print(json.dumps(receipt))

def main():
    global BATCH,CPU,accepted,digest,local_identity,read,repo_identity,write_json
    parser=argparse.ArgumentParser();parser.add_argument('--out',type=Path,required=True);parser.add_argument('--cpu',type=int,required=True)
    group=parser.add_mutually_exclusive_group(required=True);group.add_argument('--previous',type=Path);group.add_argument('--initial-empty',action='store_true')
    parser.add_argument('--previous-sha256');parser.add_argument('--source-sha256',required=True);args=parser.parse_args()
    assert hashlib.sha256(Path(__file__).read_bytes()).hexdigest()==args.source_sha256
    assert not args.out.exists() and not args.out.is_symlink() and args.out.parent.resolve()==args.out.parent
    CPU=args.cpu;assert CPU not in (102,103,104,112,113,114,115,116,117,118,119)
    assert os.sched_getaffinity(0)=={CPU} and os.getpriority(os.PRIO_PROCESS,0)>=10
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='' and all(os.environ.get(k)=='1' for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'))
    if args.previous:
        assert args.previous_sha256 and hashlib.sha256(args.previous.read_bytes()).hexdigest()==args.previous_sha256
        previous=json.loads(args.previous.read_bytes());assert previous['status']=='PARTIAL_ACCEPTED_OFFSERVER_VERIFIED' and previous['package_sha256']==PACKAGE
        prior_bytes=args.previous.read_bytes()
    else:
        assert args.previous_sha256 is None
        prior_bytes=json.dumps(dict(status='EXPLICIT_INITIAL_EMPTY_NEW_RESULT_SET_NOT_BACKUP',package_sha256=PACKAGE,accepted_job_ids=[])).encode()
    BATCH=args.out;BATCH.mkdir()
    (BATCH/'PREVIOUS_CHAIN.json').write_bytes(prior_bytes)
    sys.path.insert(0,str(RELEASE))
    from screen_common import accepted,digest,local_identity,read,repo_identity,write_json
    write_json(BATCH/'INPUT_BINDING.json',dict(previous_sha256=digest(BATCH/'PREVIOUS_CHAIN.json'),collector_sha256=digest(__file__),initial_empty=args.initial_empty,cpu=CPU))
    try: execute()
    except BaseException as error:
        write_json(BATCH/('BACKUP_FAILURE_'+str(time.time_ns())+'.json'),dict(error=repr(error),traceback=traceback.format_exc()))
        raise

if __name__=='__main__':main()
