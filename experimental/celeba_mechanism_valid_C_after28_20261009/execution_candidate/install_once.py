"""One-shot, reviewed8 CPU replay installation; no restart/retry path."""
import argparse,hashlib,json,math,os,shutil,socket,subprocess,sys,time,traceback
from pathlib import Path
from datetime import datetime,timezone
HERE=Path(__file__).resolve().parent
sys.dont_write_bytecode=True
sys.path.insert(0,str(HERE))
import batch
SERVICE='guardfed_celeba_mechanism_valid_C_after28'

# These three newer roles are not counted by the unchanged legacy classifier.
RESERVATION = dict(recovery_GPU_worker=1, recovery_coordinator=1, Hybrid_GPU=1)


def prior_completed_service_stopped():
    service = 'guardfed_celeba_mechanism_valid_C_after25'
    original_batch = '/workspace/guardfed_checks/celeba_mechanism_valid_C_after25_20261009/execution_candidate/batch.py'
    result = subprocess.run(['supervisorctl', 'status', service], capture_output=True, text=True)
    words = result.stdout.split()
    batch.require(len(words) >= 2 and words[0] == service and words[1] == 'EXITED',
        'Previously accepted C after25 service must be EXITED: ' + result.stdout + result.stderr)
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit(): continue
        try:
            argv = [s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
            batch.require(original_batch not in argv, 'Previously accepted C after25 worker remains: ' + proc.name)
        except (FileNotFoundError, ProcessLookupError): continue
    return dict(service=service, status=result.stdout.strip(), supervisor_returncode=result.returncode, original_batch_processes=0,
        prior_success_root_adoption_sha256='0d1661bdc21025b957fa4e6ac39d4c8924cb50ab1b91920268119941312a615a')


def budget_snapshot(snapshot):
    total=snapshot['total_effective_nominal'];quota=snapshot['actual_quota_cores']
    batch.require(isinstance(total,(int,float)) and not isinstance(total,bool) and math.isfinite(total) and total>=0
        and isinstance(quota,(int,float)) and not isinstance(quota,bool) and math.isfinite(quota) and quota>0, 'Invalid actual quota/reservation')
    batch.require(total+sum(RESERVATION.values())<=quota, 'Conservative legacy plus three exceeds actual quota')
    return dict(snapshot, conservative_additional_reservations=RESERVATION,
        conservative_total_including_this8=total+sum(RESERVATION.values()),
        additional_three_are_reservation_not_measured_usage=True)


def assert_cpu_available(rows, own_pid):
    occupied=[r for r in rows if r['pid']!=own_pid and len(r['cpus'])<=16 and set(r['cpus']).intersection(batch.CPUS)]
    batch.require(not occupied, 'Restricted owner on CPU112..119: '+repr(occupied))


def fresh_cpu_scan():
    rows=[]
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit() or int(proc.name)==os.getpid():continue
        try:
            argv=[s.decode(errors='replace') for s in (proc/'cmdline').read_bytes().split(b'\0') if s]
            batch.require(str(HERE/'batch.py') not in argv,'Duplicate new batch PID '+proc.name)
            if (proc/'stat').read_text().rsplit(')',1)[1].split()[0]=='Z':continue
            for task in (proc/'task').iterdir():
                try:
                    cpus=sorted(os.sched_getaffinity(int(task.name)))
                    if len(cpus)<=16:
                        rows.append(dict(pid=int(proc.name),tid=int(task.name),cpus=cpus,argv=argv))
                except (FileNotFoundError,ProcessLookupError):continue
        except (FileNotFoundError,ProcessLookupError):continue
    assert_cpu_available(rows,os.getpid())
    return dict(utc=datetime.now(timezone.utc).isoformat(),allowed_cpus=batch.CPUS,
        own_pid_excluded=os.getpid(),restricted_owner_count=0,restricted_tasks_observed=rows)


def main(draft_sha256):
    batch.require(HERE==batch.REMOTE and os.getpriority(os.PRIO_PROCESS,0)>=10,'Wrong isolated path/priority')
    batch.require(batch.digest(HERE/'EXECUTION_DRAFT.json')==draft_sha256,'External root draft SHA mismatch')
    batch.require(batch.digest('/etc/vast-agents-guide.md')=='42be4f7a84349c7bca6f6b35c10e94d70ddeb9239bcdeaf0c56317d4ab3fd2aa','Guide changed')
    previous_completed_service=prior_completed_service_stopped()
    scope=batch.identities()
    for name in ('runs','approvals','logs','batch_resource_before.json','batch_failure.json','batch.lock','compute_worker.lock','preflight.json','start_receipt.json','APPROVED.json','installation_failure.json'):
        batch.require(not (HERE/name).exists(),'Existing output/runtime blocks retry: '+name)
    cpu_before=fresh_cpu_scan()
    from resource_extra import resource_snapshot
    before=budget_snapshot(resource_snapshot())
    rows=batch.read(batch.PREPARED/'inventory_actual136_Full100refs.json')['records']
    chosen=[r for r in rows if r['id'] in batch.SELECTED]
    batch.require(len(chosen)==8,'Wrong exact cohort')
    bridge=batch.load('preflight_sealed_bridge',batch.PREPARED/'bridge.py',batch.BRIDGE_SHA)
    deps=scope['dependency_paths'];pins={}
    for key,sha in [('v2',bridge.V2_SHA),('v3',bridge.V3_SHA),('evaluator',bridge.EVALUATOR_SHA),('evidence_v4',bridge.EVIDENCE_V4_SHA),('baseline_inventory',bridge.BASELINE_INVENTORY_SHA),('manifest',bridge.MANIFEST_SHA)]:pins[deps[key]]=sha
    pins[deps['protocol']]=batch.read(batch.PREPARED/'inventory_actual136_Full100refs.json')['mechanism_protocol_sha256']
    for row in chosen:
        for rel,sha in row['source_hashes'].items():pins[str(batch.REPO/rel)]=sha
        for p,sha in row['adapter_source_hashes'].items():pins[p]=sha
        for p,sha in row['accepted_v4_row']['files'].items():
            batch.require(p not in pins or pins[p]==sha,'Conflicting original artifact SHA');pins[p]=sha
    verified=[]
    for p,sha in pins.items():
        batch.require(batch.digest(p)==sha,'Original source/data/terminal changed: '+p)
        verified.append(dict(path=p,sha256=sha,size=Path(p).stat().st_size,resolved=str(Path(p).resolve())))
    # Same read-only module imports check the exact original inventory contract.
    bridge.validate_inventory(batch.read(batch.PREPARED/'inventory_actual136_Full100refs.json'),batch.read(deps['baseline_inventory']))
    hardware=dict(gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,utilization.gpu,memory.used,temperature.gpu','--format=csv,noheader'],capture_output=True,text=True).stdout,
        recovery=subprocess.run(['nvidia-smi','-q'],capture_output=True,text=True).stdout,
        memory_current=int(Path('/sys/fs/cgroup/memory.current').read_text()),memory_max=Path('/sys/fs/cgroup/memory.max').read_text().strip(),
        memory_events=Path('/sys/fs/cgroup/memory.events').read_text(),disk_free=shutil.disk_usage('/workspace').free)
    batch.require(hardware['disk_free']>10*1024**3,'Less than10GB free')
    batch.require('GPU Recovery Action' in hardware['recovery'] and 'None' in hardware['recovery'],'GPU Recovery unverified')
    after=budget_snapshot(resource_snapshot())
    cpu_after=fresh_cpu_scan()
    script=Path('/opt/supervisor-scripts')/(SERVICE+'.sh');config=Path('/etc/supervisor/conf.d')/(SERVICE+'.conf')
    batch.require(not script.exists() and not config.exists(),'Existing service must not be overwritten')
    approved=batch.read(HERE/'EXECUTION_DRAFT.json');approved['execution_seal_sha256']=batch.digest(HERE/'EXECUTION_SOURCE_SHA256.json')
    approved['preflight_utc']=datetime.now(timezone.utc).isoformat()
    batch.check_approval(approved,scope,batch.digest(batch.PREPARED/'SCOPE.json'),approved['execution_seal_sha256'])
    batch.save_new(HERE/'preflight.json',dict(status='PASS_EXACT37_PREFLIGHT',utc=approved['preflight_utc'],hostname=socket.gethostname(),
        previous_completed_service=previous_completed_service,resources_before=before,resources_after=after,cpu_owners_before=cpu_before,cpu_owners_after=cpu_after,external_draft_sha256=draft_sha256,hardware=hardware,verified_original_members=verified,empty_outputs=True,no_duplicate=True,
        prepared_seal_sha256=batch.OLD_SEAL_SHA,execution_seal_sha256=approved['execution_seal_sha256']))
    batch.save_new(HERE/'APPROVED.json',approved)
    approval_sha=batch.digest(HERE/'APPROVED.json')
    with (HERE/'APPROVED.sha256').open('x') as f:f.write(approval_sha+'\n')
    script.write_bytes((HERE/'service.sh.template').read_bytes());script.chmod(0o755)
    config.write_bytes((HERE/'supervisor.conf.template').read_bytes())
    steps=[]
    for cmd in (['supervisorctl','reread'],['supervisorctl','add',SERVICE],['supervisorctl','start',SERVICE]):
        result=subprocess.run(cmd,capture_output=True,text=True)
        steps.append(dict(command=cmd,returncode=result.returncode,stdout=result.stdout,stderr=result.stderr))
        batch.require(result.returncode==0,'Supervisor installation failed: '+repr(steps[-1]))
    status=subprocess.run(['supervisorctl','status',SERVICE],capture_output=True,text=True).stdout.strip()
    batch.save_new(HERE/'start_receipt.json',dict(status=status,utc=datetime.now(timezone.utc).isoformat(),steps=steps,
        script_sha256=batch.digest(script),config_sha256=batch.digest(config),approval_sha256=approval_sha,
        execution_seal_sha256=approved['execution_seal_sha256'],prepared_seal_sha256=batch.OLD_SEAL_SHA,
        original_members_verified=len(verified),resources=after,compute_threads=8,max_processes=1,selected_ids=batch.SELECTED,
        autostart=False,autorestart=False,startretries=0))
    print(json.dumps(dict(status=status,original_members_verified=len(verified),nominal_threads=after['total_effective_nominal'],approval_sha256=approval_sha)),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--draft-sha256',required=True)
    args=parser.parse_args()
    try:main(args.draft_sha256)
    except BaseException as error:
        if not (HERE/'installation_failure.json').exists():batch.save_new(HERE/'installation_failure.json',dict(error=repr(error),traceback=traceback.format_exc(),automatic_retry=False))
        raise
